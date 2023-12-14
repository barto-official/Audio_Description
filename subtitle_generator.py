#!/usr/bin/env python3

import time
import argparse
import subprocess
import sys
import os
import shutil
import datetime
import json
import csv
from typing import Optional, Tuple, Union

def install_packages(requirements_path):
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_path], check=True)
    print("Installed packages from requirements.txt")

install_packages("requirements.txt")

import whisperx

import librosa
import srt
import soundfile as sf
import transformers
import torch
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.generation.logits_process import WhisperTimeStampLogitsProcessor
from transformers.models.whisper.tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE


class WhisperForAudioCaptioning(transformers.WhisperForConditionalGeneration):

    def forward(
            self,
            input_features: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            forced_ac_decoder_ids: Optional[torch.LongTensor] = None,  # added to be ignored when passed from trainer
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        return super().forward(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    # copy-pasted and adapted from transformers.WhisperForConditionalGeneration.generate
    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            forced_ac_decoder_ids: Optional[torch.Tensor] = None,
            generation_config=None,
            logits_processor=None,
            stopping_criteria=None,
            prefix_allowed_tokens_fn=None,
            synced_gpus=False,
            return_timestamps=True,
            task="transcribe",
            language="english",
            **kwargs,
    ):
        if generation_config is None:
            generation_config = self.generation_config

        if return_timestamps is not None:
            if not hasattr(generation_config, "no_timestamps_token_id"):
                raise ValueError(
                    "You are trying to return timestamps, but the generation config is not properly set."
                    "Make sure to initialize the generation config with the correct attributes that are needed such as `no_timestamps_token_id`."
                    "For more details on how to generate the approtiate config, refer to https://github.com/huggingface/transformers/issues/21878#issuecomment-1451902363"
                )

            generation_config.return_timestamps = return_timestamps
        else:
            generation_config.return_timestamps = False

        if language is not None:
            generation_config.language = language
        if task is not None:
            generation_config.task = task

        forced_decoder_ids = []
        if task is not None or language is not None:
            if hasattr(generation_config, "language"):
                if generation_config.language in generation_config.lang_to_id.keys():
                    language_token = generation_config.language
                elif generation_config.language in TO_LANGUAGE_CODE.keys():
                    language_token = f"<|{TO_LANGUAGE_CODE[generation_config.language]}|>"
                else:
                    raise ValueError(
                        f"Unsupported language: {language}. Language should be one of:"
                        f" {list(TO_LANGUAGE_CODE.keys()) if generation_config.language in TO_LANGUAGE_CODE.keys() else list(TO_LANGUAGE_CODE.values())}."
                    )
                forced_decoder_ids.append((1, generation_config.lang_to_id[language_token]))
            else:
                forced_decoder_ids.append((1, None))  # automatically detect the language

            if hasattr(generation_config, "task"):
                if generation_config.task in TASK_IDS:
                    forced_decoder_ids.append((2, generation_config.task_to_id[generation_config.task]))
                else:
                    raise ValueError(
                        f"The `{generation_config.task}`task is not supported. The task should be one of `{TASK_IDS}`"
                    )
            else:
                forced_decoder_ids.append((2, generation_config.task_to_id["transcribe"]))  # defaults to transcribe
            if hasattr(generation_config, "no_timestamps_token_id") and not generation_config.return_timestamps:
                idx = forced_decoder_ids[-1][0] + 1 if forced_decoder_ids else 1
                forced_decoder_ids.append((idx, generation_config.no_timestamps_token_id))

        # Legacy code for backward compatibility
        elif hasattr(self.config, "forced_decoder_ids") and self.config.forced_decoder_ids is not None:
            forced_decoder_ids = self.config.forced_decoder_ids
        elif (
                hasattr(self.generation_config, "forced_decoder_ids")
                and self.generation_config.forced_decoder_ids is not None
        ):
            forced_decoder_ids = self.generation_config.forced_decoder_ids

        if generation_config.return_timestamps:
            logits_processor = [WhisperTimeStampLogitsProcessor(generation_config)]

        decoder_input_ids = None

        if len(forced_decoder_ids) > 0:
            # get the token sequence coded in forced_decoder_ids
            forced_decoder_ids.sort()
            if min(forced_decoder_ids)[0] != 0:
                forced_decoder_ids = [(0, self.config.decoder_start_token_id)] + forced_decoder_ids

            position_indices, decoder_input_ids = zip(*forced_decoder_ids)
            assert tuple(position_indices) == tuple(
                range(len(position_indices))), "forced_decoder_ids is not a (continuous) prefix, we can't handle that"

            device = self.get_decoder().device

            if forced_ac_decoder_ids is None:
                forced_ac_decoder_ids = torch.tensor([[]], device=device, dtype=torch.long)

            # enrich every sample's forced_ac_decoder_ids with Whisper's forced_decoder_ids
            batch_size = forced_ac_decoder_ids.shape[0]
            fluff_len = len(decoder_input_ids)
            decoder_input_ids = torch.tensor(decoder_input_ids, device=device, dtype=torch.long)
            decoder_input_ids = decoder_input_ids.expand((batch_size, fluff_len))
            decoder_input_ids = torch.cat([decoder_input_ids, forced_ac_decoder_ids], dim=1)

            generation_config.forced_decoder_ids = forced_decoder_ids

        return super(transformers.WhisperPreTrainedModel, self).generate(
            inputs,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            decoder_input_ids=decoder_input_ids,
            **kwargs,
        )

def extract_audio_from_video(video_path, output_audio_path):
    try:
        command = [
            'ffmpeg', 
            '-probesize', '50000000',          # Increase probesize
            '-analyzeduration', '50000000',    # Increase analyzeduration
            '-i', video_path,                  # Input video file
            '-q:a', '0',                       # Specify audio quality. 0 is the highest.
            '-map', 'a',                       # Map audio streams
            output_audio_path                  # Output audio file
        ]
        subprocess.run(command, check=True)
        print(f"Audio extracted to {output_audio_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

# Function to read CSV into a list of dictionaries
def read_csv(file_name):
    with open(file_name, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        return list(reader)


# Function to write a list of dictionaries to a CSV
def write_csv(file_name, data, fieldnames):
    with open(file_name, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def find_segments(file, model):
    print(file)
    results = model.transcribe(file)
    print("Segments detected.")
    return results


def find_non_voiced_segments(voiced_segments, audio_path):
    # Load the original audio file
    audio, sr = librosa.load(audio_path, sr=None)

    total_audio_length = len(audio) / sr

    non_voiced_segments = []
    last_end = 0

    for segment in voiced_segments:
        start = segment['start']
        if start > last_end:
            # There is a gap between the last segment and this one
            non_voiced_segments.append({'start': last_end, 'end': start})
        last_end = segment['end']

    # Check for a non-voiced segment at the end of the audio file
    if last_end < total_audio_length:
        non_voiced_segments.append({'start': last_end, 'end': total_audio_length})

    print("Non-voiced segments detected.")
    return non_voiced_segments


def run_whisperx(audio_file, model, output_dir):
    command = [
        'whisperx',
        audio_file,
        '--model', model,
        '--output_dir', output_dir,
        '--align_model', 'WAV2VEC2_ASR_LARGE_LV60K_960H'
    ]

    subprocess.run(command, check=True)
    print("WhisperX transcription complete.")


def json_to_csv(json_path, output_filename):
    # Load JSON output from whisperx
    with open(json_path, 'r') as file:
        data = json.load(file)

    # Open a CSV file for writing
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['start', 'end', 'text', 'voice'])  # Updated to include 'voice' column

        # Write each subtitle segment to CSV
        for segment in data['segments']:
            writer.writerow([segment['start'], segment['end'], segment['text'], 1])  # Set voice column value to 1
    print("JSON to CSV conversion complete.")


def process_non_voiced_segments(segments):
    processed_segments = []

    for segment in segments:
        duration = segment['end'] - segment['start']

        # Skip segments shorter than 10 seconds
        if duration < 10:
            continue

        # If the segment is between 10 and 20 seconds, keep it as is
        elif duration <= 20:
            processed_segments.append(segment)

        # If the segment is more than 20 seconds but less than 40, split into two parts
        elif duration < 40:
            midpoint = segment['start'] + duration / 2
            processed_segments.append({'start': segment['start'], 'end': midpoint})
            processed_segments.append({'start': midpoint, 'end': segment['end']})

        # If the segment is 40 seconds or longer, split into multiple 20-second segments
        else:
            current_start = segment['start']
            while current_start < segment['end']:
                current_end = min(current_start + 20, segment['end'])
                processed_segments.append({'start': current_start, 'end': current_end})
                current_start = current_end
    print("Non-voiced segments processed.")
    return processed_segments


def extract_and_save_segments(audio_file, segments, output_dir):
    audio, sr = librosa.load(audio_file, sr=None)
    saved_segments = []

    for i, segment in enumerate(segments):
        start_sample = int(segment['start'] * sr)
        end_sample = int(segment['end'] * sr)
        segment_audio = audio[start_sample:end_sample]

        segment_file = os.path.join(output_dir, f"segment_{i}.mp3")
        sf.write(segment_file, segment_audio, sr)
        saved_segments.append({
            'file': segment_file,
            'start': segment['start'],
            'end': segment['end']
        })
    print("Segments extracted and saved.")
    return saved_segments


def sounds_caption(checkpoint, segments_to_process, hugging_face_token):
    model = WhisperForAudioCaptioning.from_pretrained(checkpoint, token=hugging_face_token)
    tokenizer = transformers.WhisperTokenizer.from_pretrained(checkpoint, language="en", task="caption",
                                                              predict_timestamps=True)
    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(checkpoint)

    descriptions = []
    for segment in segments_to_process:
        input_file = segment['file']
        audio, sampling_rate = librosa.load(input_file, sr=feature_extractor.sampling_rate)
        features = feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features
        # Prepare the caption style
        style_prefix = "clotho > keyword: "
        # style_prefix = " "
        style_prefix_tokens = tokenizer("", text_target=style_prefix, return_tensors="pt",
                                        add_special_tokens=False).labels

        model.eval()
        outputs = model.generate(
            inputs=features.to(model.device),
            forced_ac_decoder_ids=style_prefix_tokens,
            max_length=300,
            num_beams=4,
            return_timestamps=True,
            early_stopping=True
        )
        caption = tokenizer.decode(outputs[0], decode_with_timestamps=True, skip_special_tokens=True)
        descriptions.append({
            'start': segment['start'],
            'end': segment['end'],
            'text': caption,
            'voice': 0
        })
    print("Sounds captions generated.")
    return descriptions


def process_csv(csv_name, descriptions):
    csv_data = read_csv(csv_name)

    # Append the new descriptions
    csv_data.extend(descriptions)

    # Sort the data
    csv_data.sort(key=lambda x: float(x['start']))
    write_csv(csv_name, csv_data, fieldnames=['start', 'end', 'text', 'voice'])

def clean_csv(csv_name):
    csv_data = read_csv(csv_name)
    print(csv_data)
    # Process the csv data
    for row in csv_data:
        if row['voice'] == '0':
            # Remove the prefix and enclose in brackets
            row['text'] = '(' + row['text'].replace("clotho > keyword: ", "") + ')'
        
        del row['voice']

    # Define the new fieldnames (excluding 'Voice')
    new_fieldnames = ['start', 'end', 'text']

    # Write the updated data back to the CSV
    write_csv(csv_name, csv_data, new_fieldnames)
    print("CSV post-processed.")


def csv_to_srt(csv_file, srt_file):
    with open(csv_file, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        subtitles = []

        for i, row in enumerate(reader, 1):
            start = datetime.timedelta(seconds=float(row['start']))
            end = datetime.timedelta(seconds=float(row['end']))
            text = row['text'].replace('|', '\n')  # Replace '|' with '\n' if needed

            subtitle = srt.Subtitle(index=i, start=start, end=end, content=text)
            subtitles.append(subtitle)

        srt_content = srt.compose(subtitles)

    with open(srt_file, mode='w', newline='') as srtfile:
        srtfile.write(srt_content)

    print("CSV converted to SRT.")


def srt_to_mp4(video_input, subtitles_file, output_file):
    command = [
        'ffmpeg', '-i', video_input,
        '-vf', f"subtitles={subtitles_file}",
        output_file
    ]

    subprocess.run(command, check=True)
    print("SRT embedded in video.")


def clone_git_repo(repo_url, clone_directory=None):
    if clone_directory and os.path.exists(clone_directory):
        return
    command = ['git', 'clone', repo_url]
    if clone_directory:
        command.append(clone_directory)

    subprocess.run(command, check=True)
    print("Git repository cloned.")


def main():

    # URL of the Git repository
    # URL of the Git repository
    #repo_url = 'https://github.com/m-bain/whisperX.git'

    # Optional: specify a directory to clone into
    #clone_directory = 'whisperX'

    # Clone the repository
    #clone_git_repo(repo_url, clone_directory)
    os.chdir("whisperX")
    os.makedirs("temp_segments", exist_ok=True)

    parser = argparse.ArgumentParser(description='Process a video file to generate and embed subtitles.')
    parser.add_argument('input_video', type=str, help='Path to the input video file')
    parser.add_argument('model_size', type=str, help='Size of models used in processing: tiny | small | medium | large')
    parser.add_argument('hugging_face_token', type=str, help="Token for authentication to hugging face")

    # Parse arguments
    args = parser.parse_args()

    hugging_face_token = args.hugging_face_token
    video_file = args.input_video
    device = "cuda"
    #device='cpu'

    if not os.path.exists(args.input_video):
        print(f"Error: Input video path '{args.input_video}' does not exist.")
        sys.exit(1)

    # Extract the base name (i.e., with extension)
    base_name = os.path.basename(args.input_video)
    core_name, extension = os.path.splitext(base_name)


    video_path=base_name
    audio_path=core_name+".mp3"
    extract_audio_from_video(video_path, audio_path)



    if args.model_size == "tiny":
        model_segments = whisperx.load_model("tiny", device=device)
        model_sounds = "MU-NLPC/whisper-tiny-audio-captioning"
    elif args.model_size == "small":
        model_segments = whisperx.load_model("small", device=device)
        model_sounds = "MU-NLPC/whisper-small-audio-captioning"
    elif args.model_size == "medium":
        model_segments = whisperx.load_model("medium", device=device)
        model_sounds = "MU-NLPC/whisper-large-v2-audio-captioning"
    elif args.model_size == "large":
        model_segments = whisperx.load_model("large-v3", device=device)
        model_sounds = "MU-NLPC/whisper-large-v2-audio-captioning"
    else:
        model_segments = whisperx.load_model("medium", device=device)
        model_sounds = "MU-NLPC/whisper-large-v2-audio-captioning"

    results = find_segments(audio_path, model_segments)
    non_voiced_segments = find_non_voiced_segments(results['segments'], audio_path)
    run_whisperx(audio_path, 'medium.en', '.')
    json_path = core_name + ".json"
    csv_file_name = core_name + '.csv'  # Your CSV file

    json_to_csv(json_path, csv_file_name)
    processed_non_voiced_segments = process_non_voiced_segments(non_voiced_segments)
    segments_to_process = extract_and_save_segments(audio_path, processed_non_voiced_segments, "temp_segments")
    descriptions = sounds_caption(model_sounds, segments_to_process, hugging_face_token=hugging_face_token)
    process_csv(csv_file_name, descriptions)
    clean_csv(csv_file_name)

    srt_file_name = core_name + '.srt'  # Output SRT file
    csv_to_srt(csv_file_name, srt_file_name)
    output_filename = core_name + "_final.mp4"
    srt_to_mp4(base_name, srt_file_name, output_filename)

main()


