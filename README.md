**About the Project**

This project presents an automated solution for generating subtitles for video content. Leveraging the capabilities of WhisperX for audio transcription and FFmpeg for audio-video manipulation, the script provides a seamless way to convert spoken words and ambient sounds in a video into accurately timed subtitles. This tool is particularly useful for enhancing the accessibility of video content, making it more inclusive for viewers who are deaf or hard of hearing, as well as for those who prefer to watch videos with subtitles.

--------

**Key Features**
* Audio Extraction: Extracts the audio track from a video file, enabling focused processing of the audio content.
* Transcription of Voiced Content: Utilizes WhisperX to transcribe spoken words in the audio track, capturing the dialogues and spoken elements accurately.
* Identification and Captioning of Non-Voiced Segments: Detects non-voiced segments like silence or background noise and generates descriptive captions, providing a full auditory experience in text form.
* Subtitle File Generation: Converts the transcription and captions into an SRT file, a widely-supported subtitle format.
* Subtitle Embedding: Integrates the generated subtitles back into the original video, ensuring synchronization between the audio and the corresponding text.
* Customizability: Supports different model sizes for transcription and provides options for processing, making it adaptable to various types of video content.

----
**This Repository**

The main file is subtitle_generator.py which contains the full script to generate SDH subtitles (subtitles for hearing impaired). The pipeline takes video and outputs the video with subtitles
already embedded. Obviously, as intermediate files, subtitles are available as standalone files to grab and go. You also need to specify the size of models — tiny, small, medium, large — based on your compute capabilities. (see the model description below).

<ins>Subtitle generator with name 'testing' is the copy of the original one but without running the last function: appending subtitles to mp4 which is very time-consuming and for benchmarking it does not contribute much because this merging of subtitles with the video is model-agnostic</ins>

In **Notebooks** you can find examples of usage of the code, benchmarking (time) of different models, and checking WER of pure transcription.

In **Examples** you can find outputs of the work of the script — you will find a full movie with subtitles, pure subtitles, and pure transcription. 

A few things to consider:
1. Google Colab provides a tested environment when it comes to libraries. Local development is not guaranteed to work as expected (especially MAC with M chips)

2. (!) Important: While working with mp4 files: load them via Google Drive, do not upload directly from your computer. This is because they become corrupted because of the way Colab handles them.

3. To run the script, move the input file to /whisperX/ directory.

--------
**Models Information**

1. Transcription Model - we use WhisperX due to robust segmentation, fast processing, and VAD capabilities. For alignment, we use WAV2VEC2_ASR_LARGE_LV60K_960H — _"The large model pretrained and fine-tuned on 960 hours of Libri-Light and Librispeech on 16kHz sampled speech audio. When using the model make sure that your speech input is also sampled at 16Khz". For more details, check: https://huggingface.co/facebook/wav2vec2-large-960h-lv60
   * The WhisperX transcription using medium model due to compute capabilities constraints has WER of 2.14%. We measured the error on the Transformers movie from 1986.
     
2. Sounds Captioning Model — we use Whipser pretrained on AudioCaps and AudioSet datasets (see description below). However, pre-training was only on the subset due to compute and memory overhead. This may have significantly impacted the performance. Once pretrained, the model is fine-tuned on Clotho dataset. See github repository for more information: https://github.com/prompteus/audio-captioning


-----

**Datasets**

1. AudioSet
  * Source: YouTube videos.
  * Content: Over 2 million 10-second audio clips.
  * Annotation: Hierarchical ontology of 632 audio event classes.
  * Nature: Multi-label dataset, with an average of 2.7 classes per recording.
  * Usage: Audio classification, tagging, and source separation.


2. Clotho
  * Audio Clip Length: 10 to 30 seconds.
  * Quality: Good quality recordings (44.1kHz, 16-bit), normalized with trimmed silences.
  * Content Variety: Includes ambient sounds, animal noises, crowd murmurs, machinery, etc.
  * Captions: Each clip has five captions, created through description, editing, and scoring.
  * Crowdsourcing: Captions gathered and refined via Amazon Mechanical Turk.

3. AudioCaps
   * Focus on DCASE 2021 Challenge:
   * It serves as a superset for AudioCaps
   * A large audio captioning dataset, derived from AudioSet, comprising over 50,000 ten second audio clips.
   * Split into training, testing, and validation segments, with each clip in the training set having one caption and those in the testing and validation sets having five captions each.

Based on _https://github.com/Labbeti/aac-datasets_ :

<img center width="429" alt="Screenshot 2023-12-15 at 09 58 51" src="https://github.com/barto-official/Audio_Description/assets/125658269/3eb06ca0-478f-49c1-a4b8-1efe7164c25a">

<img center width="429" alt="Screenshot 2023-12-15 at 08 46 37" src="https://github.com/barto-official/Audio_Description/assets/125658269/137a298c-e2fb-4fd3-a01a-2387562af20f"> <br>

----
<img center width="430" alt="Screenshot 2023-12-15 at 10 05 17" src="https://github.com/barto-official/Audio_Description/assets/125658269/b68745d6-9237-4607-813a-0d9bf060b081">

**How the pipeline works:**

**1. Installing Required Packages**

The script begins by installing necessary Python packages listed in the requirements.txt file. This ensures that all dependencies are satisfied before the main processing starts. 

**2. Class Definition for Audio Captioning**

A custom class WhisperForAudioCaptioning, derived from Whisper's conditional generation model, is defined. This class is tailored for audio captioning and includes specific configurations and methods for this purpose.

**3. Extracting Audio from Video**

The script extracts the audio track from the provided video file. This is accomplished using the FFmpeg tool, which isolates the audio component and saves it as a separate file.

**4. Transcribing Audio**

The extracted audio file is then processed through WhisperX for transcription. The transcription identifies voiced segments in the audio, which are essential for generating accurate subtitles.

**5. Processing Voiced and Non-Voiced Segments**

Alongside voiced segments, the script also identifies non-voiced segments within the audio. This includes periods of silence or background noise, providing a comprehensive map of the audio track's content.

**6. Running WhisperX Transcription**

WhisperX is utilized to transcribe the audio content. The transcription results include timestamps and transcribed text for each identified segment.

**7. Converting JSON to CSV**

The output from WhisperX, typically in JSON format, is converted into a CSV format. This step makes it easier to process and manipulate the transcription data in subsequent steps.

**8. Processing and Saving Audio Segments**

Both voiced and non-voiced segments are processed and saved separately. This involves splitting longer segments for better manageability and accuracy in the final subtitle file.

**9. Generating Captions for Sounds**

For audio segments, especially non-voiced ones, captions are generated to describe the sound. This enriches the subtitles by providing context for sounds that are not spoken words.

**10. Updating and Cleaning CSV File**

The CSV file, containing transcription and caption data, is further processed. This includes cleaning up the data, removing unnecessary information, and formatting the content for subtitle conversion.

**11. Converting CSV to SRT (SubRip Subtitle File)**

The processed CSV file is converted into an SRT file, a standard format for subtitles. This file contains timestamps and corresponding subtitles ready to be embedded into the video.

**12. Embedding Subtitles into Video**

The final step involves using FFmpeg again to embed the generated SRT subtitles into the original video file. The result is a video file with integrated subtitles, synchronized with the audio track.

--------

**Technology Stack**
* Python: The primary programming language used for the script.
* WhisperX: A robust tool for audio transcription, capable of handling different languages and accents.
* FFmpeg: A powerful multimedia framework used for processing audio and video files.
* Librosa and Soundfile: Libraries for audio processing and file handling.
* Transformers: Library for advanced machine learning models, used for custom audio captioning tasks.

---------

**Testing Metrics** 

_Test Conducted using Small Models_

For testing the accuracy of our audio captioning model we used the evaluation category of the clotho dataset in order to compare how our generated captions fare against the actual captions from the clotho dataset. In order to do this we made sure to normalize the generated subtitles in order to remove all the special characters and make the letters all lowercase in order to have a fair comparison. The metrics that we used are the following:

* **BLEU1, BLEU2, BLEU3, BLEU4:** These are different variations of the BLEU (Bilingual Evaluation Understudy) score, which measures how similar a machine-translated text is to a set of high-quality reference translations. The numbers 1, 2, 3, and 4 refer to the n-gram size used in the calculation. BLEU1 considers only individual words (unigrams), BLEU2 includes pairs of consecutive words (bigrams), and so on. Higher n-gram BLEU scores (like BLEU4) are more stringent, as they consider longer phrases and thus a closer match to the reference text.

    * **Results:** BLEU1 = 0.458441, BLEU2 = 0.274022, BLEU3 = 0.171906, BLEU4 = 0.110351

* **ROUGEL:** This stands for Recall-Oriented Understudy for Gisting Evaluation using Longest Common Subsequence. ROUGE scores are used primarily for evaluating text summarization. ROUGEL specifically measures the longest common subsequence between the generated text and the reference text, considering the order of words. It's good for assessing the overall structure and content preservation in the generated text.

    * **Results:** ROUGEL = 0.216043

* **METEOR:** This stands for Metric for Evaluation of Translation with Explicit Ordering. It's another metric for evaluating translation quality, similar to BLEU but with some differences. METEOR considers synonyms and stemming, and it balances precision (how many words in the generated text are in the reference) with recall (how many words in the reference are captured in the generated text). It generally aligns more closely with human judgment than BLEU.

    * **Results:** METEOR = 0.341802
 
* **WER**: Measuring WER <ins>on pure transcription (without audio captioning)</ins> by comparing our transcription with original subtitles (.srt vs .srt) of Transformers movie from 1986.
  * **Results**: WER=2.14% 


-----------

**Limitations during Development**
1. Compute Power — we use basic Windows/Mac computers and Google Colab. Due to that fact, we mainly operated on small/medium models whose results can be enhanced by 20-30% if using more computing power for large models. Also, audio-captioning model was pre-trained only on 1/20 of the dataset possible (100k out of 2M)  which is the next aspect that can be improved but only if processing resources are available. Given the size of the model, it's extremely difficult to do it on proprietary user device.

2. Metrics controversy — metrics, especially for Audio-Captoning, are to provide more background information and some benchmarks but they don't specifically address the "accuracy" of the model. Also, because we couldn't use medium/large models, we see a big space for improvement. Second improvement can come from pre-training the model on more data (see the first point)

3. Subjectivity of Audio-captoning — difficulty in judging the model, especially for audio-captoning, comes from the fact that current subtitles for hearing impaired have relatively low-frequency of audio captions (mostly the main sounds in the movie). We implemented them on most of the possible subsegments of the audio/video. Thus it's difficult to compare the model to 'ground-truth'. Also, the labelling of sounds is very subjective and not one answer is possible that would accurately describe the scene. Furthermore, testing using 'captions' rather than keywords is bring additional obstacles. Compare: "The man is walking on the green grass" and "The man is walking on the grass". Technically, these sentences are not the same but in practice they are the same. This is the same probelm as with TTS (Text-to-Speech). As per Hugging Face: _"Different speakers may choose to emphasize different parts of the sentence, for example. This makes TTS models hard to evaluate. Because of this, the L1 or MSE loss value isn't actually very meaningful — there are multiple ways to represent the same text to a spectrogram. This is why TTS models are typically evaluated by human listeners, using a metric known as MOS or mean opinion score._"

----------

Further Work (What to Improve):
1. Use more compute power, if possible, to use large models which was impossible during development and using the author's hardware.
2. Pre-train Whisperx on full datasets, not samples (AudioCaps and AudioSets). Please, refer to https://github.com/prompteus/audio-captioning for more information. Pre-training has been done on ~100k examples out of 2M possible. Memory and time are the main obstacles to make that happen. We estimate to boost performance of sounds captioning to improve by ~20%.
3. Enhance Speaker Diarization — first, run a classifier on the whole movie to classify speakers and then include Diarization so instead of "Speaker 1" you would have, for example, "John"
4. If you would like, there is a possibility of including ASS subtitles.


   
