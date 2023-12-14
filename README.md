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
already embedded. Obviously, as intermediate files, subtitles are available as standalone files to grab and go. You also need to specify the size of models — tiny, small, medium, large — based on your compute 
capabilities. (see the model description below).

A few things to consider:
1. Google Colab provides a tested environment when it comes to libraries. Local development is not guaranteed to work as expected (especially MAC with M chips)

2. (!) Important: While working with mp4 files: load them via Google Drive, do not upload directly from your computer. This is because they become corrupted because of the way Colab handles them.

3. To run the script, move the input file to /whisperX/ directory.

--------
**Models Information**

1. Transcription Model
2. Sounds Captioning Model
3. 

----

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

Further Work (What to Improve):
1. 