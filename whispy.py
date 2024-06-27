import os
import time
import torch
import whisper
import subprocess
import argparse

devices = ["cpu", "cuda"]
languages = ["en", "es", "fr", "de", "it", "pt"]
models = ["tiny", "base", "small", "medium", "large"]

def extract_audio(video_path, audio_path):
    if not os.path.exists(audio_path):
        print(f"Extracting audio from {video_path}...")
        start_time = time.time()
        subprocess.run([
            "ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", "-c:a", "mp3", "-b:a", "192k", "-loglevel", "error", audio_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        end_time = time.time()
        print(f"Audio extracted in {end_time - start_time:.2f} seconds.")
    else:
        print(f"Audio file already exists: {audio_path}")

def format_timestamp(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = (seconds - int(seconds)) * 1000
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{int(milliseconds):03}"

def write_srt(segments, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        for i, segment in enumerate(segments, start=1):
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"]
            file.write(f"{i}\n")
            file.write(f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n")
            file.write(f"{text.strip()}\n\n")

def generate_subtitles(video_filename, model, device, language, input_dir, output_dir, audio_dir):
    video_path = os.path.join(input_dir, video_filename)
    audio_basename = os.path.splitext(video_filename)[0]
    audio_path = os.path.join(audio_dir, audio_basename + ".mp3")

    # 1. Extract audio from video
    extract_audio(video_path, audio_path)

    # 2. Transcribe audio
    print(f"Transcribing audio from {audio_path}...")
    start_time = time.time()
    result = model.transcribe(audio_path, fp16=device.type == "cuda", language=language)
    end_time = time.time()
    print(f"Audio transcribed in {end_time - start_time:.2f} seconds.")

    # 3. Save subtitles
    srt_path = os.path.join(output_dir, audio_basename + ".srt")
    if not os.path.exists(srt_path):
        write_srt(result["segments"], srt_path)
        print(f"Subtitles written to {srt_path}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe videos with Whisper and generate subtitles")
    parser.add_argument('--model', type=str, choices=models, default="base", help=f"Whisper model to use {models}")
    parser.add_argument('--device', type=str, choices=devices, default="cuda", help=f"Device to use {devices}")
    parser.add_argument('--language', type=str, choices=languages, default="en", help=f"Transcription language code {languages}")

    parser.add_argument('--input_dir', type=str, default="input", help="Directory containing input video files")
    parser.add_argument('--output_dir', type=str, default="output", help="Directory to save output subtitle files")
    parser.add_argument('--audio_dir', type=str, default="audio_temp", help="Directory to save temporary audio files")
    
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.audio_dir, exist_ok=True)

    model = whisper.load_model(args.model).to(device)

    for video_filename in os.listdir(args.input_dir):
        try:
            generate_subtitles(video_filename, model, device, args.language, args.input_dir, args.output_dir, args.audio_dir)
        except Exception as e:
            print(f"Failed when trying to subtitle video {video_filename}: {e}")

if __name__ == "__main__":
    main()