import os
import time
import torch
import whisper
import subprocess
import gradio as gr

models = ["tiny", "base", "small", "medium", "large"]
devices = ["cpu", "cuda"]
languages = ["en", "es", "fr", "de", "it", "pt"]

def extract_audio(video_path, audio_path):
    if not os.path.exists(audio_path):
        subprocess.run([
            "ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a",
            "-c:a", "mp3", "-b:a", "192k", "-loglevel", "error", audio_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def format_timestamp(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = (seconds - int(seconds)) * 1000
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{int(milliseconds):03}"

def write_srt(segments, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        for i, segment in enumerate(segments, start=1):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()
            file.write(f"{i}\n{start} --> {end}\n{text}\n\n")

def transcribe_video(video, model_name, device_choice, language):
    basename = os.path.splitext(os.path.basename(video.name))[0]
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    audio_path = os.path.join(temp_dir, basename + ".mp3")
    srt_path = os.path.join(temp_dir, basename + ".srt")

    device = torch.device("cuda" if device_choice == "cuda" and torch.cuda.is_available() else "cpu")
    model = whisper.load_model(model_name).to(device)

    extract_audio(video.name, audio_path)

    result = model.transcribe(audio_path, fp16=device.type == "cuda", language=language)
    write_srt(result["segments"], srt_path)

    return srt_path

gr.Interface(
    fn=transcribe_video,
    inputs=[
        gr.File(label="Upload a video", file_types=["video"]),
        gr.Dropdown(models, value="base", label="Model"),
        gr.Dropdown(devices, value="cuda", label="Device"),
        gr.Dropdown(languages, value="es", label="Output")
    ],
    outputs=gr.File(label="Generated SRT file"),
    title="🎬 Subtitles generator with Whisper",
    description="Transcribes video and generates subtitles (.srt) using Whisper."
).launch()
