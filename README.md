# 🎬 Whispy – Video Transcription and Subtitle Generator

**Whispy** is a simple Python-based tool that uses OpenAI's Whisper model to transcribe video files and generate `.srt` subtitle files — all through a clean and easy-to-use web interface.

---

## 🚀 Features

- Upload videos directly via the web interface.
- Select the Whisper model (`tiny`, `base`, `small`, `medium`, `large`).
- Choose the processing device (`cpu` or `cuda`).
- Supports multiple output languages (`en`, `es`, `fr`, etc.).
- Automatically generates subtitles in `.srt` format.

---

## ✅ Requirements

- Python 3.7 or higher
- `ffmpeg` installed and available in your system's PATH

---

## 🛠️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/gpietrafesavieitez/Whispy.git
   cd Whispy

2. Create and activate a virtual environment (optional but recommended): 
    ```python -m venv venv
    # On Linux/macOS:
    source venv/bin/activate
    # On Windows:
    venv\Scripts\activate

3. Install the required Python packages:
    ```pip install -r requirements.txt

4. (Optional) For CUDA support:
    ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
