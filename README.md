# jp-audio-to-subs

A robust, multi-threaded Python automation script that leverages **`faster-whisper`** and **Kotoba-Whisper** to generate accurate Japanese `.srt` subtitle files for video libraries locally, optimized for CPU execution.

Designed specifically to run efficiently without requiring a dedicated NVIDIA GPU, making use of CTranslate2 optimizations and native Kotoba-Whisper model parameters to prevent hallucination loops.

---

## Features

- **Optimized for Japanese:** Integrates `kotoba-tech/kotoba-whisper-v2.0-faster` out of the box with correct chunking parameters (`chunk_length=15`) tailored specifically for Kotoba's architecture.
- **CPU & Intel Arc Friendly:** Runs cleanly on standard processors using CTranslate2 vector math and 8-bit integer quantization (`int8`) to keep RAM usage low and performance high.
- **Parallel Batch Processing:** Uses a `ProcessPoolExecutor` to process multiple video files concurrently across your CPU cores.
- **In-Place Subtitle Embedding (Optional):** Automatically extracts audio, transcribes it, and optionally uses FFmpeg to mux the generated `.srt` directly back into the original MP4 as a soft subtitle stream.
- **Smart Skip Logic:** Automatically checks if an `.srt` file already exists for a video to prevent redundant processing.

---

## Requirements

- Python 3.10+
- **FFmpeg** installed and accessible in your system PATH.

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/jp-audio-to-subs.git
cd jp-audio-to-subs

```

2. **Install dependencies:**

```bash
pip install faster-whisper

```

3. **Verify FFmpeg installation:**

- **Windows (via winget):** `winget install Gyan.FFmpeg`
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

---

### Troubleshooting: `TypeError: open() got an unexpected keyword argument 'metadata_errors'`

If you encounter this crash when running the transcription script:

```text
TypeError: open() got an unexpected keyword argument 'metadata_errors'

```

This is caused by a compatibility break between the stable PyPI release of `faster-whisper` and the latest version of the underlying audio decoding library (`PyAV` v19+).

To fix this and pull the updated patch from the master branch directly, run the following command in your environment:

```bash
python -m pip install --force-reinstall --no-deps git+https://github.com/SYSTRAN/faster-whisper.git

```

- **Why this works:** The `--no-deps` flag tells pip to update `faster-whisper` directly from its repository source without modifying your other library tree, bypassing the argument mismatch with PyAV.

---

## Usage

Run the script from your terminal, pointing it at the target directory containing your MP4 files:

```bash
python main.py "C:\path\to\your\videos" --device cpu --compute-type int8

```

### Command-Line Arguments

| Flag             | Default                                  | Description                                                     |
| ---------------- | ---------------------------------------- | --------------------------------------------------------------- |
| `dir`            | _(Required)_                             | Path to the target directory containing `.mp4` files.           |
| `--model`        | `kotoba-tech/kotoba-whisper-v2.0-faster` | Hugging Face model ID or local path to CTranslate2 weights.     |
| `--language`     | `ja`                                     | Spoken language code.                                           |
| `--device`       | `cpu`                                    | Target inference device (`cpu` or `cuda`).                      |
| `--compute-type` | `int8`                                   | Quantization precision (`int8`, `float16`, `float32`).          |
| `--workers`      | `2`                                      | Number of concurrent video files to process in parallel.        |
| `--cpu-threads`  | `4`                                      | Number of CPU threads allocated per whisper worker.             |
| `--in-place`     | `False`                                  | Embed soft subtitles directly into the original MP4.            |
| `--keep-srt`     | `False`                                  | Keep the standalone `.srt` file even when running `--in-place`. |

---

## License

[MIT License](https://www.google.com/search?q=LICENSE)
