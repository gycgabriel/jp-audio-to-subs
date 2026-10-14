import argparse
import subprocess
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from faster_whisper import WhisperModel

DEFAULT_MODEL_PATH = "kotoba-tech/kotoba-whisper-v2.0-faster"


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def process_video_inplace(video_path: Path, args):
    video_path = video_path.resolve()
    srt_path = video_path.with_suffix(".srt")

    if srt_path.exists() and not args.in_place:
        print(f"[SKIPPED] SRT already exists for: {video_path.name}")
        return

    unique_id = uuid.uuid4().hex[:8]
    temp_wav_path = video_path.with_name(f"{video_path.stem}_{unique_id}.temp.wav")
    temp_mp4_path = video_path.with_name(f"{video_path.stem}_{unique_id}.temp.mp4")
    generated_srt = video_path.with_name(f"{video_path.stem}_{unique_id}.srt")

    print(f"[START] {video_path.name}")

    try:
        # 1. Extract 16kHz mono WAV via FFmpeg
        # fmt:off
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-ar", "16000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                str(temp_wav_path)
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
        )
        # fmt:on

        # 2. Run faster-whisper on CPU using INT8 vector instructions
        model = WhisperModel(
            args.model,
            device=args.device,
            compute_type=args.compute_type,
            cpu_threads=args.cpu_threads,
        )

        segments, _ = model.transcribe(
            str(temp_wav_path),
            language=args.language,
            chunk_length=15,  # Optimized for Kotoba-Whisper
            condition_on_previous_text=False,  # Prevents sentence repetition loops
        )

        # Build clean SRT file
        with open(generated_srt, "w", encoding="utf-8") as f:
            for idx, seg in enumerate(segments, start=1):
                start_str = format_timestamp(seg.start)
                end_str = format_timestamp(seg.end)
                f.write(f"{idx}\n{start_str} --> {end_str}\n{seg.text.strip()}\n\n")

        if not generated_srt.exists():
            raise FileNotFoundError(
                f"Expected generated SRT not found at: {generated_srt}"
            )

        generated_srt.replace(srt_path)

        # 3. Optional In-Place Embedding
        if args.in_place:
            # fmt:off
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(video_path),
                    "-i", str(srt_path),
                    "-c", "copy",
                    "-c:s", "mov_text",
                    str(temp_mp4_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
            )
            # fmt:on

            temp_mp4_path.replace(video_path)

            if not args.keep_srt:
                srt_path.unlink(missing_ok=True)

            print(f"[EMBEDDED IN-PLACE] {video_path.name}")
        else:
            print(f"[SRT CREATED] {srt_path.name}")

    except Exception as e:
        print(f"[FAILED] {video_path.name}: {e}")
        if args.in_place and srt_path.exists() and not args.keep_srt:
            srt_path.unlink(missing_ok=True)
    finally:
        if temp_wav_path.exists():
            temp_wav_path.unlink(missing_ok=True)
        if temp_mp4_path.exists():
            temp_mp4_path.unlink(missing_ok=True)
        if generated_srt.exists():
            generated_srt.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Parallel Batch Subtitle Generator for Intel/CPU Execution"
    )
    parser.add_argument("dir", type=str, help="Target directory containing MP4 files")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--language", type=str, default="ja")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--compute-type",
        type=str,
        default="int8",
        choices=["int8", "float32", "float16"],
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of concurrent video files to process",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help="CPU threads per faster-whisper instance",
    )
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--keep-srt", action="store_true")
    args = parser.parse_args()

    target_dir = Path(args.dir).resolve()
    mp4_files = list(target_dir.rglob("*.mp4"))

    print(
        f"Found {len(mp4_files)} total MP4 files. Launching with {args.workers} worker(s)..."
    )

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(process_video_inplace, mp4, args) for mp4 in mp4_files
        ]
        for future in as_completed(futures):
            future.result()

    print("\nAll batch processing complete!")


if __name__ == "__main__":
    main()
