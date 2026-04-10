from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def log(message: str) -> None:
    print(f"[export-video] {message}", flush=True)


def require_ffmpeg() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is not available on PATH. Install ffmpeg before exporting episode videos.")
    return ffmpeg_path


def resolve_inputs(episode_json: Path | None, frames_dir: Path | None, output: Path | None) -> tuple[Path, Path]:
    if episode_json is None and frames_dir is None:
        raise RuntimeError("Either --episode-json or --frames-dir must be provided.")

    metadata_path = episode_json.resolve() if episode_json is not None else None
    if metadata_path is not None:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frames_value = metadata.get("sampled_frames_dir")
        if not frames_value:
            raise RuntimeError(f"Episode metadata does not contain sampled_frames_dir: {metadata_path}")
        frames = Path(frames_value).resolve()
        resolved_output = output.resolve() if output is not None else metadata_path.with_suffix(".mp4")
        return frames, resolved_output

    assert frames_dir is not None
    frames = frames_dir.resolve()
    resolved_output = output.resolve() if output is not None else frames.with_suffix(".mp4")
    return frames, resolved_output


def main() -> int:
    parser = argparse.ArgumentParser(description="Export an MP4 from sampled episode frames.")
    parser.add_argument("--episode-json", default=None)
    parser.add_argument("--frames-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--update-metadata",
        action="store_true",
        help="When --episode-json is used, write the generated video_path back into the episode metadata.",
    )
    args = parser.parse_args()

    episode_json = Path(args.episode_json) if args.episode_json else None
    frames_dir = Path(args.frames_dir) if args.frames_dir else None
    output = Path(args.output) if args.output else None
    frames_path, output_path = resolve_inputs(episode_json, frames_dir, output)
    if not frames_path.exists():
        raise RuntimeError(f"Frames directory does not exist: {frames_path}")
    if not list(frames_path.glob("frame_*.png")):
        raise RuntimeError(f"No sampled frame PNGs were found in {frames_path}")

    ffmpeg_path = require_ffmpeg()
    command = [
        ffmpeg_path,
        "-y",
        "-framerate",
        str(args.fps),
        "-i",
        str(frames_path / "frame_%06d.png"),
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {completed.stderr.strip() or completed.stdout.strip()}")

    if episode_json is not None and args.update_metadata:
        metadata = json.loads(episode_json.read_text(encoding="utf-8"))
        metadata["video_path"] = str(output_path)
        episode_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    log(f"video={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
