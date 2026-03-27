"""Extract frames from a video at a specified FPS for OpenDroneMap processing."""

import subprocess
import argparse
from pathlib import Path


def extract_frames(video_path: str, output_dir: str, fps: int = 5):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames with ffmpeg using zero-padded sequential naming
    pattern = str(output_dir / "frame_%04d.jpg")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-qmin", "1", "-q:v", "1",  # highest JPEG quality
            pattern,
        ],
        check=True,
    )

    # Build sorted img_list.txt (lexicographic order matches numeric order due to zero-padding)
    frames = sorted(output_dir.glob("frame_*.jpg"))
    img_list_path = output_dir.parent / "img_list.txt"
    img_list_path.write_text("\n".join(f.name for f in frames) + "\n")

    print(f"Extracted {len(frames)} frames to {output_dir}")
    print(f"Image list written to {img_list_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--output-dir", default=None, help="Output directory for frames")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second to extract")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(Path(args.video).parent.parent / "images")

    extract_frames(args.video, args.output_dir, args.fps)
