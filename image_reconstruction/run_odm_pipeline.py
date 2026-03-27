"""End-to-end pipeline: video -> frames -> ODM -> orthophoto."""

import argparse
import shutil
import subprocess
from pathlib import Path

from extract_frames import extract_frames

ODM_PROJECT = Path(__file__).parent / "odm_project"
IMAGES_DIR = ODM_PROJECT / "images"
OUTPUTS_DIR = ODM_PROJECT / "outputs"

# Directories and files created by ODM that should be cleaned between runs
ODM_ARTIFACTS = [
    "opensfm", "odm_filterpoints", "odm_meshing", "odm_texturing_25d",
    "odm_orthophoto", "odm_georeferencing", "odm_report", "odm_dem",
    "cameras.json", "benchmark.txt", "log.json", "options.json", "images.json",
    "img_list.txt",
]


def clean_odm_outputs():
    """Remove all ODM processing artifacts from previous runs."""
    for name in ODM_ARTIFACTS:
        path = ODM_PROJECT / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()
    print("Cleaned previous ODM outputs.")


def clean_frames():
    """Delete extracted frames from images directory."""
    if IMAGES_DIR.exists():
        shutil.rmtree(IMAGES_DIR)
    print("Cleaned extracted frames.")


def run_odm(feature_quality: str = "medium", orthophoto_resolution: float = 0.1):
    """Run OpenDroneMap via Docker."""
    cmd = [
        "docker", "run", "--platform", "linux/amd64",
        "-ti", "--rm",
        "-v", f"{ODM_PROJECT.resolve()}:/datasets/code",
        "opendronemap/odm",
        "--project-path", "/datasets",
        "--feature-quality", feature_quality,
        "--ignore-gsd",
        "--skip-3dmodel",
        "--crop", "0",
        "--orthophoto-resolution", str(orthophoto_resolution),
        "code",
    ]
    print(f"Running ODM: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_pipeline(video_path: str, fps: int = 5, orthophoto_resolution: float = 0.1):
    """Full pipeline: extract frames, run ODM, save output, cleanup."""
    video_path = Path(video_path).resolve()
    video_stem = video_path.stem

    print(f"\n{'='*60}")
    print(f"Processing: {video_path.name}")
    print(f"{'='*60}\n")

    # Step 1: Clean previous run
    clean_odm_outputs()
    clean_frames()

    # Step 2: Extract frames
    print(f"\nExtracting frames at {fps} FPS...")
    extract_frames(str(video_path), str(IMAGES_DIR), fps=fps)

    # Step 3: Run ODM
    print("\nRunning ODM...")
    run_odm(orthophoto_resolution=orthophoto_resolution)

    # Step 4: Copy output
    ortho_src = ODM_PROJECT / "odm_orthophoto" / "odm_orthophoto.tif"
    if not ortho_src.exists():
        print(f"ERROR: Orthophoto not found at {ortho_src}")
        return

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ortho_dst = OUTPUTS_DIR / f"{video_stem}.tif"
    shutil.copy2(ortho_src, ortho_dst)
    print(f"\nSaved orthophoto to: {ortho_dst}")

    # Step 5: Clean up frames
    clean_frames()

    print(f"\nDone: {video_stem}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video to orthophoto pipeline")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second to extract (default: 5)")
    parser.add_argument("--orthophoto-resolution", type=float, default=2,
                        help="Orthophoto resolution in cm/pixel (default: 2)")
    args = parser.parse_args()

    run_pipeline(args.video, fps=args.fps, orthophoto_resolution=args.orthophoto_resolution)
