#!/usr/bin/env python3
"""
compile_images.py

Turn a sequence of images into a GIF and/or a video.

Defaults tuned for PowerPoint on Windows + Mac:
- Video defaults to MP4 (H.264 / yuv420p / faststart) via ffmpeg when available
- Default fps = 30
- ffmpeg default args: -movflags +faststart -crf 18 -preset medium

Examples
--------
# Video (defaults to mp4) using ffmpeg if available:
python compile_images.py -i "pics/*.png" --video -o talk_movie

# Exact total duration (seconds). Note: timing may be VFR-like for ffmpeg concat list.
python compile_images.py -i "pics/*.png" --video --time 5 -o talk_movie

# Force OpenCV (if you want AVI/MJPG etc.):
python compile_images.py -i "pics/*.png" --video --opencv --fourcc MJPG -o talk_movie.avi

# GIF only
python compile_images.py -i "pics/*.png" --gif --time 5 -o anim

EXAMPLE for smaller gifs:
python ffmpeg_compile_images.py --gif-width 800 --gif-colors 96
--gif -t 5 --ffmpeg -i dplot_ -o ../V03_10_dpfull_5s
"""

from __future__ import annotations

import argparse
import glob
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


# -----------------------------
# Sorting utilities
# -----------------------------
_nat_re = re.compile(r"(\d+)")


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in _nat_re.split(s)]


# -----------------------------
# Timing utilities
# -----------------------------
@dataclass(frozen=True)
class Timing:
    fps: float
    frame_duration_s: float  # seconds per frame


def compute_timing(n_frames: int, fps: Optional[float], total_time: Optional[float]) -> Timing:
    if n_frames <= 0:
        raise ValueError("n_frames must be > 0")

    if fps is not None and total_time is not None:
        raise ValueError("Choose only one of --fps or --time.")

    if total_time is not None:
        if total_time <= 0:
            raise ValueError("--time must be > 0")
        fps_calc = n_frames / total_time
        return Timing(fps=fps_calc, frame_duration_s=total_time / n_frames)

    if fps is not None:
        if fps <= 0:
            raise ValueError("--fps must be > 0")
        return Timing(fps=fps, frame_duration_s=1.0 / fps)

    # Default tuned for PowerPoint-friendly videos
    fps_default = 30.0
    return Timing(fps=fps_default, frame_duration_s=1.0 / fps_default)


def gif_duration_ms(frame_duration_s: float, clamp_ms: int = 20) -> int:
    """
    GIF frame delay is stored in 1/100 s (10ms) units; viewers often clamp small delays.
    Round to 10ms ticks and clamp.
    """
    ms = frame_duration_s * 1000.0
    ms_q = int(round(ms / 10.0) * 10)
    return max(clamp_ms, ms_q)


# -----------------------------
# Image size utilities
# -----------------------------
def get_image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as im:
        return im.size  # (w, h)


def compute_target_size(paths: List[Path]) -> Tuple[int, int]:
    max_w = 0
    max_h = 0
    for p in paths:
        w, h = get_image_size(p)
        max_w = max(max_w, w)
        max_h = max(max_h, h)
    return max_w, max_h


def pad_to_size(im: Image.Image, target_w: int, target_h: int) -> Image.Image:
    w, h = im.size
    if (w, h) == (target_w, target_h):
        return im
    out = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    x0 = (target_w - w) // 2
    y0 = (target_h - h) // 2
    out.paste(im, (x0, y0))
    return out


def resize_to_size(im: Image.Image, target_w: int, target_h: int) -> Image.Image:
    if im.size == (target_w, target_h):
        return im
    return im.resize((target_w, target_h), resample=Image.BICUBIC)


# -----------------------------
# GIF writer (Pillow)
# -----------------------------
def write_gif(paths: List[Path], out_gif: Path, timing: Timing, resize_mode: str) -> None:
    if not paths:
        raise ValueError("No images to write.")

    target_w = target_h = None
    if resize_mode in {"pad", "scale"}:
        target_w, target_h = compute_target_size(paths)

    frames: List[Image.Image] = []
    for p in paths:
        with Image.open(p) as im:
            frame = im.convert("RGB")
            if resize_mode == "pad":
                frame = pad_to_size(frame, target_w, target_h)  # type: ignore[arg-type]
            elif resize_mode == "scale":
                frame = resize_to_size(frame, target_w, target_h)  # type: ignore[arg-type]
            frames.append(frame.copy())

    dur_ms = gif_duration_ms(timing.frame_duration_s)

    out_gif.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=dur_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


# -----------------------------
# GIF writer (ffmpeg)
# -----------------------------
def write_gif_ffmpeg(
    paths: List[Path],
    out_gif: Path,
    timing: Timing,
    resize_mode: str,
    gif_fps: float,
    gif_width: int,
    gif_colors: int,
    gif_dither: str,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH (needed for --gif-ffmpeg).")
    if not paths:
        raise ValueError("No images to write.")

    out_gif.parent.mkdir(parents=True, exist_ok=True)

    # Compute target size for pad/scale if requested
    vf_size = ""
    if resize_mode in {"pad", "scale"}:
        target_w, target_h = compute_target_size(paths)
        if resize_mode == "pad":
            vf_size = (
                f",scale={target_w}:{target_h}:force_original_aspect_ratio=decrease"
                f",pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2"
            )
        else:
            vf_size = f",scale={target_w}:{target_h}"
    else:
        # Optional width scaling for GIF size reduction
        if gif_width and gif_width > 0:
            vf_size = f",scale={gif_width}:-1:flags=lanczos"

    # Ensure sane bounds
    gif_colors = max(2, min(256, int(gif_colors)))
    gif_fps = float(gif_fps) if gif_fps > 0 else float(timing.fps)

    # Dither mapping
    dither_opt = "none" if gif_dither == "none" else gif_dither

    frame_dur = timing.frame_duration_s
    with tempfile.TemporaryDirectory() as td:
        list_path = Path(td) / "concat_list.txt"
        with list_path.open("w", encoding="utf-8") as f:
            for p in paths[:-1]:
                f.write(f"file '{p.as_posix()}'\n")
                f.write(f"duration {frame_dur:.9f}\n")
            f.write(f"file '{paths[-1].as_posix()}'\n")

        # One-pass palettegen+paletteuse pipeline
        # fps filter controls how many frames go into GIF (big size lever)
        filter_complex = (
            f"[0:v]fps={gif_fps}{vf_size},split[s0][s1];"
            f"[s0]palettegen=max_colors={gif_colors}:stats_mode=diff[p];"
            f"[s1][p]paletteuse=dither={dither_opt}"
        )

        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-filter_complex", filter_complex,
            "-loop", "0",
            str(out_gif),
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg GIF failed (exit {e.returncode}). Command was:\n  {' '.join(cmd)}") from e


# -----------------------------
# Video writer (OpenCV)
# -----------------------------
def write_video_opencv(
    paths: List[Path],
    out_video: Path,
    timing: Timing,
    resize_mode: str,
    fourcc: Optional[str],
) -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is not available. Install it or use ffmpeg (default).")
    if not paths:
        raise ValueError("No images to write.")

    if resize_mode in {"pad", "scale"}:
        target_w, target_h = compute_target_size(paths)
    else:
        target_w, target_h = get_image_size(paths[0])
        for p in paths[1:]:
            w, h = get_image_size(p)
            if (w, h) != (target_w, target_h):
                raise ValueError(
                    "Frame sizes are not uniform. Use --resize pad/scale or use ffmpeg (default)."
                )

    out_video.parent.mkdir(parents=True, exist_ok=True)

    if fourcc is None:
        ext = out_video.suffix.lower()
        fourcc = "MJPG" if ext == ".avi" else "mp4v"

    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*fourcc),
        float(timing.fps),
        (int(target_w), int(target_h)),
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"OpenCV VideoWriter failed for '{out_video}'. Try --fourcc MJPG (avi) or mp4v (mp4), or use ffmpeg."
        )

    try:
        for p in paths:
            frame = cv2.imread(str(p))
            if frame is None:
                raise RuntimeError(f"Failed to read image: {p}")

            h, w = frame.shape[:2]
            if resize_mode == "pad":
                top = (target_h - h) // 2
                bottom = target_h - h - top
                left = (target_w - w) // 2
                right = target_w - w - left
                frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            elif resize_mode == "scale":
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

            writer.write(frame)
    finally:
        writer.release()


# -----------------------------
# Video writer (ffmpeg)
# -----------------------------
def write_video_ffmpeg(
    paths: List[Path],
    out_video: Path,
    timing: Timing,
    resize_mode: str,
    codec: Optional[str],
    ffmpeg_args: List[str],
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH. Install/load ffmpeg or use --opencv.")
    if not paths:
        raise ValueError("No images to write.")

    out_video.parent.mkdir(parents=True, exist_ok=True)

    vf = None
    if resize_mode in {"pad", "scale"}:
        target_w, target_h = compute_target_size(paths)
        if resize_mode == "pad":
            vf = (
                f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2"
            )
        else:
            vf = f"scale={target_w}:{target_h}"

    # Default codec choices: PowerPoint-friendly MP4
    if codec is None:
        ext = out_video.suffix.lower()
        codec = "mpeg4" if ext == ".avi" else "libx264"

    frame_dur = timing.frame_duration_s

    with tempfile.TemporaryDirectory() as td:
        list_path = Path(td) / "concat_list.txt"
        with list_path.open("w", encoding="utf-8") as f:
            for p in paths[:-1]:
                f.write(f"file '{p.as_posix()}'\n")
                f.write(f"duration {frame_dur:.9f}\n")
            f.write(f"file '{paths[-1].as_posix()}'\n")

        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
        ]

        if vf:
            cmd += ["-vf", vf]

        cmd += ["-c:v", codec]

        # H.264 compatibility (PowerPoint/QuickTime)
        if codec in {"libx264", "h264"}:
            cmd += ["-pix_fmt", "yuv420p"]

        # Keep an output rate metadata; helps some players
        cmd += ["-r", f"{timing.fps:.6f}"]

        # Add recommended defaults (faststart, crf, preset...) unless user overrides them
        cmd += ffmpeg_args

        cmd += [str(out_video)]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed (exit {e.returncode}). Command was:\n  {' '.join(cmd)}") from e


# -----------------------------
# Outputs
# -----------------------------
def resolve_outputs(out_arg: str, want_gif: bool, want_video: bool) -> Tuple[Optional[Path], Optional[Path]]:
    """
    If -o includes an extension, honor it.
    If -o has no extension, default video to .mp4 and gif to .gif.
    """
    out = Path(out_arg)
    suf = out.suffix.lower()

    gif_path = None
    vid_path = None

    if suf in {".gif", ".mp4", ".avi", ".mkv", ".mov"}:
        if suf == ".gif":
            gif_path = out if want_gif else None
            vid_path = out.with_suffix(".mp4") if want_video else None
        else:
            vid_path = out if want_video else None
            gif_path = out.with_suffix(".gif") if want_gif else None
        return gif_path, vid_path

    # Base name: default video to mp4 (PowerPoint-friendly)
    base = out
    if want_gif:
        gif_path = base.with_suffix(".gif")
    if want_video:
        vid_path = base.with_suffix(".mp4")
    return gif_path, vid_path


# -----------------------------
# CLI
# -----------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compile sequential images into a GIF and/or video.")

    p.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help='Glob for input images (quote it). Example: "pics/*inc*.png"',
    )
    p.add_argument(
        "-o",
        "--out",
        default="output",
        help="Output base name OR full filename. If no extension is provided, video defaults to .mp4.",
    )

    p.add_argument("--gif", action="store_true", help="Write a GIF.")
    p.add_argument("--video", action="store_true", help="Write a video.")

    p.add_argument("--fps", type=float, default=None, help="Frames per second. Default: 30")
    p.add_argument("-t", "--time", type=float, default=None, help="Total duration (seconds).")

    p.add_argument(
        "--resize",
        choices=["none", "pad", "scale"],
        default="none",
        help="Handle non-uniform input sizes: none (require uniform), pad, or scale.",
    )

    # Engine selection (defaults: prefer ffmpeg if available)
    p.add_argument("--ffmpeg", action="store_true", help="Force ffmpeg for video output.")
    p.add_argument("--opencv", action="store_true", help="Force OpenCV for video output.")

    # OpenCV options
    p.add_argument("--fourcc", default=None, help="OpenCV fourcc (e.g., MJPG for .avi, mp4v for .mp4).")

    # ffmpeg options (defaults tuned for PowerPoint-friendly MP4)
    p.add_argument("--codec", default=None, help="ffmpeg video codec (default: libx264 for mp4, mpeg4 for avi).")
    p.add_argument(
        "--ffmpeg-args",
        default="-movflags +faststart -crf 18 -preset medium",
        help="Extra ffmpeg args string. Default is PowerPoint-friendly: "
             "'-movflags +faststart -crf 18 -preset medium'",
    )

    p.add_argument("-v", "--verbose", action="store_true", help="Verbose output.")


    # GIFS
    p.add_argument("--gif-ffmpeg", action="store_true",
               help="Use ffmpeg palettegen/paletteuse for GIF (smaller/better).")

    p.add_argument("--gif-fps", type=float, default=15.0,
                help="GIF frames per second (default: 15).")

    p.add_argument("--gif-width", type=int, default=960,
                help="GIF width in pixels (keep aspect). 0 = keep original size. Default: 960.")

    p.add_argument("--gif-colors", type=int, default=128,
                help="Max GIF palette colors (2-256). Default: 128.")

    p.add_argument("--gif-dither", choices=["none", "bayer", "floyd_steinberg"], default="bayer",
                help="Dithering method for ffmpeg GIF. Default: bayer.")

    args = p.parse_args(argv)

    if not args.gif and not args.video:
        p.error("Choose at least one of --gif or --video.")
    if args.fps is not None and args.time is not None:
        p.error("Choose only one of --fps or --time.")
    if args.opencv and args.ffmpeg:
        p.error("Choose only one of --opencv or --ffmpeg.")

    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # matches = glob.glob(args.input)
    matches = glob.glob('*' + args.input + '*.png')
    matches = sorted(matches, key=natural_key)
    paths = [Path(m).resolve() for m in matches]

    if not paths:
        print(f"ERROR: No images matched input glob: {args.input}", file=sys.stderr)
        return 2

    timing = compute_timing(len(paths), args.fps, args.time)

    out_gif, out_vid = resolve_outputs(args.out, args.gif, args.video)

    # Decide engine: default prefer ffmpeg if available (best PPT compatibility)
    ffmpeg_available = shutil.which("ffmpeg") is not None
    use_ffmpeg_for_gif = args.gif_ffmpeg or ffmpeg_available  # default: yes if available
    use_ffmpeg = False
    if args.video:
        if args.opencv:
            use_ffmpeg = False
        elif args.ffmpeg:
            use_ffmpeg = True
        else:
            use_ffmpeg = ffmpeg_available

    if args.verbose:
        print(f"Matched {len(paths)} images.")
        print(f"Timing: fps={timing.fps:.6f}, frame_duration={timing.frame_duration_s:.6f} s")
        print(f"Resize mode: {args.resize}")
        if args.video:
            print(f"Video engine: {'ffmpeg' if use_ffmpeg else 'opencv'} "
                  f"(ffmpeg_available={ffmpeg_available})")
        if out_vid is not None:
            print(f"Video output: {out_vid}")
        if out_gif is not None:
            print(f"GIF output: {out_gif}")

    # GIF
    if args.gif and out_gif is not None:
        if use_ffmpeg_for_gif:
            # default gif fps: min(video_fps, 15) unless user overrides
            gif_fps = args.gif_fps if args.gif_fps is not None else min(timing.fps, 15.0)
            write_gif_ffmpeg(
                paths, out_gif, timing,
                resize_mode=args.resize,
                gif_fps=gif_fps,
                gif_width=args.gif_width,
                gif_colors=args.gif_colors,
                gif_dither=args.gif_dither,
            )
        else:
            write_gif(paths, out_gif, timing, resize_mode=args.resize)

    # Video
    if args.video and out_vid is not None:
        if use_ffmpeg:
            ff_args = args.ffmpeg_args.strip().split() if args.ffmpeg_args.strip() else []
            write_video_ffmpeg(
                paths,
                out_vid,
                timing,
                resize_mode=args.resize,
                codec=args.codec,
                ffmpeg_args=ff_args,
            )
        else:
            write_video_opencv(
                paths,
                out_vid,
                timing,
                resize_mode=args.resize,
                fourcc=args.fourcc,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
