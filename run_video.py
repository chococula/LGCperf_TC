"""Turn the frames saved for one measurement run into an MP4 and a GIF that play in real time.

The video runs from the pre-roll before Power ON (frames with negative elapsed time) until the
first frame that reaches the Home similarity threshold, i.e. where the measured timer stops.
Frames are the files written by Omedia_TC01: frames/frame_<n>_<elapsed>ms_sim<score>.jpg.
Usage for runs recorded before frames carried an overlay:
    python run_video.py C:/Temp/Omedia_TC01_01_... --overlay --threshold 0.7
"""
import argparse
import os
import re
import shutil
import subprocess
import tempfile

import cv2

FRAME_PATTERN = re.compile(r"frame_(p?\d+)_(-?\d+(?:\.\d+)?)ms_sim(-?\d+(?:\.\d+)?)\.jpg$")
END_HOLD_SECONDS = 1.5
VIDEO_FPS = 30
GIF_WIDTH = 480
GIF_FPS = 10


def frame_entries(frames_dir):
    entries = []
    for name in os.listdir(frames_dir):
        match = FRAME_PATTERN.match(name)
        if match:
            entries.append((match.group(1), float(match.group(2)), float(match.group(3)),
                            os.path.join(frames_dir, name)))
    return sorted(entries, key=lambda entry: entry[1])


def detection_index(entries, threshold):
    """Index of the first frame after Power ON that reaches the threshold after a frame below it."""
    armed = False
    for index, (_, elapsed_ms, score, _) in enumerate(entries):
        if elapsed_ms < 0:
            continue
        if score < threshold:
            armed = True
        elif armed:
            return index
    return None


def _ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _overlay_frames(entries, work_dir, threshold, run_label, draw_overlay):
    """Draw timer and similarity (taken from each file name) onto copies of the frames.

    Returns (path, elapsed_ms) pairs for the frames that could be read.
    """
    armed = False
    detected_ms = None
    paths = []
    for index, (number, elapsed_ms, score, path) in enumerate(entries):
        if elapsed_ms < 0:
            lines = [
                (f"{run_label}  BEFORE POWER ON  {elapsed_ms / 1000.0:.3f} s", (255, 255, 255), 1.4),
                (f"Similarity {score:.3f} / {threshold:.3f}", (0, 255, 255), 1.0),
            ]
        else:
            if score < threshold:
                armed = True
            if armed and detected_ms is None and score >= threshold:
                detected_ms = elapsed_ms
            color = (0, 255, 0) if score >= threshold else (0, 255, 255)
            lines = [
                (f"{run_label}  #{int(number) if number.isdigit() else number}  TIMER {elapsed_ms / 1000.0:.3f} s", color, 1.4),
                (f"Similarity {score:.3f} / {threshold:.3f}", color, 1.0),
            ]
            if detected_ms is not None:
                lines.append((f"HOME DETECTED at {detected_ms / 1000.0:.3f} s", (0, 255, 0), 1.0))
        frame = cv2.imread(path)
        if frame is None:
            continue
        out_path = os.path.join(work_dir, f"{index:05d}.jpg")
        cv2.imwrite(out_path, draw_overlay(frame, lines))
        paths.append((out_path, elapsed_ms))
    return paths


def _write_mp4(paths, times, mp4_path):
    """Place frames on a constant VIDEO_FPS timeline by their capture time, so playback is real time."""
    first = cv2.imread(paths[0])
    height, width = first.shape[0] // 2 * 2, first.shape[1] // 2 * 2
    process = subprocess.Popen([
        _ffmpeg(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(VIDEO_FPS), "-i", "-",
        "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", mp4_path,
    ], stdin=subprocess.PIPE)
    start = times[0]
    end = times[-1] + END_HOLD_SECONDS
    index = 0
    current = None
    for tick in range(int(round((end - start) * VIDEO_FPS)) + 1):
        moment = start + tick / VIDEO_FPS
        while index < len(paths) and times[index] <= moment + 1e-9:
            current = cv2.imread(paths[index])[:height, :width]
            index += 1
        process.stdin.write(current.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise subprocess.CalledProcessError(process.returncode, "ffmpeg")


def make_run_videos(run_dir, add_overlay=False, threshold=0.7, run_label="RUN", draw_overlay=None):
    """Create run.mp4 and run.gif in run_dir. Returns {'mp4': path, 'gif': path} or None."""
    frames_dir = os.path.join(run_dir, "frames")
    if not os.path.isdir(frames_dir):
        print(f"[VIDEO] No frames folder in {run_dir}; skipping.")
        return None
    entries = frame_entries(frames_dir)
    if not entries:
        print(f"[VIDEO] No frames in {frames_dir}; skipping.")
        return None
    end = detection_index(entries, threshold)
    if end is not None:
        entries = entries[:end + 1]

    work_dir = tempfile.mkdtemp(prefix="run_video_")
    try:
        if add_overlay:
            if draw_overlay is None:
                from Omedia_TC01 import draw_overlay
            pairs = _overlay_frames(entries, work_dir, threshold, run_label, draw_overlay)
        else:
            pairs = [(entry[3], entry[1]) for entry in entries]
        paths = [path for path, _ in pairs]
        times = [elapsed_ms / 1000.0 for _, elapsed_ms in pairs]
        mp4_path = os.path.join(run_dir, "run.mp4")
        gif_path = os.path.join(run_dir, "run.gif")
        _write_mp4(paths, times, mp4_path)
        subprocess.run([
            _ffmpeg(), "-y", "-loglevel", "error", "-i", mp4_path,
            "-vf", f"fps={GIF_FPS},scale={GIF_WIDTH}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer",
            "-loop", "0", gif_path,
        ], check=True)
    except (subprocess.CalledProcessError, OSError, ImportError) as error:
        print(f"[VIDEO] Could not create the run video: {error}")
        return None
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"[VIDEO] Saved {mp4_path} ({os.path.getsize(mp4_path) / 1e6:.1f} MB) and "
          f"{gif_path} ({os.path.getsize(gif_path) / 1e6:.1f} MB)")
    return {'mp4': mp4_path, 'gif': gif_path}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", help="Run output folder(s) that contain a frames/ folder")
    parser.add_argument("--overlay", action="store_true", help="Draw timer and similarity from file names")
    parser.add_argument("--threshold", type=float, default=0.7, help="Home similarity threshold (default 0.7)")
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        match = re.search(r"_(\d{2})_\d{8}_\d{6}", os.path.basename(os.path.normpath(run_dir)))
        label = f"RUN {int(match.group(1))}" if match else "RUN"
        make_run_videos(run_dir, add_overlay=args.overlay, threshold=args.threshold, run_label=label)


if __name__ == "__main__":
    main()
