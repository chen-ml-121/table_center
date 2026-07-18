#!/usr/bin/env python3
"""Interactively capture fisheye checkerboard calibration images.

Default target: GP340-25-12x9 checkerboard (11x8 inner corners).
Press Enter to save a frame when all corners are detected; press q to quit.
"""

import argparse
import time
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True, help="V4L2 index, e.g. 0 or 4")
    parser.add_argument("--name", required=True, help="Camera name, e.g. left or right")
    parser.add_argument("--output", type=Path, default=Path("calibration_images"))
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--cols", type=int, default=11, help="Inner corner columns")
    parser.add_argument("--rows", type=int, default=8, help="Inner corner rows")
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = args.output / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open /dev/video{args.device}")

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if (actual_width, actual_height) != (args.width, args.height):
        cap.release()
        raise RuntimeError(
            f"Requested {args.width}x{args.height}, camera returned "
            f"{actual_width}x{actual_height}"
        )

    pattern_size = (args.cols, args.rows)
    count = len(list(save_dir.glob("*.png")))
    window_name = f"Fisheye calibration: {args.name} /dev/video{args.device}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print(f"Camera: /dev/video{args.device}")
    print(f"Mode: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS, MJPG")
    print(f"Pattern: {args.cols}x{args.rows} inner corners")
    print(f"Output: {save_dir.resolve()}")
    print("Enter: save image (only when detected), q/Esc: quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: failed to read frame")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCornersSB(
                gray,
                pattern_size,
                flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE,
            )

            preview = frame.copy()
            if found:
                cv2.drawChessboardCorners(preview, pattern_size, corners, found)
                status = "DETECTED - press Enter to save"
                color = (0, 255, 0)
            else:
                status = "NOT DETECTED"
                color = (0, 0, 255)

            cv2.putText(preview, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
            cv2.putText(
                preview,
                f"saved: {count}",
                (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (255, 255, 0),
                2,
            )
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (10, 13):
                if not found:
                    print("Not saved: all 11x8 inner corners must be detected")
                    continue
                timestamp_ms = int(time.time() * 1000)
                path = save_dir / f"{args.name}_{count:03d}_{timestamp_ms}.png"
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"Failed to save {path}")
                count += 1
                print(f"Saved [{count}]: {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
