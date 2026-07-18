#!/usr/bin/env python3
"""Capture synchronized checkerboard images and Franka O_T_EE poses."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("robot_ip", help="Franka hostname or IP")
    parser.add_argument("--device", type=int, required=True, help="V4L2 index: left=0, right=4")
    parser.add_argument("--name", required=True, choices=("left", "right"))
    parser.add_argument("--output", type=Path, default=root / "hand_eye_data")
    parser.add_argument(
        "--bridge",
        type=Path,
        default=root / "build_franka_bridge" / "franka_state_bridge",
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--cols", type=int, default=11)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--square-size", type=float, default=0.025)
    parser.add_argument(
        "--robot-samples", type=int, default=10,
        help="O_T_EE readings saved for each image (default: 10)",
    )
    parser.add_argument(
        "--robot-sample-interval-ms", type=float, default=10.0,
        help="Delay between O_T_EE readings (default: 10 ms)",
    )
    return parser.parse_args()


def read_robot_matrix(process):
    request_ns = time.time_ns()
    process.stdin.write("read\n")
    process.stdin.flush()
    line = process.stdout.readline()
    response_ns = time.time_ns()
    if not line:
        error = process.stderr.read().strip()
        raise RuntimeError("Franka bridge stopped: " + (error or "unknown error"))
    values = json.loads(line)
    matrix = np.asarray(values, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError("Franka returned non-finite O_T_EE")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9):
        raise RuntimeError("Franka returned an invalid homogeneous matrix")
    return matrix, (request_ns + response_ns) // 2, response_ns - request_ns


def rotation_distance(left, right):
    relative = left[:3, :3].T @ right[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def representative_pose_index(matrices):
    """Return the SE(3) medoid index, robust against an occasional stale read."""
    scores = []
    for candidate in matrices:
        score = 0.0
        for other in matrices:
            translation_error = np.linalg.norm(candidate[:3, 3] - other[:3, 3])
            rotation_error = rotation_distance(candidate, other)
            score += translation_error + 0.05 * rotation_error
        scores.append(score)
    return int(np.argmin(scores))


def read_robot_pose_batch(process, count, interval_seconds):
    matrices = []
    timestamps = []
    durations = []
    for sample_index in range(count):
        matrix, timestamp_ns, duration_ns = read_robot_matrix(process)
        matrices.append(matrix)
        timestamps.append(timestamp_ns)
        durations.append(duration_ns)
        if sample_index + 1 < count and interval_seconds > 0:
            time.sleep(interval_seconds)
    representative = representative_pose_index(matrices)
    return matrices, timestamps, durations, representative


def write_manifest(path, args, samples):
    payload = {
        "schema_version": 1,
        "arm": args.name,
        "robot_ip": args.robot_ip,
        "robot_pose": "base_T_EE_measured_O_T_EE",
        "matrix_layout": "row_major_4x4",
        "camera_device": f"/dev/video{args.device}",
        "image_size": [args.width, args.height],
        "checkerboard_inner_corners": [args.cols, args.rows],
        "square_size_m": args.square_size,
        "robot_samples_per_image": args.robot_samples,
        "robot_sample_interval_ms": args.robot_sample_interval_ms,
        "representative_pose_method": "SE3_medoid",
        "samples": samples,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    if args.robot_samples < 1:
        raise ValueError("--robot-samples must be at least 1")
    if args.robot_sample_interval_ms < 0:
        raise ValueError("--robot-sample-interval-ms cannot be negative")
    bridge = args.bridge.expanduser().resolve()
    if not bridge.is_file():
        raise FileNotFoundError(f"Franka bridge not found: {bridge}")

    session_dir = args.output / args.name
    image_dir = session_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = session_dir / "samples.json"
    samples = []
    if manifest_path.exists():
        samples = json.loads(manifest_path.read_text(encoding="utf-8")).get("samples", [])

    process = subprocess.Popen(
        [str(bridge), args.robot_ip],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    cap = None
    try:
        ready = process.stdout.readline().strip()
        if ready != "READY":
            error = process.stderr.read().strip()
            raise RuntimeError("Failed to connect to Franka: " + (error or ready))

        cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open /dev/video{args.device}")

        actual_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if actual_size != (args.width, args.height):
            raise RuntimeError(f"Camera returned {actual_size}, expected {(args.width, args.height)}")

        pattern = (args.cols, args.rows)
        window = f"Hand-eye capture: {args.name}"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1280, 720)
        print(f"Connected to Franka {args.robot_ip} and /dev/video{args.device}")
        print("Enter: save image + O_T_EE, Backspace: delete last, q/Esc: quit")

        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCornersSB(
                gray,
                pattern,
                flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE,
            )

            preview = frame.copy()
            color = (0, 255, 0) if found else (0, 0, 255)
            if found:
                cv2.drawChessboardCorners(preview, pattern, corners, found)
            cv2.putText(
                preview,
                ("DETECTED" if found else "NOT DETECTED") + f"  saved: {len(samples)}",
                (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2,
            )
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key in (8, 127):
                if samples:
                    removed = samples.pop()
                    image_path = session_dir / removed["image"]
                    if image_path.exists():
                        image_path.unlink()
                    write_manifest(manifest_path, args, samples)
                    print(f"Deleted sample {removed['index']}")
                continue
            if key not in (10, 13):
                continue
            if not found:
                print("Not saved: all checkerboard corners must be detected")
                continue

            matrices, timestamps_ns, durations_ns, representative_index = read_robot_pose_batch(
                process,
                args.robot_samples,
                args.robot_sample_interval_ms / 1000.0,
            )
            robot_matrix = matrices[representative_index]
            index = len(samples)
            image_name = f"sample_{index:03d}.png"
            image_path = image_dir / image_name
            if not cv2.imwrite(str(image_path), frame):
                raise RuntimeError(f"Failed to save {image_path}")
            samples.append({
                "index": index,
                "image": str(Path("images") / image_name),
                "captured_at": datetime.now().astimezone().isoformat(timespec="microseconds"),
                "robot_timestamp_ns": timestamps_ns[representative_index],
                "robot_request_duration_ns": durations_ns[representative_index],
                "base_T_EE": robot_matrix.tolist(),
                "base_T_EE_representative_index": representative_index,
                "base_T_EE_samples": [matrix.tolist() for matrix in matrices],
                "robot_timestamps_ns": timestamps_ns,
                "robot_request_durations_ns": durations_ns,
            })
            write_manifest(manifest_path, args, samples)
            print(f"Saved sample {index}: {image_path}")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if process.poll() is None:
            try:
                process.stdin.write("q\n")
                process.stdin.flush()
                process.wait(timeout=3)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                process.terminate()


if __name__ == "__main__":
    main()
