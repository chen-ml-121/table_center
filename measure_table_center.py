#!/usr/bin/env python3
"""Measure an AprilTag table-center frame in a Franka base coordinate system."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from capture_hand_eye import read_robot_matrix, read_robot_pose_batch, rotation_distance


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("robot_ip")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--name", choices=("left", "right"), required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--hand-eye", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag-id", type=int, default=0)
    parser.add_argument("--tag-size", type=float, default=0.120, help="Black boundary in meters")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--robot-samples", type=int, default=10)
    parser.add_argument("--discard-robot-reads", type=int, default=1)
    parser.add_argument("--robot-sample-interval-ms", type=float, default=10.0)
    parser.add_argument(
        "--bridge", type=Path,
        default=root / "build_franka_bridge" / "franka_state_bridge",
    )
    return parser.parse_args()


def transform(rotation, translation):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(rotation).reshape(3, 3)
    result[:3, 3] = np.asarray(translation).reshape(3)
    return result


def tag_object_points(size):
    half = size / 2.0
    # OpenCV IPPE_SQUARE order: top-left, top-right, bottom-right, bottom-left.
    return np.asarray([
        [-half, half, 0.0],
        [half, half, 0.0],
        [half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float64)


def fisheye_project(points, camera_T_object, K, D):
    rvec, _ = cv2.Rodrigues(camera_T_object[:3, :3])
    object_points = np.ascontiguousarray(
        np.asarray(points, dtype=np.float64).reshape(1, -1, 3)
    )
    rvec = np.ascontiguousarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.ascontiguousarray(
        camera_T_object[:3, 3], dtype=np.float64
    ).reshape(3, 1)
    projected, _ = cv2.fisheye.projectPoints(
        object_points,
        rvec,
        tvec,
        np.ascontiguousarray(K, dtype=np.float64),
        np.ascontiguousarray(D, dtype=np.float64),
    )
    return projected.reshape(-1, 2)


def estimate_tag_pose(corners, size, K, D):
    objects = tag_object_points(size)
    normalized = cv2.fisheye.undistortPoints(
        corners.reshape(-1, 1, 2).astype(np.float64), K, D
    )
    success, rvec, tvec = cv2.solvePnP(
        objects,
        normalized,
        np.eye(3, dtype=np.float64),
        None,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success or float(tvec.reshape(-1)[2]) <= 0:
        return None, None
    rotation, _ = cv2.Rodrigues(rvec)
    camera_T_tag = transform(rotation, tvec)
    projected = fisheye_project(objects, camera_T_tag, K, D)
    rmse = float(np.sqrt(np.mean(np.sum((corners - projected) ** 2, axis=1))))
    return camera_T_tag, rmse


def pose_medoid_index(poses):
    scores = []
    for candidate in poses:
        score = 0.0
        for other in poses:
            score += np.linalg.norm(candidate[:3, 3] - other[:3, 3])
            score += 0.05 * rotation_distance(candidate, other)
        scores.append(score)
    return int(np.argmin(scores))


def summarize(measurements):
    poses = [np.asarray(item["base_T_tag"], dtype=np.float64) for item in measurements]
    medoid = pose_medoid_index(poses)
    center = poses[medoid]
    translations = np.asarray([pose[:3, 3] for pose in poses])
    translation_errors = np.linalg.norm(translations - center[:3, 3], axis=1) * 1000.0
    rotation_errors = np.asarray([
        np.degrees(rotation_distance(center, pose)) for pose in poses
    ])
    return {
        "method": "SE3_medoid",
        "measurement_index": medoid,
        "base_T_table": center.tolist(),
        "center_xyz_m": center[:3, 3].tolist(),
        "translation_median_mm": float(np.median(translation_errors)),
        "translation_max_mm": float(np.max(translation_errors)),
        "rotation_median_deg": float(np.median(rotation_errors)),
        "rotation_max_deg": float(np.max(rotation_errors)),
    }


def write_output(path, args, K, D, EE_T_camera, measurements):
    result = {
        "schema_version": 1,
        "arm": args.name,
        "robot_ip": args.robot_ip,
        "tag_family": "tag36h11",
        "tag_id": args.tag_id,
        "tag_size_m": args.tag_size,
        "convention": "base_T_tag = base_T_EE @ EE_T_camera @ camera_T_tag",
        "K": K.tolist(),
        "D": D.reshape(-1).tolist(),
        "EE_T_camera": EE_T_camera.tolist(),
        "measurements": measurements,
        "summary": summarize(measurements) if measurements else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    intrinsics = json.loads(args.intrinsics.read_text(encoding="utf-8"))
    hand_eye = json.loads(args.hand_eye.read_text(encoding="utf-8"))
    K = np.asarray(intrinsics["K"], dtype=np.float64)
    D = np.asarray(intrinsics["D"], dtype=np.float64).reshape(4, 1)
    EE_T_camera = np.asarray(hand_eye["EE_T_camera"], dtype=np.float64)
    bridge = args.bridge.expanduser().resolve()
    if not bridge.is_file():
        raise FileNotFoundError(f"Franka bridge not found: {bridge}")

    measurements = []
    if args.output.exists():
        measurements = json.loads(args.output.read_text(encoding="utf-8")).get("measurements", [])

    process = subprocess.Popen(
        [str(bridge), args.robot_ip], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, bufsize=1,
    )
    cap = None
    try:
        ready = process.stdout.readline().strip()
        if ready != "READY":
            raise RuntimeError("Failed to connect to Franka: " + (process.stderr.read().strip() or ready))

        cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open /dev/video{args.device}")

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        window = f"Table center: {args.name}"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1280, 720)
        print("Enter: measure table center, Backspace: delete last, q/Esc: quit")

        latest_pose = None
        latest_rmse = None
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            corners_list, ids, _ = detector.detectMarkers(frame)
            latest_pose = None
            latest_rmse = None
            selected_corners = None
            if ids is not None:
                for corners, marker_id in zip(corners_list, ids.reshape(-1)):
                    if int(marker_id) == args.tag_id:
                        selected_corners = corners.reshape(4, 2).astype(np.float64)
                        latest_pose, latest_rmse = estimate_tag_pose(
                            selected_corners, args.tag_size, K, D
                        )
                        break

            preview = frame.copy()
            if latest_pose is not None:
                cv2.polylines(preview, [selected_corners.astype(np.int32)], True, (0, 255, 0), 3)
                origin_and_axes = np.asarray([
                    [0, 0, 0], [0.04, 0, 0], [0, 0.04, 0], [0, 0, 0.04]
                ], dtype=np.float64)
                pixels = fisheye_project(origin_and_axes, latest_pose, K, D).astype(int)
                origin = tuple(pixels[0])
                cv2.line(preview, origin, tuple(pixels[1]), (0, 0, 255), 3)
                cv2.line(preview, origin, tuple(pixels[2]), (0, 255, 0), 3)
                cv2.line(preview, origin, tuple(pixels[3]), (255, 0, 0), 3)
                text = f"ID {args.tag_id} RMSE {latest_rmse:.3f}px saved {len(measurements)}"
                color = (0, 255, 0)
            else:
                text = f"Tag ID {args.tag_id} not detected  saved {len(measurements)}"
                color = (0, 0, 255)
            cv2.putText(preview, text, (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key in (8, 127):
                if measurements:
                    measurements.pop()
                    write_output(args.output, args, K, D, EE_T_camera, measurements)
                    print("Deleted last measurement")
                continue
            if key not in (10, 13):
                continue
            if latest_pose is None:
                print("Not saved: requested AprilTag is not detected")
                continue

            for _ in range(args.discard_robot_reads):
                read_robot_matrix(process)
                time.sleep(args.robot_sample_interval_ms / 1000.0)
            matrices, timestamps, durations, representative = read_robot_pose_batch(
                process, args.robot_samples, args.robot_sample_interval_ms / 1000.0
            )
            base_T_EE = matrices[representative]
            base_T_tag = base_T_EE @ EE_T_camera @ latest_pose
            measurements.append({
                "index": len(measurements),
                "captured_at": datetime.now().astimezone().isoformat(timespec="microseconds"),
                "tag_pnp_rmse_px": latest_rmse,
                "camera_T_tag": latest_pose.tolist(),
                "base_T_EE": base_T_EE.tolist(),
                "base_T_EE_representative_index": representative,
                "base_T_EE_samples": [matrix.tolist() for matrix in matrices],
                "robot_timestamps_ns": timestamps,
                "robot_request_durations_ns": durations,
                "base_T_tag": base_T_tag.tolist(),
                "center_xyz_m": base_T_tag[:3, 3].tolist(),
            })
            write_output(args.output, args, K, D, EE_T_camera, measurements)
            summary = summarize(measurements)
            xyz = summary["center_xyz_m"]
            print(
                f"Saved {len(measurements)}: center=({xyz[0]:.6f}, {xyz[1]:.6f}, "
                f"{xyz[2]:.6f}) m, spread median/max="
                f"{summary['translation_median_mm']:.3f}/"
                f"{summary['translation_max_mm']:.3f} mm"
            )
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
