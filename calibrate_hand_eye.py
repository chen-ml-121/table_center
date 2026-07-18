#!/usr/bin/env python3
"""Calibrate an eye-in-hand camera from fixed-checkerboard observations."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Directory containing samples.json")
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pnp-rmse", type=float, default=1.0)
    return parser.parse_args()


def transform(rotation, translation):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    result[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return result


def rotation_angle(rotation):
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def rotation_medoid(rotations):
    scores = [
        sum(rotation_angle(candidate.T @ other) for other in rotations)
        for candidate in rotations
    ]
    return int(np.argmin(scores))


def evaluate_hand_eye(base_T_EE, camera_T_target, EE_T_camera):
    base_T_target = [
        base_EE @ EE_T_camera @ camera_target
        for base_EE, camera_target in zip(base_T_EE, camera_T_target)
    ]
    translations = np.asarray([pose[:3, 3] for pose in base_T_target])
    rotations = [pose[:3, :3] for pose in base_T_target]
    center_translation = np.median(translations, axis=0)
    center_rotation = rotations[rotation_medoid(rotations)]
    translation_errors = np.linalg.norm(translations - center_translation, axis=1)
    rotation_errors = np.asarray(
        [rotation_angle(center_rotation.T @ rotation) for rotation in rotations]
    )
    return {
        "base_T_target": base_T_target,
        "center": transform(center_rotation, center_translation),
        "translation_errors_m": translation_errors,
        "rotation_errors_rad": rotation_errors,
    }


def solve_method(base_T_EE, camera_T_target, method):
    R_gripper2base = [pose[:3, :3] for pose in base_T_EE]
    t_gripper2base = [pose[:3, 3].reshape(3, 1) for pose in base_T_EE]
    R_target2cam = [pose[:3, :3] for pose in camera_T_target]
    t_target2cam = [pose[:3, 3].reshape(3, 1) for pose in camera_T_target]
    R_camera2gripper, t_camera2gripper = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=method,
    )
    return transform(R_camera2gripper, t_camera2gripper)


def main():
    args = parse_args()
    manifest_path = args.data / "samples.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    intrinsics = json.loads(args.intrinsics.read_text(encoding="utf-8"))

    K = np.asarray(intrinsics["K"], dtype=np.float64)
    D = np.asarray(intrinsics["D"], dtype=np.float64).reshape(4, 1)
    cols, rows = manifest["checkerboard_inner_corners"]
    square_size = float(manifest["square_size_m"])
    pattern = (int(cols), int(rows))
    object_points = np.zeros((cols * rows, 3), dtype=np.float64)
    object_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    base_T_EE = []
    camera_T_target = []
    sample_indices = []
    pnp_rmse = []
    rejected = []

    for sample in manifest["samples"]:
        image_path = args.data / sample["image"]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            rejected.append({"index": sample["index"], "reason": "image_read_failed"})
            continue
        found, corners = cv2.findChessboardCornersSB(
            image,
            pattern,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE,
        )
        if not found:
            rejected.append({"index": sample["index"], "reason": "corners_not_found"})
            continue

        # Convert fisheye pixels to normalized pinhole coordinates, then solve PnP.
        normalized = cv2.fisheye.undistortPoints(
            corners.astype(np.float64), K, D
        ).reshape(-1, 1, 2)
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            normalized,
            np.eye(3, dtype=np.float64),
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            rejected.append({"index": sample["index"], "reason": "solvepnp_failed"})
            continue

        projected_normalized, _ = cv2.projectPoints(
            object_points, rvec, tvec, np.eye(3), None
        )
        # Report error in original fisheye pixels for an interpretable threshold.
        projected_pixels = cv2.fisheye.distortPoints(
            projected_normalized.astype(np.float64), K, D
        )
        error = float(
            np.sqrt(np.mean(np.sum(
                (corners.reshape(-1, 2) - projected_pixels.reshape(-1, 2)) ** 2,
                axis=1,
            )))
        )
        if error > args.max_pnp_rmse:
            rejected.append({
                "index": sample["index"], "reason": "high_pnp_rmse", "rmse_px": error
            })
            continue

        rotation, _ = cv2.Rodrigues(rvec)
        base_T_EE.append(np.asarray(sample["base_T_EE"], dtype=np.float64))
        camera_T_target.append(transform(rotation, tvec))
        sample_indices.append(int(sample["index"]))
        pnp_rmse.append(error)

    if len(base_T_EE) < 10:
        raise SystemExit(f"Only {len(base_T_EE)} valid samples; at least 10 are required")

    candidates = []
    for name, method in METHODS.items():
        EE_T_camera = solve_method(base_T_EE, camera_T_target, method)
        evaluation = evaluate_hand_eye(base_T_EE, camera_T_target, EE_T_camera)
        translation_mm = evaluation["translation_errors_m"] * 1000.0
        rotation_deg = np.degrees(evaluation["rotation_errors_rad"])
        score = float(np.median(translation_mm) + np.median(rotation_deg))
        candidates.append({
            "method": name,
            "score": score,
            "EE_T_camera": EE_T_camera,
            "base_T_target": evaluation["center"],
            "translation_median_mm": float(np.median(translation_mm)),
            "translation_max_mm": float(np.max(translation_mm)),
            "rotation_median_deg": float(np.median(rotation_deg)),
            "rotation_max_deg": float(np.max(rotation_deg)),
        })

    candidates.sort(key=lambda item: item["score"])
    best = candidates[0]
    result = {
        "schema_version": 1,
        "arm": manifest.get("arm"),
        "convention": "base_T_target = base_T_EE @ EE_T_camera @ camera_T_target",
        "selected_method": best["method"],
        "EE_T_camera": best["EE_T_camera"].tolist(),
        "camera_T_EE": np.linalg.inv(best["EE_T_camera"]).tolist(),
        "base_T_checkerboard": best["base_T_target"].tolist(),
        "valid_sample_indices": sample_indices,
        "pnp_rmse_px": {
            "median": float(np.median(pnp_rmse)),
            "max": float(np.max(pnp_rmse)),
        },
        "consistency": {
            "translation_median_mm": best["translation_median_mm"],
            "translation_max_mm": best["translation_max_mm"],
            "rotation_median_deg": best["rotation_median_deg"],
            "rotation_max_deg": best["rotation_max_deg"],
        },
        "method_comparison": [
            {key: value for key, value in item.items() if key not in {"EE_T_camera", "base_T_target"}}
            for item in candidates
        ],
        "rejected": rejected,
        "intrinsics_file": str(args.intrinsics),
        "data_manifest": str(manifest_path),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Valid samples: {len(sample_indices)} / {len(manifest['samples'])}")
    print(f"PnP RMSE: median={np.median(pnp_rmse):.4f} px max={np.max(pnp_rmse):.4f} px")
    for candidate in candidates:
        print(
            f"{candidate['method']:10s} "
            f"translation median/max={candidate['translation_median_mm']:.3f}/"
            f"{candidate['translation_max_mm']:.3f} mm  "
            f"rotation median/max={candidate['rotation_median_deg']:.3f}/"
            f"{candidate['rotation_max_deg']:.3f} deg"
        )
    print(f"Selected: {best['method']}")
    print("EE_T_camera =")
    print(best["EE_T_camera"])
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
