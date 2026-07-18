#!/usr/bin/env python3
"""Calibrate one OpenCV fisheye camera from checkerboard images."""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cols", type=int, default=11, help="Inner corner columns")
    parser.add_argument("--rows", type=int, default=8, help="Inner corner rows")
    parser.add_argument("--square-size", type=float, default=0.025, help="Meters")
    return parser.parse_args()


def image_files(directory):
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in suffixes)


def main():
    args = parse_args()
    files = image_files(args.images)
    if not files:
        raise SystemExit(f"No images found in {args.images}")

    pattern = (args.cols, args.rows)
    object_template = np.zeros((1, args.cols * args.rows, 3), np.float64)
    object_template[0, :, :2] = (
        np.mgrid[0 : args.cols, 0 : args.rows].T.reshape(-1, 2) * args.square_size
    )

    object_points = []
    image_points = []
    accepted = []
    rejected = []
    image_size = None

    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            rejected.append({"file": str(path), "reason": "read_failed"})
            continue
        size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = size
        elif size != image_size:
            rejected.append({"file": str(path), "reason": f"size_{size}"})
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCornersSB(
            gray,
            pattern,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE,
        )
        if not found:
            rejected.append({"file": str(path), "reason": "corners_not_found"})
            continue

        object_points.append(object_template.copy())
        image_points.append(corners.reshape(1, -1, 2).astype(np.float64))
        accepted.append(path)

    print(f"Images: {len(files)}")
    print(f"Accepted: {len(accepted)}")
    print(f"Rejected: {len(rejected)}")
    if len(accepted) < 20:
        raise SystemExit("Fewer than 20 valid images; collect more diverse images")

    flags = (
        cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
        | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_CHECK_COND
        | cv2.fisheye.CALIB_FIX_SKEW
    )
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        200,
        1e-9,
    )

    while True:
        # OpenCV's fisheye initializer can fail on near-circular, very-wide-FoV
        # images when starting from an all-zero K. Obtain a robust pinhole
        # estimate only as an initialization, then optimize the fisheye model.
        pinhole_objects = [
            obj.reshape(-1, 3).astype(np.float32) for obj in object_points
        ]
        pinhole_images = [
            img.reshape(-1, 1, 2).astype(np.float32) for img in image_points
        ]
        _, K, _, _, _ = cv2.calibrateCamera(
            pinhole_objects, pinhole_images, image_size, None, None
        )
        K = K.astype(np.float64)
        D = np.zeros((4, 1), dtype=np.float64)
        try:
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                object_points,
                image_points,
                image_size,
                K,
                D,
                None,
                None,
                flags=flags,
                criteria=criteria,
            )
            break
        except cv2.error as exc:
            match = re.search(r"input array (\d+)", str(exc))
            if not match:
                raise
            bad_index = int(match.group(1))
            bad_path = accepted.pop(bad_index)
            object_points.pop(bad_index)
            image_points.pop(bad_index)
            rejected.append({"file": str(bad_path), "reason": "ill_conditioned_view"})
            print(f"Removed ill-conditioned view: {bad_path}")
            if len(accepted) < 20:
                raise SystemExit("Too few valid images after removing unstable views")

    per_view = []
    all_squared_error = 0.0
    all_point_count = 0
    for path, obj, observed, rvec, tvec in zip(
        accepted, object_points, image_points, rvecs, tvecs
    ):
        projected, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, K, D)
        residual = observed.reshape(-1, 2) - projected.reshape(-1, 2)
        rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        per_view.append({"file": str(path), "rmse_px": rmse})
        all_squared_error += float(np.sum(residual * residual))
        all_point_count += residual.shape[0]

    reprojection_rmse = float(np.sqrt(all_squared_error / all_point_count))
    per_view.sort(key=lambda item: item["rmse_px"], reverse=True)

    result = {
        "camera_model": "opencv_fisheye",
        "image_width": image_size[0],
        "image_height": image_size[1],
        "pattern_inner_corners": [args.cols, args.rows],
        "square_size_m": args.square_size,
        "valid_images": len(accepted),
        "rejected_images": len(rejected),
        "opencv_rms_px": float(rms),
        "reprojection_rmse_px": reprojection_rmse,
        "K": K.tolist(),
        "D": D.reshape(-1).tolist(),
        "worst_views": per_view[:10],
        "rejected": rejected,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Image size: {image_size[0]}x{image_size[1]}")
    print(f"OpenCV RMS: {rms:.4f} px")
    print(f"Reprojection RMSE: {reprojection_rmse:.4f} px")
    print("K =")
    print(K)
    print("D =")
    print(D.reshape(-1))
    print("Worst views:")
    for item in per_view[:5]:
        print(f"  {item['rmse_px']:.4f} px  {item['file']}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
