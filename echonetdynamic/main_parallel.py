#!/usr/bin/env python3
import os
import cv2
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import Set

from data_generator_functions import dataset_path, get_reference_cone


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process ultrasound images for speckle removal research",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--angle", type=float, default=80, help="Cone angle for resizing in degrees"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: {dataset_path}/processed_cut_{angle})",
    )
    parser.add_argument(
        "--reference_patient",
        type=str,
        default="0XFEBEEFF93F6FEB9",
        help="Patient ID to use as reference for cone extraction",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads for parallel processing",
    )
    return parser.parse_args()


def process_patient(
    patient: str,
    processed_patients: Set[str],
    processed_patients_file: str,
    cone,
    resized_cone,
    top_point,
    angle: float,
    dest_path: str,
) -> bool:
    """Process a single patient's ultrasound data.

    Args:
        patient: Patient identifier
        processed_patients: Set of already processed patients
        processed_patients_file: File to track processed patients
        cone: Reference cone mask
        resized_cone: Resized reference cone mask
        top_point: Reference cone apex point
        angle: Cone angle for resizing
        dest_path: Output directory path

    Returns:
        True if processing was successful, False otherwise
    """
    global cases, wrong_cases, good_cases

    if patient in processed_patients:
        return False

    video_path = os.path.join(dataset_path, patient, "video.npy")
    if not os.path.exists(video_path):
        return False

    all_img = np.load(video_path)
    cases += 1
    c, rc, patient_top_point = get_reference_cone(patient, angle)

    if patient_top_point is None:
        return False

    top_point_diff_0 = abs(patient_top_point[0] - top_point[0])
    top_point_diff_1 = abs(patient_top_point[1] - top_point[1])

    if top_point_diff_0 > 3 or top_point_diff_1 > 3:
        wrong_cases += 1
        return False

    seq = 0
    for time in range(all_img.shape[0]):
        if time % 2 != 0:
            continue
        np_img = all_img[time]
        src = np_img.astype(np.uint8)

        # Save processed images
        output_prefix = os.path.join(dest_path, f"{patient}")

        cv2.imwrite(f"{output_prefix}_whole_segment_{seq}.png", cone * src)
        cv2.imwrite(f"{output_prefix}_red_segment_{seq}.png", resized_cone * src)
        cv2.imwrite(
            f"{output_prefix}_difference_{seq}.png", (cone - resized_cone) * src
        )
        cv2.imwrite(f"{output_prefix}_img_{seq}.png", np_img)

        # Save numpy arrays
        np.save(f"{output_prefix}_difference_{seq}.npy", cone - resized_cone)
        np.save(f"{output_prefix}_red_mask_{seq}.npy", resized_cone)
        np.save(f"{output_prefix}_blue_mask_{seq}.npy", cone)
        seq += 1

    good_cases += 1

    # Save patient to processed list
    with open(processed_patients_file, "a") as f:
        f.write(f"{patient}\n")
    print(f"Processed: {patient}, Good: {good_cases}")
    return True


def main():
    """Main entry point for the script."""
    global cases, wrong_cases, good_cases
    cases = 0
    wrong_cases = 0
    good_cases = 0

    # Parse command-line arguments
    args = parse_arguments()
    angle = args.angle

    # Set up output directory
    dataset_dest_path = args.output_dir or os.path.join(
        dataset_path, f"processed_cut_{angle}"
    )
    os.makedirs(dataset_dest_path, exist_ok=True)

    # Tracking file for processed patients
    processed_patients_file = os.path.join(
        dataset_path, f"processed_patients_{angle}.txt"
    )

    # Load reference cone
    cone, resized_cone, top_point = get_reference_cone(args.reference_patient, angle)
    if cone is None or resized_cone is None or top_point is None:
        print(
            f"Error: Failed to load reference cone from patient {args.reference_patient}"
        )
        return 1

    print(
        f"Using reference cone from patient {args.reference_patient} with angle {angle}"
    )

    # Load processed patients
    processed_patients = set()
    if os.path.exists(processed_patients_file):
        with open(processed_patients_file, "r") as f:
            processed_patients = set(f.read().splitlines())
        print(f"Found {len(processed_patients)} already processed patients")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        patients = [
            patient
            for patient in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, patient))
        ]

        tasks = [
            (
                patient,
                processed_patients,
                processed_patients_file,
                cone,
                resized_cone,
                top_point,
                angle,
                dataset_dest_path,
            )
            for patient in patients
        ]

        # Execute tasks in parallel
        executor.map(lambda args: process_patient(*args), tasks)

    print(f"TOTAL CASES: {cases}")
    print(f"WRONG CASES: {wrong_cases}")
    print(f"GOOD CASES: {good_cases}")

    return 0


if __name__ == "__main__":
    exit(main())
