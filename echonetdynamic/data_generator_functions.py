"""
Ultrasound Image Processing Utilities for Speckle Removal

This module provides functions for processing ultrasound images from the EchoNet-Dynamic dataset,
including cone detection, triangle generation, and geometric transformations.
"""

import os
import math
import numpy as np
import cv2
from scipy.ndimage import label
from typing import Tuple, Optional

# Default dataset path can be overridden by environment variables or command-line arguments
DEFAULT_DATASET_PATH = "/results/mgazda/EchoNet-Dynamic-Processed"
dataset_path = os.environ.get("ECHONET_DATASET_PATH", DEFAULT_DATASET_PATH)


def set_dataset_path(path: str) -> None:
    """
    Set the dataset path globally for this module.

    Args:
        path: New path to the dataset
    """
    global dataset_path
    dataset_path = path
    print(f"Dataset path set to: {dataset_path}")


def get_dataset_path() -> str:
    """
    Get the current dataset path.

    Returns:
        Current dataset path
    """
    return dataset_path


def get_reference_cone(
    patient: str, resizing_cone_angle: float = 60
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[float, float]]]:
    """
    Extract ultrasound cone from patient's video data.

    Args:
        patient: Patient identifier
        resizing_cone_angle: Angle for cone resizing in degrees

    Returns:
        Tuple containing:
            - Binary mask of the original cone
            - Binary mask of the resized cone
            - Apex point coordinates (intersection point)
    """
    video_path = os.path.join(dataset_path, patient, "video.npy")

    try:
        all_img = np.load(video_path)
        thresholded = np.zeros((all_img.shape[1], all_img.shape[2]), dtype=np.uint8)

        # Accumulate thresholded frames
        for time in range(all_img.shape[0]):
            np_img = all_img[time]
            src = np_img.astype(np.uint8)
            _, thresh = cv2.threshold(src, 20, 255, cv2.THRESH_BINARY)
            thresholded += thresh

        # Create binary mask
        binary = thresholded > 50
        binary_thresholded = np.int32(binary) * 255
        binary_thresholded = binary_thresholded.astype(np.uint8)

        # Extract largest connected component
        binary_array = (binary_thresholded == 255).astype(int)
        labeled_array, num_clusters = label(binary_array)
        unique_labels, counts = np.unique(labeled_array, return_counts=True)
        cluster_counts = dict(zip(unique_labels, counts))
        largest_cluster_label = max(
            cluster_counts, key=lambda x: cluster_counts[x] if x != 0 else -1
        )
        binary_thresholded = (labeled_array == largest_cluster_label).astype(
            np.uint8
        ) * 255

        # Find cone boundary points
        left_row_1 = 0
        left_row_2 = 0
        right_row_1 = 0
        right_row_2 = 0
        row_ids = [20, 45, 71, 81]

        for row in range(binary_thresholded.shape[0]):
            if binary_thresholded[row, row_ids[0]] == 255:
                left_row_1 = row
                break

        for row in range(binary_thresholded.shape[0]):
            if binary_thresholded[row, row_ids[1]] == 255:
                left_row_2 = row
                break

        for row in range(binary_thresholded.shape[0]):
            if (
                row + 5 < binary_thresholded.shape[0]
                and binary_thresholded[row, row_ids[2]] == 255
                and binary_thresholded[row + 5, row_ids[2]] == 255
            ):
                right_row_1 = row
                break

        for row in range(binary_thresholded.shape[0]):
            if binary_thresholded[row, row_ids[3]] == 255:
                right_row_2 = row
                break

        # Calculate cone parameters
        p1 = (row_ids[0], left_row_1)
        p2 = (row_ids[1], left_row_2)
        p3 = (row_ids[2], right_row_1)
        p4 = (row_ids[3], right_row_2)

        intersection = calculate_intersection(p1, p2, p3, p4)
        if intersection is None:
            return None, None, None

        radius, _ = get_radius(intersection, binary_thresholded)

        # Generate cone masks
        left_point = draw_line_with_length(
            intersection, (row_ids[1], left_row_2), radius
        )
        right_point = draw_line_with_length(
            intersection, (row_ids[3], right_row_2), radius
        )

        binary_thresholded = generate_usg_triangle(
            (intersection[1], intersection[0]),
            (left_point[1], left_point[0]),
            (right_point[1], right_point[0]),
            binary_thresholded,
        )

        # Create resized cone
        left_point = draw_line_with_length(
            intersection, (row_ids[1], left_row_2), radius + 20
        )
        right_point = draw_line_with_length(
            intersection, (row_ids[3], right_row_2), radius + 20
        )

        right_point_rotated = rotate_point(
            intersection, right_point, resizing_cone_angle / 2
        )
        left_point_rotated = rotate_point(
            intersection, left_point, -(resizing_cone_angle / 2)
        )

        binary_thresholded_resized = select_points_in_triangle(
            (intersection[0], intersection[1]),
            (left_point_rotated[0], left_point_rotated[1]),
            (right_point_rotated[0], right_point_rotated[1]),
            binary_thresholded,
        )

        # Normalize to 0-1 range
        binary_thresholded = binary_thresholded / 255
        binary_thresholded_resized = binary_thresholded_resized / 255

        return binary_thresholded, binary_thresholded_resized, intersection

    except Exception as e:
        print(f"Error processing patient {patient}: {str(e)}")
        return None, None, None


def calculate_intersection(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    """
    Calculate the intersection point of two lines defined by points p1, p2 and p3, p4.

    Args:
        p1, p2: (x, y) coordinates of the first line.
        p3, p4: (x, y) coordinates of the second line.

    Returns:
        (x, y): The intersection point if it exists.
        None: If the lines are parallel or coincident.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    # Calculate the determinants
    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # If determinant is zero, the lines are parallel or coincident
    if det == 0:
        return None

    # Calculate intersection point
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det

    return (px, py)


def get_radius(
    apex: Tuple[float, float], binary_mask: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate the radius from apex to the furthest non-zero point in the mask.

    Args:
        apex: Coordinates of the apex point (x, y)
        binary_mask: Binary mask array

    Returns:
        Tuple containing:
            - Maximum radius (distance)
            - Standard deviation of all radii
    """
    non_zero_idxs = []
    for i in range(binary_mask.shape[1]):
        mask_column = binary_mask[:, i]
        last_non_zero_indices = np.where(mask_column != 0)[0]
        if len(last_non_zero_indices) > 0:
            last_non_zero_index = last_non_zero_indices[-1]
            non_zero_idxs.append((i, last_non_zero_index))

    radii = [math.dist(apex, idx) for idx in non_zero_idxs]
    return np.max(radii) if radii else 0, np.std(radii) if radii else 0


def generate_usg_triangle(
    midpoint: Tuple[float, float],
    left_point: Tuple[float, float],
    right_point: Tuple[float, float],
    arr: np.ndarray,
) -> np.ndarray:
    """
    Generate ultrasound triangle mask in the provided array.

    Args:
        midpoint: Apex point coordinates (y, x)
        left_point: Left boundary point coordinates (y, x)
        right_point: Right boundary point coordinates (y, x)
        arr: Array to draw the triangle in

    Returns:
        Array with drawn triangle (255 inside triangle)
    """
    result = arr.copy()

    # Iterate through each point in the grid
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            if is_point_in_triangle((x, y), midpoint, left_point, right_point):
                result[x, y] = 255  # Point inside the triangle

    return result


def select_points_in_triangle(
    v1: Tuple[float, float],
    v2: Tuple[float, float],
    v3: Tuple[float, float],
    array: np.ndarray,
) -> np.ndarray:
    """
    Select only non-zero points within a triangle and return a 2D array with selected pixels.

    Args:
        v1, v2, v3: Vertices of the triangle (x, y)
        array: A 2D array with values 0 or non-zero

    Returns:
        A 2D array with the same shape as input, containing only selected pixels
    """
    # Create an output array initialized to 0
    result_array = np.zeros_like(array, dtype=array.dtype)

    # Iterate over all points in the array
    rows, cols = array.shape
    for row in range(rows):
        for col in range(cols):
            # Check if the point is non-zero and inside the triangle
            if array[row, col] != 0 and is_point_in_triangle((col, row), v1, v2, v3):
                result_array[row, col] = array[row, col]

    return result_array


def is_point_in_triangle(
    pt: Tuple[float, float],
    v1: Tuple[float, float],
    v2: Tuple[float, float],
    v3: Tuple[float, float],
) -> bool:
    """
    Check if a point is inside the triangle defined by three vertices.

    Args:
        pt: The point to check (x, y)
        v1, v2, v3: Vertices of the triangle (x, y)

    Returns:
        True if the point is inside the triangle, False otherwise
    """
    # Convert points to vectors
    x, y = pt
    x1, y1 = v1
    x2, y2 = v2
    x3, y3 = v3

    # Calculate areas using the determinant method
    # Area of the full triangle
    det_t = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    # Area of sub-triangle with pt, v2, v3
    det1 = (x2 - x) * (y3 - y) - (x3 - x) * (y2 - y)
    # Area of sub-triangle with v1, pt, v3
    det2 = (x - x1) * (y3 - y1) - (x3 - x1) * (y - y1)
    # Area of sub-triangle with v1, v2, pt
    det3 = (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1)

    # Check if the point lies inside by ensuring all determinants have the same sign
    return (det1 >= 0 and det2 >= 0 and det3 >= 0) or (
        det1 <= 0 and det2 <= 0 and det3 <= 0
    )


def calculate_bisecting_point(
    apex: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    scale: float = 0.5,
) -> Tuple[float, float]:
    """
    Calculate a point p3 inside the cone, ensuring it's on the angular bisector.

    Args:
        apex: (x, y) coordinates of the apex
        p1: (x, y) coordinates of the first point
        p2: (x, y) coordinates of the second point
        scale: Scaling factor to position p3 inside the cone (0 < scale <= 1)

    Returns:
        Coordinates of p3 inside the cone
    """
    apex = np.array(apex)
    p1 = np.array(p1)
    p2 = np.array(p2)

    # Vectors from apex to p1 and apex to p2
    v1 = p1 - apex
    v2 = p2 - apex

    # Normalize the vectors
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    if v1_norm == 0 or v2_norm == 0:
        return tuple(apex)

    v1 /= v1_norm
    v2 /= v2_norm

    # Bisector vector (average of normalized vectors)
    bisector = v1 + v2
    bisector_norm = np.linalg.norm(bisector)

    if bisector_norm == 0:
        return tuple(apex)

    bisector /= bisector_norm

    # Scale the bisector vector to place p3 inside the cone
    p3 = apex + bisector * scale * v1_norm

    return tuple(p3)


def draw_line_with_length(
    start_point: Tuple[float, float], through_point: Tuple[float, float], length: float
) -> Tuple[float, float]:
    """
    Calculates the endpoint of a line with a specific length, starting at start_point and passing through through_point.

    Args:
        start_point: (x, y) coordinates of the starting point
        through_point: (x, y) coordinates of the point the line passes through
        length: Desired length of the line

    Returns:
        (x, y) coordinates of the endpoint
    """
    # Convert points to NumPy arrays
    p1 = np.array(start_point, dtype=np.float64)
    p2 = np.array(through_point, dtype=np.float64)

    # Calculate the direction vector
    direction = p2 - p1
    direction_norm = np.linalg.norm(direction)

    if direction_norm == 0:
        return start_point

    # Normalize direction vector
    direction /= direction_norm

    # Scale the direction vector to the desired length
    scaled_vector = direction * length

    # Calculate the endpoint
    endpoint = p1 + scaled_vector
    return tuple(endpoint)


def rotate_point(
    center: Tuple[float, float], point: Tuple[float, float], angle: float
) -> Tuple[float, float]:
    """
    Rotates a point around a given center by a specified angle.

    Args:
        center: (x, y) coordinates of the center of rotation
        point: (x, y) coordinates of the point to rotate
        angle: Angle of rotation in degrees (counterclockwise)

    Returns:
        The (x, y) coordinates of the rotated point
    """
    # Convert the angle from degrees to radians
    angle_rad = math.radians(angle)

    # Translate point to the origin (center becomes the origin)
    translated_x = point[0] - center[0]
    translated_y = point[1] - center[1]

    # Apply the rotation matrix
    rotated_x = translated_x * math.cos(angle_rad) - translated_y * math.sin(angle_rad)
    rotated_y = translated_x * math.sin(angle_rad) + translated_y * math.cos(angle_rad)

    # Translate the point back to its original position
    final_x = rotated_x + center[0]
    final_y = rotated_y + center[1]

    return (final_x, final_y)


def calculate_normal(
    point1: Tuple[float, float], point2: Tuple[float, float]
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Calculate the normal vector of a line given two points.

    Args:
        point1: (x1, y1) coordinates of the first point
        point2: (x2, y2) coordinates of the second point

    Returns:
        Two normal vectors perpendicular to the line
    """
    x1, y1 = point1
    x2, y2 = point2

    # Compute the vector along the line
    dx = x2 - x1
    dy = y2 - y1

    # Compute the two normals
    normal1 = (-dy, dx)  # Rotate vector by 90° counterclockwise
    normal2 = (dy, -dx)  # Rotate vector by 90° clockwise

    return normal1, normal2
