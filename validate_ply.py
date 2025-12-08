import pickle
import numpy as np
import os
import re
import open3d as o3d
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement

# Loads 3D points from a PLY file.
def load_ply_file(ply_file):
    try:
        ply_data = PlyData.read(ply_file)
        vertices = ply_data['vertex']
        points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        return points
    except Exception as e:
        print(f"Error reading PLY file {ply_file}: {e}")
        return None

# Saves 3D points with ground truth and predicted points in different colors to a PLY file
def save_colored_ply_file(gt_points, pred_points, filename):
    with open(filename, 'w') as f:
        num_points = gt_points.shape[0] + pred_points.shape[0]
        # Write PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        # Write ground truth points (yellow: 255, 255, 0)
        for point in gt_points:
            f.write(f"{point[0]} {point[1]} {point[2]} 255 255 0\n")
        # Write predicted points (blue: 0, 0, 255)
        for point in pred_points:
            f.write(f"{point[0]} {point[1]} {point[2]} 0 0 255\n")

# Saves 3D points with ground truth, predicted points, and red lines connecting them to a PLY file
def save_ply_with_lines(gt_points, pred_points, filename):
    num_gt_points = gt_points.shape[0]
    num_pred_points = pred_points.shape[0]

    if num_gt_points != num_pred_points:
        raise ValueError("Ground truth and predicted points must have the same number of points.")

    # Filter valid point pairs
    valid_indices = [
        i for i in range(num_gt_points)
        if not (np.any(np.isnan(gt_points[i])) or np.any(np.isnan(pred_points[i])) or
                np.any(np.isinf(gt_points[i])) or np.any(np.isinf(pred_points[i]))
        )
    ]
    gt_points = gt_points[valid_indices]
    pred_points = pred_points[valid_indices]

    # Create vertex list with ground truth (yellow) and predicted (blue) points
    vertices = []
    for point in gt_points:
        vertices.append((*point, 255, 255, 0))  # GT points in yellow
    for point in pred_points:
        vertices.append((*point, 0, 0, 255))  # Predicted points in blue

    # Create edge list connecting corresponding GT and predicted points with red lines
    edges = [(i, i + len(gt_points)) for i in range(len(gt_points))]

    # Save to PLY
    try:
        with open(filename, 'w') as f:
            # Write PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write(f"element edge {len(edges)}\n")
            f.write("property int vertex1\n")
            f.write("property int vertex2\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            # Write vertex data
            for vertex in vertices:
                f.write(f"{vertex[0]} {vertex[1]} {vertex[2]} {vertex[3]} {vertex[4]} {vertex[5]}\n")

            # Write edge data (red lines)
            for edge in edges:
                f.write(f"{edge[0]} {edge[1]} 255 0 0\n")
    except Exception as e:
        print(f"Error writing PLY file {filename}: {e}")

# Saves predicted 3D points to a PLY file.
def save_predicted_ply(pred_points, filename):
    try:
        with open(filename, 'w') as f:
            # Write PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {pred_points.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            # Write predicted points (blue: 0, 0, 255)
            for point in pred_points:
                f.write(f"{point[0]} {point[1]} {point[2]} 0 0 255\n")
    except Exception as e:
        print(f"Error writing predicted PLY file {filename}: {e}")

# Combines predicted positions and ground truth positions into PLY files with and without lines. 
def combine_prediction_and_groundtruth(pkl_file_path, gt_folder_path, output_dir_no_lines, output_dir_with_lines, pred_ply_dir):
    try:
        # Load the pickle file
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)

        # Ensure output directories exist
        os.makedirs(output_dir_no_lines, exist_ok=True)
        os.makedirs(output_dir_with_lines, exist_ok=True)
        os.makedirs(pred_ply_dir, exist_ok=True)

        # Extract predicted positions
        pred_pos = data[0]['pred_pos']  # Access the first trajectory (predicted)
        num_frames = len(pred_pos)

        for frame_idx in range(num_frames):
            # Load the corresponding ground truth PLY file
            gt_file = os.path.join(gt_folder_path, f"frame_{frame_idx + 1:04d}.ply")
            if not os.path.exists(gt_file):
                print(f"Ground truth PLY file not found: {gt_file}")
                continue
            gt_points = load_ply_file(gt_file)
            if gt_points is None:
                continue

            # Get the predicted points for the current frame
            pred_points = pred_pos[frame_idx]

            # Save PLY files without and with lines
            filename_no_lines = os.path.join(output_dir_no_lines, f"frame_{frame_idx + 1:04d}_comparison.ply")
            filename_with_lines = os.path.join(output_dir_with_lines, f"frame_{frame_idx + 1:04d}_comparison_with_lines.ply")
            pred_filename = os.path.join(pred_ply_dir, f"frame_{frame_idx + 1:04d}_predicted.ply")

            # save_colored_ply_file(gt_points, pred_points, filename_no_lines)
            # save_ply_with_lines(gt_points, pred_points, filename_with_lines)
            save_predicted_ply(pred_points, pred_filename)

    except Exception as e:
        print(f"Error during processing: {e}")

def save_ply_with_corresponding_lines(points_1, points_2, points_3, filename):
    """
    Saves three sets of points to a PLY file with lines connecting corresponding vertices only.

    Args:
        points_1 (numpy.ndarray): First set of 3D points of shape (num_points, 3).
        points_2 (numpy.ndarray): Second set of 3D points of shape (num_points, 3).
        points_3 (numpy.ndarray): Third set of 3D points of shape (num_points, 3).
        filename (str): Path to save the PLY file.
    """
    if not (points_1.shape[0] == points_2.shape[0] == points_3.shape[0]):
        raise ValueError("All point sets must have the same number of vertices.")

    vertices = []
    edges = []

    # Add vertices from all sets
    for point in points_1:
        vertices.append((*point, 0, 0, 255))  # Blue color
    for point in points_2:
        vertices.append((*point, 255, 165, 0))  # Orange color
    for point in points_3:
        vertices.append((*point, 0, 255, 0))  # Green color

    # Add edges (red) connecting corresponding points only if the points are not NaN
    num_points = points_1.shape[0]
    for i in range(num_points):
        if not (np.any(np.isnan(points_1[i])) or np.any(np.isnan(points_2[i])) or np.any(np.isnan(points_3[i]))):
            edges.append((i, i + num_points, 255, 0, 0))  # Between points_1 and points_2
            edges.append((i, i + 2 * num_points, 255, 0, 0))  # Between points_1 and points_3

    # Write to PLY file
    try:
        with open(filename, 'w') as f:
            # PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write(f"element edge {len(edges)}\n")
            f.write("property int vertex1\n")
            f.write("property int vertex2\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            # Write vertices
            for vertex in vertices:
                f.write(f"{vertex[0]} {vertex[1]} {vertex[2]} {vertex[3]} {vertex[4]} {vertex[5]}\n")

            # Write edges
            for edge in edges:
                f.write(f"{edge[0]} {edge[1]} {edge[2]} {edge[3]} {edge[4]}\n")
    except Exception as e:
        print(f"Error writing PLY file {filename}: {e}")


def save_combined_ply(gt_folder_path, pred_ply_dir, dataset_name, trajectory_number, exp_name, file_name, step_size=1):
    """
    Saves three PLY files with lines connecting only corresponding vertices into a new PLY file.

    Args:
        gt_folder_path (str): Path to the ground truth PLY folder.
        pred_ply_dir (str): Path to the predicted PLY folder.
        dataset_name (str): Dataset name.
        trajectory_number (str): Trajectory number.
        exp_name (str): Experiment name.
        file_name (str): File name identifier.
        step_size (int): Number of steps forward for evaluation.
    """

    # Define PLY file paths based on step_size
    ply_file_1 = f"{gt_folder_path}/frame_0001.ply"
    ply_file_2 = f"{gt_folder_path}/frame_{(1 + step_size):04d}.ply"
    ply_file_3 = f"{pred_ply_dir}/frame_{(1 + step_size):04d}_predicted.ply"

    if not (os.path.exists(ply_file_1) and os.path.exists(ply_file_2) and os.path.exists(ply_file_3)):
        print(f"Skipping step {step_size}: Missing PLY file -> {ply_file_1}, {ply_file_2}, or {ply_file_3}")
        return None

    # Output path for combined PLY file
    output_ply = f"./result/{dataset_name}/trajectory_{trajectory_number}/{exp_name}/eval/{file_name}/pred_gt_frame0_{step_size}_steps.ply"
    os.makedirs(os.path.dirname(output_ply), exist_ok=True)

    # Load and save combined PLY
    points_1 = load_ply_file(ply_file_1)
    points_2 = load_ply_file(ply_file_2)
    points_3 = load_ply_file(ply_file_3)

    if points_1 is None or points_2 is None or points_3 is None:
        print("Could not load one or more PLY files.")
        return None

    if not (points_1.shape[0] == points_2.shape[0] == points_3.shape[0]):
        print("All PLY files must have the same number of vertices.")
        return None

    save_ply_with_corresponding_lines(points_1, points_2, points_3, output_ply)

    return output_ply


# Normalize two point clouds using a shared centroid and scale.
def normalize_point_clouds(points1, points2):
    """Normalize two point clouds using a shared centroid and scale."""
    all_points = np.vstack((points1, points2))  # Combine both clouds
    centroid = np.mean(all_points, axis=0)
    scale = np.std(all_points)
    scale = max(scale, 1e-6)  # Avoid division by zero

    return (points1 - centroid) / scale, (points2 - centroid) / scale

# Compute Chamfer Distance between two point clouds.
def compute_chamfer_distance(points1, points2):
    """Compute Chamfer Distance between two point clouds."""
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)

    dist1, _ = tree1.query(points2)  # Nearest neighbor distances (points2 → points1)
    dist2, _ = tree2.query(points1)  # Nearest neighbor distances (points1 → points2)

    chamfer = np.mean(dist1**2) + np.mean(dist2**2)
    return chamfer, np.mean(dist1**2), np.mean(dist2**2)  # Return parts separately

# Compute overlap ratio based on nearest neighbor distance threshold.
def compute_overlap_ratio(points1, points2, threshold=0.01):
    """Compute overlap ratio based on nearest neighbor distance threshold."""
    tree = cKDTree(points2)
    distances, _ = tree.query(points1)
    return np.sum(distances <= threshold) / len(points1)

# Compute per-point Euclidean error between two point clouds.
def compute_per_point_error(points1, points2):
    """Compute per-point Euclidean error between two point clouds."""
    assert points1.shape == points2.shape, "Point clouds must have the same shape"
    return np.sqrt(np.sum((points1 - points2) ** 2, axis=1))  # Per-point distance

def evaluate_point_clouds(gt_folder_path, pred_ply_dir, file_name, trajectory_number, exp_name, dataset_name, step_size=1, threshold=0.01):
    """Evaluate two PLY point clouds and compute various distance metrics."""
    
    # Define PLY file paths based on step_size
    ply_path1 = f"{gt_folder_path}/frame_{(1 + step_size):04d}.ply"
    ply_path2 = f"{pred_ply_dir}/frame_{(1 + step_size):04d}_predicted.ply"

    if not os.path.exists(ply_path1) or not os.path.exists(ply_path2):
        print(f"Skipping step {step_size}: Files not found -> {ply_path1} or {ply_path2}")
        return None

    # Load point clouds
    pc1 = o3d.io.read_point_cloud(ply_path1)
    pc2 = o3d.io.read_point_cloud(ply_path2)

    # Convert to NumPy arrays
    points1 = np.asarray(pc1.points)
    points2 = np.asarray(pc2.points)

    # Normalize point clouds
    points1, points2 = normalize_point_clouds(points1, points2)

    # Compute metrics
    chamfer_dist, chamfer_part1, chamfer_part2 = compute_chamfer_distance(points1, points2)
    overlap = compute_overlap_ratio(points1, points2, threshold=threshold)
    per_point_errors = compute_per_point_error(points1, points2)

    # Compute summary statistics for per-point errors
    mean_error = np.mean(per_point_errors)
    max_error = np.max(per_point_errors)
    std_dev_error = np.std(per_point_errors)

    # Save results to file
    result_path = f"./result/{dataset_name}/trajectory_{trajectory_number}/{exp_name}/eval/{file_name}/step_{step_size}_prediction_comparison.txt"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    with open(result_path, "w") as f:
        f.write(f"Step Size: {step_size}\n")
        f.write(f"Chamfer Distance: {chamfer_dist}\n")
        f.write(f"Chamfer Part 1 (P1 → P2): {chamfer_part1}\n")
        f.write(f"Chamfer Part 2 (P2 → P1): {chamfer_part2}\n")
        f.write(f"Overlap Ratio: {overlap}\n")
        f.write(f"Mean Per-Point Error: {mean_error}\n")
        f.write(f"Max Per-Point Error: {max_error}\n")
        f.write(f"Std Dev of Per-Point Errors: {std_dev_error}\n")

    # Print results
    # print(f"Results saved to {result_path}")
    print(f"Step Size: {step_size}")
    print(f"Chamfer Distance: {chamfer_dist}")
    # print(f"Chamfer Part 1 (P1 → P2): {chamfer_part1}")
    # print(f"Chamfer Part 2 (P2 → P1): {chamfer_part2}")
    # print(f"Overlap Ratio: {overlap}")
    # print(f"Mean Per-Point Error: {mean_error}")
    # print(f"Max Per-Point Error: {max_error}")
    # print(f"Std Dev of Per-Point Errors: {std_dev_error}")

    return {
        "Step Size": step_size,
        "Chamfer Distance": chamfer_dist,
        "Chamfer Part 1": chamfer_part1,
        "Chamfer Part 2": chamfer_part2,
        "Overlap Ratio": overlap,
        "Mean Error": mean_error,
        "Max Error": max_error,
        "Std Dev Error": std_dev_error
    }

# Extract trajectory number from the file name.
def extract_trajectory_number(file_name):
    match = re.search(r'data_([\d\.]+(?:_noise_[\d\.]+)?)_ts_', file_name)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Could not extract trajectory number from file name: {file_name}")


# Process the evaluation of the PLY files.
def process_evaluation(file_name, exp_name, dataset_name, eval_dataset_name, trajectory_number=None, step_size=1):
    # Extract trajectory number from file_name
    gt_trajectory_number = extract_trajectory_number(file_name)
    
    if trajectory_number is None:
        # Extract trajectory number from file_name
        trajectory_number = gt_trajectory_number
    
    # Update ground truth folder path
    gt_folder_path = f"./{eval_dataset_name}_data/trajectory_{gt_trajectory_number}/{file_name}/output_ply"
    
    # Define other required paths
    pkl_file_path = f"./result/{dataset_name}/trajectory_{trajectory_number}/{exp_name}/eval/{file_name}.pkl"
    output_dir_no_lines = f"./result/{dataset_name}/trajectory_{trajectory_number}/{exp_name}/eval/{file_name}/comparison_ply"
    output_dir_with_lines = f"./result/{dataset_name}/trajectory_{trajectory_number}/{exp_name}/eval/{file_name}/comparison_ply_with_lines"
    pred_ply_dir = f"./result/{dataset_name}/trajectory_{trajectory_number}/{exp_name}/eval/{file_name}/pred_ply"

    # Run first part: Generate comparison and predicted PLY files
    combine_prediction_and_groundtruth(pkl_file_path, gt_folder_path, 
                                     output_dir_no_lines, output_dir_with_lines, pred_ply_dir)

    # Run third part: Evaluate PLY files
    for step_size in range(1, step_size + 1):
        print(f"Evaluating step size {step_size}")
        save_combined_ply(gt_folder_path, pred_ply_dir, dataset_name, trajectory_number, exp_name, file_name, step_size)
        evaluate_point_clouds(gt_folder_path, pred_ply_dir, file_name, trajectory_number, exp_name, dataset_name, step_size)


# if __name__ == "__main__":
#     exp_trajectory_number = '6'
#     file_name = 'gtd_input_data_6_ts_21_gs_11_cotracker_accel'
#     exp_name = 'patched_loss_130k_steps_1'
#     process_evaluation(file_name, exp_name, exp_trajectory_number)