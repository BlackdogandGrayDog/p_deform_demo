import cv2
import torch
import numpy as np
import open3d as o3d
import os
import re
import OpenEXR
import Imath
import matplotlib.pyplot as plt
import PythonCDT as cdt
from cotracker.utils.visualizer import read_video_from_images
from cotracker.predictor import CoTrackerPredictor
from copy_npz import process_trajectory

# Load Camera Intrinsics from File
def load_intrinsics(file_path):
    with open(file_path, 'r') as f:
        values = [list(map(float, line.split()[:3])) for line in f]  # Only keep first 3 values
    return np.array(values)

# Load Stereo Extrinsics from File
def load_extrinsics(file_path):
    with open(file_path, 'r') as f:
        values = [float(num) for line in f for num in line.split()]
    return np.array([[values[0], values[1], values[2], values[3]],
                     [values[4], values[5], values[6], values[7]],
                     [values[8], values[9], values[10], values[11]]])

# Function to create grid points on an image for initialization
def create_grid_points(image_shape, grid_size, margin_ratio=0.05):
    h, w = image_shape[:2]
    margin_x = int(w * margin_ratio)  # Calculate horizontal margin
    margin_y = int(h * margin_ratio)  # Calculate vertical margin
    x_coords = np.linspace(margin_x, w - margin_x, grid_size, dtype=int)  # X coordinates for the grid
    y_coords = np.linspace(margin_y, h - margin_y, grid_size, dtype=int)  # Y coordinates for the grid
    grid_points = np.array([(x, y) for y in y_coords for x in x_coords])  # Create grid points
    return grid_points

# Creates grid points within a specific region of interest
def create_grid_points_filtered_hamlyn_5(image, grid_size, margin_ratio=0.05):
    h, w = image.shape[:2]
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)

    # Generate grid points within margins
    x_coords = np.linspace(margin_x, w - margin_x, grid_size, dtype=int)
    y_coords = np.linspace(margin_y, h - margin_y, grid_size, dtype=int)
    grid_points = np.array([(x, y) for y in y_coords for x in x_coords])

    # Define your final ROI polygon
    roi = np.array([
        (147, 17), (147, 128), (231, 182), (231, 288), 
        (360, 288), (360, 138), (297, 138), (297, 70)
    ], dtype=np.int32)
    

    # Create a binary mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [roi], 1)  # fill with 1s

    # Keep only points **inside** the mask (i.e., where mask value is 1)
    filtered_points = [pt for pt in grid_points if mask[pt[1], pt[0]] == 1]

    return np.array(filtered_points)

# Creates grid points only inside the specified mask region
def create_grid_points_filtered_hamlyn_4(image, grid_size, margin_ratio=0.05):
    h, w = image.shape[:2]
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)

    # Generate grid points within margins
    x_coords = np.linspace(margin_x, w - margin_x, grid_size, dtype=int)
    y_coords = np.linspace(margin_y, h - margin_y, grid_size, dtype=int)
    grid_points = np.array([(x, y) for y in y_coords for x in x_coords])

    # Define your final ROI polygon
    roi = np.array([
        (44, 69), (158, 38), (360, 38), (360, 288),
        (39, 288), (39, 268), (86, 268), (106, 240),
        (106, 173), (39, 113)
    ], dtype=np.int32)

    # Create a binary mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [roi], 1)  # fill with 1s

    # Keep only points **inside** the mask (i.e., where mask value is 1)
    filtered_points = [pt for pt in grid_points if mask[pt[1], pt[0]] == 1]

    return np.array(filtered_points)

# Creates grid points only inside the specified mask region (hamlyn 6 ROI)
def create_grid_points_filtered_hamlyn_6(image, grid_size, margin_ratio=0.05):
    h, w = image.shape[:2]
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)

    # Generate grid points within margins
    x_coords = np.linspace(margin_x, w - margin_x, grid_size, dtype=int)
    y_coords = np.linspace(margin_y, h - margin_y, grid_size, dtype=int)
    grid_points = np.array([(x, y) for y in y_coords for x in x_coords])

    # Updated ROI for Hamlyn 6
    roi = np.array([
        (40, 70), (155, 81), (155, 28), (360, 28),
        (360, 167), (330, 215), (330, 288),
        (111, 288), (111, 165), (40, 100)
    ], dtype=np.int32)

    # Create mask and filter points inside ROI
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [roi], 1)

    filtered_points = [pt for pt in grid_points if mask[pt[1], pt[0]] == 1]
    return np.array(filtered_points)

# Creates grid points only inside the specified mask region (hamlyn 9 ROI)
def create_grid_points_filtered_hamlyn_9(image, grid_size, margin_ratio=0.05):
    h, w = image.shape[:2]
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)

    # Generate grid points within margins
    x_coords = np.linspace(margin_x, w - margin_x, grid_size, dtype=int)
    y_coords = np.linspace(margin_y, h - margin_y, grid_size, dtype=int)
    grid_points = np.array([(x, y) for y in y_coords for x in x_coords])

    # Define your new final ROI polygon
    roi = np.array([
        (44, 69), (158, 48), (360, 68), (360, 288),
        (65, 288), (65, 268), (86, 268), (116, 240),
        (116, 173), (39, 113)
    ], dtype=np.int32)

    # Create a binary mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [roi], 1)  # fill with 1s

    # Keep only points **inside** the mask
    filtered_points = [pt for pt in grid_points if mask[pt[1], pt[0]] == 1]

    return np.array(filtered_points)

# Computes Depth Map from Stereo Images or Uses a Ground Truth EXR File
def compute_depth(left_img, right_img, K, baseline, trj_num, use_sgm=False, disp_path=None):
    f = K[0, 0]  # focal length

    if use_sgm:
        # Convert images to grayscale
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        numDisparities = 128 if str(trj_num) in ['4', '5', '6'] else 64
        print(f"[SGBM] numDisparities: {numDisparities}")

        # StereoSGBM matcher
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=numDisparities,
            blockSize=15,
            P1=4 * 3 * 9**2,
            P2=16 * 3 * 9**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=100,
            speckleRange=16
        )

        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        disparity[disparity <= 0] = np.nan
        print("✅ Computed disparity using SGBM.")
    else:
        if disp_path is None or not os.path.exists(disp_path):
            raise FileNotFoundError(f"❌ Disparity map file not found: {disp_path}")
        
        disparity = np.load(disp_path).astype(np.float32)
        print(f"✅ Loaded disparity from {disp_path}")

    # Convert disparity to depth
    depth_map = (f * baseline) / disparity
    # print("✅ Converted disparity to depth.")

    # Apply percentile-based depth filtering (your original logic)
    depth_min = np.nanpercentile(depth_map, 5)
    depth_max = np.nanpercentile(depth_map, 95)
    valid_mask = (depth_map >= depth_min) & (depth_map <= depth_max) & ~np.isnan(depth_map)
    filtered_depth_map = np.where(valid_mask, depth_map, np.nan)
    # print(f"✅ Applied depth filtering: [{depth_min:.2f}, {depth_max:.2f}]")

    return filtered_depth_map, depth_map, disparity

# Function to track features using KLT optical flow
def track_features_klt(original_image, tracked_image, valid_pixels, apply_motion_filter=True):
    lk_params = dict(winSize=(21, 21), maxLevel=3, 
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    points_to_track = np.array(valid_pixels, dtype=np.float32).reshape(-1, 1, 2)
    tracked_points, status, _ = cv2.calcOpticalFlowPyrLK(original_image, tracked_image, points_to_track, None, **lk_params)

    points_to_track = points_to_track.reshape(-1, 2)
    tracked_points = tracked_points.reshape(-1, 2)
    status = status.flatten()

    if apply_motion_filter:
        # print("Applying motion filtering (removing high displacement points).")
        shifts = np.linalg.norm(tracked_points - points_to_track, axis=1)
        motion_threshold = np.percentile(shifts, 100)
        motion_mask = (shifts <= motion_threshold)
    else:
        # print("Motion filtering is disabled.")
        motion_mask = np.ones_like(status, dtype=bool)

    return points_to_track, tracked_points, status, motion_mask

# Uses CoTracker to track features
def track_features_cotracker(original_image, tracked_image, valid_pixels, model, apply_motion_filter=True):
    video = read_video_from_images(original_image, tracked_image).cuda()

    # Convert valid pixels to CoTracker input format
    points_to_track = np.array(valid_pixels, dtype=np.float32).reshape(-1, 2)
    queries_list = [[0., float(x), float(y)] for x, y in points_to_track]
    queries = torch.tensor(queries_list, dtype=torch.float32).cuda()

    # Perform tracking with CoTracker
    pred_tracks, pred_visibility = model(video, queries=queries[None], grid_size=0, backward_tracking=False)

    # Extract tracked points (first and last frame)
    tracked_points = np.array([[track[0].item(), track[1].item()] for track in pred_tracks[0][-1]])

    # Extract visibility mask from the last frame
    visibility_mask = pred_visibility[0][-1].cpu().numpy().astype(bool)  # Shape: [N]

    # Status based on visibility (1 if visible, 0 otherwise)
    status = visibility_mask.astype(np.uint8)

    if apply_motion_filter:
        # Compute motion shifts and apply additional filtering
        shifts = np.linalg.norm(tracked_points - points_to_track, axis=1)
        motion_threshold = np.percentile(shifts, 100)
        motion_mask = (shifts <= motion_threshold)
    else:
        motion_mask = visibility_mask

    return points_to_track, tracked_points, status, motion_mask

# Function to save a 3D point cloud to a file
def save_point_cloud(points_3d, filename):
    pcd = o3d.geometry.PointCloud()  # Initialize Open3D PointCloud
    pcd.points = o3d.utility.Vector3dVector(points_3d)  # Add points to the PointCloud
    o3d.io.write_point_cloud(filename, pcd)  # Save the PointCloud to a file

# Function to visualize 2D points on an image
def visualize_2d_points(image, valid_pixels, output_filename):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Show the image
    plt.scatter(valid_pixels[:, 0], valid_pixels[:, 1], c='yellow', s=10, label="Tracked Points")  # Overlay tracked points
    plt.title("Tracked Points on Frame")
    plt.legend()
    plt.axis("off")
    plt.savefig(output_filename)  # Save visualization
    plt.close()

#  Function to remove Nan points from world positions
def remove_nan_points(world_positions):
    # Load the frames
    frame_keys = sorted(world_positions.keys())
    all_frames = [world_positions[key] for key in frame_keys]
    all_points = np.stack(all_frames, axis=0)  # Shape: (num_frames, num_points, 3)

    # Find points with NaN values across any frame
    valid_points_mask = ~np.any(np.isnan(all_points), axis=(0, 2))  # Shape: (num_points,)
    
    # Filter world positions
    filtered_world_positions = {key: world_positions[key][valid_points_mask] for key in frame_keys}
    
    return filtered_world_positions, valid_points_mask

# Computes dense optical flow between two frames using TV-L1
def compute_dense_optical_flow_tv_l1(I_prev, I_t):
    prev_gray = cv2.cvtColor(I_prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(I_t, cv2.COLOR_BGR2GRAY)
    # Initialize TV-L1 optical flow
    optical_flow = cv2.optflow.createOptFlow_DualTVL1()
    # Compute dense optical flow
    flow = optical_flow.calc(prev_gray, curr_gray, None)
    return flow

# Computes dense optical flow between two frames using NeuFlow
def compute_dense_optical_flow_neuf(flow_path):
    flow = np.load(flow_path)
    print("✅ Loaded flow from ", flow_path)
    return flow

# Extracts and filters patched flow
def extract_and_filter_patched_flow(image_positions, flow_maps, patch_size):
    half_patch = patch_size // 2
    frame_keys = sorted(flow_maps.keys())  # Start from frame_0002
    num_frames = len(frame_keys)
    
    # Get the number of tracked points from the first frame in `flow_maps`
    sample_frame = frame_keys[0]
    num_points = image_positions[sample_frame].shape[0]

    # Initialize patch storage with NaNs
    patched_flow = {frame_key: np.full((num_points, patch_size, patch_size, 2), np.nan, dtype=np.float32) for frame_key in frame_keys}
    valid_patch_mask = np.ones(num_points, dtype=bool)  # Assume all points are valid initially

    # === Step 1: Identify out-of-bounds patches ===
    for frame_key in frame_keys:
        flow_map = flow_maps[frame_key]  # (H, W, 2)
        H, W, _ = flow_map.shape
        points_2d = image_positions[frame_key]  # (num_points, 2)

        for i, (x, y) in enumerate(points_2d):
            x, y = int(x), int(y)

            # If patch is out-of-bounds in any frame, mark as invalid
            if (x - half_patch < 0 or x + half_patch >= W or
                y - half_patch < 0 or y + half_patch >= H):
                valid_patch_mask[i] = False

    # === Step 2: Apply Filtering to All Frames ===
    filtered_image_positions = {
        frame_key: image_positions[frame_key][valid_patch_mask] for frame_key in image_positions.keys()
    }
    patched_flow = {
        frame_key: patched_flow[frame_key][valid_patch_mask] for frame_key in frame_keys
    }

    # === Step 3: Extract Valid Patches ===
    for frame_key in frame_keys:
        flow_map = flow_maps[frame_key]  # (H, W, 2)
        points_2d = filtered_image_positions[frame_key]  # (filtered_num_points, 2)

        for i, (x, y) in enumerate(points_2d):
            x, y = int(x), int(y)
            patched_flow[frame_key][i] = flow_map[
                y - half_patch:y + half_patch , 
                x - half_patch:x + half_patch, 
                :
            ]

    return filtered_image_positions, valid_patch_mask, patched_flow

# Function to track points across frames and generate 3D world positions
def track_points_and_generate_world_positions(
    start_frame_index, end_frame_index, folder_path, valid_pixels, K, baseline, trj_num,
    output_folder, points_3d_0, points_3d_1, points_2d_0, points_2d_1, 
    use_cotracker=True, use_neuf=True, model=None, patch_size=6):
    # Load sorted image file lists
    images_left = sorted(os.listdir(os.path.join(folder_path, "image01")))
    images_right = sorted(os.listdir(os.path.join(folder_path, "image02")))
    disp_paths = sorted(os.listdir(os.path.join(folder_path, "disparity_map")))
    flow_paths = sorted(os.listdir(os.path.join(folder_path, "flow_map")))

    # Initialize tracked points and create output subfolders
    tracked_points = valid_pixels.copy()
    mesh_point = valid_pixels.copy()
    all_tracked_points = []  # Save all frames' valid_tracked_points
    output_ply_folder = os.path.join(output_folder, "output_ply")
    output_png_folder = os.path.join(output_folder, "output_png")
    os.makedirs(output_ply_folder, exist_ok=True)
    os.makedirs(output_png_folder, exist_ok=True)

    # === Initialize World Positions (3D) and Image Positions (2D) ===
    world_positions = {
        "frame_0000": points_3d_0,
        "frame_0001": points_3d_1
    }

    image_positions = {
        "frame_0000": points_2d_0,
        "frame_0001": points_2d_1
    }
    
    flow_maps = {}
    
    depth_maps = {}

    # Iterate through frames
    for frame_index in range(start_frame_index, end_frame_index + 1):
        I_t_left = cv2.imread(os.path.join(folder_path, "image01", images_left[frame_index]))
        I_t_right = cv2.imread(os.path.join(folder_path, "image02", images_right[frame_index]))

        # Compute Dense Optical Flow (from frame_{t-1} → frame_t) using TV-L1
        I_prev = cv2.imread(os.path.join(folder_path, "image01", images_left[frame_index - 1]))
        if use_neuf:
            flow_map = compute_dense_optical_flow_neuf(os.path.join(folder_path, "flow_map", flow_paths[frame_index-1]))
        else:
            flow_map = compute_dense_optical_flow_tv_l1(I_prev, I_t_left)
        flow_maps[f"frame_{frame_index:04d}"] = flow_map.astype(np.float32)

        if frame_index > start_frame_index:
            # I_prev = cv2.imread(os.path.join(folder_path, "image01", images_left[frame_index - 1]))
            I_t = I_t_left
            apply_motion_filter = frame_index <= 1000

            if use_cotracker:
                print("Using CoTracker for feature tracking.")
                _, tracked_points, status, motion_mask = track_features_cotracker(
                    I_prev, I_t, tracked_points, model, apply_motion_filter=apply_motion_filter
                )
            else:
                print("Using KLT for feature tracking.")
                I_prev_gray = cv2.cvtColor(I_prev, cv2.COLOR_BGR2GRAY)
                I_t_gray = cv2.cvtColor(I_t, cv2.COLOR_BGR2GRAY)
                _, tracked_points, status, motion_mask = track_features_klt(
                    I_prev_gray, I_t_gray, tracked_points, apply_motion_filter=apply_motion_filter
                )
            
            # Reshape tracked points
            tracked_points = tracked_points.reshape(-1, 2)

            # === Filter Tracked Points ===
            valid_mask = (
                (status.flatten() == 1) & (motion_mask) & 
                (tracked_points[:, 0] >= 0) & (tracked_points[:, 0] < I_t_left.shape[1]) &
                (tracked_points[:, 1] >= 0) & (tracked_points[:, 1] < I_t_left.shape[0])
            )

            valid_tracked_points = tracked_points.copy()
            valid_tracked_points[~valid_mask] = np.nan
        else:
            valid_tracked_points = tracked_points.copy()

        # Save valid_tracked_points for the current frame
        all_tracked_points.append(valid_tracked_points)
        
        # === Compute Depth Map for the Current Frame ===
        disp_path = os.path.join(folder_path, "disparity_map", disp_paths[frame_index])
        filtered_depth_map, depth_map, disparity = compute_depth(I_t_left, I_t_right, K, baseline, trj_num, disp_path=disp_path)
        depth_maps[f"frame_{frame_index:04d}"] = depth_map
        # disparity_maps[f"frame_{frame_index:04d}"] = disparity
        # === Compute 3D Points ===
        points_3d = np.full((len(valid_pixels), 3), np.nan)

        valid_mask = ~np.isnan(valid_tracked_points[:, 0]) & ~np.isnan(valid_tracked_points[:, 1])
        Z = np.full(len(valid_tracked_points), np.nan)
        valid_indices = valid_mask & (
            (valid_tracked_points[:, 0] < filtered_depth_map.shape[1]) &
            (valid_tracked_points[:, 1] < filtered_depth_map.shape[0])
        )
        Z[valid_indices] = filtered_depth_map[
            valid_tracked_points[valid_indices, 1].astype(int),
            valid_tracked_points[valid_indices, 0].astype(int)
        ]

        # Compute 3D points for valid depth values
        depth_valid_mask = ~np.isnan(Z)
        points_3d[depth_valid_mask] = np.column_stack((
            (valid_tracked_points[depth_valid_mask, 0] - K[0, 2]) * Z[depth_valid_mask] / K[0, 0],
            (valid_tracked_points[depth_valid_mask, 1] - K[1, 2]) * Z[depth_valid_mask] / K[1, 1],
            Z[depth_valid_mask]
        ))

        # Store 3D world positions and 2D image positions
        world_positions[f"frame_{frame_index:04d}"] = points_3d.astype(np.float32)
        image_positions[f"frame_{frame_index:04d}"] = valid_tracked_points.astype(np.float32)

    # Apply valid mask to all frames, including frame_0000
    filtered_world_positions, valid_points_mask = remove_nan_points(world_positions)

    # Apply valid mask to mesh_point and image positions
    filtered_mesh_point = mesh_point[valid_points_mask]
    filtered_image_positions = {frame_key: image_positions[frame_key][valid_points_mask] for frame_key in image_positions}
    
    # === Second Round Filtering (Patch Extraction) ===
    filtered_image_positions, valid_patch_mask, patched_flow = extract_and_filter_patched_flow(
        image_positions=filtered_image_positions,
        flow_maps=flow_maps,
        patch_size=patch_size
    )
    
    # Apply valid mask to world positions & mesh points
    filtered_world_positions = {frame_key: filtered_world_positions[frame_key][valid_patch_mask] for frame_key in filtered_world_positions}
    filtered_mesh_point = filtered_mesh_point[valid_patch_mask]
    
    # Save .npz world positions
    world_positions_path = os.path.join(output_folder, "world_positions.npz")
    np.savez(world_positions_path, **filtered_world_positions)
    print(f"World positions saved to {world_positions_path}")

    # Save .npz image positions (2D points)
    image_positions_path = os.path.join(output_folder, "image_positions.npz")
    np.savez(image_positions_path, **filtered_image_positions)
    print(f"Image positions saved to {image_positions_path}")
    
    # Save .npz dense optical flow maps
    flow_maps_path = os.path.join(output_folder, "flow_maps.npz")
    np.savez(flow_maps_path, **flow_maps)
    print(f"Flow maps saved to {flow_maps_path}")
    
    # Save .npz patched flow maps
    patched_flow_path = os.path.join(output_folder, "patched_flow.npz")
    np.savez(patched_flow_path, **patched_flow)
    print(f"Patched flow maps saved to {patched_flow_path}")

    # Save .npz depth maps
    depth_maps_path = os.path.join(output_folder, "depth_maps.npz")
    np.savez(depth_maps_path, **depth_maps)
    print(f"Depth maps saved to {depth_maps_path}")
    
    # # Save .npz disparity maps
    # disparity_maps_path = os.path.join(output_folder, "disparity_maps.npz")
    # np.savez(disparity_maps_path, **disparity_maps)
    # print(f"Disparity maps saved to {disparity_maps_path}")

    # === Save 3D & 2D Visualizations ===
    print(f"Saving 3D points and visualizations...")
    for frame_index, frame_points in enumerate(all_tracked_points, start=start_frame_index):
        frame_key_3d = f"frame_{frame_index:04d}"
        frame_key_2d = f"frame_{frame_index-1:04d}"
        points_3d = filtered_world_positions[frame_key_3d]

        # Apply the valid points mask to each frame's tracked points
        valid_tracked_points_filtered = frame_points[valid_points_mask]

        # Read the corresponding image for visualization
        I_t_left = cv2.imread(os.path.join(folder_path, "image01", images_left[frame_index]))
        save_point_cloud(points_3d, os.path.join(output_ply_folder, f"{frame_key_2d}.ply"))
        visualize_2d_points(I_t_left, valid_tracked_points_filtered, os.path.join(output_png_folder, f"{frame_key_2d}.png"))
    print("Saved 3D points and visualizations.")
    
    return filtered_mesh_point
    
# Function to generate a 2D triangular mesh save it in OFF format and process to .npz
def generate_and_process_2d_mesh(valid_pixels, output_path):
    # Initialize constrained Delaunay triangulation
    triangulation = cdt.Triangulation(
        cdt.VertexInsertionOrder.AS_PROVIDED,
        cdt.IntersectingConstraintEdges.NOT_ALLOWED,
        0.0
    )
    
    # Convert valid pixels to CDT vertices and perform triangulation
    vertices_2d = [cdt.V2d(x, y) for x, y in valid_pixels]
    triangulation.insert_vertices(vertices_2d)
    triangulation.erase_super_triangle()
    
    # Save the mesh in OFF format
    with open(f"{output_path}.off", "w") as f:
        f.write("OFF\n")  # Write OFF file header
        f.write(f"{len(valid_pixels)} {triangulation.triangles_count()} 0\n")  # Write vertex and triangle count
        for x, y in valid_pixels:
            f.write(f"{x} {y} 0\n")  # Write vertex coordinates
        for triangle in triangulation.triangles_iter():
            v0, v1, v2 = triangle.vertices
            f.write(f"3 {int(v0)} {int(v1)} {int(v2)}\n")  # Write triangle indices
    print(f"2D mesh saved to {output_path}.off")
    
    # Extract mesh data for .npz format
    mesh_pos = np.array([[v.x, v.y] for v in triangulation.vertices_iter()], dtype=np.float32)
    cells = np.array([[t.vertices[0], t.vertices[1], t.vertices[2]] for t in triangulation.triangles_iter()], dtype=np.int32)
    
    # Save the mesh data as .npz
    np.savez(f"{output_path}.npz", mesh_pos=mesh_pos, cells=cells)
    print(f"Mesh position (mesh_pos) and faces (cells) saved to {output_path}.npz")
    os.remove(f"{output_path}.off")

# Loads mesh, world positions, image positions, and patched flow from .npz files.
def load_mesh_world_image_flow_data(mesh_npz_file, world_positions_npz_file, image_positions_npz_file, patched_flow_npz_file):
    mesh_data = np.load(mesh_npz_file)
    world_positions_data = np.load(world_positions_npz_file)
    image_positions_data = np.load(image_positions_npz_file)
    patched_flow_data = np.load(patched_flow_npz_file)

    return {
        "cells": mesh_data["cells"],
        "mesh_pos": mesh_data["mesh_pos"]
    }, {
        key: world_positions_data[key] for key in world_positions_data.keys()
    }, {
        key: image_positions_data[key] for key in image_positions_data.keys()
    }, {
        key: patched_flow_data[key] for key in patched_flow_data.keys()
    }

# Creates and saves dataset with patched flow, including average target patched flow
def create_and_save_dataset(mesh_npz_file, world_positions_npz_file, image_positions_npz_file, patched_flow_npz_file, output_file, tensor_output_file):
    # Load data
    mesh_data, world_positions, image_positions, patched_flow = load_mesh_world_image_flow_data(
        mesh_npz_file, world_positions_npz_file, image_positions_npz_file, patched_flow_npz_file
    )

    sorted_frames = sorted(world_positions.keys())  # Frames 0000 onwards
    sorted_patch_frames = sorted(patched_flow.keys())  # Frames 0002 onwards
    num_frames = len(sorted_frames) - 3  # Adjusted for prev_prev|world_pos
    num_patch_frames = len(sorted_patch_frames)  # Excludes last frame

    # Dictionary format dataset
    dataset = {}
    for i in range(2, len(sorted_frames) - 1):  # Start from frame_0002
        current_frame_key = sorted_frames[i]
        prev_frame_key = sorted_frames[i - 1]
        prev_prev_frame_key = sorted_frames[i - 2]
        target_frame_key = sorted_frames[i + 1]

        world_pos = world_positions[current_frame_key].astype(np.float32)
        prev_world_pos = world_positions[prev_frame_key].astype(np.float32)
        prev_prev_world_pos = world_positions[prev_prev_frame_key].astype(np.float32)
        target_world_pos = world_positions[target_frame_key].astype(np.float32)

        image_pos = image_positions[current_frame_key].astype(np.float32)
        prev_image_pos = image_positions[prev_frame_key].astype(np.float32)
        prev_prev_image_pos = image_positions[prev_prev_frame_key].astype(np.float32)
        target_image_pos = image_positions[target_frame_key].astype(np.float32)  # ✅ New: Target image pos

        # Include patched flow only if it's not the last dataset frame
        if current_frame_key in patched_flow and target_frame_key in patched_flow:
            patched_flow_current = patched_flow[current_frame_key].astype(np.float32)  # Shape: (num_points, 10, 10, 2)
            target_patched_flow = patched_flow[target_frame_key].astype(np.float32)  # Shape: (num_points, 10, 10, 2)

            # Compute the average of the 10x10 patch → Shape: (num_points, 2)
            avg_target_patched_flow = np.mean(target_patched_flow, axis=(1, 2))  # Averaging over spatial dimensions
        else:
            patched_flow_current = None
            target_patched_flow = None
            avg_target_patched_flow = None

        # Create a node_type array
        node_type = np.zeros((mesh_data["mesh_pos"].shape[0], 1), dtype=np.int32)

        # Construct the dataset for the current frame
        dataset[current_frame_key] = {
            "cells": mesh_data["cells"].astype(np.int32),
            "mesh_pos": mesh_data["mesh_pos"].astype(np.float32),
            "node_type": node_type.astype(np.int32),
            "world_pos": world_pos,
            "prev|world_pos": prev_world_pos,
            "prev_prev|world_pos": prev_prev_world_pos,
            "target|world_pos": target_world_pos,
            "image_pos": image_pos,
            "prev|image_pos": prev_image_pos,
            "prev_prev|image_pos": prev_prev_image_pos,
            "target|image_pos": target_image_pos,  # ✅ New: Added target|image_pos
        }

        if patched_flow_current is not None:
            dataset[current_frame_key]["patched_flow"] = patched_flow_current
            dataset[current_frame_key]["target|patched_flow"] = target_patched_flow
            dataset[current_frame_key]["avg_target|patched_flow"] = avg_target_patched_flow  # Add the new feature

    # Save dictionary format dataset
    np.savez(output_file, **{f"{frame_key}/{key}": value for frame_key, frame_data in dataset.items() for key, value in frame_data.items()})
    print(f"Entire dictionary format dataset saved to {output_file}")

    # Stacking list format dataset into 3D tensors with explicit dtype
    stacked_tensors = {
        "cells": np.stack([mesh_data["cells"].astype(np.int32)] * num_frames, axis=0),
        "mesh_pos": np.stack([mesh_data["mesh_pos"].astype(np.float32)] * num_frames, axis=0),
        "node_type": np.stack([dataset[sorted_frames[i]]["node_type"].astype(np.int32) for i in range(2, len(sorted_frames) - 1)], axis=0),
        "world_pos": np.stack([dataset[sorted_frames[i]]["world_pos"].astype(np.float32) for i in range(2, len(sorted_frames) - 1)], axis=0),
        "prev|world_pos": np.stack([dataset[sorted_frames[i]]["prev|world_pos"].astype(np.float32) for i in range(2, len(sorted_frames) - 1)], axis=0),
        "prev_prev|world_pos": np.stack([dataset[sorted_frames[i]]["prev_prev|world_pos"].astype(np.float32) for i in range(2, len(sorted_frames) - 1)], axis=0),
        "target|world_pos": np.stack([dataset[sorted_frames[i]]["target|world_pos"].astype(np.float32) for i in range(2, len(sorted_frames) - 1)], axis=0),
        "image_pos": np.stack([dataset[sorted_frames[i]]["image_pos"].astype(np.float32) for i in range(2, len(sorted_frames) - 1)], axis=0),
        "prev|image_pos": np.stack([dataset[sorted_frames[i]]["prev|image_pos"].astype(np.float32) for i in range(2, len(sorted_frames) - 1)], axis=0),
        "prev_prev|image_pos": np.stack([dataset[sorted_frames[i]]["prev_prev|image_pos"].astype(np.float32) for i in range(2, len(sorted_frames) - 1)], axis=0),
        "target|image_pos": np.stack([dataset[sorted_frames[i]]["target|image_pos"].astype(np.float32) for i in range(2, len(sorted_frames) - 1)], axis=0),
    }

    # Add patched flow only for frames that have it (excluding last frame)
    if num_patch_frames > 0:
        stacked_tensors["patched_flow"] = np.stack(
            [dataset[sorted_patch_frames[i]]["patched_flow"].astype(np.float32) for i in range(num_patch_frames - 1)], axis=0
        )
        stacked_tensors["target|patched_flow"] = np.stack(
            [dataset[sorted_patch_frames[i]]["target|patched_flow"].astype(np.float32) for i in range(num_patch_frames - 1)], axis=0
        )
        stacked_tensors["avg_target|patched_flow"] = np.stack(
            [dataset[sorted_patch_frames[i]]["avg_target|patched_flow"].astype(np.float32) for i in range(num_patch_frames - 1)], axis=0
        )

    # Save stacked tensors to .npz
    np.savez(tensor_output_file, **stacked_tensors)
    print(f"Stacked tensors saved to {tensor_output_file}")

    # Remove intermediate files
    os.remove(mesh_npz_file)
    os.remove(world_positions_npz_file)
    os.remove(image_positions_npz_file)
    os.remove(patched_flow_npz_file)

    return stacked_tensors

# Generates a dataset for a specific trajectory and grid size
def dataset_generation_hamlyn(traj_num, start_frame_index, end_frame_index, grid_size, use_cotracker, patch_size):
    data_folder = f"./stereo_datasets/stereo_hamlyn/rectified{traj_num}/"
    intrinsics_file = os.path.join(data_folder, "intrinsics.txt")
    extrinsics_file = os.path.join(data_folder, "extrinsics.txt")

    # === Load Intrinsics & Extrinsics ===
    K = load_intrinsics(intrinsics_file)
    extrinsics = load_extrinsics(extrinsics_file)
    baseline = abs(extrinsics[0, 3]) / 1000  # Convert from mm to meters
    print(f"Using Stereo Baseline: {baseline:.6f} meters")

    # === Create Output Folder ===
    tracker_type = 'cotracker' if use_cotracker else 'klt'
    
    output_folder = f"./hamlyn_data/trajectory_{traj_num}/st_input_data_{traj_num}_ts_{end_frame_index-1}_gs_{grid_size}_{tracker_type}_accel"

    os.makedirs(output_folder, exist_ok=True)

    # === Load Stereo Image Pairs for Three Frames ===
    I_2_left = cv2.imread(os.path.join(data_folder, "image01/0000000002.jpg"))
    I_2_right = cv2.imread(os.path.join(data_folder, "image02/0000000002.jpg"))
    I_1_left = cv2.imread(os.path.join(data_folder, "image01/0000000001.jpg"))
    I_1_right = cv2.imread(os.path.join(data_folder, "image02/0000000001.jpg"))
    I_0_left = cv2.imread(os.path.join(data_folder, "image01/0000000000.jpg"))
    I_0_right = cv2.imread(os.path.join(data_folder, "image02/0000000000.jpg"))
    
    # Initialize grid points depending on trajectory
    if traj_num in {"5"} or traj_num.startswith("5"):
        grid_points = create_grid_points_filtered_hamlyn_5(I_1_left, grid_size=grid_size, margin_ratio=0.12)
        print("Using filtered 5 grid points")
    elif traj_num in {"4"} or traj_num.startswith("4"):
        grid_points = create_grid_points_filtered_hamlyn_4(I_1_left, grid_size=grid_size, margin_ratio=0.12)
        print("Using filtered 4 grid points")
    elif traj_num in {"6"} or traj_num.startswith("6"):
        grid_points = create_grid_points_filtered_hamlyn_6(I_1_left, grid_size=grid_size, margin_ratio=0.12)
        print("Using filtered 6 grid points")
    # elif traj_num in {"7", "8", "9", "10", "14", "15"}:
    #     grid_points = create_grid_points_filtered_hamlyn_9(I_1_left, grid_size=grid_size, margin_ratio=0.12)
    #     print("Using filtered 9 grid points")
    elif traj_num in {"12"} or traj_num.startswith("12"):
        grid_points = create_grid_points(I_1_left.shape, grid_size=grid_size, margin_ratio=0.15)
        print("Using filtered 12 grid points")
    else:
        grid_points = create_grid_points(I_1_left.shape, grid_size=grid_size, margin_ratio=0.12)
        print("Using unfiltered grid points")
        
    filtered_depth_map_2, _, _ = compute_depth(I_2_left, I_2_right, K, baseline, traj_num, disp_path=os.path.join(data_folder, "disparity_map/0000000002.npy"))
    valid_pixels_2 = grid_points[~np.isnan(filtered_depth_map_2[grid_points[:, 1], grid_points[:, 0]])]

    # === Feature Tracking: I_2 → I_1 → I_0 ===
    if use_cotracker:
        print("Using CoTracker for feature tracking.")
        model = CoTrackerPredictor(checkpoint="./cotracker/models/cotracker2.pth").cuda()

        # Track from I_2 → I_1
        original_points, tracked_points_1, status_1, motion_mask_1 = track_features_cotracker(
            I_2_left, I_1_left, valid_pixels_2, model
        )

        # Track from I_1 → I_0
        tracked_points_1, tracked_points_0, status_0, motion_mask_0 = track_features_cotracker(
            I_1_left, I_0_left, tracked_points_1, model
        )
    else:
        print("Using KLT for feature tracking.")
        model = None

        # Track from I_2 → I_1
        original_points, tracked_points_1, status_1, motion_mask_1 = track_features_klt(
            cv2.cvtColor(I_2_left, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(I_1_left, cv2.COLOR_BGR2GRAY),
            valid_pixels_2
        )

        # Track from I_1 → I_0
        tracked_points_1, tracked_points_0, status_0, motion_mask_0 = track_features_klt(
            cv2.cvtColor(I_1_left, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(I_0_left, cv2.COLOR_BGR2GRAY),
            tracked_points_1
        )

    # === Filter Valid Tracked Points Across Frames (Right After Tracking) ===
    valid_mask_1 = (
        (status_1.flatten() == 1) & (motion_mask_1) &
        (tracked_points_1[:, 0] >= 0) & (tracked_points_1[:, 0] < I_1_left.shape[1]) &
        (tracked_points_1[:, 1] >= 0) & (tracked_points_1[:, 1] < I_1_left.shape[0])
    )

    valid_mask_0 = (
        (status_0.flatten() == 1) & (motion_mask_0) &
        (tracked_points_0[:, 0] >= 0) & (tracked_points_0[:, 0] < I_0_left.shape[1]) &
        (tracked_points_0[:, 1] >= 0) & (tracked_points_0[:, 1] < I_0_left.shape[0])
    )

    valid_original = original_points[valid_mask_1 & valid_mask_0].reshape(-1, 2)
    valid_tracked_1 = tracked_points_1[valid_mask_1 & valid_mask_0].reshape(-1, 2)
    valid_tracked_0 = tracked_points_0[valid_mask_1 & valid_mask_0].reshape(-1, 2)
    
    # === Compute Depth Maps for I_1 and I_0 (After Tracking and Filtering) ===
    filtered_depth_map_1, _, _ = compute_depth(I_1_left, I_1_right, K, baseline, traj_num, disp_path=os.path.join(data_folder, "disparity_map/0000000001.npy"))
    filtered_depth_map_0, _, _ = compute_depth(I_0_left, I_0_right, K, baseline, traj_num, disp_path=os.path.join(data_folder, "disparity_map/0000000000.npy"))

    # === Filter Tracked Points Again Based on Valid Depths ===
    valid_depth_1 = ~np.isnan(filtered_depth_map_1[valid_tracked_1[:, 1].astype(int), valid_tracked_1[:, 0].astype(int)])
    valid_depth_0 = ~np.isnan(filtered_depth_map_0[valid_tracked_0[:, 1].astype(int), valid_tracked_0[:, 0].astype(int)])

    valid_mask = valid_depth_1 & valid_depth_0
    valid_original = valid_original[valid_mask]
    valid_tracked_1 = valid_tracked_1[valid_mask]
    valid_tracked_0 = valid_tracked_0[valid_mask]

    # === Compute 3D Points for I_1 and I_0 ===
    Z_1 = filtered_depth_map_1[valid_tracked_1[:, 1].astype(int), valid_tracked_1[:, 0].astype(int)]
    Z_0 = filtered_depth_map_0[valid_tracked_0[:, 1].astype(int), valid_tracked_0[:, 0].astype(int)]

    points_3d_1 = np.column_stack(((valid_tracked_1[:, 0] - K[0, 2]) * Z_1 / K[0, 0], 
                                (valid_tracked_1[:, 1] - K[1, 2]) * Z_1 / K[1, 1], 
                                Z_1))

    points_3d_0 = np.column_stack(((valid_tracked_0[:, 0] - K[0, 2]) * Z_0 / K[0, 0], 
                                (valid_tracked_0[:, 1] - K[1, 2]) * Z_0 / K[1, 1], 
                                Z_0))



    ## Perform tracking and generate world positions
    filtered_mesh_points = track_points_and_generate_world_positions(
        start_frame_index, end_frame_index, data_folder, valid_original, K, baseline, traj_num, 
        output_folder, points_3d_0, points_3d_1, valid_tracked_0, valid_tracked_1, 
        use_cotracker=use_cotracker, model=model, patch_size=patch_size
    )

    # Generate and process 2D mesh
    generate_and_process_2d_mesh(filtered_mesh_points, os.path.join(output_folder, "2d_mesh"))

    # Create and save SLAM dataset in one step
    ds_tensor = create_and_save_dataset(
        os.path.join(output_folder, "2d_mesh.npz"),
        os.path.join(output_folder, "world_positions.npz"),
        os.path.join(output_folder, "image_positions.npz"),
        os.path.join(output_folder, "patched_flow.npz"),
        os.path.join(output_folder, "dic_dataset.npz"),
        os.path.join(output_folder, f"{os.path.basename(output_folder)}.npz")
    )

    print("Dataset creation and processing completed.")

# Automatically generate datasets for all trajectories and grid sizes
def auto_generate_datasets_hamlyn(traj_id, dataset_mode):
    # Define constants
    if dataset_mode == "train":
        traj_nums = [f"{traj_id * 1000 + i}" for i in range(31)]  # train sequences use end_frame_index = 18
    elif dataset_mode == "eval":
        traj_nums = [f"{traj_id * 1000 + 100 + i}" for i in range(6)] # eval sequences use end_frame_index = 12


    start_frame_index = 2
    if dataset_mode is "train":
        end_frame_index = 18 # for train sequences
    else:
        end_frame_index = 12 # for eval sequences
    use_cotracker = True
    patch_size = 6
    
    # Loop over different trajectory numbers and grid sizes
    for traj_num in traj_nums:
        for grid_size in range(10, 21):
            print(f"Generating dataset for Trajectory {traj_num}, Grid Size {grid_size}, Patch Size {patch_size}...")
            dataset_generation_hamlyn(traj_num=traj_num, start_frame_index=start_frame_index, 
                                      end_frame_index=end_frame_index, grid_size=grid_size, 
                                      use_cotracker=use_cotracker, patch_size=patch_size)
    
    if dataset_mode == "train":
        process_trajectory(traj_id)


if __name__ == "__main__":
    # ==== Trj_11 TRAIN ====
    traj_id = 11 
    dataset_mode = "train" 
    print(f"\n>>> Generating {dataset_mode} datasets for Trajectory {traj_id}...<<<\n")
    auto_generate_datasets_hamlyn(traj_id, dataset_mode)

    # ==== Trj_11 EVAL ====
    dataset_mode = "eval"
    print(f"\n>>> Generating {dataset_mode} datasets for Trajectory {traj_id}...<<<\n")
    auto_generate_datasets_hamlyn(traj_id, dataset_mode)
    
    # ==== Trj_12 Finetune TRAIN ====
    traj_id = 12
    dataset_mode = "train"
    print(f"\n>>> Generating {dataset_mode} datasets for Trajectory {traj_id}...<<<\n")
    auto_generate_datasets_hamlyn(traj_id, dataset_mode)
    
    # ==== Trj_12 Finetune EVAL ====
    dataset_mode = "eval"
    print(f"\n>>> Generating {dataset_mode} datasets for Trajectory {traj_id}...<<<\n")
    auto_generate_datasets_hamlyn(traj_id, dataset_mode)
