import os
import cv2
import numpy as np
import glob
import open3d as o3d
from scipy.interpolate import RectBivariateSpline

from feature_extract_and_matching import get_feature_tracker
from estimate_camera_pose import estimate_correct_camera_pose, triangulate, cheirality_condition_check
from perspective_n_point import perspective_n_points_ransac, perspective_n_points_nonlinear
from reconstruct import find_missing_reconstruction, triangulate_nonlinear, perform_bundle_adjustment, check_projection_error


if __name__ == '__main__':
    np.random.seed(100)
    # K matrix
    K = np.asarray([
        [716.70, 0, 481.27],
        [0, 721.90, 362.52],
        [0, 0, 1]
    ])
    num_images = 6
    w_im = 960
    h_im = 720

    # Load input images
    Im = np.empty((num_images, h_im, w_im, 3), dtype=np.uint8)

    image_folder = "E:\\UMD\\ENPM673\\Final Project\\final_sfm\\data_images"
    # image_folder = "E:\\UMD\ENPM673\\Final Project\\Working\\clock_images"
    im_file = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    print(im_file)
    for i in range(num_images):
        # im_file = 'sfm_images/{:d}.jpg'.format(i)
        print(f"Loading image {i}")
        im = cv2.imread(im_file[i])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        Im[i, :, :, :] = im

    print("Images loaded")

    # Build feature track
    track = get_feature_tracker(Im, K)
    print("Track building over")

    track1 = track[0, :, :]
    track2 = track[1, :, :]

    # Estimate ﬁrst two camera poses
    R, C, X = estimate_correct_camera_pose(track1, track2)

    output_dir = 'reconstruction_output'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Set of camera poses
    P = np.zeros((num_images, 3, 4))

    # Set first two camera poses
    P[0] = np.eye(3, 4)
    P[1] = np.hstack([R, -(R @ C).reshape((3, 1))])
    # print("camera 1, 2 initialized")
    # print("checking projection error")
    # check_projection_error(1, X, track, K, P)
    ransac_iter = 200
    ransac_threshold = 0.01
    for i in range(2, num_images):
        print(f"adding camera number {i+1}")
        # Estimate new camera pose
        track_i = track[i, :, :]

        R, C, inlier = perspective_n_points_ransac(
            X, track_i, ransac_iter, ransac_threshold)
        inlier_idx = np.where(inlier)[0]
        print(f"number of inliers for camera{i+1}: {len(inlier_idx)}")
        # breakpoint()
        R, C = perspective_n_points_nonlinear(
            R, C, X[inlier_idx, :], track_i[inlier_idx, :])

        P[i] = np.hstack([R, -(R @ C).reshape((3, 1))])

        for j in range(i):
            print(f"matching camera {j+1} with {i+1}")
            # Fine new points to reconstruct
            track_j = track[j, :, :]
            track_j_mask = np.logical_and(
                track_j[:, 0] != -1, track_j[:, 1] != -1)

            new_points = np.logical_and(
                find_missing_reconstruction(X, track_i), track_j_mask)
            new_points_idx = np.where(new_points)[0]
            print(
                f"new points added with camera {i+1} in iteration {j+1}: {len(new_points_idx)}")
            # Triangulate points
            new_X = triangulate(
                P[i], P[j], track_i[new_points, :], track_j[new_points, :])
            new_X = triangulate_nonlinear(
                new_X, P[i], P[j], track_i[new_points, :], track_j[new_points, :])

            # Filter out points based on cheirality
            valid_idx = cheirality_condition_check(P[i], P[j], new_X)
            # Update 3D points
            X[new_points_idx[valid_idx], :] = new_X[valid_idx, :]
        # Run bundle adjustment

        valid_ind = X[:, 0] != -1
        X_ba = X[valid_ind, :]
        track_ba = track[:i + 1, valid_ind, :]
        P_new, X_new = perform_bundle_adjustment(
            P[:i + 1, :, :], X_ba, track_ba)
        P[:i + 1, :, :] = P_new
        X[valid_ind, :] = X_new

        P[:i+1, :, :] = P_new
        X[valid_ind, :] = X_new

        ###############################################################
        # Save the camera coordinate frames as meshes for visualization
        m_cam = None
        for j in range(i+1):
            R_d = P[j, :, :3]
            C_d = -R_d.T @ P[j, :, 3]
            T = np.eye(4)
            T[:3, :3] = R_d
            T[:3, 3] = C_d
            m = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
            m.transform(T)
            if m_cam is None:
                m_cam = m
            else:
                m_cam += m
        o3d.io.write_triangle_mesh(
            '{}/cameras_{}.ply'.format(output_dir, i+1), m_cam)

        # Save the reconstructed points as point cloud for visualization
        X_new_h = np.hstack([X_new, np.ones((X_new.shape[0], 1))])
        colors = np.zeros_like(X_new)
        for j in range(i, -1, -1):
            x = X_new_h @ P[j, :, :].T
            x = x / x[:, 2, np.newaxis]
            mask_valid = (x[:, 0] >= -1) * (x[:, 0] <= 1) * \
                (x[:, 1] >= -1) * (x[:, 1] <= 1)
            uv = x[mask_valid, :] @ K.T
            for k in range(3):
                interp_fun = RectBivariateSpline(np.arange(h_im), np.arange(
                    w_im), Im[j, :, :, k].astype(float)/255, kx=1, ky=1)
                colors[mask_valid, k] = interp_fun(
                    uv[:, 1], uv[:, 0], grid=False)

        ind = np.sqrt(np.sum(X_ba ** 2, axis=1)) < 200
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X_new[ind]))
        pcd.colors = o3d.utility.Vector3dVector(colors[ind])
        o3d.io.write_point_cloud(
            '{}/points_{}.ply'.format(output_dir, i+1), pcd)

        print(f"checking projection error of {i+1}th camera")
        check_projection_error(i, X, track, K, P, inlier=inlier)
