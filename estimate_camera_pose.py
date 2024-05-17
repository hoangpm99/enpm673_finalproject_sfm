import numpy as np

from feature_extract_and_matching import estimate_essential_matrix_RANSAC

# Function to get four possible camera poses from the essential matrix
def get_camera_poses_from_e(E):
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    r1 = U @ W @ Vt
    r2 = U @ W.T @ Vt
    if np.linalg.det(r1) < 0:  # Ensure proper rotation matrix with determinant 1
        r1 = -r1
    if np.linalg.det(r2) < 0:
        r2 = -r2

    t1 = U[:, 2]
    t2 = -U[:, 2]

    # Camera centers calculated as -R.T * t
    c1 = -r1.T @ t1
    c2 = -r1.T @ t2
    c3 = -r2.T @ t1
    c4 = -r2.T @ t2

    R_set = np.array([r1, r1, r2, r2])
    C_set = np.array([c1, c2, c3, c4])

    return R_set, C_set

# Function to triangulate 3D points
def triangulate(P1, P2, track1, track2):
    assert track1.shape == track2.shape, "track1 and track2 must have same shape"
    n = track1.shape[0]

    X = -1 * np.ones((n, 3))
    for i in range(n):
        x1, y1 = track1[i]
        x2, y2 = track2[i]
        if x1 == -1 or x2 == -1 or y1 == -1 or y2 == -1:
            continue
        A = np.array([
            x1 * P1[2] - P1[0],
            y1 * P1[2] - P1[1],
            x2 * P2[2] - P2[0],
            y2 * P2[2] - P2[1]
        ])

        _, _, Vt = np.linalg.svd(A)
        point_3d = Vt[-1]
        point_3d /= point_3d[-1]
        X[i] = point_3d[:-1]

    return X

# Function to check the cheirality condition
def cheirality_condition_check(P1, P2, X):
    r1, t1 = P1[:, :3], P1[:, 3]
    r2, t2 = P2[:, :3], P2[:, 3]
    c1, c2 = -r1.T @ t1, -r2.T @ t2

    mask1 = np.sum(X, axis=-1) != -3
    mask2 = np.logical_and(
        np.dot(r1[2], (X - c1).T) > 0, np.dot(r2[2], (X - c2).T) > 0)

    valid_index = np.logical_and(mask1, mask2)

    return valid_index

# Function to estimate the correct camera pose
def estimate_correct_camera_pose(track1, track2):
    # Only use the features that are visible in both images
    feature_mask = np.logical_and(
        np.sum(track1, axis=-1) != -2, np.sum(track2, axis=-1) != -2)
    x1, x2 = track1[feature_mask], track2[feature_mask]

    # Estimate the essential matrix
    E, _ = estimate_essential_matrix_RANSAC(x1, x2, 500, None)
    # Get 4 possible camera poses
    R_set, C_set = get_camera_poses_from_e(E)

    P1 = np.eye(3, 4)
    valid_points = 0

    for i in range(4):
        P2 = np.hstack([R_set[i], -(R_set[i] @ C_set[i]).reshape((3, 1))])
        # Triangulate points
        X_3d = triangulate(P1, P2, track1, track2)
        
        # Check cheirality condition
        valid_index = cheirality_condition_check(P1, P2, X_3d)
        print(f"Num valid points: {np.sum(valid_index)} for camera pose {i}")
        if np.sum(valid_index) > valid_points:
            valid_points = np.sum(valid_index)
            R = R_set[i]
            C = C_set[i]
            X = -1 * np.ones((track1.shape[0], 3))
            X[valid_index] = X_3d[valid_index]

    return R, C, X
