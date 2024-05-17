import numpy as np
from scipy.optimize import least_squares

from utils_functions import rotation_to_quaterion
from utils_functions import quaternion_to_rotation

# Funtion to find the missing reconstruction


def find_missing_reconstruction(X, track_i):

    X_mask = np.logical_and(np.logical_and(
        X[:, 0] != -1, X[:, 1] != -1), X[:, 2] != -1)
    track_i_mask = np.logical_and(track_i[:, 0] != -1, track_i[:, 1] != -1)
    new_point = np.logical_and(track_i_mask, ~X_mask)

    return new_point

# Function to triangulate 3D points non-linearly


def triangulate_nonlinear(X, P1, P2, x1, x2):

    R1 = P1[:, :3]
    C1 = -R1.T @ P1[:, 3]
    R2 = P2[:, :3]
    C2 = -R2.T @ P2[:, 3]

    p1 = np.concatenate([C1, rotation_to_quaterion(R1)])
    p2 = np.concatenate([C2, rotation_to_quaterion(R2)])

    lamb = 0.005
    n_iter = 10
    X_new = X.copy()
    for i in range(X.shape[0]):
        pt = X[i, :]
        for j in range(n_iter):
            proj1 = R1 @ (pt - C1)
            proj1 = proj1[:2] / proj1[2]
            proj2 = R2 @ (pt - C2)
            proj2 = proj2[:2] / proj2[2]

            dfdX1 = compute_point_jacobian(pt, p1)
            dfdX2 = compute_point_jacobian(pt, p2)

            H1 = dfdX1.T @ dfdX1 + lamb * np.eye(3)
            H2 = dfdX2.T @ dfdX2 + lamb * np.eye(3)

            J1 = dfdX1.T @ (x1[i, :] - proj1)
            J2 = dfdX2.T @ (x2[i, :] - proj2)

            delta_pt = np.linalg.inv(H1) @ J1 + np.linalg.inv(H2) @ J2
            pt += delta_pt

        X_new[i, :] = pt

    return X_new

# Function to compute the point Jacobian


def compute_point_jacobian(X, p):
    R = quaternion_to_rotation(p[3:])
    C = p[:3]
    x = R @ (X - C)

    u = x[0]
    v = x[1]
    w = x[2]
    du_dc = R[0, :]
    dv_dc = R[1, :]
    dw_dc = R[2, :]

    dfdX = np.stack([
        (w * du_dc - u * dw_dc) / (w**2),
        (w * dv_dc - v * dw_dc) / (w**2)
    ], axis=0)

    return dfdX

# Setup for bundle adjustment


def bundle_adjustment_setup(P, X, track):

    n_cameras = P.shape[0]
    n_points = X.shape[0]

    n_projs = np.sum(track[:, :, 0] != -1)
    b = np.zeros((2*n_projs,))
    S = np.zeros((2*n_projs, 7*n_cameras+3*n_points), dtype=bool)
    k = 0
    camera_index = []
    point_index = []
    for i in range(n_cameras):
        for j in range(n_points):
            if track[i, j, 0] != -1:
                if i not in (0, 1):
                    S[2*k: 2*(k+1), 7*i: 7*(i+1)] = 1
                S[2*k: 2*(k+1), 7*n_cameras+3*j: 7*n_cameras+3*(j+1)] = 1
                b[2*k: 2*(k+1)] = track[i, j, :]
                camera_index.append(i)
                point_index.append(j)
                k += 1
    camera_index = np.asarray(camera_index)
    point_index = np.asarray(point_index)

    z = np.zeros((7*n_cameras+3*n_points,))
    for i in range(n_cameras):
        R = P[i, :, :3]
        C = -R.T @ P[i, :, 3]
        q = rotation_to_quaterion(R)
        p = np.concatenate([C, q])
        z[7*i: 7*(i+1)] = p

    z[7*n_cameras:] = X.ravel()

    return z, b, S, camera_index, point_index

# Function to calculate the reprojection error


def calculate_reprojection_error(z, b, n_cameras, n_points, camera_index, point_index):

    n_projs = camera_index.shape[0]
    f = np.zeros((2*n_projs,))
    for k, (i, j) in enumerate(zip(camera_index, point_index)):
        p = z[7*i: 7*(i+1)]
        X = z[7*n_cameras+3*j: 7*n_cameras+3*(j+1)]
        R = quaternion_to_rotation(p[3:] / np.linalg.norm(p[3:]))
        C = p[:3]
        proj = R @ (X - C)
        proj = proj / proj[2]
        f[2*k: 2*(k+1)] = proj[:2]
    err = b - f

    return err

# Update the poses and 3D points


def update_poses_and_3d_points(z, n_cameras, n_points):

    P_new = np.empty((n_cameras, 3, 4))
    for i in range(n_cameras):
        p = z[7*i: 7*(i+1)]
        q = p[3:]
        R = quaternion_to_rotation(q / np.linalg.norm(q))
        C = p[:3]
        P_new[i, :, :] = R @ np.hstack([np.eye(3), -C[:, np.newaxis]])

    X_new = np.reshape(z[7*n_cameras:], (-1, 3))

    return P_new, X_new


# Function to perform bundle adjustment
def perform_bundle_adjustment(P, X, track):

    n_cameras = P.shape[0]
    n_points = X.shape[0]

    z0, b, S, camera_index, point_index = bundle_adjustment_setup(P, X, track)

    res = least_squares(
        lambda x: calculate_reprojection_error(
            x, b, n_cameras, n_points, camera_index, point_index),
        z0,
        jac_sparsity=S,
        verbose=2
    )
    z = res.x

    err0 = calculate_reprojection_error(
        z0, b, n_cameras, n_points, camera_index, point_index)
    err = calculate_reprojection_error(
        z, b, n_cameras, n_points, camera_index, point_index)
    print('Reprojection error {} -> {}'.format(np.linalg.norm(err0), np.linalg.norm(err)))

    P_new, X_new = update_poses_and_3d_points(z, n_cameras, n_points)

    return P_new, X_new


def check_projection_error(idx, X, track, K, P, inlier=None):
    is_X_valid = np.all(X != -1, axis=1)
    if inlier is not None:
        is_inlier = np.zeros_like(is_X_valid)
        inlier_idx = np.where(inlier)[0]
        is_inlier[inlier_idx] = True
        is_X_valid = np.logical_and(is_X_valid, is_inlier)
    X_valid = X[is_X_valid]
    X_valid_h = np.hstack([X_valid, np.ones((X_valid.shape[0], 1))])
    x_proj_h = P[idx] @ X_valid_h.T
    x_proj = x_proj_h[:2] / x_proj_h[2]
    x_valid = track[idx][is_X_valid].T

    x_proj_K = K @ x_proj_h
    x_proj_K = x_proj_K[:2] / x_proj_K[2]
    x_proj_K = x_proj_K.T
    x_valid_K = K @ np.vstack([x_valid, np.ones((1, x_valid.shape[1]))])
    x_valid_K = x_valid_K[:2] / x_valid_K[2]
    x_valid_K = x_valid_K.T
    print("Ideal                        Projected")
    print(np.hstack([x_valid_K[:20], x_proj_K[:20]]))
    print(np.linalg.norm(x_proj_K - x_valid_K))

    return