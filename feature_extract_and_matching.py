import cv2
import matplotlib.pyplot as plt
import numpy as np

# Function to match features between two images after SIFT feature extraction


def feature_matching_sift(location1, descriptor1, location2, descriptor2):
    # Compute the pairwise distances between descriptors
    distances = np.sqrt(np.sum(np.square(descriptor1[:, np.newaxis] - descriptor2), axis=2))

    # Find the nearest neighbor for each descriptor in descriptor1
    nn_indexes = np.argmin(distances, axis=1)

    # Find the nearest neighbor for each descriptor in descriptor2
    nn_distances = np.min(distances, axis=1)

    # Compute the ratio of the nearest neighbor distance to the second nearest neighbor distance
    second_nn_distances = np.partition(distances, 1, axis=1)[:, 1]
    ratio = nn_distances / second_nn_distances

    # Select the matches based on the ratio test
    match_indexes = np.where(ratio < 0.8)[0]

    # Get the matched keypoint locations and indexes
    x1 = location1[match_indexes]
    x2 = location2[nn_indexes[match_indexes]]
    ind1 = match_indexes

    return x1, x2, ind1


def estimate_essential_matrix(x1, x2):
    # Construct the A matrix
    A = []
    for p1, p2 in zip(x1, x2):
        x, y = p1
        xp, yp = p2
        A.append([xp*x, xp*y, xp, yp*x, yp*y, yp, x, y, 1])
    A = np.array(A)

    # Perform SVD on A
    _, _, V = np.linalg.svd(A)

    # Extract the essential matrix from the last column of V
    E = V[-1].reshape(3, 3)

    # Enforce the rank 2 constraint
    U, S, Vt = np.linalg.svd(E)
    S = np.array([1, 1, 0])

    # Reconstruct the essential matrix
    E = U @ np.diag(S) @ Vt
    return E


def estimate_essential_matrix_RANSAC(x1, x2, ransac_iter, ransac_threshold):
    # Set the default values for ransac_iter and ransac_threshold
    if ransac_iter is None:
        # Calculate the number of iterations based on desired probability and inlier ratio
        p = 0.99  # Desired probability of success
        e = 0.5  # Expected inlier ratio
        s = 8    # Number of samples required for 8-point algorithm
        ransac_iter = int(np.log(1 - p) / np.log(1 - (1 - e)**s))
        print(f"Number of RANSAC iterations: {ransac_iter}")

    if ransac_threshold is None:
        # Calculate the threshold based on the median of errors
        x1_hom = np.hstack((x1, np.ones((x1.shape[0], 1))))
        x2_hom = np.hstack((x2, np.ones((x2.shape[0], 1))))
        E_initial = estimate_essential_matrix(x1, x2)
        errors = np.abs(np.diag(x2_hom @ E_initial @ x1_hom.T))
        ransac_threshold = 2 * np.median(errors)
        print(f"RANSAC threshold: {ransac_threshold}")

    max_inliers = []
    best_E = None

    # Perform RANSAC iterations
    for _ in range(ransac_iter):
        # randomly select 8 pairs of matched keypoints to perform 8 points algorithm
        indexes = np.random.choice(x1.shape[0], 8, replace=False)
        E = estimate_essential_matrix(x1[indexes], x2[indexes])
        # calculate the error for each pair of matched keypoints
        x1_hom = np.hstack((x1, np.ones((x1.shape[0], 1))))
        x2_hom = np.hstack((x2, np.ones((x2.shape[0], 1))))

        errors = np.abs(np.diag(x2_hom @ E @ x1_hom.T))

        inliers = np.where(errors < ransac_threshold)[0]

        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            best_E = E

    E = best_E
    inlier_num = max_inliers

    return E, inlier_num


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines '''
    R, C, _ = img1.shape
    
    for R, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -R[2]/R[1]])
        x1, y1 = map(int, [C, -(R[2]+R[0]*C)/R[1]])
        img3 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img3 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img4 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img3, img4


def get_feature_tracker(img, K):

    print("Building feature tracker")

    assert K[2][2] == 1, "Last element of the last row of K is 1.0."

    loc_list = []
    des_list = []

    N = img.shape[0]
    sift = cv2.SIFT_create()

    # Extract SIFT features for each image
    print("Extracting SIFT features")
    for i in range(N):
        # Convert to grayscale
        gray_i = cv2.cvtColor(img[i], cv2.COLOR_RGB2GRAY)
        kp, des = sift.detectAndCompute(gray_i, None)
        loc = np.array([loc.pt for loc in kp])
        loc_list.append(loc)
        des_list.append(des)
        print(f"Extracted {loc.shape[0]} SIFT features from image {i+1}")

    track = np.empty((N, 0, 2))

    # Match features between images for each pair of images
    for i in range(N):
        track_i = -1 * np.ones((N, loc_list[i].shape[0], 2))
        for j in range(i+1, N):
            x1, x2, ind1 = feature_matching_sift(
                loc_list[i], des_list[i], loc_list[j], des_list[j])
            print(
                f'Got {x1.shape[0]} matches: image {i+1} and image {j+1}')
            
            x1 = np.dot(np.linalg.inv(K), np.vstack(
                (x1.T, np.ones(x1.shape[0])))).T
            x2 = np.dot(np.linalg.inv(K), np.vstack(
                (x2.T, np.ones(x2.shape[0])))).T
            
            x1 = x1[:, :2]
            x2 = x2[:, :2]
            E, inlier_indexes = estimate_essential_matrix_RANSAC(
                x1, x2, 500, None)

            # get the feature indexes which are considered as inliers
            feature_indexes = ind1[inlier_indexes]

            # update the track_i so that the inliers are stored in the correct indexes
            track_i[i, feature_indexes, :] = x1[inlier_indexes]
            track_i[j, feature_indexes, :] = x2[inlier_indexes]

            # Compute the fundamental matrix
            F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
            # Find epilines and draw them
            pt1 = np.int32(
                K @ np.vstack((x1[inlier_indexes].T, np.ones(x1[inlier_indexes].shape[0]))))
            pt2 = np.int32(
                K @ np.vstack((x2[inlier_indexes].T, np.ones(x2[inlier_indexes].shape[0]))))
            pt1 = pt1[:2].T
            pt2 = pt2[:2].T
            lines1 = cv2.computeCorrespondEpilines(pt2.reshape(-1, 1, 2), 2, F)

            I = img.copy()
            lines1 = lines1.reshape(-1, 3)
            img5, img6 = drawlines(I[i], I[j], lines1, pt1, pt2)
            # Find epilines and draw
            lines2 = cv2.computeCorrespondEpilines(pt1.reshape(-1, 1, 2), 1, F)
            lines2 = lines2.reshape(-1, 3)
            img3, img4 = drawlines(I[j], I[i], lines2, pt2, pt1)
            plt.subplot(121), plt.imshow(img5)
            plt.title(f'epiline of image{i+1} for image {j+1}',
                      fontsize=10), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(img3)
            plt.title(f'epiline of image{j+1} for image {i+1}',
                      fontsize=10), plt.xticks([]), plt.yticks([])
            plt.savefig(f'./epipolar_visuals/epipolar_{i}_{j}.png')
            
        # check if the feature is matched in at least one
        valid_feature_mask = np.sum(track_i[i], axis=-1) != -2
        track_i = track_i[:, valid_feature_mask]
        print(f'Found {track_i.shape[1]} features in image {i+1}')
        track = np.concatenate((track, track_i), axis=1)

    return track
