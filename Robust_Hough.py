from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

def edge(img):
    img_cpy = np.copy(img)
    x_size, y_size = img.shape
    for row in range(1, x_size-1):
        for col in range(1, y_size-1):
            if img[row, col] == 1:
                if (img[row+1, col] == 1) and (img[row-1, col] == 1) and (img[row, col+1] == 1) and (img[row, col-1] == 1):
                    img_cpy[row, col] = 0
    return img_cpy

def DerivativeEdges(img):
    derivative_img = edge(img)
    return derivative_img


def CalculateHoughMatrix(edge_image, num_rhos=180, num_thetas=360):
    '''
    The input image shall be a logic image, i.e. only 0 and 1
    '''
    # Extract Image size
    edge_height, edge_width = edge_image.shape[:2]
    edge_height_half, edge_width_half = edge_height / 2, edge_width / 2
    d = np.sqrt(np.square(edge_height) + np.square(edge_width))
    dtheta = 180 / num_thetas
    drho = (2 * d) / num_rhos
    thetas = np.arange(-90, 90, step=dtheta)
    rhos = np.arange(-d, d, step=drho)
    # Calculate Cosine and Sinus values of the axis
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))
    # Calculate the edge point indices from the edge image
    edge_points = np.argwhere(edge_image != 0)
    # Transform the axis to support negative rhos and thetas
    edge_points = edge_points - np.array([[edge_height_half, edge_width_half]])
    # Calculate the Rho Values in a Matrix formation
    rho_values = np.matmul(edge_points, np.array([sin_thetas, cos_thetas]))
    # Calculate the Hough Matrix
    accumulator = np.zeros((len(rhos), len(rhos)))
    accumulator, theta_vals, rho_vals = np.histogram2d(np.tile(thetas, rho_values.shape[0]), rho_values.ravel(), bins=[thetas, rhos])
    accumulator = np.transpose(accumulator)
    return accumulator, rhos, thetas

def CalculateHoughPeaks(H, numPeaks, threshold):
    done = False
    new_H = np.copy(H)
    peaks = []
    # Define neighborhood size
    nhood = np.array((H.shape[0] / 50, H.shape[1] / 50))
    nhood = [max(2 * np.ceil(elem / 2) + 1, 1) for elem in nhood]
    # Calculate neighborhood center
    nhood_center = [(elem-1)/2 for elem in nhood]
    THETA = 1
    RHO = 0
    while not done:
        # Extract the coordinates of rho and theta from the new Hough Matrix
        index = np.unravel_index(np.argmax(new_H), np.array(new_H).shape)
        max_val = new_H[index]
        # Check if this value sustain our predefined threshold for being an actual line in the original image
        if max_val > threshold:
            peaks.append(index)
            # Calculate the centers for both Rho and Theta from the chosen index
            theta_center, rho_center = index[THETA], index[RHO]
            # Calculate the neighborhood for both Rho and Theta from the chosen index around the current center
            theta_low_bound, rho_low_bound = theta_center - nhood_center[THETA], rho_center - nhood_center[RHO]
            theta_high_bound, rho_high_bound = theta_center + nhood_center[THETA], rho_center + nhood_center[RHO]
            
            # Generate a Grid of Rhos and Thetas Combination around the maximal point
            [rhos, thetas] = np.meshgrid(np.arange(np.max((0, rho_low_bound)), np.min((rho_high_bound, new_H.shape[RHO]))+1),
                                   np.arange(theta_low_bound, theta_high_bound + 1))
            '''
            The following Code just changes the values around the following peak to be 0 so we will not choose 
            any of them in the next round.
            '''
            rhos = rhos.ravel()
            thetas = thetas.ravel()

            theta_too_low = np.argwhere(thetas < 0)
            thetas[theta_too_low] = new_H.shape[1] + thetas[theta_too_low]
            rhos[theta_too_low] = new_H.shape[0] - rhos[theta_too_low]

            theta_too_high = np.argwhere(thetas >= new_H.shape[THETA])
            thetas[theta_too_high] = thetas[theta_too_high] - new_H.shape[THETA]
            rhos[theta_too_high] = new_H.shape[RHO] - rhos[theta_too_high]
            new_H[rhos.astype(np.int64), thetas.astype(np.int64)] = 0
            done = len(peaks) == numPeaks
        else:
            done = True
    return np.array(peaks)



def applyKmenas(thetas, rhos):
    sil = []
    kmax = 3
    expanded_theta = np.expand_dims(thetas, -1)
    for k in range(2, kmax + 1):
        kmeans = KMeans(n_init=20, random_state=1, max_iter=1000, tol=1e-6, n_clusters=k).fit(expanded_theta)
        labels = kmeans.labels_
        sil.append(silhouette_score(expanded_theta, labels))
    if np.argmax(sil) == 1:
        Centeroids = kmeans.cluster_centers_
        ClusterMapping = kmeans.labels_
        args = np.argsort(Centeroids, axis=0)
        left_center = args[0]
        right_center = args[2]
        num_left = sum(ClusterMapping == left_center)
        num_right = sum(ClusterMapping == right_center)
        if num_left < num_right:
            thetas = thetas[ClusterMapping != left_center]
            rhos = rhos[ClusterMapping != left_center]
        else:
            thetas = thetas[ClusterMapping != right_center]
            rhos = rhos[ClusterMapping != right_center]

    clusterSize = 4
    kmeans = KMeans(n_clusters=clusterSize).fit(np.column_stack((thetas, rhos)))
    Centeroids = kmeans.cluster_centers_
    args = np.argsort(np.abs(Centeroids[:, 0]), axis=0)
    peaks = Centeroids[args, :]
    return peaks


def FindIntersectionsInXYRepresentation(peaks):
    def find_x(rhos, thetas):
        assert len(rhos) == 2
        assert len(thetas) == 2
        rad_thetas = np.deg2rad(thetas)
        denom = np.zeros((2,1))
        numerator = np.zeros((2, 1))
        for idx, data in enumerate(zip(rhos, rad_thetas)):
            rho, theta = data
            denom[idx] = np.cos(theta) / np.sin(theta)
            numerator[idx] = rho / np.sin(theta)
        '''
        Calculate the following equation
        X = (rho0 / sin(theta0) - rho1 / sin(theta1)) / (cos(theta0) / sin(theta0) - cos(theta1) / sin(theta1))
        '''
        x = (numerator[0] - numerator[1]) / (denom[0] - denom[1])
        return x

    def find_y(rho, theta, x):
        rad_theta = np.deg2rad(theta)
        '''
        Calculate the following equation
        Y = rho/sin(theta)-x*cos(theta)/sin(theta);
        '''
        y = rho/np.sin(rad_theta) - x*np.cos(rad_theta)/np.sin(rad_theta)
        return y

    assert peaks.shape[0] == 4
    X = 0
    Y = 1
    THETA = 0
    RHO = 1
    corners_estimation = np.zeros_like(peaks)
    # Sort according to theta
    sorted_args = np.argsort(peaks[:, THETA], axis=0)
    sorted_peaks = peaks[sorted_args]
    sorted_peaks[np.abs(sorted_peaks[:, THETA]) < 0.001, THETA] = 0.001
    # Generate intersection pairs in a list formation
    [THETAS, RHOS] = np.meshgrid(range(0, 2), range(2, 4))
    peaks_pairs = [(sorted_peaks[i], sorted_peaks[j]) for i, j in zip(THETAS.ravel(), RHOS.ravel())]
    for corner_idx, pair in enumerate(peaks_pairs):
        thetas = []
        rhos = []
        thetas.extend([peak[THETA] for peak in pair])
        rhos.extend([peak[RHO] for peak in pair])
        corners_estimation[corner_idx][X] = find_x(rhos, thetas)
        corners_estimation[corner_idx][Y] = find_y(rho=pair[0][RHO], theta=pair[0][THETA], x=corners_estimation[corner_idx][X])
    return corners_estimation


def closest_point(edge_img, corners):
    # Find the indices where the edge appears in the mask
    doc_edges_indices = np.where(edge_img == 1)
    # Transform it to matrix representation
    doc_edges_indices = np.array(list(doc_edges_indices)).T
    # Find the distances in x axis
    x_dist = np.square((np.expand_dims(corners[:, 0], axis=1) - np.expand_dims(doc_edges_indices[:, 0], axis=0)))
    # Find the distances in y axis
    y_dist = np.square((np.expand_dims(corners[:, 1], axis=1) - np.expand_dims(doc_edges_indices[:, 1], axis=0)))
    # Find the approximated distance ~ we dont use square root and mean in order to accelerate  
    distance_vec = x_dist + y_dist
    # Find the minimum distance indices
    idx = np.argmin(distance_vec, axis=1)
    # Fine tune the corner estimation
    closest_point = doc_edges_indices[idx]
    return closest_point


def plot_hough(H, estimated_peaks, hough_peaks, T, R):
    """
    This function prints the Hough Matrix
    """
    fig, ax = plt.subplots(1, 1)
    ax.imshow(np.log1p(H))
    ax.scatter(hough_peaks[:, 1], hough_peaks[:, 0], c='red', label="Hough peaks")
    newThetaAxis = (np.digitize(estimated_peaks[:, 0], T)) 
    newRhoAxis = np.digitize(estimated_peaks[:, 1], R)
    ax.scatter(newThetaAxis, newRhoAxis, c='yellow', label="Estimated lines")
    ax.set_title("Hough Matrix")
    plt.xticks(list(range(0,360,20)),list(range(-90,90,10)))
    plt.yticks(list(range(0,180,10)),list(range(-90,90,10)))
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\rho$')
    plt.legend()
    plt.show()
    
    
def plot_img_and_mask(img, mask, corners):
    """
    This function prints an img,  its ground truth mask, and also prints a given corners indices over them
    """
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[0].scatter(corners[:, 0], corners[:, 1], c='red')
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[:, :, i])
            ax[i + 1].scatter(corners[:, 0], corners[:, 1], c='red')
    else:
        ax[1].set_title(f'Ground Truth Mask')
        ax[1].imshow(mask)
        ax[1].scatter(corners[:, 0], corners[:, 1], c='red')
    plt.xticks([]), plt.yticks([])
    plt.show()
    
    
def e2e_algorithm(img, numPeaks=20):
    # End to End Corner Extraction from the masked image
    edge_image = DerivativeEdges(img)
    H, R, T = CalculateHoughMatrix(edge_image)
    hough_peaks = CalculateHoughPeaks(H, numPeaks, threshold=0.1 * np.max(H))
    thetas = T[hough_peaks[:, 1]]
    rhos = R[hough_peaks[:, 0]]
    estimated_peaks = applyKmenas(thetas, rhos)
    corners_estimation = FindIntersectionsInXYRepresentation(estimated_peaks)
    corners_estimation[:, 0] += img.shape[1]/2
    corners_estimation[:, 1] += img.shape[0]/2
    corners_estimation_closest_point = np.fliplr(closest_point(edge_image, np.fliplr(corners_estimation)))
    return corners_estimation_closest_point, corners_estimation, H, T, R, hough_peaks, estimated_peaks
   