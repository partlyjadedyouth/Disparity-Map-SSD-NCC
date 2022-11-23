import os, sys
import numpy as np
from matplotlib import pyplot as plt
import cv2

def SSD(t, x, r, c, w):
    """
    This function is for computing SSD, 
    which stands for sum of squared distance.

    t : template image (w x w)
    x : target image
    r, c : pixel position on x
    w : window size (== 2*w0+1)
    patch : w x w patched image on x[r, c]
    """

    w0 = w // 2
    
    ### START CODE HERE ###
    patch = x[r-w0:r+w0+1, c-w0:c+w0+1]
    ssd = np.sum((t - patch)**2)
    ### END CODE HERE ###
    
    return ssd

def disparity_SSD_c(I_L, I_R):
    """
    This function is for finding a corresponding pixel by using SSD,
    and compute disparity map.

    I_L, I_R : two rectified images
    w : window size (== 2*w0+1)
    max_dixparity : disparity should be smaller than this value
    D : disparity map to be returned
    """
    assert I_L.shape == I_R.shape
    
    w = 11                  # you can change this
    assert(w % 2 == 1)
    w0 = w // 2
    max_disparity = 20      # you can change this
    
    ### START CODE HERE ###
    D = np.zeros_like(I_L)
    img_row, img_col = I_L.shape
    
    # Traverse image
    for r in range(w0, img_row - w0):
        for c in range(w0, img_col - w0):
            patch_L = I_L[r-w0:r+w0+1, c-w0:c+w0+1] # Template image on I_L

            # Range of the column to perform SSD
            min_c = w0 if c - max_disparity < w0 else c - max_disparity
            max_c = img_col - w0 if c + max_disparity > img_col - w0 else c + max_disparity

            min_ssd = float('inf')  # Minimum value of SSD
            min_ssd_col = float('-inf') # Column of min_ssd 

            # Search through the row
            for i in range(min_c, max_c):
                ssd = SSD(patch_L, I_R, r, i, w)
                if ssd < min_ssd:
                    min_ssd = ssd
                    min_ssd_col = i

            D[r, c] = abs(c - min_ssd_col)
    ### END CODE HERE ###
    
    return D

def disparity_SSD_d(I_L, I_R):
    """
    This function is for finding a corresponding pixel 
    by using NCC(Normalized Cross Correlation),
    and compute disparity map.

    I_L, I_R : two rectified images
    w : window size (== 2*w0+1)
    max_dixparity : disparity should be smaller than this value
    D : disparity map to be returned
    """
    assert I_L.shape == I_R.shape
    
    w = 11                  # you can change this
    assert(w % 2 == 1)
    w0 = w // 2
    max_disparity = 15      # you can change this
    
    ### START CODE HERE ###
    D = np.zeros_like(I_L)
    img_row, img_col = I_L.shape
    
    # Traverse image
    for r in range(w0, img_row - w0):
        for c in range(w0, img_col - w0):
            patch_L = I_L[r-w0:r+w0+1, c-w0:c+w0+1] # Template image on I_L
            mean_patch_L = np.average(patch_L)
            norm_patch_L = np.sum((patch_L - mean_patch_L)**2)

            # Range of the column to perform SSD
            min_c = w0 if c - max_disparity < w0 else c - max_disparity
            max_c = img_col - w0 if c + max_disparity > img_col - w0 else c + max_disparity

            max_ncc = float('-inf')  # Minimum value of NCC
            max_ncc_col = float('-inf') # Column of max_ncc

            # Search through the row and compute NCC
            for i in range(min_c, max_c):
                patch_R = I_R[r-w0:r+w0+1, i-w0:i+w0+1]
                mean_patch_R = np.average(patch_R)
                norm_patch_R = np.sum((patch_R - mean_patch_R)**2)
                
                ncc = np.sum(np.multiply(patch_L - mean_patch_L, patch_R - mean_patch_R)) / np.sqrt((norm_patch_L * norm_patch_R))
                if ncc > max_ncc:
                    max_ncc = ncc
                    max_ncc_col = i

            D[r, c] = abs(c - max_ncc_col)
    ### END CODE HERE ###
    
    return D


# Do not modify this
def save_disparity_map(filename, D):
    D = (np.clip(D / 64 * 255.0, 0, 255)).astype(np.uint8)
    cv2.imwrite(filename, D)
    
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('wrong arguments')
        exit(0)
        
    problem = sys.argv[1] # c or d
    l_image = sys.argv[2] # L1
    r_image = sys.argv[3] # R1 or R2
    
    # Read images
    data_dir = './data/stereo'
    save_dir = './results'
    os.makedirs(save_dir, exist_ok=True)

    I_L = cv2.cvtColor(cv2.imread(os.path.join(data_dir, f'{l_image}.png')), cv2.COLOR_BGR2GRAY)
    I_R = cv2.cvtColor(cv2.imread(os.path.join(data_dir, f'{r_image}.png')), cv2.COLOR_BGR2GRAY)

    if problem == 'c':
        D = disparity_SSD_c(I_L, I_R)
    elif problem == 'd':
        D = disparity_SSD_d(I_L, I_R)
    else:
        raise ValueError
    
    save_disparity_map(os.path.join(save_dir, f'disparity_{problem}_{l_image}{r_image}.png'), D)