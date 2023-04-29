import cv2
import numpy as np
import sys
from showdetail import showdetails,print_progress_bar


def bilateral_filter(img, diameter, sigma_color, sigma_space,channel_id):
    # Convert the input image to float32
    img = np.float32(img)

    # Compute the spatial kernel
    radius = diameter // 2
    spatial_kernel = np.zeros((diameter, diameter))
    for i in range(diameter):
        for j in range(diameter):
            spatial_distance = np.sqrt((i - radius) ** 2 + (j - radius) ** 2)
            spatial_kernel[i, j] = np.exp(-np.square(spatial_distance) / (2 * np.square(sigma_space)))

    # Initialize the output image
    output = np.zeros_like(img)
    H=img.shape[0]
    W=img.shape[1]
    # Iterate over each pixel in the input image
    for i in range(radius, H - radius):
        for j in range(radius, W - radius):
            # Extract the local patch around the current pixel
            patch = img[i - radius:i + radius + 1, j - radius:j + radius + 1]

            # Compute the range kernel using the patch and the current pixel
            range_kernel = np.exp(-np.square(patch - img[i, j]) / (2 * np.square(sigma_color)))

            # Compute the bilateral filter response
            fast_bilateral_filter = spatial_kernel * range_kernel
            normalization_factor = np.sum(fast_bilateral_filter)
            output[i, j] = np.sum(fast_bilateral_filter * patch) / normalization_factor
            print_progress_bar(channel_id*(H-diameter)*(W-diameter)+(i-radius)*(W-diameter)+(j-radius)+1,(H-diameter)*(W-diameter)*3,"start","complete")

    # Convert the output image back to uint8
    output = np.uint8(np.clip(output, 0, 255))

    return output

def bilateral_filter_color(img, diameter, sigma_color, sigma_space):
    # Split the input image into its color channels
    b, g, r = cv2.split(img)

    # Apply the fast bilateral filter to each color channel
    b_filtered = bilateral_filter(b, diameter, sigma_color, sigma_space,channel_id=0)
    g_filtered = bilateral_filter(g, diameter, sigma_color, sigma_space,channel_id=1)
    r_filtered = bilateral_filter(r, diameter, sigma_color, sigma_space,channel_id=2)

    # Merge the filtered color channels back into a color image
    output = cv2.merge((b_filtered, g_filtered, r_filtered))

    return output

if __name__=="__main__":
    # Load the input image
    img = cv2.imread('data\\lamp\\lamp_ambient.tif')
    # Set the filter parameters
    diameter = 15
    SPACE_K_LIST=[15,30,60]
    INTENSITY_K_LIST=[0.05,0.15,0.25]
    for sigma_space in SPACE_K_LIST:
        for sigma_color in INTENSITY_K_LIST:
            print("bilater filter by sigma_space:"+str(sigma_space)+" sigma_color:"+str(sigma_color))
            output = bilateral_filter_color(img, diameter, sigma_color, sigma_space)
            savename="data\\lamp\\Bilater_flitering\\"+"bilateral_space"+str(sigma_space)+"_intensity"+str(sigma_color).split(".")[1]+".jpg"
            cv2.imwrite(savename, output)