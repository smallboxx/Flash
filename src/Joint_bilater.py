import cv2
import numpy as np
import sys

# progress_bar
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')

def joint_bilateral_filter(img,img_guide,diameter, sigma_color, sigma_space,channel_id):
    # Convert the input image to float32
    img = np.float32(img)
    img_guide = np.float32(img_guide)

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

            # Compute the fast bilateral filter response
            bilateral_filter = spatial_kernel * range_kernel
            normalization_factor = np.sum(bilateral_filter)
            output[i, j] = np.sum(bilateral_filter * img_guide[i - radius:i + radius + 1, j - radius:j + radius + 1]) / normalization_factor
            print_progress_bar(channel_id*(H-diameter)*(W-diameter)+(i-radius)*(W-diameter)+(j-radius)+1,(H-diameter)*(W-diameter)*3,"start","complete\n")

    # Convert the output image back to uint8
    output = np.uint8(np.clip(output, 0, 255))

    return output

def joint_bilateral_filter_color(img,flash_img,diameter, sigma_color, sigma_space):
    # Split the input image into its color channels
    b, g, r = cv2.split(img)
    b_guide,g_guide,r_guide = cv2.split(flash_img)

    # Apply the fast bilateral filter to each color channel
    b_filtered = joint_bilateral_filter(b,b_guide ,diameter, sigma_color, sigma_space,channel_id=0)
    g_filtered = joint_bilateral_filter(g,g_guide ,diameter, sigma_color, sigma_space,channel_id=1)
    r_filtered = joint_bilateral_filter(r,r_guide ,diameter, sigma_color, sigma_space,channel_id=2)

    # Merge the filtered color channels back into a color image
    output = cv2.merge((b_filtered, g_filtered, r_filtered))

    return output

if __name__=="__main__":
    # Load the input image
    img = cv2.imread('data\\lamp\\lamp_ambient.tif')
    flash_img = cv2.imread('data\\lamp\\lamp_flash.tif')
    # Set the filter parameters
    diameter = 15
    SPACE_K_LIST=[15,30,60]
    INTENSITY_K_LIST=[0.05,0.15,0.25]
    for sigma_space in SPACE_K_LIST:
        for sigma_color in INTENSITY_K_LIST:
            print("bilater filter by sigma_space:"+str(sigma_space)+" sigma_color:"+str(sigma_color))
            output = joint_bilateral_filter_color(img,flash_img, diameter, sigma_color, sigma_space)
            savename="data\\lamp\\Bilater_flitering\\"+"joint_bilateral_space"+str(sigma_space)+"_intensity"+str(sigma_color).split(".")[1]+".jpg"
            cv2.imwrite(savename, output)