import cv2
import numpy as np
from showdetail import showdetails,print_progress_bar

def joint_bilateral_filter(ambient,flash,diameter, sigma_color, sigma_space,channel_id):
    # Convert the images to float32 type
    flash = np.float32(flash)
    ambient = np.float32(ambient)

    # Apply joint bilateral filtering to the ambient image
    joint_bilateral = cv2.ximgproc.jointBilateralFilter(flash, ambient, diameter, sigma_color, sigma_space)

    return joint_bilateral

def joint_bilateral_filter_color(img,flash_img,diameter, sigma_color, sigma_space):
    # Split the input image into its color channels
    b, g, r = cv2.split(img)
    b_guide,g_guide,r_guide = cv2.split(flash_img)

    # Apply the joint bilateral filter to each color channel
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