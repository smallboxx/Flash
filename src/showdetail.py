import cv2
import numpy as np
import sys
def showdetails(img1,img2):
    img =np.subtract(img1,img2)
    return img

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')

if __name__=="__main__":
    filename="data\\lamp\\Bilater_flitering\\masking.jpg"
    img1=cv2.imread("data\\lamp\\lamp_ambient.tif")
    img2=cv2.imread(filename)
    img=showdetails(img1,img2)
    savename="data\\lamp\\mask\\"+"mask_"+filename.split("\\")[-1].split(".")[0]+".jpg"
    print(savename)
    cv2.imwrite(savename,img)