import cv2
import numpy as np
import os

# Ausgangsbild: Grauwertbild mit bimodaler Intensitätsverteilung

folder_name ='data/images_test_staphylococcus_512x512_20'

output_foldername = 'images_staphylococcus_test_20_512x512_otsu'

if not os.path.isdir(output_foldername):
    os.mkdir(output_foldername)

filename_list = os.listdir(folder_name)
# print(filename_list)

for image in range(0, len(filename_list)):
    filename = filename_list[image]

    path = folder_name + '/' + filename

    img = cv2.imread(path)
    # img = cv2.imread(path,0)

    # Grauwertbild
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img_gray

    # cv2.imshow('Image', img)
    # cv.waitKey(0)

    # global thresholding
    # ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

    # Otsu's thresholding
    # ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
   
    # show image
    # cv2.imshow('Gaussian filtered Otsu segmentation', th3)
    # cv2.waitKey(0)

    # Bild speichern
    output_filename = filename
    # output_filename2 = filename[:-4] + '_otsu_without_gaussian' + '.PNG'
    # output_filename3 = filename[:-4] + '_otsu_with_gaussian_7x7' + '.PNG'

    output_path = output_foldername + '/' + output_filename
    # output_path2 = output_foldername + '/' + output_filename2
    # output_path3 = output_foldername + '/' + output_filename3

    cv2.imwrite(output_path, th3)
    # cv2.imwrite(output_path2, th2)
    # cv2.imwrite(output_path3, th4)
