import cv2
import numpy as np
import json
from pprint import pprint
import os

def drawMarkers(midpoint_coordinates, img):
    # height, width = img.shape[:2] 
    for elem in range(len(midpoint_coordinates)):
        point = midpoint_coordinates[elem]
        x = point[0]
        y = point[1]
        cv2.line(img,(abs(x - 5), y), (x + 5, y),(0,255,0),1) # Start, Ende, Farbe, Dicke
        cv2.line(img,(x, abs(y - 5)), (x, y + 5),(0,255,0),1)
        # cv2.rectangle(img,(point),(point),(0,255,0),2) # Start, Ende, Farbe, Dicke
    return img

def getContours(img):
    coordinates = []
    # counter = 0
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # print(area)
        peri = cv2.arcLength(cnt, True)
        # print(peri)
        coordinate_midpoint_x = 0
        coordinate_midpoint_y = 0
        # counter += 1
        if ((area > 5 and area < 100) and peri < 50): 
            cv2.drawContours(imgContour, cnt, -1, (0,0,255),3)
            # peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            # print(len(approx))
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            if objCor > 4:
                object_type = "Circle"
            else: object_type = "None"

            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(imgContour,object_type,(x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,255),2)

            coordinate_midpoint_x = x + (w//2)
            coordinate_midpoint_y = y + (h//2)
            coordinates.append((coordinate_midpoint_x, coordinate_midpoint_y))
    
    # return (counter, coordinates)
    return coordinates

folder_name ='images_staphylococcus_with_points_renamed_512x512'

filename_list = os.listdir(folder_name)
print(filename_list)

# filename_list = filename_list[:10]

for image in range(0, len(filename_list)):
    filename = filename_list[image]

    path = folder_name + '/' + filename

    img = cv2.imread(path)

    # in HSV Farbraum transformieren
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # rote Punkte 0°/100%/100%
    hue_min = 0 
    hue_max = 6 # 7 
    sat_min = 170
    sat_max = 255
    val_min = 0
    val_max = 255
    lower = np.array([hue_min,sat_min,val_min])
    upper = np.array([hue_max,sat_max,val_max])
    mask_red = cv2.inRange(img_HSV, lower, upper)

    # hellblaue Punkte 180°/100%/100%
    hue_min = 70 
    hue_max = 90
    sat_min = 140 # 130
    sat_max = 255
    val_min = 160 # 220
    val_max = 255
    lower = np.array([hue_min,sat_min,val_min])
    upper = np.array([hue_max,sat_max,val_max])
    mask_cyan = cv2.inRange(img_HSV, lower, upper)

    # magenta Punkte 300°/100%/100%
    hue_min = 125
    hue_max = 255
    sat_min = 190
    sat_max = 255
    val_min = 0
    val_max = 255
    lower = np.array([hue_min,sat_min,val_min])
    upper = np.array([hue_max,sat_max,val_max])
    mask_magenta = cv2.inRange(img_HSV, lower, upper)

    mask = mask_red + mask_cyan + mask_magenta

    img_result= cv2.bitwise_and(img,img,mask=mask)

    imgContour = mask.copy()

    img_canny = cv2.Canny(mask, 50, 50)

    # cv2.imshow('Original', img)
    # cv2.imshow('HSV', img_HSV)
    # cv2.imshow('Mask', mask)
    # cv2.imshow('Result', img_result)
    # cv2.imshow('Mask red', mask_red)
    # cv2.imshow('Mask lightblue', mask_cyan)
    # cv2.imshow('Mask violet', mask_magenta)
    # cv2.imshow('Canny', img_canny)
    # cv2.imshow('Contour', imgContour)

    midpoint_coordinates = getContours(img_canny)

    # Marker zeichnen
    img_markers = drawMarkers(midpoint_coordinates, img)
    # cv2.imshow('Original with markers', img_markers)

    # Markerbild speichern
    output_filename = filename[:-4] + '.PNG'
    output_foldername_markers = 'images_staphylococcus_with_markers'

    if not os.path.isdir(output_foldername_markers):
        os.mkdir(output_foldername_markers)

    output_path_markers = output_foldername_markers + '/' + output_filename
    cv2.imwrite(output_path_markers, img_markers)

    # Dictionary erzeugen

    dict_coordinates = {}

    count = 0
    for coord in midpoint_coordinates:
        count += 1
        print('Object Nr %s x-coordinate %s y-coordinate %s' % (count, coord[0], coord[1]))
        dict_coordinates[count] = coord
    print('Colonies detected: ', len(midpoint_coordinates))

    print(dict_coordinates)

    dict_coordinates_json = json.dumps(dict_coordinates, indent = 2)

    # in Datei schreiben
    output_foldername = 'JSON_files'

    if not os.path.isdir(output_foldername):
        os.mkdir(output_foldername)

    output_filename = output_foldername + '/' + filename[:-4] + '.json'

    with open(output_filename,'w', encoding = 'utf-8') as file:
        # file.write(filename)
        file.write(dict_coordinates_json)

cv2.waitKey(0)