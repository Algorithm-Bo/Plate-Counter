import cv2
import numpy as np
import os
import json
from pprint import pprint

# configure folders
# binary result image
input_name = '32_filters_bn_batch_size_4_test_easy_1'

input_foldername = 'results/' + input_name + '/results_t'

output_foldername = 'connected_components_images/' + input_name

output_foldername_CSV = 'CSV_lists/' + input_name

json_output_foldername = 'JSON_data/' + input_name

# test images folder
image_folder = 'data/images_test_staphylococcus_512x512_20'

json_folder = json_output_foldername

output_foldername_bb = 'bounding_boxes_images/' + input_name

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cv2.imshow('labeled hue', labeled_img)
    # cv2.waitKey()

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # cv2.imshow('labeled bgr', labeled_img)
    # cv2.waitKey()

    # set bg label to black 
    labeled_img[label_hue==0] = 0

    # show image
    # cv2.imshow('labeled', labeled_img)
    # cv2.waitKey()
    
    cv2.imwrite(output_path_color, labeled_img)

filename_list = os.listdir(input_foldername)
# print(filename_list)

if not os.path.isdir(output_foldername):
    os.mkdir(output_foldername)

if not os.path.isdir(output_foldername_CSV):
    os.mkdir(output_foldername_CSV)

if not os.path.isdir(json_output_foldername):
    os.mkdir(json_output_foldername)

if not os.path.isdir(output_foldername_bb):
    os.mkdir(output_foldername_bb)

object_count_list = []
for image in range(0, len(filename_list)):
    filename = str(image) + '.PNG'
    path = input_foldername + '/' + filename

    output_filename_color = filename
    output_path_color = output_foldername + '/' + output_filename_color

    # load image
    img = cv2.imread(path, 0)

    # cv2.imshow('Original', img)
    # cv2.waitKey()
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary

    # cv2.imshow('thres_binary', img)
    # cv2.waitKey()
    list_positions = []
    n_labels, output_img, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)

    # remove background class
    sizes = stats[1:, -1]
    n_labels = n_labels - 1
      
    # color image
    imshow_components(output_img)
    
    dict_coordinates = {}

    object_count = 0
    for coord in range(0, n_labels):
        object_count += 1
        label = object_count -1

        x_upper_left = stats[label+1,cv2.CC_STAT_LEFT]
        y_upper_left = stats[label+1,cv2.CC_STAT_TOP]
        width = stats[label+1,cv2.CC_STAT_WIDTH]
        height = stats[label+1,cv2.CC_STAT_HEIGHT]
               
        pos_x = int(x_upper_left + (width/2))
        pos_y = int(y_upper_left + (height/2))
        
        area = stats[label+1,cv2.CC_STAT_AREA]
        
        x_upper_left = int(x_upper_left)
        y_upper_left = int(y_upper_left)
        width = int(width)
        height = int(height)
        area = int(area)

        position_data_list = []
        position_data_list = [pos_x, pos_y, x_upper_left, y_upper_left, width, height, area]

        dict_coordinates[object_count] = position_data_list
        """
        print('Object Nr. %s x-upper_left %s y_upper_left %s width %s height %s' % (object_count, x_upper_left, y_upper_left, width, height))
        print('Object Nr. %s x-coordinate %s y-coordinate %s area %s' % (object_count, pos_x, pos_y, area))
        print('dict_coordinates: ', dict_coordinates)
        """
    # append list
    object_count_list.append(object_count)

    # print('Colonies detected: ', object_count)
    # print('dict coordinates')
    # pprint(dict_coordinates)

    # Create JSON Dictionary
    dict_coordinates_json = json.dumps(dict_coordinates, indent = 2, sort_keys=True)

    # write in file
    json_path = json_output_foldername + '/' + filename[:-4] + '.json'

    with open(json_path,'w', encoding = 'utf-8') as file:
        # file.write(filename)
        file.write(dict_coordinates_json)
        
# Create CSV Table
CSV_path = output_foldername_CSV + '/' + 'Counted_Objects.CSV'

counter = 0

with open(CSV_path, 'w') as fp:
    for elem in object_count_list:
        string1 = 'Image ' + str(counter) + '.PNG' 
        fp.write(string1)
        fp.write(';')
        fp.write(str(elem))
        fp.write('\n')
        counter += 1

# save images with bounding boxes
filename_list = os.listdir(image_folder)
print(filename_list)

for image in range(0, len(filename_list)):
    filename = str(image) + '.PNG'

    image_path = image_folder + '/' + filename

    output_filename = filename
    
    output_path_bb = output_foldername_bb + '/' + output_filename

    # print(image_path)
    # read JSON coordinates and show in image

    json_filename = filename[:-4] + '.JSON'
    json_path = json_folder + '/' + json_filename

    if os.path.isfile(json_path):
        with open(json_path, "r") as read_file:
            data = json.load(read_file)

    # pprint(data)

    # read image
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img,(512,512))

    count = 1
    for elem in data:
        # print('Nr. %d' %count)
        coordinates = data[elem]
        x = coordinates[0]
        # print('x: %d' %x)
        y = coordinates[1]
        # print('y: %d' %y)
        x_upper_left = coordinates[2]
        # print('x_upper_left: %d' %x_upper_left)
        y_upper_left = coordinates[3]
        # print('y_upper_left: %d' %y_upper_left)
        width = coordinates[4]
        # print('width: %d' %width)
        height = coordinates[5]
        # print('height: %d' %height)
        
        img_with_bb = img_resized
        number = str(count)
        cv2.rectangle(img_with_bb,(x_upper_left,y_upper_left),(x_upper_left + width, y_upper_left + height),(0,0,255),1) # Start, Ende, Farbe, Dicke
        cv2.putText(img_with_bb,number,(x_upper_left + width, y_upper_left + height),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
        
        count += 1
        
    string1 = 'Image: ' + filename
    string2 = 'Objects counted: ' + str(count-1)

    cv2.putText(img_with_bb,string1,(10, 20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
    cv2.putText(img_with_bb,string2,(10, 50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

    # cv2.imshow('Image with Bounding Boxes', img_with_bb)
    # cv2.waitKey(0)
    cv2.imwrite(output_path_bb, img_with_bb)

