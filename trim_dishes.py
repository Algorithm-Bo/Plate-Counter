import cv2
import numpy as np
import os

def make_square(img):
    size_y, size_x = img.shape[:2] 
    if size_x > size_y:
        print('make square', img)
        add_v = np.zeros(((size_x-size_y),size_x,3))
        add_v[:] = 0,0,255 # rot
        img_square = np.vstack((img,add_v))
        return img_square
    else:
        print('make square', img)
        add_h = np.zeros((size_y,(size_y-size_x),3))
        add_h[:] = 0,0,255
        img_square = np.hstack((img,add_h))
        return img_square

folder_name = 'images_lab/pseudomonas'
folder_name_labeled = 'images_lab_with_points/pseudomonas'

output_foldername = 'images_pseudomonas'
output_foldername_labeled = 'images_pseudomonas_with_points'

if not os.path.isdir(output_foldername):
    os.mkdir(output_foldername)

if not os.path.isdir(output_foldername_labeled):
    os.mkdir(output_foldername_labeled)

filename_list = os.listdir(folder_name)
# print(filename_list)

output_names_dishes = 'names_dishes_pseudomonas.CSV'
output_size_dishes = 'size_dishes_pseudomonas.CSV'

output_foldername2 = 'images_hough_dishes_pseudomonas'

names_list_dishes = []
size_list_dishes = []
dish_number = 0

for image in range(0, len(filename_list)):
    filename = filename_list[image]

    path = folder_name + '/' + filename
    path_labeled = folder_name_labeled + '/' + filename[:2] + 'a' + filename[2:]
    # print(path_labeled)

    img = cv2.imread(path)
    img_labeled = cv2.imread(path_labeled)

    # output_filename = filename[:-4] + '.PNG'
    # output_path = output_foldername + '/' + output_filename
    # output_path_labeled = output_foldername_labeled + '/' + output_filename
    # print(output_path)

    # Grauwertbild erzeugen
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray, 5)
        
    # cv2.imshow("Image gray", img_gray)
    # cv2.waitKey(0)

    # Hough Transformation
    rows = img_gray.shape[0]
    # min_dist = abs(rows/3) -120
    min_dist = 540

    dishes = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, min_dist, param1=100, param2=20, minRadius=255, maxRadius=285)
    # dishes = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, min_dist, param1=100, param2=20, minRadius=255, maxRadius=280)
    
    if dishes is not None:
        dishes = np.uint16(np.around(dishes))

        # dish_in_image = 1
        for i in dishes[0, :]:
            center = (i[0], i[1])
            
            radius = i[2]           
            # new_output_path = output_path[:-4] + '_' + str(dish_number) + '.PNG'
            # new_output_path_labeled = output_path_labeled[:-4] + '_' + str(dish_number) + '_labeled' + '.PNG'
            
            new_output_path = output_foldername + '/' + str(dish_number) + '.PNG'
            new_output_path_labeled = output_foldername_labeled + '/' + str(dish_number) + '_with_points' + '.PNG'

            center_x = center[0]
            center_y = center[1]

            # Rand
            # border = 10
            # radius = radius + border 
                    
            # ROI festlegen
            x_start = center_x - radius
            y_start = center_y - radius
            
            if x_start < 0:
                print('x outside', )
                x_start = 0

            if y_start < 0:
                print('y outside')
                y_start = 0
                     
            new_img = img[y_start:(center_y + radius), x_start:(center_x + radius)]
            new_img_labeled = img_labeled[y_start:(center_y + radius), x_start:(center_x + radius)]

            size_y, size_x = new_img.shape[:2] 
            if size_x != size_y:
                print('no square', dish_number)
                new_img = make_square(new_img)
                new_img_labeled = make_square(new_img_labeled)

            # cv2.imshow("dish Nr." + str(dish_number) , new_img)
            # cv2.waitKey(0)

            # if dish_in_image < 10:
            cv2.imwrite(new_output_path, new_img)
            cv2.imwrite(new_output_path_labeled, new_img_labeled)
            # dish_number -= 1

            img_circles = img_labeled
            # circle center
            # cv2.circle(img_circles, center, 1, (0, 0, 255), 5)
            # circle outline
            # cv2.circle(img_circles, center, radius, (0, 0, 255), 5)
            # border
            # cv2.circle(img_circles, center, radius_b, (0, 255, 0), 5)

            filename_labeled = filename[:2] + 'a' + filename[2:]
            names_list_dishes.append(str(dish_number) + '.PNG; ' + filename_labeled)
            
            size_list_dishes.append(size_x)
            size_list_dishes.append(size_y)
            size_list_dishes.append('next')
            dish_number += 1  
            # dish_in_image +=1

    # cv2.imshow("detected circles", img)
    # cv2.waitKey(0)

    if not os.path.isdir(output_foldername2):
        os.mkdir(output_foldername2)

    output_filename2 = filename[:-4] + '_circles' + '.PNG'
    output_path2 = output_foldername2 + '/' + output_filename2

    # cv2.imwrite(output_path2, img_circles)

with open(output_names_dishes,'w') as fp:
    for elem in names_list_dishes:
        fp.write(elem)
        fp.write('\n')

with open(output_size_dishes,'w') as fp:
    for elem in size_list_dishes:
        fp.write(str(elem))
        fp.write('\n')

