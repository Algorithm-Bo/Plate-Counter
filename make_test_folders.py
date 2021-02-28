import cv2 
import os

# folders
folder_name = 'data/images_test_examples'
output_foldername = 'test_data_examples'

num_test_images = 15
start_num = 0

filename_list = os.listdir(folder_name)
print(filename_list)

# create output folder
if not os.path.isdir(output_foldername):
    os.mkdir(output_foldername)

for n in range(0, num_test_images):
    filename = str(n) + '.PNG'

    path = folder_name + '/' + filename
        
    img = cv2.imread(path)
    
    # cv2.imshow('Test', img)
    # cv2.waitKey(0)
    print(start_num)

    new_folder = output_foldername + '/' +str(n)

    if not os.path.isdir(new_folder):
        os.mkdir(new_folder)
 
    output_path_image = new_folder + '/' + str(n) + '.png'
      
    print(output_path_image)
    
    cv2.imwrite(output_path_image, img)
         
    start_num += 1
    
