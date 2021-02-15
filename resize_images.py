import cv2
import os

# input folder
folder_name ='data/images_test_staphylococcus_512x512_20'

new_x = 256
new_y = 256

output_foldername = 'data/images_test_staphylococcus_256x256_20'

if not os.path.isdir(output_foldername):
    os.mkdir(output_foldername)

filename_list = os.listdir(folder_name)
print(filename_list)

for image in range(0, len(filename_list)):
    # filename = filename_list[image]
    filename = str(image) + '.PNG'

    path = folder_name + '/' + filename

    img = cv2.imread(path)
    
    # output_filename = filename[:-4] + '_' + str(new_x) + 'x' + str(new_y) + '.PNG'
    # output_filename = str(i) + '_binary' + '.PNG'
    # output_filename = filename
    
    # output_filename = filename
    output_filename = str(image) + '.PNG'

    output_path = output_foldername + '/' + output_filename
    print(output_path)

    new_img = cv2.resize(img, (new_x, new_y), interpolation = cv2.INTER_AREA)
    # INTER_AREA for shrinking
    
    cv2.imwrite(output_path, new_img)

        