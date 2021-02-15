import cv2 
import os

# folders for original and augmented data
# images:

# folder_name1 = 'data/augmented_images_train_easy/img'
folder_name1 = 'data/augmented_images_train_easy_256_special/img'

# masks:
# folder_name2 = 'data/augmented_images_train_easy/masks'
folder_name2 = 'data/augmented_images_train_easy_256_special/masks'

# output foldername
output_foldername = 'train_data_easy_256_special_630'

filename_list = os.listdir(folder_name1)
print(filename_list)

# Ausgabeordner erstellen, falls nicht vorhanden
if not os.path.isdir(output_foldername):
    os.mkdir(output_foldername)

# Dateien nacheinander aufrufen
n = 0

for image1 in range(0, len(filename_list)):
    # filename = filename_list[image]
    filename = str(n) + '.PNG'

    path1 = folder_name1 + '/' + filename
    path2 = folder_name2 + '/' + filename
    
    img = cv2.imread(path1)
    mask = cv2.imread(path2)

    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('Original', img)
    # cv2.imshow('Mask', mask)
    # cv2.waitKey(0)

    os.chdir(output_foldername)

    new_folder = str(n)

    if not os.path.isdir(new_folder):
        os.mkdir(new_folder)

    os.chdir(new_folder)
    
    os.mkdir('image')
    os.mkdir('mask')

    # output_path_image = output_foldername + '/' + new_folder + '/image/' + filename
    # output_path_mask = output_foldername + '/' + new_folder + '/mask/' + filename

    output_path_image = 'image/' + filename
    # output_path_mask = 'mask/' + filename[:-4] + '.TIF'
    output_path_mask = 'mask/' + filename
    print(output_path_image)
    print(output_path_mask)
    
    cv2.imwrite(output_path_image, img)
    cv2.imwrite(output_path_mask, mask_gray)

    os.chdir('..')
    os.chdir('..')
    
    n += 1

