import cv2
import numpy as np
import os
import json
from pprint import pprint

input_name = '16_filters_batch_size_4_test_1'

test_mask_folder = 'results/' + input_name + '/results_t'
gt_mask_folder = 'data/gt_masks_20'

output_foldername_CSV = 'CSV_statistics/' + input_name

if not os.path.isdir(output_foldername_CSV):
    os.mkdir(output_foldername_CSV)

filename_list = os.listdir(test_mask_folder)
print(filename_list)

statistics_list = []
sum_dice = 0
sum_iou = 0

for image in range(0, len(filename_list)):
    
    filename = str(image) + '.png'
    test_mask_path = test_mask_folder + '/' + filename
    gt_mask_path = gt_mask_folder + '/' + filename

    test_mask = cv2.imread(test_mask_path)
    gt_mask = cv2.imread(gt_mask_path)
    """
    cv2.imshow('Test Mask', test_mask)
    cv2.imshow('Ground Truth Mask', gt_mask)
    cv2.waitKey(0)
    """
    test_mask = test_mask[:,:,0]
    gt_mask = gt_mask[:,:,0]

    # crop
    # test_mask = test_mask[0:256,0:256] 
    # gt_mask = gt_mask[0:256,0:256]

    height, width = test_mask.shape[:2] 
    # print(height)
    # print(width)

    img_size = width

    TP = 0 # richtig detektiert (test = 1, gt = 1)
    FP = 0 # falsch detektiert (test = 1, gt = 0)
    FN = 0 # Objekt nicht erkannt (test = 0, gt = 1)
    TN = 0 # Hintergrund erkannt (test = 0, gt = 0)

    # print(test_mask.shape)
    # print(gt_mask.shape)

    for i in range(0, img_size):
        for j in range(0, img_size):
            test_pixel = test_mask[i,j]
            gt_pixel = gt_mask[i,j]
            # print(test_pixel, gt_pixel)
            if (test_pixel==255 and gt_pixel==255):
                TP += 1
                # print('TP')
            elif (test_pixel==255 and gt_pixel==0):
                FP += 1
                # print('FP')
            elif (test_pixel==0 and gt_pixel==255):
                FN += 1
                # print('FN')
            else:
                TN += 1
                # print('TN')    

    print(filename)
    print('TP: ',TP)
    print('FP: ',FP)
    print('FN: ',FN)
    print('TN: ',TN)

    if (TP+FP) != 0:
        precision = TP/(TP+FP)
    else:
        print('Precision Error!')
        precision = 99999 
    
    if (TP+FN) != 0:
        sensitivity = TP/(TP+FN)
    else:
        print('Sensitivity Error!') 
        sensitivity = 99999  

    if (TP+TN+FP+FN) != 0:
        accuracy = (TP+TN)/(TP+TN+FP+FN)
    else:
        print('Accuracy Error!') 
        accuracy = 99999  

    if (TN+FP) != 0:
        false_positive_rate = FP/(TN+FP)
    else:
        print('FPR Error!')   
        false_positive_rate = 99999

    if ((2*TP)+FP+FN) != 0:
        dice = (2*TP)/((2*TP)+FP+FN)
    else:
        print('Dice Error!')   
        dice = 99999

    if (TP+FP+FN) != 0:
        iou = TP/(TP+FP+FN)
    else:
        print('IoU Error!')
        iou = 99999 

    print('Precision: ', precision)
    print('Sensitivity: ', sensitivity)
    print('Accuracy: ', accuracy)
    print('False positive rate: ', false_positive_rate)
    print('Dice score: ', dice)
    print('Intersection over Union: ', iou)

    stats = (TP, FP, FN, TN, precision, sensitivity, dice, iou)

    # append list
    statistics_list.append(stats)

    sum_dice = sum_dice + stats[6]
    sum_iou = sum_iou + stats[7]

# mean values
mean_dice = sum_dice/len(filename_list)
print('Mean Dice Score: ', mean_dice)
mean_iou = sum_iou/len(filename_list)
print('Mean IoU: ', mean_iou)

# Create CSV Table
CSV_path = output_foldername_CSV + '/' + 'Statistics.CSV'

counter = 0

with open(CSV_path, 'w') as fp:
    fp.write('Image')
    fp.write(';')
    fp.write('TP')
    fp.write(';')
    fp.write('FP')
    fp.write(';')
    fp.write('FN')
    fp.write(';')
    fp.write('TN')
    fp.write(';')
    fp.write('Sensitivity')
    fp.write(';')
    fp.write('Precision')
    fp.write(';')
    fp.write('Dice Score')
    fp.write(';')
    fp.write('IoU')
    fp.write('\n')

    for elem in statistics_list:
        fp.write(str(counter) + '.PNG')
        fp.write(';')
        fp.write(str(elem[0]))
        fp.write(';')
        fp.write(str(elem[1]))
        fp.write(';')
        fp.write(str(elem[2]))
        fp.write(';')
        fp.write(str(elem[3]))
        fp.write(';')
        fp.write(str(elem[4]))
        fp.write(';')
        fp.write(str(elem[5]))
        fp.write(';')
        fp.write(str(elem[6]))
        fp.write(';')
        fp.write(str(elem[7]))
        fp.write('\n')
        counter += 1

    fp.write('Mean Dice Score:')
    fp.write('\n')
    fp.write(str(mean_dice))
    fp.write('\n')
    fp.write('Mean IoU:')
    fp.write('\n')
    fp.write(str(mean_iou))

