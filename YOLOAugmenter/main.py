import cv2
from matplotlib import pyplot as plt
import albumentations as A
import multiprocessing
import threading
import random
import time
import gc
import os
from os import listdir
import numpy as np
imageLST = []
labelsLST = []
BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White
img_dir = "images"
lable_dir= "labels"
output_label_dir = "output/labels"
output_image_dir = "output/images"
print("LIST OF IMAGES")
for image in os.listdir(img_dir):
 
    # check if the image ends with png
     if (image.endswith(".jpg")):
         imageLST.append(image)
print(imageLST)
print(f'NUMBER OF IMAGES :{len(imageLST)}')
print("LIST OF LABELS")
for lable in os.listdir(lable_dir):
 
    # check if the image ends with png
    if (lable.endswith(".txt")):
        labelsLST.append(lable)
print(labelsLST)
print(f'NUMBER OF LABELS :{len(labelsLST)}')
labelsLST.sort()
imageLST.sort()
def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image

    dh, dw, _ = img.shape
    
    data = str(bboxes)
    #print(data)
    #print(bboxes)
    my_new_string = data.replace("]", "").replace("[","").replace("(","").replace(")","")
    newList=my_new_string.split(",")
    #print(newList)
    

    # Split string to float
    x = float(newList[0])
    y = float(newList[1])
    w = float(newList[2])
    h = float(newList[3])
    # Taken from https://github.com/pjreddie/darknet/blob/810d7f797bdb2f021dbe65d2524c2ff6b8ab5c8b/src/image.c#L283-L291
    # via https://stackoverflow.com/questions/44544471/how-to-get-the-coordinates-of-the-bounding-box-in-yolo-object-detection#comment102178409_44592380
    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)
    
    if l < 0:
        l = 0
    if r > dw - 1:
        r = dw - 1
    if t < 0:
        t = 0
    if b > dh - 1:
        b = dh - 1

    cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 20)
    plt.imshow(img)
    plt.show()

i = 1
b = 0
def process_images(images, output, no):
    global i,b,splitListWithNoNewLine
    thread_results = []
    for idx, image in enumerate(images):
        random.seed(i)
        i+=1
        bbox = open(f'labels/{labelsLST[b]}', "r")
        bboxread = bbox.read()
        splitList = bboxread.split(' ')
        splitListWithNoNewLine=list(map(str.strip,splitList))
        #print (splitListWithNoNewLine)
        bboxes = [[float(splitListWithNoNewLine[1]), float(splitListWithNoNewLine[2]), float(splitListWithNoNewLine[3]), float(splitListWithNoNewLine[4])]]
        category_ids = [int(splitListWithNoNewLine[0])]
        category_id_to_name = {int(splitListWithNoNewLine[0]): 'clas 0'}

        transform = A.Compose(
            [
               A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=35, val_shift_limit=15, p=0.3),
            A.RandomGamma(p=0.3),
            A.Blur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.RandomRain(p=0.3),
            A.MotionBlur(p=0.3),
            A.MultiplicativeNoise(p=0.2),
            A.RGBShift(p=0.2,r_shift_limit=15,g_shift_limit=15,b_shift_limit=15),
            A.Downscale(interpolation=cv2.INTER_LINEAR,p=0.2),
            A.ISONoise(p=0.4),
            A.OpticalDistortion(p=0.3, border_mode=cv2.BORDER_CONSTANT,interpolation=cv2.INTER_LINEAR),
            A.ImageCompression(p=0.2),
            A.OneOf([
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT,interpolation=cv2.INTER_LINEAR,
                                   p=0.5),
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, border_mode=cv2.BORDER_CONSTANT,interpolation=cv2.INTER_LINEAR,
                                   p=0.5),
            ], p=0.5),
        ],
            bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']),
        )

        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        thread_results.append((transformed['image'], transformed['bboxes'], category_ids, category_id_to_name))
    output[no] = thread_results
a = 0
c = 1
filename = 0
def save(augmented_image,label,output_image_dir,output_label_dir,splitListWithNoNewLine):  
    global filename
    augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'{output_image_dir}/{filename}.jpg', augmented_image)
    f = open(f'{output_label_dir}/{filename}.txt','w')
    data = str(label)
    #print(data)
    #print(label)
    my_new_string = data.replace("]", "").replace("[","").replace("(","").replace(")","")
    newList=my_new_string.split(",")
    #print(newList)
    newList.insert(0,splitListWithNoNewLine[0])
    toWrite=f'{newList[0]} {newList[1]} {newList[2]} {newList[3]} {newList[4]}'
    f.write(toWrite)
    f.close()
    filename +=1

def main():
    global a, DirPath, Files,b,c,filename, output_label_dir,output_image_dir,splitListWithNoNewLine
    start_time = time.perf_counter()
    num_processes = 5  # Adjust the number of processes based on your CPU cores
    num_threads_per_process = 2  # Adjust the number of threads per process based on your requirements
    
    manager = multiprocessing.Manager()
    output = manager.dict()
    processes = []
    d = 1
    while c <= len(imageLST):
        c+=1
        images = [cv2.imread(f'images/{imageLST[a]}') for _ in range(num_processes * num_threads_per_process)]
        images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
        for i in range(num_processes):
            start_idx = i * num_threads_per_process
            end_idx = start_idx + num_threads_per_process
            process_images_thread = threading.Thread(target=process_images, args=(images[start_idx:end_idx], output, i))
            process_images_thread.start()
            processes.append(process_images_thread)
            

        for process in processes:
            process.join()

        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        print("Elapsed time: ", elapsed_time)
        gc.collect()
        a +=1
        b +=1

        results = []
        for i in range(num_processes):
            results.extend(output[i])

        results = []
        for i in range(num_processes):
            results.extend(output[i])

        # Visualize the augmented images and bounding boxes
        for idx, (augmented_image, bboxes, category_ids, category_id_to_name) in enumerate(results):
            print(f"Image {idx + 1}:")
            for bbox, category_id in zip(bboxes, category_ids):
                class_name = category_id_to_name[category_id]
                x_min, y_min, w, h = bbox
                print(f"Class: {class_name}, BBox: {x_min:.4f}, {y_min:.4f}, {w:.4f}, {h:.4f}")
            print()  # Newline for separation
            save(augmented_image,bboxes,output_image_dir,output_label_dir,splitListWithNoNewLine)
            print(f'{d}IMAGES WRITTEN') 
            d+=1
            #visualize(augmented_image, bboxes, category_ids, category_id_to_name)
            
            if idx + 1 >= num_processes * num_threads_per_process:
                gc.collect()
                break
print(f'DONE')            
if __name__ == "__main__":
    main()