# YOLOAugmenter
A python program for augmenting YOLO dataset.

This program not only augments the images, but the anotations too !!!

To change how many augmented images are created for each input image, adjust the following variabels:

line 144: num_processes
line 145: num_threads_per_process

KEEP IN MIND !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

num_processes <= number of cores of your CPU

number of augmented images for one image = num_threads_per_process X num_processes

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

By default the variabels are:

line 144: num_processes = 5
line 145: num_threads_per_process = 2

This will generate 10 augmentations for one image
