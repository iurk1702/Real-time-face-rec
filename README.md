
## 1.) data_gen_base.py - takes an id number as input and captures a photo and stores it in the training folder. This file takes in new users. 

## 2.) trainer.py - Iterates over the training folder, encodes images and stores it in the 'trainer.yml' file.

## 3.) recogniser.py - Opens webcam, checks if the person standing next to it has been seen before(i.e. present in training set). If yes, shows the name in real time. If not, then captures an image and puts it in the training folder to train the model later.

## 4.) global.json - Has only one element - "id". It is a global variable that is accessible by all the files. It is initialized as global  so as to ensure no two or more training images belong to the same id.
