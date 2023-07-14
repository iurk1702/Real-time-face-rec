''''
Training Multiple Faces stored on a DataBase:
	==> Each face should have a unique numeric integer ID as 1, 2, 3, etc
	==> LBPH computed model will be saved on trainer/ directory. (if it does not exist, pls create one)
	==> for using PIL, install pillow library with "pip install pillow"
Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition
Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18, Vaarunay Kaushal - https://github.com/iurk1702
'''


import cv2
import os
#from  global_id_count import global_id_count
#print(global_id_count)
import json


cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

while(True):

    ret, img = cam.read()
    #img = cv2.flip(img, -1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1
        #access global id index value stored in json file
        # Step 1: Read the JSON file
        with open('global.json') as file:
            data = json.load(file)

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        # Step 2: Modify the value
        data['id'] =  data['id'] + 1 # Change the value as per your requirements

        # Step 3: Write the modified data back to the JSON file
        with open('global.json', 'w') as file:
            json.dump(data, file)
        #global_id_count = global_id_count + 1
        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 1: # Take 30 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()