''''
Training Multiple Faces stored on a DataBase:
	==> Each face should have a unique numeric integer ID as 1, 2, 3, etc
	==> LBPH computed model will be saved on trainer/ directory. (if it does not exist, pls create one)
	==> for using PIL, install pillow library with "pip install pillow"
Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition
Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18, Vaarunay Kaushal - https://github.com/iurk1702
'''


import cv2
import numpy as np
import os
#from  global_id_count import global_id_count
import json

count = 0

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


font = cv2.FONT_HERSHEY_SIMPLEX

# initiate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Vaarunay', 'ABC', 'XYZ', 'Joe', 'Shmo']

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:

    ret, img = cam.read()
    #img = cv2.flip(img, -1)  # Flip vertically

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        store_count = 0
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        print(id)

        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:

            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

            #generate dataset for that user
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1

            # Step 1: Read the JSON file
            with open('global.json') as file:
                data = json.load(file)

            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/User." + str(data['id']) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
            #global_id_count = global_id_count + 1
            data['id'] = data['id'] + 1  # Change the value as per your requirements

            # Step 3: Write the modified data back to the JSON file
            with open('global.json', 'w') as file:
                json.dump(data, file)

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 1:  # Take 30 face sample and stop video
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()