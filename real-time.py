import cv2
import sys
import json
import time
import numpy as np
from keras.models import model_from_json


emotion_labels = ['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral']
cascPath = sys.argv[1]

faceCascade = cv2.CascadeClassifier(cascPath)

json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights('model.h5')

def predict_emotion(face_image_gray): 
    resized_img = cv2.resize(face_image_gray, (48,48), interpolation = cv2.INTER_AREA)
    image = resized_img.reshape(1, 1, 48, 48)
    list_of_list = model.predict(image, batch_size=1, verbose=1)
    angry, fear, happy, sad, surprise, neutral = [prob for lst in list_of_list for prob in lst]
    return [angry, fear, happy, sad, surprise, neutral]

video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY,1)


    faces = faceCascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
	flags = 0
    )

    emotions = []
    for (x, y, w, h) in faces:

        face_image_gray = img_gray[y:y+h, x:x+w]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        angry, fear, happy, sad, surprise, neutral = predict_emotion(face_image_gray)
        with open('emotion.txt', 'a') as f:
            f.write('{},{},{},{},{},{},{}\n'.format(time.time(), angry, fear, happy, sad, surprise, neutral))

    cv2.imshow('Video', frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
