import cv2
import sys
import json
import numpy as np
from keras.models import model_from_json


emotions = ['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral']
cascPath = sys.argv[1]

faceCascade = cv2.CascadeClassifier(cascPath)
noseCascade = cv2.CascadeClassifier(cascPath)

json_file = open('model/model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights('model/model.h5')

def overlay_emojiface(probs):
    if max(probs) > 0.8:
        emotion = emotions[np.argmax(probs)]
        return 'emoji/{}-{}.png'.format(emotion, emotion)
    else:
        index1, index2 = np.argsort(probs)[::-1][:2]
        emotion1 = emotions[index1]
        emotion2 = emotions[index2]
        return 'emoji/{}-{}.png'.format(emotion1, emotion2)

def predict_emotion(face_image_gray): 
    resized_img = cv2.resize(face_image_gray, (48,48), interpolation = cv2.INTER_AREA)
    image = resized_img.reshape(1, 1, 48, 48)
    list_of_list = model.predict(image, batch_size=1, verbose=1)
    angry, fear, happy, sad, surprise, neutral = [prob for lst in list_of_list for prob in lst]
    return [angry, fear, happy, sad, surprise, neutral]

video_capture = cv2.VideoCapture(0)
while True:
    # Capture Frame
    ret, frame = video_capture.read()

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY,1)


    faces = faceCascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
	flags=0
    )

    for (x, y, w, h) in faces:

        face_image_gray = img_gray[y:y+h, x:x+w]
        filename = overlay_emojiface(predict_emotion(face_image_gray))

        print (filename)
        emoji = cv2.imread(filename,-1)
        # emoji = (emoji/256).astype('uint8')
        try:
            emoji.shape[2]
        except:
            emoji = emoji.reshape(emoji.shape[0], emoji.shape[1], 1)
        # print emoji.dtype
        # print emoji.shape
        orig_mask = emoji[:,:,3]
        # print orig_mask.shape
        ret1, orig_mask = cv2.threshold(orig_mask, 10, 255, cv2.THRESH_BINARY)
        orig_mask_inv = cv2.bitwise_not(orig_mask)
        emoji = emoji[:,:,0:3]
        origMustacheHeight, origMustacheWidth = emoji.shape[:2]

        roi_gray = img_gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect hidung
        nose = noseCascade.detectMultiScale(roi_gray)

        for (nx,ny,nw,nh) in nose:
            mustacheWidth =  20 * nw
            mustacheHeight = mustacheWidth * origMustacheHeight / origMustacheWidth

            x1 = nx - (mustacheWidth/4)
            x2 = nx + nw + (mustacheWidth/4)
            y1 = ny + nh - (mustacheHeight/2)
            y2 = ny + nh + (mustacheHeight/2)

            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > w:
                x2 = w
            if y2 > h:
                y2 = h

            mustacheWidth = (x2 - x1)
            mustacheHeight = (y2 - y1)

            mustache = cv2.resize(emoji, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
            mask = cv2.resize(orig_mask, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
            mask_inv = cv2.resize(orig_mask_inv, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)

            roi = roi_color[y1:y2, x1:x2]
            roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
            roi_fg = cv2.bitwise_and(mustache,mustache,mask = mask)
            dst = cv2.add(roi_bg,roi_fg)
            roi_color[y1:y2, x1:x2] = dst
            break

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
video_capture.release()
cv2.destroyAllWindows()
