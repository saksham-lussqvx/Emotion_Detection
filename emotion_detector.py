# Made by @Saksham Solanki

# How to use?

# First of all if you want detection on a photo then place the 
# photo in the same folder as the code, after that type 1.
# After that select the type of emotion you want to recognize 
# or just press enter for all after that enter the full name of the 
# file including the format of the photo such as -> trial.jpg or trial.png
# After that it'll load te desired result. For vido format the proccedure
# is almost same just you don't have to enter the name of any file, it'll load
# the camera by itself. 
#----------------------------------------------- Import Libraries -----------------------------------------------# 
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
#----------------------------------------------------------------------------------------------------------------#
#----------------------------------------------- Initialize Model -----------------------------------------------#
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
#----------------------------------------------------------------------------------------------------------------#

#_____________________________________________ Photo/Video Detectors ____________________________________________#
class detectors:
#------------------------------------------------ Photo Detector ------------------------------------------------#
    def on_Photo(photo,key):
        model.load_weights('model.h5') # load the pretrained model
        cv2.ocl.setUseOpenCL(False) # prevents openCL usage and unnecessary logging messages
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"} # dictionary which assigns each label an emotion (alphabetical order)
        
        # load the photo and remember photo should be in the same folder as code
        cap = cv2.imread(photo)
        while True:
            frame = cap
            facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # load the default face detector
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert the photo to grayscale
            faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5) # detect faces in it

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2) # reshape the img
                roi_gray = gray[y:y + h, x:x + w] # convert back to color
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0) # crop the img 
                prediction = model.predict(cropped_img) # Predict the type of emotion
                maxindex = int(np.argmax(prediction)) # get the index of type to place label
                if key == "":
                    cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                elif (maxindex) == (int(key)-1):
                    cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
#----------------------------------------------------------------------------------------------------------------#
#------------------------------------------------ Video Detector ------------------------------------------------#
    def on_video(key):
        # emotions will be displayed on your face from the webcam feed
        model.load_weights('model.h5')

        # prevents openCL usage and unnecessary logging messages
        cv2.ocl.setUseOpenCL(False)

        # dictionary which assigns each label an emotion (alphabetical order)
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

        # start the webcam feed
        cap = cv2.VideoCapture(0)
        while True:
            # Find haar cascade to draw bounding box around face
            ret, frame = cap.read()
            if not ret:
                break
            facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                if key == "":
                    cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                elif (maxindex) == (int(key)-1):
                    cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
#----------------------------------------------------------------------------------------------------------------#
#________________________________________________________________________________________________________________#

#-------------------------------------------------- Main driver -------------------------------------------------#
def main():

    print("Please select the format:")
    print("1) Photo")
    print("2) Video")
    Type = input("1/2: ")

    if Type == "1":
        print("Select any of the emotion or press enter for all:")
        print("""
        1) Angry
        2) Disgusted
        3) Fearful
        4) Happy
        5) Neutral
        6) Sad
        7) Surprised""")
        key = input(": ")
        photo = input("Enter photo name: ")
        detectors.on_Photo(photo, key)

    if Type == "2":
        print("Select any of the emotion or press enter for all:")
        print("""
        1) Angry
        2) Disgusted
        3) Fearful
        4) Happy
        5) Neutral
        6) Sad
        7) Surprised""")
        key = input(": ")
        detectors.on_video(key)
#----------------------------------------------------------------------------------------------------------------#
if __name__ == '__main__':
    main()