import datetime

import cv2
import numpy as np
import pyttsx3
import threading

def voice_alert(engine):
    engine.say("You are in an unsafe zone. Please be cautious.")
    engine.runAndWait()



engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
engine.setProperty('rate', 150)

status_list = [None, None]

capture = cv2.VideoCapture('video3.mp4') # change to integer for webcam or filename for video file

ret, frame1 = capture.read()
ret, frame2 = capture.read()

while capture.isOpened():
    status = 0
    text = "Unoccupied"
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 800:
            status = 1
            continue

        # cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(frame1, [box], 0, (0, 0, 255), 2)
        status_list.append(status)
        text = "Motion detected"

    # cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

    if text == "Motion detected":
        t = threading.Thread(target=voice_alert, args=(engine,))
        t.start()


    cv2.putText(frame1, "Room Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)
    cv2.putText(frame1, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame1.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 255), 1)

    cv2.imshow("video feed", frame1)
    frame1 = frame2
    ret, frame2 = capture.read()

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
capture.release()