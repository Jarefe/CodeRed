from cv2 import VideoCapture
import argparse
import datetime
import imutils
import time
import cv2

# construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to video file")
ap.add_argument("-a","--min-area", type=int, default=700, help="minimum area size")
args = vars(ap.parse_args())

# if video argument is none, then read from webcam
if args.get("video", None) is None:
    vs = VideoCapture(1) # change integer to choose which webcam (set to 0 if only 1 webcam)
    if not (vs.isOpened()):
        print("Camera not found")
    time.sleep(2.0)

# else read from video file
else:
    vs = cv2.VideoCapture(args["video"])

# initialize first frame in video stream
firstFrame = None


# Basic Motion Tracking
# loop over frames of video
while True:
    # grab current frame and initialize occupied/unoccupied text
    frame = vs.read()
    frame = frame if args.get("video", None) is None else frame[1]
    text = "Unoccupied"

    #if frame could not be grabbed, that is end of video
    if frame is None:
        break

    # resize frame, convert to grayscale, then blur
    _, frame = vs.read()
    (w, h, c) = frame.shape
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue

    # compute absolute difference between current and first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # dilate thresholded image to fil in holes, then find contours on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over contours
    for c in cnts:
        # if contour is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]:
            continue

        # compute bounding box for contour, draw on frame, and update text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"

    cv2.putText(frame, "Room Status: {}".format(text), (10,20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 255), 1)

    #show frame and record if user presses a key
    cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(1) & 0xFF

    # if 'q' is pressed, break loop
    if key == ord("q"):
        break

# cleanup camera and close windows
vs.release() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()