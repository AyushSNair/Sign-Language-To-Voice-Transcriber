import cv2 as cv
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math
import time

# Initialize the HandDetector with maximum one hand detection
detector = HandDetector(maxHands=2)
offset = 20
imgSize = 300
counter = 0

# Folder to save captured images
folder = r'C:\Users\Aryush\Desktop\SignToVoice\Signenv\See you later'

# Initialize the video capture
capture = cv.VideoCapture(0)

if not capture.isOpened():
    print("Couldn't detect a camera")
    exit()

while True:
    success, frame = capture.read()
    hands, frame = detector.findHands(frame)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the hand region from the frame
        imgCrop = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Display the cropped hand image and the white background image
        cv.imshow('ImageCrop', imgCrop)
        cv.imshow('ImageWhite', imgWhite)
    
    # Display the original frame with hand detection
    cv.imshow('Image', frame)

    # Check for key press
    key = cv.waitKey(1)
    if key & 0xFF == ord('q') and 'imgWhite' in locals():
        counter += 1
        cv.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Saved Image {counter}")

    if key & 0xFF == ord('e'):
        break

# Release the capture and destroy all OpenCV windows
capture.release()
cv.destroyAllWindows()
