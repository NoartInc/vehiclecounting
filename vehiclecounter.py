import cv2
import numpy as np
from time import sleep

large_min = 80
alt_min = 80
offset = 6

pos_line = 550

delay = 60

detec = []
car = 0


def capt_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


cap = cv2.VideoCapture("E:/Machine Vision/PythonProject/Vehicle Counter/jalanraya.mp4")
subtractor = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame1 = cap.read()
    temp = float(1/delay)
    sleep(temp)
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    img_sub = subtractor.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilat1 = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilat1 = cv2.morphologyEx(dilat1, cv2.MORPH_CLOSE, kernel)

    contour, h = cv2.findContours(dilat1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame1, (25, pos_line), (1200, pos_line), (255, 127, 0), 3)
    for(i, c) in enumerate(contour):
        (x, y, w, h) = cv2.boundingRect(c)
        valid_contour = (w >= large_min) and (h >= alt_min)
        if not valid_contour:
            continue

        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        center = capt_center(x, y, w, h)
        detec.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

        for (x, y) in detec:
            if y < (pos_line+offset) and y > (pos_line-offset):
                car += 1
                cv2.line(frame1, (25, pos_line), (1200, pos_line), (0, 127, 255), 3)
                detec.remove((x, y))
                print("Kendaraan terdeteksi : "+str(car))

    cv2.putText(frame1, "Kendaraan: " + str(car), (450, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Video Original", frame1)
    cv2.imshow("Detector", dilat1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
