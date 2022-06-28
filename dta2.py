import os

import cv2
import numpy as np
from numpy import asarray, zeros
import mediapipe as mp

blur = 7
k = 7
linesize = 3
max_a = 100
location = "girl_data/girl/train_img/"
images = []
posture = []

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
res = []
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for p in os.listdir(location):
        print(location + p)
        if p.split(".")[-1] == "jpg":
            im = cv2.imread(location + p)
            imc = im.copy()
            rg = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            results = holistic.process(rg)
            mp_drawing.draw_landmarks(im, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)

            mp_drawing.draw_landmarks(im, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(im, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            mp_drawing.draw_landmarks(im, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            re = asarray([results.face_landmarks, results.right_hand_landmarks, results.left_hand_landmarks,
                         results.pose_landmarks])k

            tra = cv2.cvtColor(imc, cv2.COLOR_BGR2BGRA)
            transparent_mask = zeros(tra.shape, np.uint8)

            h, w, c = tra.shape
            h = cv2.inRange(tra, tra[0][0], tra[h - 1][w - 1])
            h = cv2.bitwise_not(h)

            erode_k = 7
            h = cv2.erode(
                h, (erode_k, erode_k)
            )

            # h = cv2.GaussianBlur(h, sigmaX=blur, dst=blur, ksize=(k, k))
            c = cv2.bitwise_and(tra, tra, mask=h)

            images.append(c)
            res.append(re)

np.savez_compressed("girl_images.npz", asarray(images))
np.savez_compressed("girl_res.npz", asarray(res))
"""n=np.load("images.npz")["arr_0"]
cv2.imshow("woeks",n[0])
cv2.waitKey(-1)"""

# h = cv2.bitwise_and(im, im, mask=cv2.bitwise_not(h))Windows()
"""       
l, _ = cv2.findContours(h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
l = [i for i in l if cv2.contourArea(i) >= max_a]
print(l)
print(len(l))
cv2.drawContours(im, l, -1, (0, 87, 67))"""
