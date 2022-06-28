r"""import cv2
from numpy import asarray

blur = 1
k = 7
linesize = 3
max_a = 1400
im = cv2.imread(r"C:\Users\Admin\Downloads\Compressed\archive\girl_data\girl\train_img\1_1.jpg")
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

edgek = 40
cv2.imshow("im", im)
k = cv2.waitKey(8000)
edges = cv2.Canny(gray, edgek, edgek)
cv2.imshow("ed1", edges)
k = cv2.waitKey(8000)
edges = cv2.dilate(edges, (k, k))
cv2.imshow("ed", edges)
k = cv2.waitKey(8000)
l, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(l))
l = [i for i in l if cv2.contourArea(i) >= max_a]
print(len(l))
i
cv2.drawContours(im, l, -1, (0, 90, 89), 4)

cv2.imshow("l", im)
for i in l:
    cv2.fillConvexPoly(im, asarray(i), (0, 90, 8))

k = cv2.waitKey(8000)
cv2.imshow("li", im)
k = cv2.waitKey(8000)
if k == 27:
    cv2.destroyAllWindows()
"""
import cv2
import numpy as np
import mediapipe as mp
from numpy import asarray, zeros

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

blur = 1
k = 7
linesize = 3
max_a = 1400
im = cv2.imread(r"C:\Users\Admin\Downloads\Compressed\archive\girl_data\girl\train_img\1_1.jpg")
rg = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    results = holistic.process(im)
    print(results)
    mp_drawing.draw_landmarks(im, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)

    mp_drawing.draw_landmarks(im, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(im, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    mp_drawing.draw_landmarks(im, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    cv2.imshow("hh", im)
    re = asarray([results.face_landmarks, results.right_hand_landmarks, results.left_hand_landmarks,
                 results.pose_landmarks])
    print(re[])
    print(re.shape)

k = cv2.waitKey(-1)
print(k)
if k == 27:
    cv2.destroyAllWindows()
import redis
redis.from_url