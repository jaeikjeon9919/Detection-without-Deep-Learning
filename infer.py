import cv2
import joblib
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='handwritten digits detection')
parser.add_argument('--image_path', type=str, action='store', default="sample_image/demo.jpg")
args = parser.parse_args()


clf = joblib.load("digits_cls_normalized.pkl")

dim = (640, 480)

im = cv2.imread(args.image_path)
im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)



im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (7, 7), 0)

im_th = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

im_th = cv2.bitwise_not(im_th)
ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


rects = [cv2.boundingRect(ctr) for ctr in ctrs]

digits = []
for rect in rects:
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    x_ax, y_ax = roi.shape
    if y_ax > 20:
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)

        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (10, 10))
        roi[roi < 25] = 0

        roi = roi.reshape(1,28*28)

        roi = roi/255.0
        roi = (roi - 0.130925351) / 0.30844852402703143

        nbr = clf.predict(np.array(roi))
        digits.append(nbr)
        cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)

np.savetxt("demo_digits.txt", np.array(digits), fmt="%s")
cv2.imwrite("demo_output.png", im)

##

