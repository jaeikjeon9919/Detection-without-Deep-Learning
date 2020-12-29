import cv2
import joblib
from skimage.feature import hog
import numpy as np
import matplotlib.pyplot as plt

clf = joblib.load("digits_cls_normalized.pkl")
# pca = joblib.load("pca_feature_800.pkl")

dim = (640, 480)

im = cv2.imread("demo_5.jpg")
im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)

# plt.imshow(im)
# plt.show()

im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (7, 7), 0)


# ret, im_th = cv2.threshold(im_gray, 150, 255, cv2.THRESH_BINARY_INV)
im_th = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
# ret, im_th = cv2.threshold(im_th, 200, 255, cv2.THRESH_BINARY_INV)


plt.imshow(im_th)
plt.show()



im_th = cv2.bitwise_not(im_th)
ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


rects = [cv2.boundingRect(ctr) for ctr in ctrs]

print(rects)

for rect in rects:
    # Draw the rectangles
    # cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    x_ax, y_ax = roi.shape
    # print(rect)
    if y_ax > 20:
        # print(x_ax, y_ax)
        # desired_size = 28
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)

        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (10, 10))
        # kernel = np.ones((6, 6), np.uint8)
        # roi = cv2.medianBlur(roi, 3)
        # roi = cv2.inRange(roi, 10, 255)
        # print(roi)
        roi[roi < 25] = 0
        # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        # roi = cv2.filter2D(roi, -1, kernel)
        plt.imshow(roi)
        plt.show()

        roi = roi.reshape(1,28*28)
        # roi_pca = pca.transform(roi)
        # print(roi_pca.shape)
        # Calculate the HOG features
        #
        # blur = cv2.GaussianBlur(roi, (3, 3), 0)
        # print(blur.shape)
        # # threshhold gray region to white (255,255, 255) and sets the rest to black(0,0,0)
        # mask = cv2.inRange(blur, (0, 0, 0), (150, 150, 150))

        # roi = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
        # print(roi_hog_fd.shape)
        # print(np.array(roi).shape)
        print(roi.shape)
        roi = roi/255.0
        roi = (roi - 0.130925351)/ 0.30844852402703143
        # roi_concat = np.concatenate((roi_pca, roi), 1)
        nbr = clf.predict(np.array(roi))
        print(nbr)
        cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)

print(im.shape)
plt.imshow(im)
plt.show()

cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()

##

