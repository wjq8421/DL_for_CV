import numpy as np
import cv2

labels = ['dog', 'cat']
np.random.seed(1)

W = np.random.randn(2, 3072)
b = np.random.randn(2)

orig = cv2.imread('E:\DL_datasets\cats_dogs\cat\cat.10000.jpg')
image = cv2.resize(orig, (32, 32)).flatten()

scores = W.dot(image) + b
for (label, score) in zip(labels, scores):
    print('[INFO] {}: {:.2f}'.format(label, score))

txt = 'Label: {}'.format(labels[np.argmax(scores)])
cv2.putText(orig, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.imshow("Image", orig)
cv2.waitKey(0)
