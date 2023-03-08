import numpy as np 
import torch 
import cv2 


img = np.ones((400, 400, 3)) / 55
cv2.imshow('frame', img)

print("Hello")

cv2.waitKey(0) == 'q'
cv2.destroyAllWindows()