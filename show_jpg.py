import cv2
import sys

filename = sys.argv[1]
img = cv2.imread(filename)
cv2.imshow('img',img)

while True:
    if cv2.waitKey(1)==ord('q'):
        break
 
cv2.destroyAllWindows()
