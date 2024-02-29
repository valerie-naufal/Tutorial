import cv2

img = cv2.imread('samples/image3.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

#cv2.imshow('Gray Image', gray)
cv2.imshow('Binary Image', thresh)
cv2.imwrite('output/threshold4.jpg',thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
