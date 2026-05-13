import cv2 as cv

img=cv.imread('Results.jpg')
cv.imshow('Original',img)
cv.waitKey(0)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imwrite('Gray.jpg',gray)
cv.imshow('Gray',gray)
cv.waitKey(0)
length=img.shape[1]
width=img.shape[0]
circle=cv.circle(img, (int(length / 2), int(width / 2)), 500, 100)
cv.imshow('Circle',circle)
cv.waitKey(0)
line=cv.line(img,(0,0),(int(length / 2), int(width / 2)), (0, 0, 255), 3)
cv.imshow('Line',line)
cv.waitKey(0)

cv.imwrite('result.jpg',line)