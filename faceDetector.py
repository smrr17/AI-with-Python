import cv2 as cv
trained_face_data = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv.imread('cp.jfif')
grayscaled_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
detected_faces = trained_face_data.detectMultiScale(img)

# for a single face
# x, y, w, h = detected_faces[0]

# for multiple faces
for (x, y, w, h) in detected_faces:
    cv.rectangle(img, (x, y), (x+w, y+h), (255,255, 202), 2)
print(detected_faces)
cv.imshow('rdj',img)
cv.waitKey()