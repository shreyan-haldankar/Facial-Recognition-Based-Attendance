import cv2 as cv
import numpy as np
import face_recognition

# Loading The Images into the Face_recognition Module
imgElon = face_recognition.load_image_file('ImagesBasics/Elon Musk.jpg')
imgElon = cv.cvtColor(imgElon, cv.COLOR_BGR2RGB)

# Image Test
# imgTest = face_recognition.load_image_file('ImagesBasics/Elon Test.jpg')
# What if we change the test image to bill gates
# It detected the Face properly, and then it’s telling us that the encodings do not match
imgTest = face_recognition.load_image_file('ImagesBasics/Bill Gates.jpg')
imgTest = cv.cvtColor(imgTest, cv.COLOR_BGR2RGB)

# Finding the faces in the image and then finding their encodings
faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]

# To check where we have detected the faces
# We have to give x1, y1 and x2, y2 values in our rectangle
cv.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)
# cv.rectangle(image, point1, point2, color, thickness)
# print(faceLoc)  # (168, 425, 297, 296)
# It prints out 4 different values (top, right, bottom, left)


# Same thing can be done for the test image
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

# Final Step
# We are getting the encodings, and then we are comparing the encodings with the test image
# 128 measurements of both the faces
# Linear SVM to find out whether the images match or not
results = face_recognition.compare_faces([encodeElon], encodeTest)
# We are giving a list of known faces
# Then we are comparing it with test encodings

# Sometimes there can be a lot of images, and there can be similarities between them
#  And then you want to find out how similar these images are
# To do that we will find the distance
faceDis = face_recognition.face_distance([encodeElon], encodeTest)
# The lower the distance the better the match is
# So printing the distance
print(results, faceDis)
# We compare if they both are the same images or not
#  There's a clear difference when their faces match and don't match

# So the final step we can do here is just to display this on the final result image
cv.putText(imgTest, f"{results} {round(faceDis[0], 2)}", (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
# cv.putText(image, Text, origin, font, scale, color, thickness)

cv.imshow('Elon Musk', imgElon)
cv.imshow('Elon Test', imgTest)

cv.waitKey(0)
