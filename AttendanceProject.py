import cv2 as cv
import numpy as np
import face_recognition
import os
from datetime import datetime

# We will create a list that will get the images from our folder automatically
# Then it will generate the encodings automatically
# And then it will try to find it in our Webcam

# To do that make a folder ImagesAttendance with images of the people

# We will ask our program to find this folder and find the number of Images it has
# and import them and find the encodings for them

path = 'ImagesAttendance'
# And then we create a list of all the Images we will import
images = []
# We will use the File Names from the Images Attendance folder to output them when we get the results
classNames = []
# Now we grab the list of images

myList = os.listdir(path)
print(myList)

# Next, We are going to use this names and import the Images one by one
for cl in myList:
    curImg = cv.imread(f"{path}/{cl}")
    images.append(curImg)
    # Appending the name of the image:
    classNames.append(os.path.splitext(cl)[0])

# Printing the class names
print(classNames)


# Now we create a simple function that will create the encodings for us
def findEncodings(images):
    encodeList = []
    # We loop through all the images
    for img in images:
        # First thing is to convert it into RGB
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # Now we find the encodings
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    # Next we return the encoded List
    return encodeList


# Function to mark the attendance
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        # Now we read all the lines in our data
        # If somebody is already arrived we don't want to repeat it
        myDataList = f.readlines()
        nameList = []
        # We want to put all the names we find in this list
        for line in myDataList:
            entry = line.split(',')
            # We want to split the list in name and time
            nameList.append(entry[0])
        # print(nameList)
        # Entry 0 will be the names
        # if name is not present
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


# Calling the function to find encodings
encodeListKnown = findEncodings(images)
print("Encoding Complete")  # so we know it has done this step

# Third step is to find the matches between our encodings
# Let's initialize the webcam

cap = cv.VideoCapture(0)

while True:
    success, img = cap.read()
    # Because we are doing this in real time we want to reduce the size of our image
    # It will help us in speeding the process
    imgS = cv.resize(img, (0, 0), None, 0.25, 0.25)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # To convert into RGB

    # In the webcam we will find multiple faces, and then we will set the location of these faces
    # then we send these locations to our encodings function
    # To find the locations
    facesCurFrame = face_recognition.face_locations(imgS)

    # Next step is to find the encoding of our Webcam
    # along with image we will send location of the faces
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # Moving on to finding the matches
    # We will iterate over all the faces that we have found in our current frame
    # then we will compare all the faces with the encodings we have found
    # We will use both lists facesCurFrame and encodesCurFrame

    # To loop through both the lists simultaneously
    # One by one it will grab the faceLoc from facesCurFrame list and encodings from encodeCurFrame
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        # We compare two lists : known faces and the current encoding through webcam
        # Then we find the distance
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # since we give a list as parameter to face_distance it will return a list as well
        # lowest distance will be our best match
        # print(faceDis)
        # So What we have to do is to find the lowest element in our list of FaceDistance and that will be our best
        # match
        matchIndex = np.argmin(faceDis)
        # We can display a box around the best match with their name
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            # creating a rectangle around the face
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv.FILLED)
            # Rectangle To show the name
            cv.putText(img, name, (x1 + 6, y2 - 6), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # If the face matches we call the markAttendance function
            markAttendance(name)

    # cv.cvtColor(img, cv.COLOR_RGB2BGR)
    cv.imshow('Webcam', img)
    cv.waitKey(1)
