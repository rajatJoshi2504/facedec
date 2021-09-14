import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

path = 'img'
images = []
classnames = []
mylist = os.listdir(path)
# print(mylist)
for cl in mylist:
    curimg = cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classnames.append(os.path.splitext(cl)[0])
# print(classnames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%m/%d/%Y, %H:%M:%S")
            f.writelines(f'\n{name},{dtString}')


encodeListKnow = findEncodings(images)

print(f'Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img5 = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img5 = cv2.cvtColor(img5, cv2.COLOR_BGR2RGB)

    faceCurFrames = face_recognition.face_locations(img5)
    encodeCurFrame = face_recognition.face_encodings(img5, faceCurFrames)

    for encodeface, faceloc in zip(encodeCurFrame, faceCurFrames):
        matches = face_recognition.compare_faces(encodeListKnow, encodeface)
        facedis = face_recognition.face_distance(encodeListKnow, encodeface)
        # print(facedis)
        matchIndex = np.argmin(facedis)

        if matches[matchIndex]:
            name = classnames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)
            # print(f"Face Found : {name}")

    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# # After the loop release the cap object
# vid.release()
# # Destroy all the windows
# cv2.destroyAllWindows()