import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime 

path = 'images'
images = []
studentName = []

myList = os.listdir(path)

# Taking out image name

for list_images in myList:
    current_image = cv2.imread(f'{path}/{list_images}')
    images.append(current_image)
    studentName.append(os.path.splitext(list_images)[0])

print("Training the system..")


# function to encode the image HOGG algorithm
def faceEncoding(images):
    encodeList = []
    
    for img in images:
        imgs = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(imgs)[0]
        encodeList.append(encode)
    return encodeList


createdEncoding = faceEncoding(images)
# print("All images encoded successfully !!")
print("System trained !!")

def attendance(name):
    with open('attendance.csv','r+') as f:
        dataList = f.readlines()
        nameList = []
        for line in dataList:
            entry = line.split(',')
            nameList.append(entry[0])
        
        if name not in nameList:
            time_now = datetime.now()
            timeStr = time_now.strftime('%H:%M:%S')
            dateStr = time_now.strftime('%d-%m-%Y')
            
            f.writelines(f'{name},{timeStr},{dateStr}\n')



# using camera

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    face = cv2.resize(frame,(0,0),None,0.25,0.25)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    faceLocation = face_recognition.face_locations(face)

    encodeCurrentFrame = face_recognition.face_encodings(face,faceLocation)
    
    for encodeFace, faceLoc in zip(encodeCurrentFrame,faceLocation):
        matches = face_recognition.compare_faces(createdEncoding,encodeFace)
        faceDist = face_recognition.face_distance(createdEncoding,encodeFace)

        matchIndex = np.argmin(faceDist)
        
        if matches[matchIndex]:
            name = studentName[matchIndex].upper()
            # print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame,(x1,y1),(x2,y2),(31,191,195),2)
            cv2.rectangle(frame,(x1,y2-10),(x2,y2+30),(31,191,195),cv2.FILLED)
            cv2.putText(frame,name,(x1+25,y2+18),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
            attendance(name)
    
    cv2.imshow("camera",frame)
    if cv2.waitKey(10) == 13:
        break
capture.release()
cv2.destroyAllWindows()
            
