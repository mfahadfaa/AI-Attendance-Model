import cv2
import numpy as np
import face_recognition 
import os
from datetime import datetime
# 1. dataset folder se images aur unke naam load karna
path ="dataset"
images=[]
classnames=[]
mylist=os.listdir(path)
print(f"dataset mein ye files mili hain:{mylist}")
for cl in mylist:
    curimg = cv2.imread(f"{path}/{cl}")
    if curimg is not None:
        images.append(curimg)
        classnames.append(os.path.splitext(cl)[0])
    else:
        print(f"warning: image {cl}koload nahikiya ja saka.")
        print(f"loaded names: {classnames}")
        
        # 2. faces ko mathemical encodings mein badalne ka function def findencodings(images):
    def findencodings(images):    
        
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            # phela face jo imge mein mile uske encoding lena
            encodes = face_recognition.face_encodings(img)
            if len(encodes) > 0:
                encodeList.append(encodes[0])
            else:
                print("warning: kisi ek photo mein face nahi mil raha!")       
print("ai encoding shuru ho rahi hain, thod wait karein...")
encodeListKnown = findencodings(images)
print ("encoding complete! system ready .")
# 3 . Attendence lagane ka function (attendence folder ke ander csv file banegi)
def markAttendance (name):
              os.makedirs("attendence", exist_ok =True)# agar attendance folder nhi to ban jaye
              file_path = "attendence/Attendence.csv"
              # file check karna ya nhi banana
              if not os.path.exists(file_path):
                  with open(file_path, "w") as f:
                      mydatalist = f.readlines()
                      nameList = []
                      for line in mydatalist:
                          entry = line.split(",")
                          nameList.append(entry[0])
                          if name  not in nameList:
                              now=datetime.now()
                              dtString = now.strftime("%H:%M:%S")
                              dtString = now.strftime("%d:%m:%y")
                              f.writelines(f"{name},{dtString},{dtString},\n")
                              print(f"Attendance Marked for: {name}")
                              
                              # 4. webcam start karein
                              cap = cv2.videoCapture(0)
                              
                              while True:
                                  success, img = cap.read()
                                  if not success:
                                      print("webcam accessnahi ho raha!")
                                      break
                                  
                                  # processing speed barhane ke liye frame ko 1/4th size ka karna
                                  imgs = cv2.resize(img,(0,0),None,0.25,0.25)
                                  img = cv2.cvtColor(imgs,cv2.Color_BGR2RGB)
                                  
                                  facesCurFrame = face_recognition.face_locations(imgs)
                                  encodesCurFrame = face_recognition.face_encodings(imgs,facesCurFrame)
                                  for encodeFace, faceloc in zip(encodesCurFrame, facesCurFrame):
                                    matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
                                    faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
                                  
                                  if len(faceDis) > 0:
                                      matchIndex = np.argmin(faceDis)
                                      
                                      if matches[matchIndex]:
                                          name = classnames[matchIndex].upper()
                                          
                                          
                                          # Face par box aur naam draw karna
                                          y1,x2,y2,x1 = faceloc
                                          y1,x2,y2,x1 = y1 * 4,x2 * 4,y2 * 4,x1 * 4
                                          cv2.rectangle(img,(x1, y1),(x2, y2),(0, 255, 0),2)
                                          cv2.rectangle(img, (x1, y2 - 35),(x2, y2),(0, 255, 0),cv2.FILLED)
                                          cv2.putText(img,name,(x1 + 6,y2 - 6),cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                                          
                                          # Attendance register karna 
                                          markAttendance(name)
                                          
                                          cv2.imshow("AI Attendance System - Live",img)
                                          
                                          #Band karne ke liye keyboard per "q" dabayein
                                          if cv2.waitKey (1) & 0xFF == ord("q"):
                                              break
                                          
                                          cap.release()
                                          cv2.destroyAllWindows()
                                  
                          
            