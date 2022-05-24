import cv2
import os

#cream folderul denumit Fps unde se vor salva pozele
directory = "Fps"
parentDir = "D:/work/IntelACLabs/"
path = os.path.join(parentDir,directory)
os.mkdir(path)

#importam video-ul si mergem pe fiecare cadru, salvandu-l in fisierul creat
vc=cv2.VideoCapture('pg.mp4')
k, frame = vc.read()
i = 0
while k:
    cv2.imwrite('D:/work/IntelACLabs/Fps/pic%d.jpg' %i, frame)
    k, frame = vc.read()
    print('New frame: ', k) #verificam salvarea fiecarui cadru, afisand si numerotarea acestora
    i=i+1

vc.release()
cv2.destroyAllWindows()