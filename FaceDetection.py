import numpy as np
import cv2

def nothing(x):
    pass

cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking",0,255,nothing)
cv2.createTrackbar("LS", "Tracking",0,255,nothing)
cv2.createTrackbar("LV", "Tracking",0,255,nothing)
cv2.createTrackbar("UH", "Tracking",255,255,nothing)
cv2.createTrackbar("US", "Tracking",255,255,nothing)
cv2.createTrackbar("UV", "Tracking",255,255,nothing)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture(0)

while(True):
    # cattura frame
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h= cv2.getTrackbarPos("LH", "Tracking")
    l_s= cv2.getTrackbarPos("LS", "Tracking")
    l_v= cv2.getTrackbarPos("LV", "Tracking")

    u_h= cv2.getTrackbarPos("UH", "Tracking")
    u_s= cv2.getTrackbarPos("US", "Tracking")
    u_v= cv2.getTrackbarPos("UV", "Tracking")

    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])

    #mask = cv2.inRange(hsv, l_b, u_b)
    print (frame)


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w] #(y_cordi_start, y_cordi_emd)
        roi_color = frame[y:y+h, x:x+w]
        gr = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2BGRA)
        mask = np.zeros_like(frame)

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for(ex,ey,ew,eh) in eyes:
            color = (255, 255, 255)
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), color, 2)
            mask[y:y+h, x:x+w] = frame[y:y+h, x:x+w]
        #frame[y:y+h, x:x+w] = gr

        #recognize

        img_item = "my-image.png"
        #cv2.imshow(img_item, roi_color)
        cv2.imshow("mask", mask)

        color = (255, 255, 255)
        stroke = 2
        width = x+w #end cord x
        height = y+h #end cord y
        cv2.rectangle(mask, (x,y), (width, height), color, stroke)


    # mostra i frame
    #res = cv2.bitwise_and(frame, frame, mask=mask)
    #cv2.imshow('frame', frame)



    #cv2.imshow("res", res)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# chiude la finestra di acquisizione
cap.release()
cv2.destroyAllWindows()
