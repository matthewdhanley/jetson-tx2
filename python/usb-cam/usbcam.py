import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
        print("Cannot open camera\n")
        exit(1)

while True:
        ret, img = cap.read()
        if not ret:
                print("No image was captured by 'cap.read()'\n") #### DIES HERE ####
                break

        cv2.imshow('img', img)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
                break

cap.release()
cv2.destroyAllWindows()