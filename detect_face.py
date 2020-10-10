from random import randint
import cv2
import sys
import os
import traceback

CASCADE="Face_cascade.xml"
FACE_CASCADE=cv2.CascadeClassifier(CASCADE)

def detect_faces(image_path,display=True):

	image=cv2.imread(image_path)
	image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	faces = FACE_CASCADE.detectMultiScale(image_grey,scaleFactor=1.16,minNeighbors=5,minSize=(25,25),flags=0)

	for x,y,w,h in faces:
	    sub_img=image[y-10:y+h+10,x-10:x+w+10]
	    os.chdir("Extracted")
	    cv2.imwrite(str(randint(0,10000))+".jpg",sub_img)
	    os.chdir("../")
	    cv2.rectangle(image,(x,y),(x+w,y+h),(255, 255,0),2)

	if display:
		cv2.imshow("Faces Found",image)
		# if (cv2.waitKey(0) & 0xFF == ord('q')) or (cv2.waitKey(0) & 0xFF == ord('Q')):
		# 	cv2.destroyAllWindows()

def get_images(image_path):

	if not "Extracted" in os.listdir("."):
		os.mkdir("Extracted")

	if os.path.isdir(image_path):
		for image in os.listdir(image_path):
			try:
				print ("Processing.....",os.path.abspath(os.path.join(image_path,image)))
				detect_faces(os.path.abspath(os.path.join(image_path,image)),False)
			except Exception:
				print ("Could not process ",os.path.abspath(os.path.join(image_path,image)))
	else:
		detect_faces(image_path)
