from random import randint
import numpy as np
import cv2
import sys
import os
import traceback
from PIL import Image

sys.path.insert(1, 'background_remove/')

from background_remove.demo.background_remove import remove_background



CASCADE="Face_cascade.xml"
FACE_CASCADE=cv2.CascadeClassifier(CASCADE)

def get_faces(background):
	SCALE_FACTOR = 25.0 / 500.0
	image_grey=cv2.cvtColor(background,cv2.COLOR_BGR2GRAY)
	size = int(min(background.shape[0], background.shape[1]) * SCALE_FACTOR)
	return FACE_CASCADE.detectMultiScale(image_grey,scaleFactor=1.1,minNeighbors=5,minSize=(size,size),flags=0)



def write_input_face(background, input, x1, x2, y1, y2):

	alpha_s = input[:, :, 3] / 255.0
	alpha_l = 1.0 - alpha_s

	for c in range(0, 3):
	    background[y1:y2, x1:x2, c] = (alpha_s * input[:, :, c] +
	                              alpha_l * background[y1:y2, x1:x2, c])
def bounds_from_face(face):
	return (face[1]-10, face[1]+face[3]+10, face[0]-10, face[0]+face[2]+10)

def get_bounds_no_const(face):
	return (face[1], face[1]+face[3], face[0], face[0]+face[2])

def add_alpha(input_path):
	img = Image.open(input_path)
	if img.mode == "RGB":
		a_channel = Image.new('L', img.size, 255)   # 'L' 8-bit pixels, black and white
		img.putalpha(a_channel)
	new_path = input_path.split('.')[0] + '.png'
	img.save(new_path)
	return new_path

def write_all_faces(background_path, input_path, extract=False):

	if not "Extracted" in os.listdir("."):
		os.mkdir("Extracted")

	input_path = remove_background(input_path)


	background_image=cv2.imread(background_path)
	input_image = cv2.imread(input_path, -1)
	if extract:
		input_image_faces = get_faces(input_image)
		face = input_image_faces[0]
		a,b,c,d = bounds_from_face(face)
		input_image = input_image[a:b,c:d,:]
		a,b,c,d = get_bounds_no_const(face)
		testing_input = input_image[a:b,c:d,:]
		cv2.imwrite('test_face.png', testing_input)

	faces = get_faces(background_image)

	for face in faces:
		y1, y2, x1, x2 = bounds_from_face(face)
		dims = (x2 - x1, y2 - y1)
		resized_input = cv2.resize(input_image, dims)
		write_input_face(background_image, resized_input, x1, x2, y1, y2)
	cv2.imwrite('test_result.jpg', background_image)

def samify(background_path):
	background_image=cv2.imread(background_path)
	input_image = cv2.imread('samy_2.png', -1)

	faces = get_faces(background_image)

	for face in faces:
		y1, y2, x1, x2 = bounds_from_face(face)
		dims = (x2 - x1, y2 - y1)
		resized_input = cv2.resize(input_image, dims)
		write_input_face(background_image, resized_input, x1, x2, y1, y2)
	cv2.imwrite('test_result.jpg', background_image)

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
