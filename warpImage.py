import cv2
import utils
import imutils
import numpy as np

def warpImage(imagePath, savePath):
	image = cv2.imread(imagePath)
	
	
	grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
	gaussianBlurr = grayscale#cv2.GaussianBlur(grayscale, (5,5), 0) # maybe adjust the convolution kernel dynamically lol

	# we need to find the ends of the paper, so
	edged = cv2.Canny(gaussianBlurr, 75, 200)

	#contours
	contors = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	contors = imutils.grab_contours(contors)
 
	docCnt = None

	if len(contors)>0:
		#it exists, sort it
		contors = sorted(contors, key=cv2.contourArea)
		
		for c in contors:
			perim = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.02*perim, True)
			
			if (len(approx) == 4):
				docCnt = approx
				break
	
	warped = utils.four_point_transform(grayscale, docCnt.reshape(4,2))
	
	cv2.imwrite(savePath, warped)
	return savePath

def getMasks(image, rectangles):
	# Create an empty list to store the mask images
	masks = []
	highlight_image = np.zeros_like(image)

	# Loop through each rectangle in the rectangles list
	for rect in rectangles:
		# Create a black canvas with the same size as the input image
		mask = np.zeros_like(image)

		# Get the coordinates and dimensions of the rectangle
		x, y, w, h = rect['x'], rect['y'], rect['width'], rect['height']

		# Draw the rectangle on the mask, making the pixels white
		cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)

		# Add the mask to the list
		masks.append(mask)
  
		# Draw the rectangle on the highlight_image
		cv2.rectangle(highlight_image, (x, y), (x + w, y + h), (255, 255, 255), -1)
	cv2.imwrite('omr_sheets\\reference\\highlighted_image.png', highlight_image)

	return masks