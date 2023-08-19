import cv2
import imutils
#from four_point import splitBoxes
import utils
import warpImage
import measureConfig
import numpy as np
import json

import random


# Workflow (Grade OMR):
#	Get empty OMR picture from user
#	Get OMR details from user
#		questionsNo
#		choicesNo
#		sections (data from masking step)
#	Preprocess picture
#	Create picture masks
#	Make OMR config
#		create contours
#		get bubbles
#			sort contours
#			filter contours
#			splice contours -> bubbles
#		create table of bubbles
#		measure
#			measure bubble size (average 3 )
# 			measure margin size (average 3 )
#			make sample bubble
#			return json of it
#	do other stuff idk

isPreprocessed = False
warped = threshold = masks = processed = None

defaultFilePath = "omr_sheets\\reference\\ref3_warped.png"
defaultMask = "omr_sheets\\reference\\highlighted_image.png"

def preprocess(filename, rectangles=[]):
	global warped, threshold, masks, processed, isPreprocessed
	
	warped = cv2.imread(filename)
	grayscale = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

	threshold = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY_INV| cv2.THRESH_OTSU)[1]
	# debug case for testing
	if len(rectangles)==0:
		print("using debug fallback")
		masks = [cv2.imread(defaultMask)]
	else:
		masks = warpImage.getMasks(threshold, rectangles)
	
	processed = [cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY) if masked.shape[-1] == 3 else masked for masked in masks]
	processed = [np.uint8(masked) for masked in processed]
	processed = [cv2.bitwise_and(threshold, threshold, mask=masked) for masked in processed]
	isPreprocessed = True
	return processed

preprocess('./omr_sheets/reference/ref3_warped.png', [{'x': 158, 'y': 477, 'width': 144, 'height': 456}, {'x': 349, 'y': 474, 'width': 141, 'height': 463}, {'x': 538, 'y': 473, 'width': 144, 'height': 401}])

def preprocess_required(func):
	def wrapper(*args, **kwargs):
		# Check if preprocessing is required
		if not isPreprocessed:
			preprocess(defaultFilePath)
		
		# Call the original function with the provided arguments
		return func(*args, **kwargs)
	
	return wrapper



@preprocess_required
def makeOMRConfig(sections, filepath = ""):
	if not filepath == "":
		global defaultFilePath
		defaultFilePath = filepath
 
	sectionsConfig = []
	for i in range(len(processed)):
		masked = processed[i]
		questionsNo = sections[i]['questionsNo']
		choicesNo = sections[i]['choicesNo']
		contors = cv2.findContours(masked, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

		contors = imutils.grab_contours(contors)
  
		
			
	
		sorted_contors = sorted(contors, key=cv2.contourArea, reverse=True)
  
		# top_x_contors = sorted_contors[:questionsNo*choicesNo]
		# bubbleArray = utils.splitBoxes(masked, top_x_contors, 300)
		# sorted_bubbles = utils.sort_bubbles(bubbleArray, questionsNo, choicesNo)
		#cv2.imshow(" paste for " + str(i), utils.pasteOnlyBubble(threshold, sorted_bubbles))
		# if i == 0:
		# 	for j in range(questionsNo*choicesNo):
		# 		top_x_contors = sorted_contors[:questionsNo*choicesNo-j]
		# 		bubbleArray = utils.splitBoxes(masked, top_x_contors, 300)
		# 		sorted_bubbles = utils.sort_bubbles(bubbleArray, questionsNo, choicesNo)
		# 		cv2.imshow(str(questionsNo*choicesNo-j) + " paste for" + str(i), utils.pasteOnlyBubble(threshold, sorted_bubbles))
   
		top_x_contors = sorted_contors[:questionsNo*choicesNo-1]
  
		bubbleArray = utils.splitBoxes(masked, top_x_contors, 300)			
  
		# for i in bubbleArray:
		# 	cv2.imshow("bubble " + str(random.random()), i.image)
		
		sorted_bubbles = utils.sort_bubbles(bubbleArray, questionsNo, choicesNo)

		# for i in range(len(sorted_bubbles)):
		# 	cv2.imshow("box " + str(i+1), utils.pasteOnlyBubble(threshold, sorted_bubbles[:i]))

		#table = [sorted_bubbles[i * choicesNo:(i + 1) * choicesNo] for i in range(questionsNo)]

		table = utils.fill_blank_table(sorted_bubbles, questionsNo, choicesNo)

		config = measureConfig.measure(table)
		config = json.loads(config)
		bubbles = config["bubbles"]
		samples = config["samples"]
		sectionsConfig.append({
			'bubbles': bubbles,
			'samples': samples,
			'sectionData': {
	   			'questionsNo': questionsNo, 
		  		'choicesNo': choicesNo
			}
		})
	for section in sectionsConfig:
		create_omr_sheet(section)

	cv2.waitKey(0)


def create_omr_sheet(config):
	# Get the configuration data from the dictionary
	start_x, start_y = config['bubbles'][0]['x'], config['bubbles'][0]['y']
	bubble_width, bubble_height = config['bubbles'][0]['width'], config['bubbles'][0]['height']
	horizontal_margin = config['bubbles'][0]['horizontalMargin']
	vertical_margin = config['bubbles'][0]['verticalMargin']

	# Get the number of rows and columns from the 'bubbles' variable
	questionsNo = config['sectionData']['questionsNo']
	choicesNo = config['sectionData']['choicesNo']

	# Create a black canvas for the OMR sheet
	#omr_sheet = np.zeros((sheet_height, sheet_width), dtype=np.uint8)
	omr_sheet = np.zeros_like(threshold, dtype=np.uint8)

	for bubble in config['bubbles']:
		#try:
		row = (bubble['y'] - start_y) // (bubble_height + vertical_margin)
		col = (bubble['x'] - start_x) // (bubble_width + horizontal_margin)
		# Get the image data from the config
		sample_image = config['samples'][col]

		# Calculate the adjusted bubble width for pasting
		adjusted_bubble_width = min(bubble_width, len(sample_image[0]))

		# Calculate the carryover (if any) for adjusting the pasting position
		carryover = (bubble_width - adjusted_bubble_width) // 2

		# Paste the image onto the OMR sheet
		for i in range(len(sample_image)):
			omr_sheet[bubble['y'] + i][bubble['x'] + carryover:bubble['x'] + carryover + adjusted_bubble_width] = \
				sample_image[i][:adjusted_bubble_width]
		# except Exception as e:
		# 	print("whoops")
		# 	print(bubble['y'], start_y, bubble_height, vertical_margin)
		# 	print(bubble['x'], start_x, bubble_width, horizontal_margin)
		

	# Save the OMR sheet as an image file
	#cv2.imshow('omr_sheet.png', omr_sheet)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	
makeOMRConfig([{'questionsNo': 14, 'choicesNo': 4}, {'questionsNo': 14, 'choicesNo': 4}, {'questionsNo': 11, 'choicesNo': 4}], './omr_sheets/reference/ref3_warped.png')

# grade = "idk man i dont work here"

# def gradeOMR(shouldShow=False):
#	 return grade

# grade = "idk man i dont work here"

# def gradeOMR(shouldShow):
# 	def cvshow(text, image):
# 		if (shouldShow):
# 			cv2.imshow(text, image)
	
	
# 	Answer_key = {0:0, 1:0, 2:0, 3:0, 4:0}

# 	questionsNo = 14

# 	choicesNo = 4


# 	warped = cv2.imread(".\\omr_sheets\\reference\\ref4_warped.png")

# 	grayscale = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
# 	cvshow("gray", grayscale)

# 	threshold = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY_INV| cv2.THRESH_OTSU)[1]

# 	mask = cv2.imread(".\\omr_sheets\\reference\\highlighted_image.png")
# 	# literally just signalling
	# mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

	# # Threshold the "mask" image to create a binary mask (0 for black, 255 for white)
	# _, mask_binary = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)

	# # Convert "mask_binary" to 3 channels to match the "threshold" image
	# mask_binary_3_channels = cv2.merge((mask_binary, mask_binary, mask_binary))

	# mask_binary_grayscale = cv2.cvtColor(mask_binary_3_channels, cv2.COLOR_BGR2GRAY)

	# mask_binary_grayscale_resized = cv2.resize(mask_binary_grayscale, (threshold.shape[1], threshold.shape[0]))
 
	# masked = cv2.bitwise_and(threshold, threshold, mask=mask_binary_grayscale_resized)

# 	contors = cv2.findContours(masked, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# 	contors = imutils.grab_contours(contors)
 
 
# 	sorted_contors = sorted(contors, key=cv2.contourArea, reverse=True)
 
# 	top_x_contors = sorted_contors[:questionsNo*choicesNo]
 
# 	cvshow("masked", masked)
 

# 	print(str(len(top_x_contors)) + " uh? " + str(questionsNo*choicesNo))

 
# 	bubbleArray = utils.splitBoxes(masked, top_x_contors, 300)
 
 

# 	# print(len(bubbleArray))
	
	
# 	bubbleArray = utils.regularize(bubbleArray)
	
# 	cvshow("Pls", utils.pasteOnlyBubble(threshold, bubbleArray))
 
# 	sorted_bubbles = utils.sort_bubbles(bubbleArray, questionsNo, choicesNo)
 
# 	table = [sorted_bubbles[i * choicesNo:(i + 1) * choicesNo] for i in range(questionsNo)]
	
# 	answer = [utils.detect_answer(question) for question in table]
 
# 	print(answer)
 
# 	print(len(sorted_bubbles))
# 	for i in range(1,len(sorted_bubbles)+1)[::-1]:
# 		pls = utils.pasteOnlyBubble(threshold, sorted_bubbles[:i])
# 		cvshow("maybe? " + str(i), pls)
# 	#outlined = [four_point.pasteOnlyBubble(threshold, [bubble]) for bubble in sorted_bubbles]

# 	# for i, bubble in enumerate(outlined):
# 	# 	if (i<8):
# 	# 		cvshow(f"contor {i+1}", bubble)
	
# 	# outlinedBubbles = [four_point.Bubble(0, 0, bubble, bubble.shape[0], bubble.shape[1]) for bubble in outlined]
	
	


# 	# cv2.drawContours(warped, contors, -1, (0, 255, 0), 2)

# 	# cvshow("second", warped)
# 	# cvshow("first", four_point.four_point_transform(grayscale, docCnt.reshape(4,2)))


# 	# pixelValues = np.zeros((questionsNo, choicesNo))

# 	# print(len(contors))

# 	#cvshow("orig", warped)

# 	# bubbleArray = four_point.splitBoxes(masked, contors, 300)
	
# 	# bubbleArray = four_point.regularize(bubbleArray)
 
# 	# import time
	
# 	# start_time = time.time()

# 	# bubbleArray = four_point.combine(bubbleArray, 45)

# 	# end_time = time.time()
 
# 	# execution_time = end_time - start_time

# 	# print(f"Execution time for combination: {execution_time:.6f} seconds")
# 	# print(len(bubbleArray))
 
# 	# # print(four_point.calculate_image_similarity(bubbleArray[41].image, bubbleArray[41].image)) # should be full
# 	# # print(four_point.calculate_image_similarity(bubbleArray[41].image, bubbleArray[40].image)) # should be basically full
# 	# # print(four_point.calculate_image_similarity(bubbleArray[35].image, bubbleArray[34].image)) # should be basically full
# 	# # print(four_point.calculate_image_similarity(bubbleArray[41].image, bubbleArray[35].image))
# 	# # print(four_point.calculate_image_similarity(bubbleArray[41].image, bubbleArray[34].image))
# 	# # print(four_point.calculate_image_similarity(bubbleArray[34].image, bubbleArray[43].image))
	
# 	# #cvshow("Pls", four_point.paste(bubbleArray))
# 	# for i, bubble in enumerate(bubbleArray):
# 	# 	#print([bubble.width, bubble.height])
# 	# 	cvshow(f"Bubble {i+1}", bubble.image)

# 	cv2.waitKey(0)
# 	return grade
# gradeOMR(True)