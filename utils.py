import numpy as np
import cv2
import operator

def adjust_contrast(image, contrast):
    # Create a CLAHE object with the specified contrast parameter
    clahe = cv2.createCLAHE(clipLimit=contrast, tileGridSize=(8, 8))

    # Apply CLAHE to the grayscale image
    enhanced_img = clahe.apply(image)

    return enhanced_img

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

def convert_to_binary(image):
	# Convert the image to binary format (white pixels as 1, black pixels as 0)
	_, binary_image = cv2.threshold(image, 1, 1, cv2.THRESH_BINARY)
	return binary_image

def is_omr_bubble(contour, min_area_threshold=100):
	# Calculate the aspect ratio of the contour's bounding rectangle
	x, y, w, h = cv2.boundingRect(contour)
	aspect_ratio = float(w) / h

	# Check if the contour is approximately circular or square based on aspect ratio and vertices
	# You can adjust the aspect_ratio_threshold and vertex_threshold based on your specific case
	aspect_ratio_threshold = 0.7  # Experiment with different values

	# Check if the aspect ratio is close to 1 (for circles or squares)
	is_circular_or_square = 1 - aspect_ratio_threshold <= aspect_ratio <= 1 + aspect_ratio_threshold
 
	contour_area = cv2.contourArea(contour)
	is_big_enough = contour_area > min_area_threshold
 
  # Calculate the solidity of the contour
	hull = cv2.convexHull(contour)
	hull_area = cv2.contourArea(hull)
	solidity = contour_area / hull_area
	
	# Check if the solidity is within a certain range (smaller than text)
	solidity_threshold = 0.9  # Experiment with different values
	is_within_solidity_range = solidity > solidity_threshold

	# The contour is considered an OMR bubble if it passes both criteria
	return is_circular_or_square and is_big_enough and is_within_solidity_range
 
# def splitBoxes(image, contours_list, min_area_threshold=100, max_merge_distance=10):
# 	bubble_arrays = []

# 	# Convert contours to rectangles (bounding boxes) for cv2.groupRectangles
# 	rects = [cv2.boundingRect(cnt) for cnt in contours_list]

# 	# Combine close contours into a single contour using cv2.groupRectangles
# 	rects, _ = cv2.groupRectangles(rectList=rects, groupThreshold=1, eps=max_merge_distance)
 
# 	canvas = np.zeros_like(image)

# 	for rect in rects:
# 		x, y, w, h = rect
# 		# Extract the combined bubble region from the original image
# 		bubble = image[y:y+h, x:x+w]
  
# 		contour = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
		
# 		if is_omr_bubble(contour, min_area_threshold):
	  
			

#			 # Paste the bubble onto the canvas
# 			canvas[y:y+h, x:x+w] = bubble

# 	bubble_arrays.append(canvas)

# 	return bubble_arrays

# gives only contour in bubble form
def splitBoxes(image, contours_list, min_area_threshold=100, max_merge_distance=10):
	bubble_objects = []

	for contour in contours_list:
		x, y, w, h = cv2.boundingRect(contour)
		# Extract the combined bubble region from the original image
		bubble = image[y:y+h, x:x+w]

		# Check if the bubble meets the OMR bubble criteria
		contour_points = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
		if True:#is_omr_bubble(contour_points, min_area_threshold):
			custom_object = Bubble(x, y, bubble, w, h)
			bubble_objects.append(custom_object)

	return bubble_objects

# def splitBoxes(image, contours_list, min_area_threshold=100):
# 	canvas = np.zeros_like(image)

# 	for contour in contours_list:
# 		x, y, w, h = cv2.boundingRect(contour)
# 		# Extract the combined bubble region from the original image
# 		bubble = image[y:y+h, x:x+w]

# 		# Check if the bubble meets the OMR bubble criteria
# 		contour_points = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
# 		if is_omr_bubble(contour_points, min_area_threshold):
# 			# Place the bubble onto the canvas in the appropriate position
# 			canvas[y:y+h, x:x+w] = bubble

# 	return [canvas]

class Bubble:
	def __init__(self, x, y, image, width, height):
		self.x = x
		self.y = y
		self.image = image
		self.width = width
		self.height = height
	
	@property
	def area(self):
		return self.width * self.height

# def splitBoxes(image, contours_list, min_area_threshold=100):
# 	bubble_objects = []

# 	for contour in contours_list:
# 		x, y, w, h = cv2.boundingRect(contour)
# 		# Create an empty canvas with the same size as the original image
# 		bubble_canvas = np.zeros_like(image)

# 		# Draw the contour on the canvas
# 		cv2.drawContours(bubble_canvas, [contour], 0, (255), -1)

# 		# Extract the combined bubble region from the original image
# 		bubble = image[y:y+h, x:x+w]

# 		# Check if the bubble meets the OMR bubble criteria
# 		contour_points = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
# 		if is_omr_bubble(contour_points, min_area_threshold):
# 			# Replace the bubble region with the contour on the canvas
# 			bubble_canvas[y:y+h, x:x+w] = bubble

# 			# Create a Bubble object with the canvas containing the contour
# 			bubble_obj = Bubble(x, y, bubble_canvas, bubble_canvas.shape[1], bubble_canvas.shape[0])
# 			bubble_objects.append(bubble_obj)

# 	return bubble_objects

def regularize(bubbleArray):
	# Find the maximum width and height among all Bubble objects
	max_width = max(bubble.width for bubble in bubbleArray)
	max_height = max(bubble.height for bubble in bubbleArray)

	# Regularize each Bubble object to have the same dimensions
	regularized_bubble_array = []
	for bubble in bubbleArray:
		# Calculate the padding needed to resize the Bubble object
		width_pad = max_width - bubble.width
		height_pad = max_height - bubble.height

		# Create an empty canvas with the maximum dimensions
		canvas = np.zeros((max_height, max_width), dtype=np.uint8)

		# Calculate the position to paste the Bubble image centered on the canvas
		x_offset = width_pad // 2
		y_offset = height_pad // 2

		# Paste the Bubble image onto the canvas
		canvas[y_offset:y_offset + bubble.height, x_offset:x_offset + bubble.width] = bubble.image

		# Update the Bubble object with the regularized image and dimensions
		bubble.image = canvas
		bubble.width = max_width
		bubble.height = max_height

		regularized_bubble_array.append(bubble)

	return regularized_bubble_array

def sort_bubbles(bubbleArray, rowNo, colNo):
    if len(bubbleArray) == 0 or rowNo <= 0 or colNo <= 0:
        return bubbleArray
    
    # Calculate the band size for x and y coordinates
    x_min = min(bubble.x for bubble in bubbleArray)
    x_max = max(bubble.x for bubble in bubbleArray)
    y_min = min(bubble.y for bubble in bubbleArray)
    y_max = max(bubble.y for bubble in bubbleArray)

    # You can adjust these factors to control the band sizes
    x_band_factor = 0.1  # Adjust as needed
    y_band_factor = 0.1  # Adjust as needed

    x_band_size = (x_max - x_min) * x_band_factor / colNo
    y_band_size = (y_max - y_min) * y_band_factor / rowNo

    # Sort the bubbles using custom sorting key
    sorted_bubbles = sorted(bubbleArray, key=lambda bubble: ((bubble.y - y_min) // y_band_size, (bubble.x - x_min) // x_band_size))
    
    return sorted_bubbles

def fill_blank_table(sorted_bubbles, questionsNo, choicesNo):
    table = []

    for i in range(questionsNo):
        row_start = i * choicesNo
        row_end = (i + 1) * choicesNo
        row_bubbles = sorted_bubbles[row_start:row_end]
        
        if len(row_bubbles) < choicesNo:
            row_bubbles += [None] * (choicesNo - len(row_bubbles))
        
        table.append(row_bubbles)

    return table

def replace_outliers_with_average(data, threshold=3.5):
    data = np.array(data)  # Convert the list to a NumPy array
    median = round(np.median(data))  # Round the median value

    # Calculate Median Absolute Deviation (MAD)
    mad = np.median(np.abs(data - median))
    if mad == 0:
        z_scores = np.zeros_like(data)  # Set z_scores to zero if MAD is zero
    else:
        z_scores = 0.6745 * (data - median) / mad

    outliers = np.abs(z_scores) > threshold

    # Use np.where() to update the values based on the outliers condition
    data = np.where(outliers, median, data)

    return data.tolist()  # Convert back to a list before returning

def detect_answer(bubbleArray):
    max_white_pixels = 0
    max_index = 0

    for i, bubble in enumerate(bubbleArray):
        white_pixels = cv2.countNonZero(bubble.image)
        if white_pixels > max_white_pixels:
            max_white_pixels = white_pixels
            max_index = i

    return max_index
 
def calculate_image_similarity(image1, image2):
	# Convert the images to grayscale (if they are not already)
	if len(image1.shape) == 3:
		image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
	else:
		image1_gray = image1
	if len(image2.shape) == 3:
		image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
	else:
		image2_gray = image2

	# Calculate the correlation between the images
	correlation = cv2.matchTemplate(image1_gray, image2_gray, cv2.TM_CCOEFF_NORMED)

	# Get the maximum correlation value and its location
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(correlation)

	# Return the maximum correlation value (normalized for comparison)
	#return random.random()
	return max_val

def calculate_bubble_similarity(bubble1, bubble2):
    return calculate_image_similarity(bubble1.image, bubble2.image)

def combinate(obj_list, operation):
    n = len(obj_list)

    # Create an empty upper triangular matrix filled with zeros
    matrix = np.zeros((n, n), dtype=int)

    # Populate the upper triangular matrix with results of applying the function
    for i in range(n):
        for j in range(i, n):
            result = operation(obj_list[i], obj_list[j])
            matrix[i, j] = result

    # Fill the lower triangular part of the matrix with the values from the upper triangular part
    matrix = matrix + matrix.T - np.diag(np.diag(matrix))

    return matrix	

# def combine(bubbleArray, needLength):
    
#     combinations = combinate(bubbleArray, calculate_bubble_similarity)
    

def combine(bubbleArray, needLength):
	calls = 0
	if len(bubbleArray) <= needLength:
		return bubbleArray

	# Create a copy of bubbleArray so that we can modify it
	combined_bubbles = bubbleArray.copy()

	while len(combined_bubbles) > needLength:
		# Initialize variables to store the indices of the most similar images
		most_similar_idx1 = 0
		most_similar_idx2 = 1
		max_similarity = -1

		# Find the most similar pair of images
		for i in range(len(combined_bubbles)):
			for j in range(i + 1, len(combined_bubbles)):
				similarity = calculate_image_similarity(combined_bubbles[i].image, combined_bubbles[j].image)
				calls+= 1
				if similarity > max_similarity:
					max_similarity = similarity
					most_similar_idx1 = i
					most_similar_idx2 = j

		# Merge the two most similar images into one
		combined_image = np.maximum(combined_bubbles[most_similar_idx1].image, combined_bubbles[most_similar_idx2].image)
		x = min(combined_bubbles[most_similar_idx1].x, combined_bubbles[most_similar_idx2].x)
		y = min(combined_bubbles[most_similar_idx1].y, combined_bubbles[most_similar_idx2].y)
		width = max(combined_bubbles[most_similar_idx1].x + combined_bubbles[most_similar_idx1].width,
					combined_bubbles[most_similar_idx2].x + combined_bubbles[most_similar_idx2].width) - x
		height = max(combined_bubbles[most_similar_idx1].y + combined_bubbles[most_similar_idx1].height,
					 combined_bubbles[most_similar_idx2].y + combined_bubbles[most_similar_idx2].height) - y

		combined_bubble = Bubble(x, y, combined_image, width, height)

		# Remove the two merged images from the list and add the combined image
		combined_bubbles.pop(most_similar_idx1)
		combined_bubbles.pop(most_similar_idx2 - 1)  # As we already removed one element, adjust the index
		combined_bubbles.append(combined_bubble)
	print(calls)

	return combined_bubbles[:needLength]


def paste(bubbleArray):
	combined_image = np.zeros_like(bubbleArray[0])
	for bubble in bubbleArray:
		combined_image = cv2.bitwise_or(combined_image, bubble.image)
	return combined_image

def pasteOnlyBubble(image, bubbles):
    canvas = np.zeros_like(image)
    for bubble in bubbles:
        x, y = bubble.x, bubble.y
        width, height = bubble.width, bubble.height

        # Ensure the bubble coordinates are within the canvas bounds
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + width, canvas.shape[1]), min(y + height, canvas.shape[0])

        # Calculate the ROI in the bubble image (may be cropped)
        roi_x1, roi_y1 = max(0, -x), max(0, -y)
        roi_x2, roi_y2 = min(width, width - (x + width - canvas.shape[1])), min(height, height - (y + height - canvas.shape[0]))

        # Paste the bubble onto the canvas
        canvas_roi = canvas[y1:y2, x1:x2]
        bubble_roi = bubble.image[roi_y1:roi_y2, roi_x1:roi_x2]

        # Resize the bubble ROI if necessary
        if bubble_roi.shape[0] < canvas_roi.shape[0] or bubble_roi.shape[1] < canvas_roi.shape[1]:
            bubble_roi = cv2.resize(bubble_roi, (canvas_roi.shape[1], canvas_roi.shape[0]))

        # Combine the bubble with the canvas using transparency (alpha blending)
        canvas[y1:y2, x1:x2] = cv2.addWeighted(canvas_roi, 1.0, bubble_roi, 1.0, 0.0)

    return canvas


# def combine(bubbleArray, needLength):

# you have to write this function, this function should output an array of bubble objects, the length of this bubble must be equal to needLength, this function should work by calling calculate_image_similarity() on images in bubbleArray, and combining those which are most similar