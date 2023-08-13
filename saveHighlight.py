# saveHighlight.py

import cv2, os
import numpy as np

def save_highlight_data(rectangle_data):
    # Extract the image height and width from the last element of the list
    image_height = rectangle_data[-1]['imageHeight']
    image_width = rectangle_data[-1]['imageWidth']
    # Remove the last element, which contains image metadata
    rectangle_data.pop()

    # Create a white canvas with the specified image size
    canvas = np.zeros((image_height, image_width), dtype=np.uint8) * 255

    # Draw black rectangles on the canvas based on the rectangle data
    for rect in rectangle_data:
        x, y, w, h = rect['x'], rect['y'], rect['width'], rect['height']
        cv2.rectangle(canvas, (x, y), (x + w, y + h), 255, -1)

    # Save the generated image in the "omr_sheets/reference" directory
    save_path = os.path.join('omr_sheets', 'reference', 'highlighted_image.png')
    cv2.imwrite(save_path, canvas)