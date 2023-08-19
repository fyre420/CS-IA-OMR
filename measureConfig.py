import json
import operator
import numpy as np
import utils

def predict_bubble(row, col, table):
    # In this example, let's assume we predict the bubble based on the bubble above it
    if row > 0 and table[row - 1][col] is not None:
        above_bubble = table[row - 1][col]
        predicted_bubble = utils.Bubble(
            x=above_bubble.x,
            y=above_bubble.y + above_bubble.height,  # Place it below the above bubble
            image=np.zeros((10, 10)),  # Example image shape
            width=10,
            height=10
        )
        return predicted_bubble
    else:
        # If no bubble above, provide default values
        return utils.Bubble(x=0, y=0, image=np.zeros((10, 10)), width=10, height=10)

def measure(table):
    # Check if the table is not empty
    if not table or not table[0]:
        raise ValueError("Invalid table")

    # Create a list to store the measurements for each bubble
    bubble_measurements = []

    for row in range(len(table)):
        for col in range(len(table[row])):
            bubble = table[row][col]

            if bubble is None:
                # You need to replace this with your own prediction logic
                predicted_bubble = predict_bubble(row, col, table)#{'x': 0, 'y': 0, 'image': np.zeros((bubble_height, bubble_width))}
                bubble = predicted_bubble

            # Calculate the bubble dimensions
            bubble_width = len(bubble.image[0])
            bubble_height = len(bubble.image)

            # Calculate the horizontal margin for this bubble
            horizontal_margin = 0
            if col < len(table[row]) - 1 and table[row][col + 1] is not None:
                next_bubble = table[row][col + 1]
                horizontal_margin = next_bubble.x - (bubble.x + bubble_width)

            # Calculate the vertical margin for this bubble
            vertical_margin = 0
            if row < len(table) - 1 and table[row + 1][col] is not None:
                next_row_bubble = table[row + 1][col]
                vertical_margin = next_row_bubble.y - (bubble.y + bubble_height)

            # Add the measurements to the list
            bubble_measurements.append({
                'x': bubble.x,
                'y': bubble.y,
                'width': bubble_width,
                'height': bubble_height,
                'horizontalMargin': horizontal_margin,
                'verticalMargin': vertical_margin,
            })


    # Get the image data for each column and convert to list
    column_samples = []
    for col_idx in range(len(table[0])):
        # Get the first non-None bubble in the column
        column_bubble = None
        for row_idx in range(len(table)):
            if col_idx < len(table[row_idx]) and table[row_idx][col_idx] is not None:
                column_bubble = table[row_idx][col_idx]
                break

        if column_bubble is not None:
            column_samples.append(column_bubble.image.tolist())  # Convert ndarray to list

    # Create a dictionary to store all the measurements and samples
    result = {
        'bubbles': bubble_measurements,
        'samples': column_samples,
    }

    # Convert the dictionary to JSON format
    return json.dumps(result)
