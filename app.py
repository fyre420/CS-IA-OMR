# app.py

from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
import os

import scan

import subprocess
import json

from saveHighlight import save_highlight_data

import warpImage

app = Flask(__name__)
app.debug = True






def is_safe_filename(filename):
	# Define a list of allowed characters and patterns for the filename
	allowed_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_ "
	allowed_extensions = [
		".bmp", ".dib",
		".jpeg", ".jpg", ".jpe",
		".jp2",
		".png",
		".webp",
		".pbm", ".pgm", ".ppm", ".pxm", ".pnm",
		".sr", ".ras",
		".tiff", ".tif",
		".exr",
		".hdr", ".pic"
	]

	# Split the filename into the base name and the extension
	base_name, extension = os.path.splitext(filename)

	# Check if the extension is allowed
	if extension.lower() not in allowed_extensions:
		return False

	# Check if the base name contains only allowed characters
	if all(char in allowed_characters for char in base_name):
		return True
	else:
		return False
@app.route('/')
def index():
	# Replace 'index.html' with the name of your HTML file
	return render_template('index.html')

@app.route('/app.js', methods = ['GET'])
def serve_app_js():
	js_file = os.path.join(os.path.dirname(__file__), 'app.js')
	return send_file(js_file)


@app.route('/api/get_data', methods = ['GET'])
def get_data():
	# Your Python code here to process the request and return data
	data = {
		'message': 'word'#gradeOMR(False),
	}
	return jsonify(data)

@app.route('/api/process_data', methods=['POST'])
def process_data():
	try:
		data = request.get_json()

		# Retrieve the required data from the JSON
		filepath = data.get('filepath')
		rectangles = data.get('rectangles')
		sectionData = data.get('sectionData')
		print(filepath, rectangles, sectionData)

		# # Run scan.preprocess
		scan.preprocess(filepath, rectangles)
		# # Get scan.makeOMRConfig
		scan.makeOMRConfig(sectionData, filepath)
		# Return a success response
		return jsonify({'message': 'Data processed successfully'})

	except Exception as e:
		return jsonify({'error': f'Error processing data: {str(e)}'}), 500

@app.route('/api/upload_image', methods = ['POST'])
def upload_image():
	if 'imageFile' not in request.files:
		return jsonify({'error': 'No file part'})

	file = request.files['imageFile']

	if file.filename == '':
		return jsonify({'error': 'No selected file'})

	if not is_safe_filename(file.filename):
		return jsonify({'error': 'Unsafe filename, cannot save'})

	# Save the uploaded file to a temporary location
	file_path = 'temp_image.jpg'
	file.save(file_path)
 
	filename, extension = os.path.splitext(file.filename)

	# Append '_warped' to the filename and concatenate with the original extension
	warped_filename = filename + '_warped' + extension

	# Save the warped image with the new filename
	warpImage.warpImage(file_path, os.path.join('omr_sheets', 'reference', warped_filename))

	# Remove the temporary image file
	os.remove(file_path)

	return jsonify({'message': 'Image processed and saved successfully', 'filename': warped_filename})

@app.route('/api/delete_file', methods=['POST'])
def delete_file():
	data = request.get_json()
	filename = data.get('filename')  # Use data.get('filename') instead of data.filename
	try:
		# Check if the filename is safe
		if not is_safe_filename(filename):  # Use filename here
			raise OSError(f'Unsafe filename: {filename}')

		# Perform your file processing or save operations here
		if filename:
			file_path = os.path.join('omr_sheets', 'reference', filename)
			try:
				os.remove(file_path)
				return jsonify({'message': f'File {filename} deleted successfully'})
			except OSError as e:
				return jsonify({'error': f'Failed to delete file: {str(e)}'}), 500
		else:
			return jsonify({'error': 'Invalid filename provided'}), 400
	# If everything is successful, return a success response

	except OSError as e:
		# Handle the unsafe filename error
		return jsonify({'error': str(e)}), 400
		

	

@app.route('/omr_sheets/<path:filename>', methods=['GET'])
def serve_image(filename):
	return send_from_directory('omr_sheets', filename)

@app.route('/api/save_highlight', methods=['POST'])
def save_highlight():
	try:
		data = request.json

		# Call the function from saveHighlight.py to process the highlight data
		save_highlight.save_highlight_data(data)

		return jsonify({'message': 'Highlight data received and processed.'})

	except Exception as e:
		return jsonify({'error': str(e)})

if __name__ == '__main__':
	app.run(debug=True, port=3000)