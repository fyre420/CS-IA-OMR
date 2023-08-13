const emptyOMRUpload = document.getElementById('emptyOMRUpload');
const imageInput = document.getElementById('emptyOMR');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext("2d")
const clearButton = document.getElementById('clearButton');
const clearHighlightsButton = document.getElementById('clearHighlights');
const form = document.getElementById('userInputForm');
const numQuestionsInput = document.getElementById('questionsNo');
const choicesPerQuestionInput = document.getElementById('choicesNo');
const warnHighlights = document.getElementById('warnHighlights')

function toggleHide(elements=elemToHide) {
	for (const element of elements) {
		if (element.style.display = 'none') {
			element.style.display = 'block'
			continue
		}
		element.style.display = 'none'
	}
}

elemToHide = [warnHighlights, canvas, clearHighlightsButton, clearButton, form]

// Event listener for image upload
imageCopy = null;
postData = null;
imageInput.addEventListener('change', async (event) => {
	event.preventDefault()

	const formData = new FormData(emptyOMRUpload);
	try {
		// Show the rest of form after the image is uploaded
		toggleHide(elemToHide)

		// Image upload and retrieval
		const response = await fetch('/api/upload_image', {
			method: 'POST',
			body: formData,
		});

		if (response.ok) {
			const data = await response.json();
			postData = data
			ctx.clearRect(0, 0, canvas.width, canvas.height);
			const imageUrl = "omr_sheets/reference/" + data.filename;

			// Create a new Image element and set its src to the image URL
			const image = new Image();
			// Create a Promise to ensure the image is loaded
			const imageLoaded = new Promise((resolve, reject) => {
				image.onload = function() {
					resolve();
				};
				image.onerror = function() {
					reject('Error loading image.');
				};
			});

			// Set the src to trigger the loading of the image
			image.src = imageUrl;

			// Wait for the image to be loaded before drawing it on the canvas
			await imageLoaded;

			canvas.width = image.width;
			canvas.height = image.height;

			// Draw the image on the canvas
			ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
			console.log('Image loaded.');

			imageCopy = image.cloneNode()
		} else {
			console.error('Error:', response.status, response.statusText);

			// Handle the error response from the server
			const errorData = await response.json();
			console.error('Server Error:', errorData.error);

			toggleHide(elemToHide)
		}
		

	} catch (error) {
		console.error('Error:', error);
		alert("uploadfunc had an error, look at console")
	}
});

clearHighlightsButton.addEventListener('click', async () => {
	rectangles = []
	isDrawing = false;
	startX, startY, endX, endY = undefined;
	redrawImage()

	form.innerHTML = `
		<br id = "brick">
		<button type="submit">Scan</button>
	`
});

clearButton.addEventListener('click', async () => {
	emptyOMRUpload.reset();
	canvas.width = 0
	canvas.height = 0
	ctx.fillStyle = '#FFFFFF'; // White color
	ctx.fillRect(0, 0, canvas.width, canvas.height);

	toggleHide(elemToHide)

	rectangles = []
	isDrawing = false;
	startX, startY, endX, endY = undefined;
	redrawImage()

	form.innerHTML = `
		<br id = "brick">
		<button type="submit">Scan</button>
	`

	// Set imageCopy to null to release the reference to the image object
	imageCopy = null;

	// Get the filename from the response data of the upload_image endpoint
	const filename = postData.filename;

	// Send a POST request to the Flask app to delete the image file
	try {
		const response = await fetch('/api/delete_file', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({ filename }),
		});

		if (!response.ok) {
			console.error('Error:', response.status, response.statusText);
		} else {
			console.log('Image file deleted successfully.');
		}
	} catch (error) {
		console.error('Error:', error);
	}
});


let rectangles = [];

let isDrawing = false;
let startX, startY, endX, endY;

function redrawImage() {
	// Redraw the image on the canvas

	ctx.drawImage(imageCopy, 0, 0, canvas.width, canvas.height)
	for (const rect of rectangles) {
		ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
	}
}

canvas.addEventListener('mousedown', (event) => {
	isDrawing = true;
	startX = event.offsetX;
	startY = event.offsetY;
});

canvas.addEventListener('mousemove', (event) => {
	if (isDrawing) {
		// Update the end coordinates
		endX = event.offsetX;
		endY = event.offsetY;

		// Redraw the image with the current rectangle
		redrawImage();

		// Draw the current rectangle
		ctx.fillStyle = 'rgba(0, 0, 255, 0.3)';
		ctx.fillRect(startX, startY, endX - startX, endY - startY);
	}
});

sections = []

canvas.addEventListener('mouseup', (event) => {

	if (isDrawing) {
		isDrawing = false;

		// Calculate the width and height of the rectangle
		const width = endX - startX;
		const height = endY - startY;

		// Add the rectangle coordinates to the rectangles array
		
		addRectangle({ x: startX, y: startY, width, height })
	}
});

const inputData = [];



function addRectangle(rect) {
	rectangles.push(rect);

	const form = document.getElementById('userInputForm');

	const formHTML = `
		<div>
			<h2>Section ${rectangles.length}</h2>
			<label for="numQuestions">Enter the number of questions in the page:</label>
			<input type="number" name="numQuestions" id="questionsNo" min="3" required>
			<br>
			<label for="choicesPerQuestion">Enter how many choices per question:</label>
			<input type="number" name="choicesPerQuestion" id="choicesNo" min="2" value="4" required>
		</div>
	`;

	const newSection = document.createElement('div');
	newSection.innerHTML = formHTML;
	sections.push(newSection)
	const brick = document.getElementById('brick')
	form.insertBefore(newSection, brick);
}
function validateFormSection(section) {
    const numQuestionsInput = section.querySelector('#questionsNo');
    const choicesPerQuestionInput = section.querySelector('#choicesNo');

    const numQuestions = parseInt(numQuestionsInput.value, 10);
    const choicesPerQuestion = parseInt(choicesPerQuestionInput.value, 10);

    if (isNaN(numQuestions) || isNaN(choicesPerQuestion)) {
        numQuestionsInput.setCustomValidity('Please enter valid numbers for numQuestions and choicesPerQuestion.');
        choicesPerQuestionInput.setCustomValidity('Please enter valid numbers for numQuestions and choicesPerQuestion.');
        return false;
    }

    // Custom validation rules, adjust as needed
    if (numQuestions < 3) {
        numQuestionsInput.setCustomValidity('Number of questions must be at least 3.');
        return false;
    } else {
        numQuestionsInput.setCustomValidity('');
    }

    if (choicesPerQuestion < 2) {
        choicesPerQuestionInput.setCustomValidity('Number of choices per question must be at least 2.');
        return false;
    } else {
        choicesPerQuestionInput.setCustomValidity('');
    }
	inputData.push({questionsNo: numQuestions, choicesNo: choicesPerQuestion})
    return true;
}

questionsNo = 0
choicesNo = 4

form.addEventListener('submit', async (event) => {
	event.preventDefault();

	for (const section of sections) {
		if (!validateFormSection(section)) return
	}

	if (rectangles.length == 0) {
		alert('Please highlight the part of the image which contains the questions.');
		return;
	}

	/* Need to return:
		filename, rectangles, rectangleLength, questionsNo, choicesNo
	*/
	const rectangleData = rectangles.map(rect => {
		return {
			x: rect.x,
			y: rect.y,
			width: rect.width,
			height: rect.height,
		};
	});

	// Create the data object with the required properties
	const data = {
		filepath: `./omr_sheets/reference/${postData.filename}`,
		rectangles: rectangleData, // Use the newly created 'rectangleData' array
		sectionData: inputData
	};

	try {
		// Send the data as JSON to the server using the fetch API
		const response = await fetch('/api/process_data', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify(data)
		});

		if (!response.ok) {
			console.error('Error:', response.status, response.statusText);
		} else {
			console.log('Data processed successfully.');
			// Display the result or perform further actions if needed
		}
	} catch (error) {
		console.error('Error:', error);
	}
});

// const getDataButton = document.getElementById('getDataButton');
// const dataDisplay = document.getElementById('dataDisplay');

// getDataButton.addEventListener('click', async () => {
// 	const response = await fetch('/api/get_data');
	
// 	if (!response.ok) {
// 		console.error('Error:', response.status, response.statusText);
// 		return;
// 	}

// 	const data = await response.json();
// 	dataDisplay.textContent = data.message;
// });

// imageCopy = null;

// const fileUploadForm = document.getElementById('fileUploadForm');
// // Event listener for form submission
// fileUploadForm.addEventListener('submit', async (event) => {
// 	event.preventDefault();

// 	const formData = new FormData(fileUploadForm);
// 	try {
// 		const response = await fetch('/api/upload_image', {
// 			method: 'POST',
// 			body: formData,
// 		});

// 		if (!response.ok) {
// 			console.error('Error:', response.status, response.statusText);
// 			return;
// 		}


// 		ctx.clearRect(0, 0, canvas.width, canvas.height);

// 		const data = await response.json();
// 		const imageUrl = "omr_sheets/reference/" + data.filename;

// 		// Create a new Image element and set its src to the image URL
// 		const image = new Image();
// 		 // Create a Promise to ensure the image is loaded
// 		const imageLoaded = new Promise((resolve, reject) => {
// 			image.onload = function() {
// 				resolve();
// 			};
// 			image.onerror = function() {
// 				reject('Error loading image.');
// 			};
// 		});

// 		// Set the src to trigger the loading of the image
// 		image.src = imageUrl;

// 		// Wait for the image to be loaded before drawing it on the canvas
// 		await imageLoaded;

// 		canvas.width = image.width;
//		 canvas.height = image.height;

// 		// Draw the image on the canvas
// 		ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
// 		console.log('Image loaded.');

// 		imageCopy = image.cloneNode()
		

// 	} catch (error) {
// 		console.error('Error:', error);
// 	}
// });

// const canvas = document.getElementById('canvas');
// const ctx = canvas.getContext('2d');
// const clearButton = document.getElementById('clearButton');
// const saveHighlightButton = document.getElementById('saveButton')



// clearButton.addEventListener('click', () => {
// 	ctx.clearRect(0,0,canvas.width,canvas.height)
// 	rectangles = []
// });
// // Store the rectangles' coordinates
// let rectangles = [];

// let isDrawing = false;
// let startX, startY, endX, endY;


// function clearCanvas() {
//	 // Clear the rectangles array and redraw the image
//	 rectangles = [];
//	 redrawImage();
// }
// // 

// function redrawImage() {
// 	 // Redraw the image on the canvas

// 	ctx.drawImage(imageCopy, 0, 0, canvas.width, canvas.height)
// 	for (const rect of rectangles) {
// 		ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
// 	}
// }

// canvas.addEventListener('mousedown', (event) => {
// 	 isDrawing = true;
// 	 startX = event.offsetX;
// 	 startY = event.offsetY;
// });

// canvas.addEventListener('mousemove', (event) => {
// 	 if (isDrawing) {
// 		 // Update the end coordinates
// 		 endX = event.offsetX;
// 		 endY = event.offsetY;

// 		 // Redraw the image with the current rectangle
// 		 redrawImage();

// 		 // Draw the current rectangle
// 		 ctx.fillStyle = 'rgba(0, 0, 255, 0.3)';
// 		 ctx.fillRect(startX, startY, endX - startX, endY - startY);
// 	 }
// });

// canvas.addEventListener('mouseup', (event) => {

// 	 if (isDrawing) {
// 		 isDrawing = false;

// 		 // Calculate the width and height of the rectangle
// 		 const width = endX - startX;
// 		 const height = endY - startY;

// 		 // Add the rectangle coordinates to the rectangles array
// 		 rectangles.push({ x: startX, y: startY, width, height });
// 	 }
// });


// saveHighlightButton.addEventListener('click', (event) => {
// 	// Make sure there are rectangles to save
// 	if (rectangles.length === 0) {
// 		console.log('No rectangles to save.');
// 		return;
// 	}

// 	// Create an array of rectangle data
// 	const rectangleData = rectangles.map(rect => {
// 		return {
// 			x: rect.x,
// 			y: rect.y,
// 			width: rect.width,
// 			height: rect.height,
// 		};
// 	});

// 	rectangleData.push({
// 		 imageHeight: imageCopy.height,
// 		 imageWidth: imageCopy.width,
// 	 });

// 	// Send the rectangle data to the Flask server using an HTTP POST request
// 	fetch('/api/save_highlight', {
// 		method: 'POST',
// 		headers: {
// 			'Content-Type': 'application/json',
// 		},
// 		body: JSON.stringify(rectangleData),
// 	})
// 	.then(response => {
// 		if (!response.ok) {
// 			console.error('Error:', response.status, response.statusText);
// 		} else {
// 			console.log('Highlight data saved successfully.');
// 		}
// 	})
// 	.catch(error => {
// 		console.error('Error:', error);
// 	});
// });