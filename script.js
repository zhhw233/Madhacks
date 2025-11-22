// Get DOM elements first (after DOM loads)
const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const audioPlayback = document.getElementById("audioPlayback");
const predictButton = document.getElementById("predictButton");
const predictionDiv = document.getElementById("prediction");

// Add stop button handler
stopButton.addEventListener("click", () => {
	mediaRecorder.stop();
	mediaRecorder.stream.getTracks().forEach((track) => track.stop()); // Stop mic access
	startButton.disabled = false;
	stopButton.disabled = true;
});

let mediaRecorder;
let audioChunks = [];
let recordedBlob = null;



startButton.addEventListener("click", async () => {
	try {
		const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
		mediaRecorder = new MediaRecorder(stream);
		mediaRecorder.start();

        const MAX_DURATION = 20000;
        stopTimeout = setTimeout(() => {
            if (mediaRecorder.state === "recording") {
                mediaRecorder.stop();
            }
        }, MAX_DURATION);

		mediaRecorder.ondataavailable = (event) => {
			audioChunks.push(event.data);
		};

		mediaRecorder.onstop = () => {
            clearTimeout(stopTimeout);

			const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
			const audioUrl = URL.createObjectURL(audioBlob);
			audioPlayback.src = audioUrl;
			recordedBlob = audioBlob; // Save for prediction
			predictButton.disabled = false; // Enable predict button
			audioChunks = []; // Clear chunks for next recording
		};

		startButton.disabled = true;
		stopButton.disabled = false;
	} catch (err) {
		console.error("Error accessing microphone:", err);
	}
});

// Predict accent button handler
predictButton.addEventListener("click", async () => {
	if (!recordedBlob) {
		predictionDiv.textContent = "No recording available";
		return;
	}

	predictionDiv.textContent = "Analyzing...";
	predictButton.disabled = true;

	try {
		// Create FormData to send audio file
		const formData = new FormData();
		formData.append("audio", recordedBlob, "recording.webm");

		// Send to your backend API
		const response = await fetch("/predict", {
			method: "POST",
			body: formData,
		});

		const result = await response.json();

		// Display prediction
		predictionDiv.innerHTML = `
                    <h3>Prediction Results:</h3>
                    <p>Accent: ${result.accent}</p>
                    <p>Age: ${result.age}</p>
                    <p>Confidence: ${result.confidence}%</p>
                `;
	} catch (err) {
		console.error("Error predicting:", err);
		predictionDiv.textContent = "Error making prediction. Please try again.";
	} finally {
		predictButton.disabled = false;
	}
});
