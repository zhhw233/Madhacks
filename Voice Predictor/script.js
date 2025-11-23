// Get DOM elements first (after DOM loads)
const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const audioPlayback = document.getElementById("audioPlayback");
const predictButton = document.getElementById("predictButton");
const predictionDiv = document.getElementById("prediction");
const countdownDisplay = document.getElementById("countdown");

let mediaRecorder;
let audioChunks = [];
let recordedBlob = null;
let stopTimeout;
let countdownInterval;
let timeLeft = 30;
const TARGET_SR = 22050;

/* DNA PARTICLES — Soft Orb Helix */
const dnaContainer = document.getElementById("dnaParticles");

function createOrb(xOffset, delay) {
    const orb = document.createElement("div");
    orb.classList.add("dna-orb");

    // Horizontal helix center
    const baseX = window.innerWidth / 2;
    orb.style.left = (baseX + xOffset) + "px";

    // Start at random vertical positions BELOW the screen
    const startY = window.innerHeight + Math.random() * 300;
    orb.style.top = startY + "px";

    // Random orb size variation
    const size = Math.random() * 10 + 10;
    orb.style.width = size + "px";
    orb.style.height = size + "px";

    // Vary the animation duration a bit
    const duration = Math.round(Math.random() * 4 + 6);
    orb.style.animationDuration = duration + "s";
    orb.style.animationDelay = delay + "s";

    dnaContainer.appendChild(orb);

    // Cleanup
    setTimeout(() => orb.remove(), duration * 1000 + 2000);
}


// Generate strands continuously
setInterval(() => {
    const t = Date.now() / 600;  // smooth phase offset

    // Helix offset: sin wave creates the twist
    const leftX = Math.sin(t) * 60 - 120;
    const rightX = Math.cos(t) * 60 + 120;

    createOrb(leftX, Math.random() * 1);
    createOrb(rightX, Math.random() * 1);
}, 300);


// Stop button handler
stopButton.addEventListener("click", () => {
	if (mediaRecorder && mediaRecorder.state === "recording") {
		mediaRecorder.stop();
		mediaRecorder.stream.getTracks().forEach((track) => track.stop());
	}

	clearTimeout(stopTimeout);
	clearInterval(countdownInterval);

	if (countdownDisplay) {
		countdownDisplay.textContent = "Stopped";
		countdownDisplay.style.color = "white";

	}

	startButton.disabled = false;
	stopButton.disabled = true;
});

// Start button handler
startButton.addEventListener("click", async () => {
	try {
		const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
		mediaRecorder = new MediaRecorder(stream);
		audioChunks = [];
		timeLeft = 30;

		mediaRecorder.start();

		// Update countdown display
		if (countdownDisplay) {
			countdownDisplay.textContent = `${timeLeft}s`;
			countdownDisplay.style.color = "white";

		}

		// Auto-stop after 30 seconds
		const MAX_DURATION = 30000;
		stopTimeout = setTimeout(() => {
			if (mediaRecorder.state === "recording") {
				mediaRecorder.stop();
				mediaRecorder.stream.getTracks().forEach((track) => track.stop());
			}
		}, MAX_DURATION);

		// Start countdown timer
		countdownInterval = setInterval(() => {
			timeLeft--;
			if (timeLeft <= 0) {
				clearInterval(countdownInterval);
				if (countdownDisplay) {
					countdownDisplay.textContent = "0s";
					countdownDisplay.style.color = "white";

				}
			} else {
				if (countdownDisplay) {
					countdownDisplay.textContent = `${timeLeft}s`;
					countdownDisplay.style.color = "white";

				}
			}
		}, 1000);

		mediaRecorder.ondataavailable = (event) => {
			audioChunks.push(event.data);
		};

		mediaRecorder.onstop = () => {
			clearTimeout(stopTimeout);
			clearInterval(countdownInterval);

			if (countdownDisplay) {
				countdownDisplay.textContent = "Done";
				countdownDisplay.style.color = "white";

			}

			const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
			const audioUrl = URL.createObjectURL(audioBlob);
			audioPlayback.src = audioUrl;
			recordedBlob = audioBlob;
			predictButton.disabled = false;
			audioChunks = [];

			startButton.disabled = false;
			stopButton.disabled = true;
		};

		startButton.disabled = true;
		stopButton.disabled = false;
	} catch (err) {
		console.error("Error accessing microphone:", err);
		alert("Error accessing microphone. Please allow microphone access.");
	}
});

// Predict accent button handler
predictButton.addEventListener("click", async () => {
	if (!recordedBlob) {
		predictionDiv.textContent = "No recording available";
		predictionDiv.textContent.style = "white";
		
		return;
	}

	predictionDiv.textContent = "Analyzing...";
	predictionDiv.textContent.style = "white";
	predictButton.disabled = true;

	try {
		// Convert recorded WebM/Opus blob to WAV (PCM16, mono, TARGET_SR) in-browser
		async function convertBlobToWav(blob) {
			const arrayBuffer = await blob.arrayBuffer();
			const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
			const decoded = await audioCtx.decodeAudioData(arrayBuffer.slice(0));

			// Resample using OfflineAudioContext to TARGET_SR
			const offlineCtx = new (window.OfflineAudioContext || window.webkitOfflineAudioContext)(decoded.numberOfChannels, Math.ceil(decoded.duration * TARGET_SR), TARGET_SR);
			const src = offlineCtx.createBufferSource();
			src.buffer = decoded;
			src.connect(offlineCtx.destination);
			src.start(0);
			const rendered = await offlineCtx.startRendering();

			// PCM16 WAV encoding
			function encodeWAV(buf) {
				const numChannels = buf.numberOfChannels;
				const sampleRate = buf.sampleRate;
				const samples = buf.length;
				const bytesPerSample = 2;
				const blockAlign = numChannels * bytesPerSample;
				const buffer = new ArrayBuffer(44 + samples * blockAlign);
				const view = new DataView(buffer);

				function writeString(view, offset, string) {
					for (let i = 0; i < string.length; i++) {
						view.setUint8(offset + i, string.charCodeAt(i));
					}
				}

				/* RIFF identifier */
				writeString(view, 0, 'RIFF');
				/* file length */
				view.setUint32(4, 36 + samples * blockAlign, true);
				/* RIFF type */
				writeString(view, 8, 'WAVE');
				/* format chunk identifier */
				writeString(view, 12, 'fmt ');
				/* format chunk length */
				view.setUint32(16, 16, true);
				/* sample format (raw) */
				view.setUint16(20, 1, true);
				/* channel count */
				view.setUint16(22, numChannels, true);
				/* sample rate */
				view.setUint32(24, sampleRate, true);
				/* byte rate (sampleRate * blockAlign) */
				view.setUint32(28, sampleRate * blockAlign, true);
				/* block align (channel count * bytesPerSample) */
				view.setUint16(32, blockAlign, true);
				/* bits per sample */
				view.setUint16(34, 16, true);
				/* data chunk identifier */
				writeString(view, 36, 'data');
				/* data chunk length */
				view.setUint32(40, samples * blockAlign, true);

				// Write interleaved PCM samples
				let offset = 44;
				const channelData = [];
				for (let ch = 0; ch < numChannels; ch++) channelData.push(buf.getChannelData(ch));

				for (let i = 0; i < samples; i++) {
					for (let ch = 0; ch < numChannels; ch++) {
						let sample = channelData[ch][i];
						// clamp
						sample = Math.max(-1, Math.min(1, sample));
						view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
						offset += 2;
					}
				}
				return buffer;
			}

			const wavBuffer = encodeWAV(rendered);
			return new Blob([wavBuffer], { type: 'audio/wav' });
		}

		const wavBlob = await convertBlobToWav(recordedBlob);

		// Create FormData to send WAV audio file
		const formData = new FormData();
		formData.append("audio", wavBlob, "recording.wav");

		// Send to backend API
		const response = await fetch("http://localhost:5001/predict", {
			method: "POST",
			body: formData,
		});

		const result = await response.json();

		// Display prediction with individual confidence scores
		predictionDiv.innerHTML = `
			<h3>Prediction Results:</h3>
			<p>Accent: ${result.accent} (${result.accent_confidence}%)</p>
			<p>Age: ${result.age} (${result.age_confidence}%)</p>
			<p>Sex: ${result.sex} (${result.sex_confidence}%)</p>
		`;
	} catch (err) {
		console.error("Error predicting:", err);
		predictionDiv.textContent = "Error making prediction. Please try again.";
		predictionDiv.textContent.style = "white";
	} finally {
		predictButton.disabled = false;
	}
});
