/**
 * SignBridge AI - Frontend Application
 * Real-time sign language recognition with sentence building and TTS
 */

const API_BASE_URL = 'http://localhost:5000';

// DOM Elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const cameraOverlay = document.getElementById('camera-overlay');
const cameraWrapper = document.getElementById('camera-wrapper');
const btnStart = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');
const predictionText = document.getElementById('prediction-text');
const confidenceBar = document.getElementById('confidence-bar');
const confidenceValue = document.getElementById('confidence-value');
const sentenceDisplay = document.getElementById('sentence-display');
const btnAddWord = document.getElementById('btn-add-word');
const btnAddSpace = document.getElementById('btn-add-space');
const btnDeleteWord = document.getElementById('btn-delete-word');
const btnClear = document.getElementById('btn-clear');
const btnSpeak = document.getElementById('btn-speak');
const historyList = document.getElementById('history-list');
const statusBadge = document.getElementById('status-badge');

// Application State
let stream = null;
let isCameraActive = false;
let predictionInterval = null;
let sentence = '';
let recentGestures = [];
let gestureHoldCount = 0;
let lastGesture = null;
const HOLD_THRESHOLD = 8;
const FRAME_INTERVAL = 300;
const MAX_HISTORY = 10;

/**
 * Initialize the application
 */
document.addEventListener('DOMContentLoaded', () => {
    checkServerHealth();
    setupEventListeners();
    updateSentenceDisplay();
});

/**
 * Check server health status
 */
async function checkServerHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json'
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            statusBadge.textContent = 'Online';
            statusBadge.classList.add('online');
            console.log('Server health:', data);
        } else {
            throw new Error('Server unhealthy');
        }
    } catch (error) {
        console.error('Server health check failed:', error);
        statusBadge.textContent = 'Offline';
        statusBadge.classList.remove('online');
    }
}

/**
 * Setup all event listeners
 */
function setupEventListeners() {
    btnStart.addEventListener('click', startCamera);
    btnStop.addEventListener('click', stopCamera);
    btnAddWord.addEventListener('click', addCurrentWord);
    btnAddSpace.addEventListener('click', addSpace);
    btnDeleteWord.addEventListener('click', deleteLastWord);
    btnClear.addEventListener('click', clearSentence);
    btnSpeak.addEventListener('click', speakSentence);
}

/**
 * Start the camera and begin predictions
 */
async function startCamera() {
    try {
        // Request camera access
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: false
        });
        
        // Set video source
        video.srcObject = stream;
        
        // Wait for video to be ready
        await new Promise((resolve) => {
            video.onloadedmetadata = () => {
                resolve();
            };
        });
        
        await video.play();
        isCameraActive = true;
        
        // Update UI
        cameraOverlay.classList.add('hidden');
        cameraWrapper.classList.add('active');
        btnStart.disabled = true;
        btnStop.disabled = false;
        
        // Start prediction loop
        predictionInterval = setInterval(captureAndPredict, FRAME_INTERVAL);
        
        console.log('Camera started successfully');
        
    } catch (error) {
        console.error('Error accessing camera:', error);
        cameraOverlay.innerHTML = `
            <div class="overlay-content">
                <p>Error accessing camera.</p>
                <p style="font-size: 0.9rem; margin-top: 0.5rem;">Please allow camera permissions.</p>
            </div>
        `;
    }
}

/**
 * Stop the camera and predictions
 */
function stopCamera() {
    // Stop all tracks
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    isCameraActive = false;
    
    // Clear prediction interval
    if (predictionInterval) {
        clearInterval(predictionInterval);
        predictionInterval = null;
    }
    
    // Reset hold logic
    gestureHoldCount = 0;
    lastGesture = null;
    
    // Update UI
    cameraOverlay.classList.remove('hidden');
    cameraWrapper.classList.remove('active');
    btnStart.disabled = false;
    btnStop.disabled = true;
    
    // Reset prediction display
    predictionText.textContent = '—';
    confidenceBar.style.width = '0%';
    confidenceValue.textContent = '0%';
    
    console.log('Camera stopped');
}

/**
 * Capture frame and send to server for prediction
 */
async function captureAndPredict() {
    if (!isCameraActive || !stream || video.paused || video.ended) {
        return;
    }
    
    try {
        // Set canvas dimensions to match video
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        
        // Draw video frame to canvas
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Convert to base64 JPEG
        const imageData = canvas.toDataURL('image/jpeg', 0.8);
        
        // Send to server
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({ image: imageData })
        });
        
        if (!response.ok) {
            throw new Error(`Prediction failed: ${response.status}`);
        }
        
        const data = await response.json();
        updatePrediction(data);
        
    } catch (error) {
        console.error('Prediction error:', error);
    }
}

/**
 * Update prediction display and handle gesture hold logic
 */
function updatePrediction(data) {
    const gesture = data.gesture || data.prediction || '—';
    const confidence = data.confidence || 0;
    
    // Update display
    predictionText.textContent = gesture;
    confidenceBar.style.width = `${confidence}%`;
    confidenceValue.textContent = `${Math.round(confidence)}%`;
    
    // Handle gesture hold logic
    if (gesture !== '—' && gesture !== 'None' && gesture !== lastGesture) {
        // New gesture detected
        gestureHoldCount = 1;
        lastGesture = gesture;
    } else if (gesture === lastGesture && gesture !== '—' && gesture !== 'None') {
        // Same gesture continuing
        gestureHoldCount++;
        
        // If held for threshold frames, add to history
        if (gestureHoldCount === HOLD_THRESHOLD) {
            addToHistory(gesture);
        }
    } else if (gesture === '—' || gesture === 'None') {
        // No gesture detected
        gestureHoldCount = 0;
        lastGesture = null;
    }
}

/**
 * Add gesture to recent history
 */
function addToHistory(gesture) {
    // Avoid duplicates at the start
    if (recentGestures.length > 0 && recentGestures[0] === gesture) {
        return;
    }
    
    // Add to beginning of array
    recentGestures.unshift(gesture);
    
    // Keep only max items
    if (recentGestures.length > MAX_HISTORY) {
        recentGestures = recentGestures.slice(0, MAX_HISTORY);
    }
    
    // Update display
    renderHistory();
}

/**
 * Render history tags
 */
function renderHistory() {
    historyList.innerHTML = '';
    
    recentGestures.forEach((gesture, index) => {
        const tag = document.createElement('span');
        tag.className = 'history-tag';
        tag.textContent = gesture;
        tag.style.animationDelay = `${index * 0.05}s`;
        historyList.appendChild(tag);
    });
}

/**
 * Add current prediction to sentence
 */
function addCurrentWord() {
    const currentWord = predictionText.textContent.trim();
    if (currentWord && currentWord !== '—' && currentWord !== 'None') {
        sentence += currentWord;
        updateSentenceDisplay();
        
        // Visual feedback
        sentenceDisplay.style.animation = 'none';
        setTimeout(() => {
            sentenceDisplay.style.animation = '';
        }, 10);
    }
}

/**
 * Add space to sentence
 */
function addSpace() {
    sentence += ' ';
    updateSentenceDisplay();
}

/**
 * Delete last word from sentence
 */
function deleteLastWord() {
    // Remove trailing spaces first
    sentence = sentence.trimEnd();
    
    // Find last word and remove it
    const lastSpaceIndex = sentence.lastIndexOf(' ');
    if (lastSpaceIndex === -1) {
        sentence = '';
    } else {
        sentence = sentence.substring(0, lastSpaceIndex);
    }
    
    updateSentenceDisplay();
}

/**
 * Clear entire sentence
 */
function clearSentence() {
    sentence = '';
    updateSentenceDisplay();
}

/**
 * Update sentence display
 */
function updateSentenceDisplay() {
    if (sentence.trim() === '') {
        sentenceDisplay.innerHTML = '<span class="placeholder-text">Your sentence will appear here...</span>';
    } else {
        // Escape HTML to prevent XSS
        const escapedSentence = sentence
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
        sentenceDisplay.textContent = escapedSentence;
    }
}

/**
 * Send sentence to TTS endpoint
 */
async function speakSentence() {
    const textToSpeak = sentence.trim();
    
    if (!textToSpeak) {
        console.log('No text to speak');
        return;
    }
    
    try {
        // Disable button and show loading state
        btnSpeak.disabled = true;
        const originalText = btnSpeak.innerHTML;
        btnSpeak.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10" stroke-dasharray="60" stroke-dashoffset="20">
                    <animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="1s" repeatCount="indefinite"/>
                </circle>
            </svg>
            Speaking...
        `;
        
        const response = await fetch(`${API_BASE_URL}/speak`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({ text: textToSpeak })
        });
        
        if (!response.ok) {
            throw new Error(`TTS failed: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('TTS response:', data);
        
    } catch (error) {
        console.error('TTS error:', error);
        // Fallback to browser TTS if server fails
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(textToSpeak);
            utterance.rate = 1;
            utterance.pitch = 1;
            window.speechSynthesis.speak(utterance);
        }
    } finally {
        // Restore button
        btnSpeak.disabled = false;
        btnSpeak.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
                <path d="M15.54 8.46 Q19 12 15.54 15.54"/>
                <path d="M19.07 4.93 Q24 12 19.07 19.07"/>
            </svg>
            Speak Sentence
        `;
    }
}

/**
 * Handle page unload - cleanup
 */
window.addEventListener('beforeunload', () => {
    stopCamera();
});

/**
 * Handle visibility change - pause/resume camera
 */
document.addEventListener('visibilitychange', () => {
    if (document.hidden && isCameraActive) {
        // Optionally pause predictions when tab is hidden
        console.log('Tab hidden, predictions paused');
    }
});
