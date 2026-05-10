const API_BASE_URL = 'http://localhost:5000';

const video           = document.getElementById('video');
const canvas          = document.getElementById('canvas');
const cameraOverlay   = document.getElementById('camera-overlay');
const cameraWrapper   = document.getElementById('camera-wrapper');
const btnStart        = document.getElementById('btn-start');
const btnStop         = document.getElementById('btn-stop');
const predictionText  = document.getElementById('prediction-text');
const confidenceBar   = document.getElementById('confidence-bar');
const confidenceValue = document.getElementById('confidence-value');
const sentenceDisplay = document.getElementById('sentence-display');
const btnAddWord      = document.getElementById('btn-add-word');
const btnAddSpace     = document.getElementById('btn-add-space');
const btnDeleteWord   = document.getElementById('btn-delete-word');
const btnClear        = document.getElementById('btn-clear');
const btnSpeak        = document.getElementById('btn-speak');
const historyList     = document.getElementById('history-list');
const statusBadge     = document.getElementById('status-badge');

let stream             = null;
let isCameraActive     = false;
let predictionInterval = null;
let sentence           = '';
let recentGestures     = [];
let gestureHoldCount   = 0;
let lastGesture        = null;
let constructTimer     = null;
let isConstructing     = false;  // prevents overlap

const HOLD_THRESHOLD  = 5;
const FRAME_INTERVAL  = 200;
const MAX_HISTORY     = 15;
const CONSTRUCT_DELAY = 4000;
const SPEECH_DURATION = 5000;

document.addEventListener('DOMContentLoaded', () => {
    checkServerHealth();
    setupEventListeners();
    updateSentenceDisplay();
    createRecentMessagesBlock();
});

async function checkServerHealth() {
    try {
        const res = await fetch(`${API_BASE_URL}/health`);
        if (res.ok) {
            statusBadge.textContent = 'Online';
            statusBadge.classList.add('online');
        } else throw new Error();
    } catch {
        statusBadge.textContent = 'Offline';
        statusBadge.classList.remove('online');
    }
}

function setupEventListeners() {
    btnStart.addEventListener('click', startCamera);
    btnStop.addEventListener('click', stopCamera);
    btnAddWord.addEventListener('click', addCurrentWord);
    btnAddSpace.addEventListener('click', addSpace);
    btnDeleteWord.addEventListener('click', deleteLastWord);
    btnClear.addEventListener('click', clearSentence);
    btnSpeak.addEventListener('click', speakSentence);
}

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
            audio: false
        });
        video.srcObject = stream;
        await new Promise(resolve => { video.onloadedmetadata = () => resolve(); });
        await video.play();
        isCameraActive = true;
        cameraOverlay.classList.add('hidden');
        cameraWrapper.classList.add('active');
        btnStart.disabled = true;
        btnStop.disabled  = false;
        predictionInterval = setInterval(captureAndPredict, FRAME_INTERVAL);
    } catch {
        cameraOverlay.innerHTML = `
            <div class="overlay-content">
                <p>Error accessing camera.</p>
                <p style="font-size:0.9rem;margin-top:0.5rem;">Please allow camera permissions.</p>
            </div>`;
    }
}

function stopCamera() {
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    isCameraActive = false;
    if (predictionInterval) { clearInterval(predictionInterval); predictionInterval = null; }
    if (constructTimer)     { clearTimeout(constructTimer);      constructTimer     = null; }
    gestureHoldCount = 0;
    lastGesture      = null;
    isConstructing   = false;
    cameraOverlay.classList.remove('hidden');
    cameraWrapper.classList.remove('active');
    btnStart.disabled           = false;
    btnStop.disabled            = true;
    predictionText.textContent  = '—';
    confidenceBar.style.width   = '0%';
    confidenceValue.textContent = '0%';
}

async function captureAndPredict() {
    if (!isCameraActive || !stream || video.paused || video.ended) return;
    try {
        canvas.width  = video.videoWidth  || 640;
        canvas.height = video.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageData = canvas.toDataURL('image/jpeg', 0.8);
        const response  = await fetch(`${API_BASE_URL}/predict`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ frame: imageData })
        });
        if (!response.ok) throw new Error();
        updatePrediction(await response.json());
    } catch { /* silent fail */ }
}

function updatePrediction(data) {
    const gesture    = data.gesture    || '—';
    const confidence = data.confidence || 0;

    predictionText.textContent  = gesture;
    confidenceBar.style.width   = `${confidence * 100}%`;
    confidenceValue.textContent = `${Math.round(confidence * 100)}%`;

    if (gesture === '—' || gesture === 'None' || gesture === 'No_Gesture') {
        gestureHoldCount = 0;
        return;
    }

    if (gesture !== lastGesture) {
        gestureHoldCount = 1;
        lastGesture      = gesture;
    } else {
        gestureHoldCount++;
        if (gestureHoldCount === HOLD_THRESHOLD) {
            addToHistory(gesture);
        }
    }
}

function addToHistory(gesture) {
    if (gesture === 'No_Gesture') return;
    if (recentGestures.length > 0 && recentGestures[0] === gesture) return;

    // If currently constructing/speaking don't clear — just accumulate
    recentGestures.unshift(gesture);
    if (recentGestures.length > MAX_HISTORY) {
        recentGestures = recentGestures.slice(0, MAX_HISTORY);
    }
    renderHistory();

    // Only start timer if not already constructing
    if (!isConstructing) {
        if (constructTimer) clearTimeout(constructTimer);
        constructTimer = setTimeout(autoConstructSentence, CONSTRUCT_DELAY);
    }
}

async function autoConstructSentence() {
    const words = [...recentGestures].reverse().filter(g => g !== 'No_Gesture');
    if (words.length === 0) return;

    isConstructing = true;

    try {
        const res  = await fetch(`${API_BASE_URL}/construct`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ words: words })
        });
        const data = await res.json();

        if (data.sentence) {
            // Show in main display
            sentence = data.sentence;
            updateSentenceDisplay();

            // After speech duration move to recent messages
            setTimeout(() => {
                addToRecentMessages(data.sentence);

                // Clear display and gestures
                sentence       = '';
                recentGestures = [];
                updateSentenceDisplay();
                renderHistory();
                isConstructing = false;

                // If new gestures came in during speech, start timer again
                if (recentGestures.length > 0) {
                    constructTimer = setTimeout(autoConstructSentence, CONSTRUCT_DELAY);
                }
            }, SPEECH_DURATION);
        }
    } catch (err) {
        console.error('Auto construct error:', err);
        isConstructing = false;
    }
}

function createRecentMessagesBlock() {
    const historyBox    = document.querySelector('.history-box');
    const block         = document.createElement('div');
    block.id            = 'recent-messages-block';
    block.style.cssText = `
        margin-top:1rem;padding:1.25rem;
        background:rgba(0,0,0,0.2);
        border:1px solid rgba(108,99,255,0.2);
        border-radius:12px;display:none;`;
    block.innerHTML = `
        <span style="font-size:0.75rem;color:var(--text-muted);
            text-transform:uppercase;letter-spacing:1px;
            display:block;margin-bottom:0.75rem;">Recent messages</span>
        <div id="recent-messages-list"></div>`;
    historyBox.after(block);
}

function addToRecentMessages(text) {
    const block = document.getElementById('recent-messages-block');
    const list  = document.getElementById('recent-messages-list');
    block.style.display = 'block';
    const msg = document.createElement('div');
    msg.style.cssText = `
        padding:0.75rem 1rem;
        background:rgba(108,99,255,0.1);
        border:1px solid rgba(108,99,255,0.3);
        border-radius:8px;margin-bottom:0.5rem;
        font-size:0.95rem;color:var(--text-primary);
        line-height:1.5;animation:fadeIn 0.4s ease;`;
    msg.textContent = text;
    list.prepend(msg);
    while (list.children.length > 5) list.removeChild(list.lastChild);
}

function renderHistory() {
    historyList.innerHTML = '';
    recentGestures.forEach((gesture, i) => {
        const tag               = document.createElement('span');
        tag.className           = 'history-tag';
        tag.textContent         = gesture;
        tag.style.animationDelay = `${i * 0.05}s`;
        historyList.appendChild(tag);
    });
}

function addCurrentWord() {
    const w = predictionText.textContent.trim();
    if (w && w !== '—' && w !== 'None' && w !== 'No_Gesture') {
        sentence += (sentence ? ' ' : '') + w;
        updateSentenceDisplay();
    }
}

function addSpace()       { sentence += ' '; updateSentenceDisplay(); }
function clearSentence()  { sentence  = '';  updateSentenceDisplay(); }

function deleteLastWord() {
    sentence = sentence.trimEnd();
    const i  = sentence.lastIndexOf(' ');
    sentence = i === -1 ? '' : sentence.substring(0, i);
    updateSentenceDisplay();
}

function updateSentenceDisplay() {
    if (sentence.trim() === '') {
        sentenceDisplay.innerHTML = '<span class="placeholder-text">Your sentence will appear here...</span>';
    } else {
        sentenceDisplay.textContent = sentence
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
    }
}

async function speakSentence() {
    const text = sentence.trim();
    if (!text) return;
    try {
        btnSpeak.disabled    = true;
        btnSpeak.textContent = 'Speaking...';
        const res = await fetch(`${API_BASE_URL}/speak`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ text })
        });
        if (!res.ok) throw new Error();
    } catch {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
        }
    } finally {
        btnSpeak.disabled  = false;
        btnSpeak.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
                <path d="M15.54 8.46 Q19 12 15.54 15.54"/>
                <path d="M19.07 4.93 Q24 12 19.07 19.07"/>
            </svg>
            Speak Sentence`;
    }
}

window.addEventListener('beforeunload', () => stopCamera());
document.addEventListener('visibilitychange', () => {
    if (document.hidden && isCameraActive) console.log('Tab hidden');
}); 