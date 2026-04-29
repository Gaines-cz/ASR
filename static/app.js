/**
 * AutoGLM ASR Web Client - Frontend Application
 */

(function() {
    'use strict';

    // VAD Configuration
    const VAD_STATE = {
        IDLE: 'idle',
        SPEECH: 'speech',
        SILENCE: 'silence'
    };

    const VAD_CONFIG = {
        ENERGY_THRESHOLD: 0.01,
        SPEECH_END_FRAMES: 15,      // ~750ms silence to confirm speech end
        MIN_SPEECH_MS: 300,         // minimum speech duration
        WINDOW_SIZE_MS: 3000,        // 3 second window
        OVERLAP_MS: 1500,            // 50% overlap
        SAMPLE_RATE: 16000
    };

    // Sliding Window Buffer Class
    class SlidingWindowBuffer {
        constructor(options = {}) {
            this.windowSizeMs = options.windowSizeMs || VAD_CONFIG.WINDOW_SIZE_MS;
            this.overlapMs = options.overlapMs || VAD_CONFIG.OVERLAP_MS;
            this.sampleRate = options.sampleRate || VAD_CONFIG.SAMPLE_RATE;
            this.samples = [];
            this.windowSamples = Math.floor(this.windowSizeMs * this.sampleRate / 1000);
            this.overlapSamples = Math.floor(this.overlapMs * this.sampleRate / 1000);
        }

        push(newSamples) {
            this.samples.push(...newSamples);
        }

        getBufferDurationMs() {
            return (this.samples.length / this.sampleRate) * 1000;
        }

        shouldSend() {
            return this.getBufferDurationMs() >= this.windowSizeMs;
        }

        getSendWindow() {
            if (this.samples.length < this.windowSamples) {
                return null;
            }

            // Get the latest window
            const window = this.samples.slice(-this.windowSamples);

            // Keep overlap samples for next window
            this.samples = window.slice(-this.overlapSamples);

            return new Float32Array(window);
        }

        clear() {
            this.samples = [];
        }
    }

    // State management
    const state = {
        mode: 'idle', // idle, recording, processing
        isRecording: false,
        sessionId: null,
        chunkIndex: 0,
        mediaStream: null,
        audioContext: null,
        processor: null,
        audioChunks: [],
        processingPromise: Promise.resolve(),
        fileTranscriptionResult: null,  // Store file transcription result

        // VAD state
        vadState: VAD_STATE.IDLE,
        silenceFrames: 0,
        speechStartTime: 0,
        slidingBuffer: null
    };

    // DOM Elements
    const elements = {
        // File transcription
        fileInput: document.getElementById('file-input'),
        transcribeBtn: document.getElementById('transcribe-btn'),
        fileResult: document.getElementById('file-result'),
        exportFileBtn: document.getElementById('export-file-btn'),

        // Recording controls
        startRecordingBtn: document.getElementById('start-recording-btn'),
        stopRecordingBtn: document.getElementById('stop-recording-btn'),
        recordingStatus: document.getElementById('recording-status'),
        sessionInfo: document.getElementById('session-info'),
        sessionId: document.getElementById('session-id'),

        // Transcript displays
        liveTranscript: document.getElementById('live-transcript'),
        finalTranscript: document.getElementById('final-transcript'),
        exportFinalBtn: document.getElementById('export-final-btn'),

        // Error display
        errorSection: document.getElementById('error-section'),
        errorMessage: document.getElementById('error-message')
    };

    // ============================================
    // Utility Functions
    // ============================================

    function showError(message, errorCode) {
        elements.errorSection.classList.remove('hidden');
        elements.errorMessage.textContent = errorCode
            ? `[${errorCode}] ${message}`
            : message;
    }

    function hideError() {
        elements.errorSection.classList.add('hidden');
        elements.errorMessage.textContent = '';
    }

    function showElement(el) {
        el.classList.remove('hidden');
    }

    function hideElement(el) {
        el.classList.add('hidden');
    }

    function setPlaceholder(el, text) {
        el.innerHTML = `<p class="placeholder">${text}</p>`;
    }

    function appendToLiveTranscript(text) {
        const placeholder = elements.liveTranscript.querySelector('.placeholder');
        if (placeholder) {
            placeholder.remove();
        }

        const p = document.createElement('p');
        p.textContent = text;
        p.className = 'chunk-text';
        elements.liveTranscript.appendChild(p);
        elements.liveTranscript.scrollTop = elements.liveTranscript.scrollHeight;
    }

    function setFinalTranscript(text) {
        elements.finalTranscript.textContent = text;
        showElement(elements.exportFinalBtn);
    }

    function setButtonState(btn, disabled) {
        btn.disabled = disabled;
    }

    // ============================================
    // API Functions
    // ============================================

    async function transcribeFile(file, prompt = null) {
        const formData = new FormData();
        formData.append('file', file);
        if (prompt) {
            formData.append('prompt', prompt);
        }

        const response = await fetch('/api/transcribe/file', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Transcription failed');
        }

        return data.data;
    }

    async function createSession() {
        const response = await fetch('/api/transcribe/session', {
            method: 'POST'
        });

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Failed to create session');
        }

        return data.data.session_id;
    }

    async function transcribeChunk(sessionId, chunkIndex, audioBlob, mimeType, windowSizeMs, overlapMs) {
        const formData = new FormData();
        formData.append('session_id', sessionId);
        formData.append('chunk_index', chunkIndex);
        formData.append('file', audioBlob, 'chunk.wav');
        formData.append('mime_type', mimeType);
        formData.append('window_size_ms', windowSizeMs);
        formData.append('overlap_ms', overlapMs);

        const response = await fetch('/api/transcribe/chunk', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Chunk transcription failed');
        }

        return data.data;
    }

    async function finalizeSession(sessionId) {
        const formData = new FormData();
        formData.append('session_id', sessionId);

        const response = await fetch('/api/transcribe/finalize', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Finalization failed');
        }

        return data.data;
    }

    // ============================================
    // File Transcription
    // ============================================

    async function handleFileTranscription() {
        const file = elements.fileInput.files[0];

        if (!file) {
            showError('请选择音频文件', 'NO_FILE');
            return;
        }

        hideError();
        setButtonState(elements.transcribeBtn, true);
        elements.transcribeBtn.textContent = '转译中...';
        hideElement(elements.fileResult);
        hideElement(elements.exportFileBtn);

        try {
            const result = await transcribeFile(file);
            state.fileTranscriptionResult = result.transcript;
            showElement(elements.fileResult);
            elements.fileResult.textContent = result.transcript;
            showElement(elements.exportFileBtn);
        } catch (error) {
            showError(error.message, 'TRANSCRIBE_ERROR');
        } finally {
            setButtonState(elements.transcribeBtn, false);
            elements.transcribeBtn.textContent = '开始';
        }
    }

    // ============================================
    // Recording Functions
    // ============================================

    async function startRecording() {
        try {
            // Request microphone permission
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

            // Create session
            state.sessionId = await createSession();
            state.chunkIndex = 0;
            state.audioChunks = [];

            // Initialize VAD state
            state.vadState = VAD_STATE.IDLE;
            state.silenceFrames = 0;
            state.speechStartTime = 0;
            state.slidingBuffer = new SlidingWindowBuffer({
                windowSizeMs: VAD_CONFIG.WINDOW_SIZE_MS,
                overlapMs: VAD_CONFIG.OVERLAP_MS,
                sampleRate: VAD_CONFIG.SAMPLE_RATE
            });

            // Show session info
            elements.sessionId.textContent = state.sessionId;
            showElement(elements.sessionInfo);

            // Use Web Audio API to capture raw PCM for better real-time processing
            const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: VAD_CONFIG.SAMPLE_RATE });
            await audioContext.resume(); // Browsers start suspended
            const source = audioContext.createMediaStreamSource(stream);
            const processor = audioContext.createScriptProcessor(4096, 1, 1);

            // Store audio context, processor, and stream for cleanup
            state.audioContext = audioContext;
            state.processor = processor;
            state.mediaStream = stream;
            state.isRecording = true;
            state.mode = 'recording';

            console.log(`[VAD] Recording started with window=${VAD_CONFIG.WINDOW_SIZE_MS}ms, overlap=${VAD_CONFIG.OVERLAP_MS}ms`);

            processor.onaudioprocess = (event) => {
                if (!state.isRecording) return;

                const inputBuffer = event.inputBuffer;
                const samples = inputBuffer.getChannelData(0);

                // VAD: Calculate energy
                let energy = 0;
                for (let i = 0; i < samples.length; i++) {
                    energy += samples[i] * samples[i];
                }
                energy = Math.sqrt(energy / samples.length);

                const isSpeech = energy >= VAD_CONFIG.ENERGY_THRESHOLD;

                // VAD State Machine
                switch (state.vadState) {
                    case VAD_STATE.IDLE:
                        if (isSpeech) {
                            state.vadState = VAD_STATE.SPEECH;
                            state.speechStartTime = Date.now();
                            state.slidingBuffer.clear();
                            console.log('[VAD] Speech started, energy:', energy.toFixed(4));
                        }
                        // Always accumulate samples during IDLE to build up buffer
                        state.slidingBuffer.push(Array.from(samples));
                        break;

                    case VAD_STATE.SPEECH:
                        // Add samples to sliding buffer
                        state.slidingBuffer.push(Array.from(samples));

                        if (isSpeech) {
                            state.silenceFrames = 0;
                        } else {
                            state.silenceFrames++;
                            // Check if silence duration exceeds threshold
                            if (state.silenceFrames >= VAD_CONFIG.SPEECH_END_FRAMES) {
                                const speechDuration = Date.now() - state.speechStartTime;
                                if (speechDuration >= VAD_CONFIG.MIN_SPEECH_MS) {
                                    console.log('[VAD] Speech ended, duration:', speechDuration, 'ms');
                                    state.vadState = VAD_STATE.SILENCE;
                                    state.silenceFrames = 0;
                                    // Send any remaining audio in buffer
                                    sendBufferedAudio();
                                } else {
                                    console.log('[VAD] Speech too short, ignoring, duration:', speechDuration, 'ms');
                                    state.vadState = VAD_STATE.IDLE;
                                    state.slidingBuffer.clear();
                                }
                            }
                        }
                        break;

                    case VAD_STATE.SILENCE:
                        // Continue adding samples but check if speech resumes
                        state.slidingBuffer.push(Array.from(samples));

                        if (isSpeech) {
                            console.log('[VAD] Speech resumed');
                            state.vadState = VAD_STATE.SPEECH;
                            state.silenceFrames = 0;
                        }
                        break;
                }

                // Check if sliding buffer is ready to send
                if (state.slidingBuffer.shouldSend()) {
                    sendBufferedAudio();
                }
            };

            source.connect(processor);
            processor.connect(audioContext.destination);

            // Update UI
            setButtonState(elements.startRecordingBtn, true);
            setButtonState(elements.stopRecordingBtn, false);
            elements.recordingStatus.textContent = '录音中...';
            elements.recordingStatus.classList.add('recording');

            // Reset transcript areas
            setPlaceholder(elements.liveTranscript, '转写结果将在这里实时显示...');
            setPlaceholder(elements.finalTranscript, '最终转写结果将显示在这里...');
            hideElement(elements.exportFinalBtn);
            hideError();

        } catch (error) {
            if (error.name === 'NotAllowedError') {
                showError('麦克风权限被拒绝，请在浏览器设置中允许访问麦克风', 'MIC_PERMISSION_DENIED');
            } else {
                showError(`无法访问麦克风: ${error.message}`, 'MIC_ACCESS_ERROR');
            }
        }
    }

    function sendBufferedAudio() {
        const window = state.slidingBuffer.getSendWindow();
        if (!window || window.length === 0) return;

        const wavBlob = encodeWav(window, VAD_CONFIG.SAMPLE_RATE);
        const chunkIndex = state.chunkIndex++;
        console.log('[VAD] Sending chunk', chunkIndex, 'with', window.length, 'samples');
        state.audioChunks.push(wavBlob);
        processNextChunk();
    }

    function encodeWav(samples, sampleRate) {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);

        // WAV header
        writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + samples.length * 2, true);
        writeString(view, 8, 'WAVE');
        writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(view, 36, 'data');
        view.setUint32(40, samples.length * 2, true);

        // Write samples
        let offset = 44;
        for (let i = 0; i < samples.length; i++) {
            const sample = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }

        return new Blob([buffer], { type: 'audio/wav' });
    }

    function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    async function processNextChunk() {
        if (state.audioChunks.length === 0 || !state.isRecording) return;

        const chunk = state.audioChunks.shift();
        if (!chunk) return;

        const chunkIndex = state.chunkIndex++;
        state.processingPromise = state.processingPromise.then(async () => {
            if (!state.isRecording) return;
            try {
                const result = await transcribeChunk(
                    state.sessionId,
                    chunkIndex,
                    chunk,
                    'audio/wav',
                    VAD_CONFIG.WINDOW_SIZE_MS,
                    VAD_CONFIG.OVERLAP_MS
                );
                appendToLiveTranscript(result.merged_text);
            } catch (error) {
                console.error('Chunk transcription error:', error);
            }
        });
    }

    function stopRecording() {
        if (!state.isRecording) return;

        state.isRecording = false;

        // Stop audio processing
        if (state.processor) {
            state.processor.disconnect();
            state.processor = null;
        }
        if (state.audioContext) {
            state.audioContext.close();
            state.audioContext = null;
        }
        if (state.mediaStream) {
            state.mediaStream.getTracks().forEach(track => track.stop());
            state.mediaStream = null;
        }

        // Update UI
        setButtonState(elements.startRecordingBtn, false);
        setButtonState(elements.stopRecordingBtn, true);
        elements.recordingStatus.textContent = '处理中...';
        elements.recordingStatus.classList.remove('recording');

        state.mode = 'processing';

        // Finalize session
        finalizeSession(state.sessionId).then(result => {
            setFinalTranscript(result.final_text);
            state.mode = 'idle';
            elements.recordingStatus.textContent = '完成';
            setButtonState(elements.startRecordingBtn, false);
            setButtonState(elements.stopRecordingBtn, true);
        }).catch(error => {
            showError(error.message, 'FINALIZE_ERROR');
            state.mode = 'idle';
        });
    }

    // ============================================
    // Export Functions
    // ============================================

    function exportToFile(text, filename) {
        const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    function exportFileResult() {
        if (state.fileTranscriptionResult) {
            const filename = `音频文件转写_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.txt`;
            exportToFile(state.fileTranscriptionResult, filename);
        }
    }

    function exportFinalResult() {
        const text = elements.finalTranscript.textContent;
        if (text) {
            const filename = `实时录音转录_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.txt`;
            exportToFile(text, filename);
        }
    }

    // ============================================
    // Event Listeners
    // ============================================

    function initEventListeners() {
        // File transcription
        elements.transcribeBtn.addEventListener('click', handleFileTranscription);
        elements.exportFileBtn.addEventListener('click', exportFileResult);

        // Recording controls
        elements.startRecordingBtn.addEventListener('click', startRecording);
        elements.stopRecordingBtn.addEventListener('click', stopRecording);

        // Export buttons
        elements.exportFinalBtn.addEventListener('click', exportFinalResult);

        // File input change
        elements.fileInput.addEventListener('change', () => {
            hideError();
            hideElement(elements.fileResult);
            hideElement(elements.exportFileBtn);
        });
    }

    // ============================================
    // Initialization
    // ============================================

    function init() {
        initEventListeners();
        console.log('AutoGLM ASR Web Client initialized');
    }

    // Start the application
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
