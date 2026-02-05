import argparse

from flask import Flask, render_template_string, send_from_directory
import numpy as np
import os
import glob

from song_analyzer import SongAnalyzer

app = Flask(__name__)

# Directory for audio files
AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio')
os.makedirs(AUDIO_DIR, exist_ok=True)


def get_available_instruments():
    """Get a list of available audio instrument folders"""
    # Check for subdirectories in the audio directory
    instrument_dirs = [d for d in os.listdir(AUDIO_DIR)
                       if os.path.isdir(os.path.join(AUDIO_DIR, d))]

    if not instrument_dirs:
        # Create a sample folder if none exist
        sample_dir = os.path.join(AUDIO_DIR, 'sample_instrument')
        os.makedirs(sample_dir, exist_ok=True)
        instrument_dirs = ['sample_instrument']

    return instrument_dirs


def get_audio_files(instrument):
    """Get a list of audio files for the specified instrument"""
    instrument_path = os.path.join(AUDIO_DIR, instrument)
    if not os.path.exists(instrument_path):
        return []

    # Find all audio files
    audio_files = []
    for ext in ['*.mp3', '*.wav', '*.ogg']:
        audio_files.extend(glob.glob(os.path.join(instrument_path, ext)))

    # Return just the filenames
    return [os.path.basename(f) for f in audio_files]


@app.route('/')
def index():
    parser = argparse.ArgumentParser()
    parser.add_argument('--track_path', type=str, help='Source path to track')
    parser.add_argument('--scale', type=str,default='Relative', help='Scale to use')
    parser.add_argument('--instrument', type=str,default='main', help="Track's instrument")
    parser.add_argument('--min_seg_length_sec', type=int,default=5, help="Min Segment length in sec")
    parser.add_argument('--max_seg_length_sec', type=int,default=40, help="Max Segment length in sec")
    parser.add_argument('--energy_levels', type=int,default=5, help="Number of levels")
    parser.add_argument('--resolution', type=int,default=0.1, help="Display Resolution in sec")


    args = parser.parse_args()
    src_track = args.track_path
    scale = args.scale
    # scales Relative, Normal, Percentile

    instrument = args.instrument
    levels = args.energy_levels
    min_seg_length_sec = args.min_seg_length_sec
    max_seg_length_sec = args.max_seg_length_sec

    analyzer = SongAnalyzer(normalize_values=True, scale=scale, number_of_levels=levels,
                            max_seg_length_sec=max_seg_length_sec, min_seg_length_sec=min_seg_length_sec)
    quantized_values = analyzer.single_track_run(track=src_track, instrument=instrument,
                                                 show_plot=False)

    resolution = args.resolution # in sec
    samples = int(resolution*analyzer.sample_rate)
    quantized_values = quantized_values[::samples]
    # Number of points in the slider and quantized list
    length = len(quantized_values)
    # Number of quantization steps
    steps = analyzer.number_of_levels

    # speed = 45 #1000/analyzer.sample_rate
    speed = 1000/analyzer.sample_rate
    # Playback speed in ms (time between updates)
    # speed = int(request.args.get('speed', speedup))

    audio_files = [src_track]


    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Graph Visualizer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }

        h1 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .controls {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }

        .btn {
            background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
            border: none;
            color: white;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        }

        .btn:active {
            transform: translateY(0);
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: default;
            transform: none;
        }

        .file-input {
            display: none;
        }

        .file-label {
            background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
            border: none;
            color: white;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }

        .file-label:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        }

        .time-display {
            background: rgba(255, 255, 255, 0.2);
            padding: 10px 20px;
            border-radius: 15px;
            font-family: 'Courier New', monospace;
            font-size: 18px;
            font-weight: bold;
        }

        .graph-container {
            position: relative;
            width: 100%;
            height: 400px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 15px;
            margin-bottom: 20px;
            overflow: hidden;
            box-shadow: inset 0 4px 15px rgba(0, 0, 0, 0.3);
        }

        #canvas {
            width: 100%;
            height: 100%;
            cursor: pointer;
            transition: filter 0.3s ease;
        }

        #canvas:hover {
            filter: brightness(1.1);
        }

        .info-panel {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }

        .info-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(5px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .info-card h3 {
            margin-bottom: 10px;
            color: #4ecdc4;
        }

        .progress-bar {
            width: 100%;
            height: 12px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 6px;
            margin: 10px 0;
            overflow: hidden;
            cursor: pointer;
            position: relative;
            transition: height 0.2s ease;
        }
        
        .progress-bar:hover {
            height: 16px;
            background: rgba(255, 255, 255, 0.3);
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4ecdc4, #44a08d);
            border-radius: 4px;
            width: 0%;
            transition: width 0.1s ease;
        }
         .progress-fill::after {
            content: '';
            position: absolute;
            right: -6px;
            top: 50%;
            transform: translateY(-50%);
            width: 12px;
            height: 12px;
            background: white;
            border-radius: 50%;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
            opacity: 0;
            transition: opacity 0.2s ease;
        }

        .progress-bar:hover .progress-fill::after {
            opacity: 1;
        }
        .sample-input {
            width: 100%;
            padding: 10px;
            margin: 10px 0;
            border: none;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.1);
            color: white;
            font-size: 14px;
        }

        .sample-input::placeholder {
            color: rgba(255, 255, 255, 0.6);
        }

        .error-message {
            background: rgba(255, 107, 107, 0.2);
            border: 1px solid #ff6b6b;
            color: #ff6b6b;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 Audio Graph Visualizer</h1>
        
        <div class="controls">
            <label for="audioFile" class="file-label">📁 Load Audio</label>
            <input type="file" id="audioFile" class="file-input" accept="audio/*">
            
            <button id="playBtn" class="btn" disabled>▶️ Play</button>
            <button id="pauseBtn" class="btn" disabled>⏸️ Pause</button>
            <button id="stopBtn" class="btn" disabled>⏹️ Stop</button>
            
            <div class="time-display">
                <span id="timeDisplay">00:00 / 00:00</span>
            </div>
        </div>

        <div class="error-message" id="errorMessage"></div>

        <div class="graph-container">
            <canvas id="canvas"></canvas>
        </div>

        <div class="progress-bar">
            <div class="progress-fill" id="progressFill"></div>
        </div>

        <div class="info-panel">
            <div class="info-card">
                <h3>📊 Data Input</h3>
                <textarea id="quantizedData" class="sample-input" rows="4" 
                    placeholder="Enter quantized values (comma-separated, e.g., 0.1,0.5,0.8,0.3,0.9,0.2...)">0.1,0.5,0.8,0.3,0.9,0.2,0.7,0.4,0.6,0.8,0.3,0.5,0.9,0.1,0.7,0.4,0.6,0.2,0.8,0.5</textarea>
                <button id="updateData" class="btn" style="width: 100%; margin-top: 10px;">Update Graph</button>
            </div>
            
            <div class="info-card">
                <h3>📈 Current Stats</h3>
                <p>Sample Index: <span id="currentSample">0</span></p>
                <p>Sample Value: <span id="currentValue">0.00</span></p>
                <p>Total Samples: <span id="totalSamples">0</span></p>
                <p>Progress: <span id="progressPercent">0%</span></p>
            </div>
        </div>
    </div>

    <script>
        class AudioGraphVisualizer {
            constructor() {
                this.canvas = document.getElementById('canvas');
                this.ctx = this.canvas.getContext('2d');
                this.audio = new Audio();
                this.quantizedData = [];
                this.currentSampleIndex = 0;
                this.isPlaying = false;
                this.animationId = null;
                this.isDraggingProgress = false;

                this.setupCanvas();
                this.setupEventListeners();
                this.setupDefaultData();
                this.updateDisplay();
            }

            setupCanvas() {
                const rect = this.canvas.getBoundingClientRect();
                this.canvas.width = rect.width * window.devicePixelRatio;
                this.canvas.height = rect.height * window.devicePixelRatio;
                this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
                this.canvas.style.width = rect.width + 'px';
                this.canvas.style.height = rect.height + 'px';
            }

            setupEventListeners() {
                // File input
                document.getElementById('audioFile').addEventListener('change', (e) => {
                    this.loadAudio(e.target.files[0]);
                });

                // Control buttons
                document.getElementById('playBtn').addEventListener('click', () => this.play());
                document.getElementById('pauseBtn').addEventListener('click', () => this.pause());
                document.getElementById('stopBtn').addEventListener('click', () => this.stop());
                document.getElementById('updateData').addEventListener('click', () => this.updateQuantizedData());

                // Canvas click for seeking
                this.canvas.addEventListener('click', (e) => this.seekToPosition(e));
                
                 // Progress bar interaction
                this.setupProgressBarEvents();
                
                // Audio events
                this.audio.addEventListener('loadedmetadata', () => this.onAudioLoaded());
                this.audio.addEventListener('timeupdate', () => this.onTimeUpdate());
                this.audio.addEventListener('ended', () => this.onAudioEnded());
                this.audio.addEventListener('error', (e) => this.showError('Audio loading error: ' + e.message));

                // Window resize
                window.addEventListener('resize', () => this.setupCanvas());
            }
            
                setupProgressBarEvents() {
                const progressBar = document.querySelector('.progress-bar');
                
                // Mouse events
                progressBar.addEventListener('mousedown', (e) => this.startProgressDrag(e));
                document.addEventListener('mousemove', (e) => this.handleProgressDrag(e));
                document.addEventListener('mouseup', () => this.endProgressDrag());
                
                // Touch events for mobile
                progressBar.addEventListener('touchstart', (e) => this.startProgressDrag(e.touches[0]));
                document.addEventListener('touchmove', (e) => {
                    e.preventDefault();
                    this.handleProgressDrag(e.touches[0]);
                });
                document.addEventListener('touchend', () => this.endProgressDrag());
                
                // Click to seek
                progressBar.addEventListener('click', (e) => this.seekToProgressPosition(e));
            }  
            
            startProgressDrag(event) {
                if (!this.audio.duration) return;
                
                this.isDraggingProgress = true;
                document.body.style.userSelect = 'none'; // Prevent text selection while dragging
                this.seekToProgressPosition(event);
            }

            handleProgressDrag(event) {
                if (!this.isDraggingProgress || !this.audio.duration) return;
                this.seekToProgressPosition(event);
            }

            endProgressDrag() {
                this.isDraggingProgress = false;
                document.body.style.userSelect = ''; // Re-enable text selection
            }

            seekToProgressPosition(event) {
                if (!this.audio.duration) return;
                
                const progressBar = document.querySelector('.progress-bar');
                const rect = progressBar.getBoundingClientRect();
                const x = (event.clientX || event.pageX) - rect.left;
                const progress = Math.max(0, Math.min(1, x / rect.width));
                
                // Update audio time
                this.audio.currentTime = progress * this.audio.duration;
                
                // Update sample index
                if (this.quantizedData.length > 0) {
                    this.currentSampleIndex = Math.floor(progress * this.quantizedData.length);
                }
                
                // Update display immediately
                this.updateDisplay();
            }
                      
            setupDefaultData() {
                // REPLACE THIS SECTION WITH YOUR CALCULATED LIST
                const yourCalculatedList = {{ quantized_values }};
                                
                // Convert array to string for the text area
                const defaultData = yourCalculatedList.join(',');
                document.getElementById('quantizedData').value = defaultData;
                this.updateQuantizedData();
            }

            // Method to programmatically set new data
            setQuantizedData(dataArray) {
                if (Array.isArray(dataArray)) {
                    document.getElementById('quantizedData').value = dataArray.join(',');
                    this.updateQuantizedData();
                } else {
                    this.showError('Data must be an array of numbers');
                }
            }

            loadAudio(file) {
                if (!file) return;
                
                const url = URL.createObjectURL(file);
                this.audio.src = url;
                this.hideError();
                
                // Enable controls when audio is loaded
                this.audio.addEventListener('canplaythrough', () => {
                    document.getElementById('playBtn').disabled = false;
                    document.getElementById('pauseBtn').disabled = false;
                    document.getElementById('stopBtn').disabled = false;
                }, { once: true });
            }

            updateQuantizedData() {
                const input = document.getElementById('quantizedData').value.trim();
                if (!input) {
                    this.showError('Please enter quantized data values');
                    return;
                }

                try {
                    this.quantizedData = input.split(',').map(val => {
                        const num = parseFloat(val.trim());
                        if (isNaN(num)) throw new Error(`Invalid number: ${val}`);
                        return Math.max(0, Math.min(1, num)); // Clamp between 0 and 1
                    });

                    if (this.quantizedData.length === 0) {
                        throw new Error('No valid data points found');
                    }

                    this.hideError();
                    this.drawGraph();
                    this.updateStats();
                } catch (error) {
                    this.showError('Data parsing error: ' + error.message);
                }
            }

            play() {
                if (this.audio.src && this.quantizedData.length > 0) {
                    this.audio.play().catch(e => this.showError('Playback error: ' + e.message));
                    this.isPlaying = true;
                    this.startAnimation();
                }
            }

            pause() {
                this.audio.pause();
                this.isPlaying = false;
                this.stopAnimation();
            }

            stop() {
                this.audio.pause();
                this.audio.currentTime = 0;
                this.isPlaying = false;
                this.currentSampleIndex = 0;
                this.stopAnimation();
                this.updateDisplay();
            }

            seekToPosition(event) {
                if (!this.audio.duration || this.quantizedData.length === 0) return;

                const rect = this.canvas.getBoundingClientRect();
                const x = event.clientX - rect.left;
                const progress = x / rect.width;
                
                this.audio.currentTime = progress * this.audio.duration;
                this.currentSampleIndex = Math.floor(progress * this.quantizedData.length);
                this.updateDisplay();
            }

            onAudioLoaded() {
                this.updateDisplay();
            }

            onTimeUpdate() {
                if (this.audio.duration && this.quantizedData.length > 0) {
                    const progress = this.audio.currentTime / this.audio.duration;
                    this.currentSampleIndex = Math.floor(progress * this.quantizedData.length);
                    this.updateDisplay();
                }
            }

            onAudioEnded() {
                this.isPlaying = false;
                this.stopAnimation();
                this.currentSampleIndex = 0;
                this.updateDisplay();
            }

            startAnimation() {
                const animate = () => {
                    if (this.isPlaying) {
                        this.drawGraph();
                        this.animationId = requestAnimationFrame(animate);
                    }
                };
                animate();
            }

            stopAnimation() {
                if (this.animationId) {
                    cancelAnimationFrame(this.animationId);
                    this.animationId = null;
                }
            }

            drawGraph() {
                const width = this.canvas.width / window.devicePixelRatio;
                const height = this.canvas.height / window.devicePixelRatio;
                
                // Clear canvas
                this.ctx.clearRect(0, 0, width, height);
                
                if (this.quantizedData.length === 0) return;

                // Draw grid
                this.drawGrid(width, height);
                
                // Draw quantized data as bars
                this.drawBars(width, height);
                
                // Draw cursor
                this.drawCursor(width, height);
            }

            drawGrid(width, height) {
                this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
                this.ctx.lineWidth = 1;
                
                // Horizontal lines
                for (let i = 0; i <= 10; i++) {
                    const y = (height * i) / 10;
                    this.ctx.beginPath();
                    this.ctx.moveTo(0, y);
                    this.ctx.lineTo(width, y);
                    this.ctx.stroke();
                }
                
                // Vertical lines
                const step = Math.max(1, Math.floor(this.quantizedData.length / 20));
                for (let i = 0; i < this.quantizedData.length; i += step) {
                    const x = (width * i) / this.quantizedData.length;
                    this.ctx.beginPath();
                    this.ctx.moveTo(x, 0);
                    this.ctx.lineTo(x, height);
                    this.ctx.stroke();
                }
            }

            drawBars(width, height) {
                const barWidth = width / this.quantizedData.length;
                
                this.quantizedData.forEach((value, index) => {
                    const x = (width * index) / this.quantizedData.length;
                    const barHeight = value * height;
                    const y = height - barHeight;
                    
                    // Create gradient for each bar
                    const gradient = this.ctx.createLinearGradient(0, height, 0, 0);
                    gradient.addColorStop(0, '#4ecdc4');
                    gradient.addColorStop(0.5, '#44a08d');
                    gradient.addColorStop(1, '#667eea');
                    
                    // Highlight current sample
                    if (index === this.currentSampleIndex) {
                        this.ctx.fillStyle = '#ff6b6b';
                        this.ctx.shadowColor = '#ff6b6b';
                        this.ctx.shadowBlur = 10;
                    } else {
                        this.ctx.fillStyle = gradient;
                        this.ctx.shadowBlur = 0;
                    }
                    
                    this.ctx.fillRect(x, y, barWidth - 1, barHeight);
                });
                
                this.ctx.shadowBlur = 0;
            }

            drawCursor(width, height) {
                if (this.quantizedData.length === 0) return;
                
                const x = (width * this.currentSampleIndex) / this.quantizedData.length;
                
                this.ctx.strokeStyle = '#ff6b6b';
                this.ctx.lineWidth = 3;
                this.ctx.shadowColor = '#ff6b6b';
                this.ctx.shadowBlur = 5;
                
                this.ctx.beginPath();
                this.ctx.moveTo(x, 0);
                this.ctx.lineTo(x, height);
                this.ctx.stroke();
                
                this.ctx.shadowBlur = 0;
            }

            updateDisplay() {
                this.drawGraph();
                this.updateStats();
                this.updateTimeDisplay();
                this.updateProgressBar();
            }

            updateStats() {
                document.getElementById('currentSample').textContent = this.currentSampleIndex;
                document.getElementById('currentValue').textContent = 
                    (this.quantizedData[this.currentSampleIndex] || 0).toFixed(3);
                document.getElementById('totalSamples').textContent = this.quantizedData.length;
                
                const progress = this.quantizedData.length > 0 ? 
                    ((this.currentSampleIndex / this.quantizedData.length) * 100).toFixed(1) : 0;
                document.getElementById('progressPercent').textContent = progress + '%';
            }

            updateTimeDisplay() {
                const current = this.formatTime(this.audio.currentTime || 0);
                const duration = this.formatTime(this.audio.duration || 0);
                document.getElementById('timeDisplay').textContent = `${current} / ${duration}`;
            }

            updateProgressBar() {
                const progress = this.audio.duration ? 
                    (this.audio.currentTime / this.audio.duration) * 100 : 0;
                document.getElementById('progressFill').style.width = progress + '%';
            }

            formatTime(seconds) {
                const mins = Math.floor(seconds / 60);
                const secs = Math.floor(seconds % 60);
                return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
            }

            showError(message) {
                const errorEl = document.getElementById('errorMessage');
                errorEl.textContent = message;
                errorEl.style.display = 'block';
                setTimeout(() => this.hideError(), 5000);
            }

            hideError() {
                document.getElementById('errorMessage').style.display = 'none';
            }
        }

        // Initialize the visualizer when the page loads
        document.addEventListener('DOMContentLoaded', () => {
            new AudioGraphVisualizer();
        });
    </script>
</body>
</html>
    ''',
                                  length=length,
                                  steps=steps,
                                  speed=speed,
                                  audio_files=audio_files,
                                  quantized_values=quantized_values)


@app.route('/audio/<instrument>/<filename>')
def serve_audio(instrument, filename):
    """Serve audio files from the audio directory"""
    return send_from_directory(os.path.join(AUDIO_DIR, instrument), filename)


if __name__ == '__main__':
    app.run(debug=True)
