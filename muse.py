#!/usr/bin/env python3
"""
Music Recognition and Display System for Raspberry Pi with ACRCloud Integration
"""

import os
import sys
import time
import logging
import queue
import threading
import json
import requests
import base64
import hashlib
import hmac
import io
from datetime import datetime
import sounddevice as sd
import numpy as np
import librosa
import PySimpleGUI as sg
from PIL import Image, ImageDraw

# Configuration
CONFIG = {
    "sample_rate": 44100,  # Audio sample rate in Hz
    "chunk_size": 1024,    # Audio frames per buffer
    "silence_threshold": 0.01,  # Minimum audio level to consider as sound
    "min_recording_length": 5,  # Minimum seconds to record before processing
    "max_recording_length": 15,  # Maximum seconds to record
    "api_timeout": 10,     # Timeout for API calls in seconds
    "retry_count": 3,      # Number of retries for API calls
    "cache_file": "~/.music_recognition_cache.json",  # Cache file path
    "theme": "DarkBlue3",  # GUI theme
    "api_key": "1cb43a21e75276577271ebd063b5f5d2",
    "api_secret": "zUIpjdFy4ru3vjhgyGqCOkcDdX935kCLSRT5cDW8",
    "api_url": "http://identify-us-west-2.acrcloud.com/v1/identify"
}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('music_recognition.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AudioRecorder:
    """Handles audio recording from USB microphone"""
    
    def __init__(self):
        self.recording = False
        self.audio_queue = queue.Queue()
        self.sample_rate = CONFIG["sample_rate"]
        self.chunk_size = CONFIG["chunk_size"]
        self.device_info = self._get_audio_device()
        
    def _get_audio_device(self):
        """Find and validate the USB microphone device"""
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        
        # Prefer USB devices if available
        usb_devices = [d for d in input_devices if 'usb' in d['name'].lower()]
        
        if usb_devices:
            device = usb_devices[0]
        elif input_devices:
            device = input_devices[0]
        else:
            raise RuntimeError("No audio input devices found")
            
        logger.info(f"Using audio device: {device['name']}")
        return device
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback function for sounddevice to process audio chunks"""
        if status:
            logger.warning(f"Audio stream status: {status}")
        if self.recording:
            self.audio_queue.put(indata.copy())
    
    def start_recording(self):
        """Start recording audio"""
        if self.recording:
            logger.warning("Recording already in progress")
            return
            
        self.recording = True
        self.audio_queue = queue.Queue()  # Clear previous recordings
        
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                device=self.device_info['index'],
                channels=1,  # Mono audio is sufficient for recognition
                callback=self._audio_callback,
                dtype='float32'
            )
            self.stream.start()
            logger.info("Recording started")
        except Exception as e:
            logger.error(f"Error starting audio stream: {str(e)}")
            raise
    
    def stop_recording(self):
        """Stop recording and return the recorded audio"""
        if not self.recording:
            logger.warning("No recording in progress")
            return None
            
        self.recording = False
        self.stream.stop()
        self.stream.close()
        logger.info("Recording stopped")
        
        # Combine all audio chunks
        audio_data = []
        while not self.audio_queue.empty():
            audio_data.append(self.audio_queue.get())
            
        if not audio_data:
            logger.warning("No audio data recorded")
            return None
            
        return np.concatenate(audio_data, axis=0)
    
    def is_sound_present(self, duration=1.0):
        """Check if there is sound above the silence threshold"""
        try:
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            
            rms = np.sqrt(np.mean(recording**2))
            return rms > CONFIG["silence_threshold"]
        except Exception as e:
            logger.error(f"Error detecting sound: {str(e)}")
            return False

class MusicRecognizer:
    """Handles music recognition using ACRCloud API"""
    
    def __init__(self):
        self.cache = self._load_cache()
        
    def _load_cache(self):
        """Load recognition cache from file"""
        cache_path = os.path.expanduser(CONFIG["cache_file"])
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {str(e)}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save recognition cache to file"""
        cache_path = os.path.expanduser(CONFIG["cache_file"])
        try:
            with open(cache_path, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {str(e)}")
    
    def _generate_signature(self, http_method, uri, access_key, secret_key, data_type, signature_version, timestamp):
        """Generate ACRCloud API signature"""
        string_to_sign = http_method + "\n" + uri + "\n" + access_key + "\n" + data_type + "\n" + signature_version + "\n" + timestamp
        sign = base64.b64encode(hmac.new(secret_key.encode('ascii'), string_to_sign.encode('ascii'), digestmod=hashlib.sha1).digest())
        return sign.decode('ascii')
    
    def _save_audio_temp(self, audio_data):
        """Save audio data to temporary WAV file for ACRCloud"""
        try:
            import tempfile
            import soundfile as sf
            
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"acr_temp_{int(time.time())}.wav")
            
            # Convert to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            sf.write(temp_path, audio_data, CONFIG["sample_rate"])
            return temp_path
        except Exception as e:
            logger.error(f"Error saving temp audio file: {str(e)}")
            return None
    
    def recognize_music(self, audio_data):
        """Recognize music from audio data using ACRCloud API"""
        if audio_data is None or len(audio_data) == 0:
            logger.error("No audio data provided for recognition")
            return None
            
        # First check cache
        fingerprint = self._generate_fingerprint(audio_data)
        if fingerprint is None:
            return None
            
        # Convert fingerprint to string key for caching
        fingerprint_key = json.dumps(fingerprint, sort_keys=True)
        
        if fingerprint_key in self.cache:
            logger.info("Found match in cache")
            return self.cache[fingerprint_key]
            
        # Save audio to temporary file
        temp_audio_path = self._save_audio_temp(audio_data)
        if not temp_audio_path:
            return None
            
        try:
            # Prepare ACRCloud API request
            http_method = "POST"
            data_type = "audio"
            signature_version = "1"
            timestamp = str(int(time.time()))
            
            signature = self._generate_signature(
                http_method,
                "/v1/identify",
                CONFIG["api_key"],
                CONFIG["api_secret"],
                data_type,
                signature_version,
                timestamp
            )
            
            files = [
                ('sample', ('sample.wav', open(temp_audio_path, 'rb'), 'audio/wav'))
            ]
            
            data = {
                'access_key': CONFIG["api_key"],
                'data_type': data_type,
                'signature_version': signature_version,
                'signature': signature,
                'sample_bytes': str(os.path.getsize(temp_audio_path)),
                'timestamp': timestamp
            }
            
            # Make API request
            response = requests.post(
                CONFIG["api_url"],
                files=files,
                data=data,
                timeout=CONFIG["api_timeout"]
            )
            
            # Clean up temp file
            try:
                os.remove(temp_audio_path)
            except:
                pass
            
            if response.status_code != 200:
                logger.error(f"ACRCloud API error: {response.status_code} - {response.text}")
                return None
                
            result = response.json()
            
            if result['status']['code'] != 0:
                logger.error(f"ACRCloud recognition error: {result['status']['msg']}")
                return None
                
            if 'metadata' not in result or 'music' not in result['metadata'] or len(result['metadata']['music']) == 0:
                logger.info("No music recognized in audio")
                return None
                
            # Extract the most likely match
            best_match = result['metadata']['music'][0]
            song_info = {
                'title': best_match.get('title', 'Unknown'),
                'artist': best_match.get('artists', [{}])[0].get('name', 'Unknown'),
                'album': best_match.get('album', {}).get('name', 'Unknown'),
                'duration': best_match.get('duration_ms', 0) / 1000,
                'year': best_match.get('release_date', '')[:4] or 'Unknown',
                'acr_id': best_match.get('acrid', None)
            }
            
            # Cache the result
            self.cache[fingerprint_key] = song_info
            self._save_cache()
            
            return song_info
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error processing API response: {str(e)}")
            return None
    
    def _generate_fingerprint(self, audio_data):
        """Generate simple fingerprint for caching purposes"""
        try:
            # Convert to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
                
            # Simple fingerprint based on audio statistics
            fingerprint = {
                'mean': float(np.mean(audio_data)),
                'std': float(np.std(audio_data)),
                'len': len(audio_data)
            }
            return fingerprint
        except Exception as e:
            logger.error(f"Error generating fingerprint: {str(e)}")
            return None
    
    def get_cached_results(self):
        """Return all cached recognition results"""
        return list(self.cache.values())

class MusicDisplayGUI:
    """Handles the graphical user interface"""
    
    def __init__(self):
        self.theme = CONFIG["theme"]
        sg.theme(self.theme)
        
        # Album art placeholder
        self.album_art = sg.Image(
            data=self._get_default_album_art(),
            key='-ALBUM_ART-',
            size=(200, 200)
        )
        
        # Current song display
        self.current_song_layout = [
            [sg.Text('Now Playing', font=('Helvetica', 18), key='-NOW_PLAYING-')],
            [self.album_art],
            [sg.Text('Title:', font=('Helvetica', 14)), 
             sg.Text('', size=(30, 1), key='-TITLE-')],
            [sg.Text('Artist:', font=('Helvetica', 14)), 
             sg.Text('', size=(30, 1), key='-ARTIST-')],
            [sg.Text('Album:', font=('Helvetica', 14)), 
             sg.Text('', size=(30, 1), key='-ALBUM-')],
            [sg.Text('Year:', font=('Helvetica', 14)), 
             sg.Text('', size=(10, 1), key='-YEAR-')],
            [sg.Text('Detected at:', font=('Helvetica', 10)), 
             sg.Text('', size=(20, 1), key='-DETECTION_TIME-')]
        ]
        
        # History display
        self.history_layout = [
            [sg.Text('Recently Recognized', font=('Helvetica', 16))],
            [sg.Listbox(
                values=[],
                size=(40, 10),
                key='-HISTORY-',
                enable_events=True
            )]
        ]
        
        # Control buttons
        self.control_layout = [
            [sg.Button('Start Listening', key='-START-'),
             sg.Button('Stop Listening', key='-STOP-', disabled=True),
             sg.Button('Exit', key='-EXIT-')],
            [sg.Text('Status:', font=('Helvetica', 10)),
             sg.Text('Ready', key='-STATUS-')]
        ]
        
        # Combine all layouts
        self.layout = [
            [sg.Column(self.current_song_layout, element_justification='center')],
            [sg.HorizontalSeparator()],
            [sg.Column(self.history_layout)],
            [sg.HorizontalSeparator()],
            [sg.Column(self.control_layout)]
        ]
        
        self.window = sg.Window(
            'Music Recognition System',
            self.layout,
            finalize=True,
            element_justification='center'
        )
        
    def _get_default_album_art(self):
        """Return default album art image data (a placeholder)"""
        img = Image.new('RGB', (200, 200), color='black')
        draw = ImageDraw.Draw(img)
        draw.text((50, 80), "Album Art", fill='white')
        bio = io.BytesIO()
        img.save(bio, format='PNG')
        return bio.getvalue()
    
    def update_current_song(self, song_info):
        """Update the display with current song information"""
        if song_info is None:
            return
            
        self.window['-TITLE-'].update(song_info.get('title', 'Unknown'))
        self.window['-ARTIST-'].update(song_info.get('artist', 'Unknown'))
        self.window['-ALBUM-'].update(song_info.get('album', 'Unknown'))
        self.window['-YEAR-'].update(song_info.get('year', 'Unknown'))
        self.window['-DETECTION_TIME-'].update(
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def update_history(self, history):
        """Update the history list with recognized songs"""
        display_items = [
            f"{item.get('artist', 'Unknown')} - {item.get('title', 'Unknown')}"
            for item in history
        ]
        self.window['-HISTORY-'].update(display_items)
    
    def update_status(self, message):
        """Update the status message"""
        self.window['-STATUS-'].update(message)
    
    def close(self):
        """Close the GUI window"""
        self.window.close()

class MusicRecognitionSystem:
    """Main system controller"""
    
    def __init__(self):
        self.audio_recorder = AudioRecorder()
        self.music_recognizer = MusicRecognizer()
        self.gui = MusicDisplayGUI()
        self.running = False
        self.listening = False
        
        # Load cached history
        cached_results = self.music_recognizer.get_cached_results()
        self.gui.update_history(cached_results)
        
    def run(self):
        """Main application loop"""
        self.running = True
        
        while self.running:
            event, values = self.gui.window.read(timeout=100)  # 100ms timeout
            
            if event == sg.WIN_CLOSED or event == '-EXIT-':
                self.running = False
                break
                
            elif event == '-START-':
                self._start_listening()
                
            elif event == '-STOP-':
                self._stop_listening()
                
            elif event == '-HISTORY-' and values['-HISTORY-']:
                # User selected a song from history
                selected_index = self.gui.window['-HISTORY-'].get_indexes()[0]
                cached_results = self.music_recognizer.get_cached_results()
                if selected_index < len(cached_results):
                    self.gui.update_current_song(cached_results[selected_index])
            
            # Check for sound if in listening mode
            if self.listening and not self.audio_recorder.recording:
                if self.audio_recorder.is_sound_present():
                    self._process_audio()
        
        self.gui.close()
        logger.info("Application shutdown complete")
    
    def _start_listening(self):
        """Start listening for music"""
        self.listening = True
        self.gui.window['-START-'].update(disabled=True)
        self.gui.window['-STOP-'].update(disabled=False)
        self.gui.update_status("Listening for music...")
        logger.info("Started listening mode")
    
    def _stop_listening(self):
        """Stop listening for music"""
        self.listening = False
        if self.audio_recorder.recording:
            self.audio_recorder.stop_recording()
        self.gui.window['-START-'].update(disabled=False)
        self.gui.window['-STOP-'].update(disabled=True)
        self.gui.update_status("Ready")
        logger.info("Stopped listening mode")
    
    def _process_audio(self):
        """Record and process audio"""
        self.gui.update_status("Recording audio...")
        logger.info("Starting audio recording")
        
        try:
            self.audio_recorder.start_recording()
            start_time = time.time()
            
            # Record for minimum time or until silence
            while time.time() - start_time < CONFIG["min_recording_length"]:
                time.sleep(0.1)
                
            # Optionally: Continue recording up to max length if sound continues
            while (time.time() - start_time < CONFIG["max_recording_length"] and 
                   self.audio_recorder.is_sound_present()):
                time.sleep(0.1)
                
            audio_data = self.audio_recorder.stop_recording()
            
            if audio_data is None or len(audio_data) < CONFIG["min_recording_length"] * CONFIG["sample_rate"]:
                self.gui.update_status("No valid audio captured")
                logger.warning("Recording too short or empty")
                return
                
            self.gui.update_status("Identifying song...")
            logger.info(f"Processing audio ({len(audio_data)/CONFIG['sample_rate']:.1f}s)")
            
            # Recognize the music
            song_info = self.music_recognizer.recognize_music(audio_data)
            
            if song_info:
                self.gui.update_current_song(song_info)
                self.gui.update_status("Song identified!")
                logger.info(f"Recognized: {song_info.get('artist')} - {song_info.get('title')}")
                
                # Update history
                cached_results = self.music_recognizer.get_cached_results()
                self.gui.update_history(cached_results)
            else:
                self.gui.update_status("Song not recognized")
                logger.warning("Failed to recognize song")
                
        except Exception as e:
            self.gui.update_status(f"Error: {str(e)}")
            logger.error(f"Error processing audio: {str(e)}")
            if self.audio_recorder.recording:
                self.audio_recorder.stop_recording()

def main():
    """Application entry point"""
    try:
        logger.info("Starting Music Recognition System")
        app = MusicRecognitionSystem()
        app.run()
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()