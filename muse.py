import pyaudio
import struct
import math
import time
import requests
import json
import tkinter as tk
import hmac
import hashlib
import base64
from config import API_KEY, API_SECRET, API_URL

# ---------------------------
# Global configuration values, updated with your ACRCloud data
# ---------------------------
CHUNK = 1024                # Number of audio frames per buffer
FORMAT = pyaudio.paInt16    # Audio format (16-bit integer)
CHANNELS = 1                # Use mono channel for processing
RATE = 44100                # Sample rate (Hz)
RECORD_SECONDS = 10         # Duration for recording snippet
THRESHOLD = 500             # RMS threshold for detecting audio

# ACRCloud credentials - update these with your acquired values
# Note: "66829" (project/account ID) and "Muse" are not required in the request.
# API_KEY      # Access Key
# API_SECRET  # Access Secret
# API_URL = "http://identify-us-west-2.acrcloud.com/v1/identify"  # Full endpoint URL

# ---------------------------
# Helper function to create ACRCloud signature
# ---------------------------
def create_signature(access_key, access_secret, timestamp):
    """
    Creates a signature required by ACRCloud.
    
    Args:
        access_key (str): Your ACRCloud access key.
        access_secret (str): Your ACRCloud access secret.
        timestamp (int): The current Unix timestamp.
    
    Returns:
        tuple: (signature, data_type, signature_version)
    """
    http_method = "POST"
    http_uri = "/v1/identify"
    data_type = "audio"           # Using audio fingerprinting
    signature_version = "1"

    string_to_sign = "\n".join([http_method, http_uri, access_key, data_type, signature_version, str(timestamp)])
    # Compute HMAC-SHA1 signature and then base64 encode it
    hmac_res = hmac.new(access_secret.encode('utf-8'), string_to_sign.encode('utf-8'), digestmod=hashlib.sha1).digest()
    signature = base64.b64encode(hmac_res).decode('utf-8')
    return signature, data_type, signature_version

# ---------------------------
# Existing functions (compute_rms, record_audio_chunk, update_display) remain unchanged...
# ---------------------------
def compute_rms(audio_data):
    count = len(audio_data) // 2
    format_str = f"{count}h"
    try:
        samples = struct.unpack(format_str, audio_data)
    except struct.error as e:
        print("Error unpacking audio data:", e)
        return 0
    sum_squares = sum(sample**2 for sample in samples)
    rms = math.sqrt(sum_squares / count) if count > 0 else 0
    return rms

def record_audio_chunk(p, stream, record_seconds):
    frames = []
    num_frames = int(RATE / CHUNK * record_seconds)
    for i in range(num_frames):
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        except Exception as e:
            print(f"Error reading audio stream at frame {i}: {e}")
            return None
    return b''.join(frames)

def update_display(metadata):
    window = tk.Tk()
    window.title("Now Playing")
    window.geometry("480x320")
    
    try:
        music_list = metadata.get("metadata", {}).get("music", [])
        if not music_list:
            raise ValueError("No music data found in metadata.")
        
        song_data = music_list[0]
        song_title = song_data.get("title", "Unknown Title")
        song_artist = song_data.get("artists", [{}])[0].get("name", "Unknown Artist")
        album = song_data.get("album", {}).get("name", "Unknown Album")
        
        display_text = f"Title: {song_title}\nArtist: {song_artist}\nAlbum: {album}"
    except Exception as e:
        print("Error parsing song metadata:", e)
        display_text = "Error retrieving song details"
    
    label = tk.Label(window, text=display_text, font=("Helvetica", 16))
    label.pack(expand=True)
    window.after(10000, window.destroy)
    window.mainloop()

# ---------------------------
# Updated API Call Function for ACRCloud
# ---------------------------
def call_recognition_api(audio_data):
    """
    Calls the ACRCloud music recognition API with the recorded audio snippet.
    
    Args:
        audio_data (bytes): The recorded audio snippet.
    
    Returns:
        dict or None: Parsed JSON metadata if recognition is successful, else None.
    """
    try:
        timestamp = int(time.time())
        signature, data_type, signature_version = create_signature(API_KEY, API_SECRET, timestamp)
        
        # Prepare the payload according to ACRCloud's requirements
        data = {
            'access_key': API_KEY,
            'data_type': data_type,
            'signature': signature,
            'signature_version': signature_version,
            'timestamp': timestamp
        }
        
        # The API expects the audio file under the key 'sample'
        files = {
            'sample': ('audio.wav', audio_data, 'audio/wav')
        }
        
        response = requests.post(API_URL, files=files, data=data, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('status', {}).get('msg') == 'Success':
                return result
            else:
                print("Recognition error:", result.get('status', {}).get('msg'))
                return None
        else:
            print("HTTP error:", response.status_code)
            return None
    except requests.RequestException as e:
        print("API request error:", e)
        return None
    except json.JSONDecodeError as e:
        print("Error decoding JSON response:", e)
        return None

def main():
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    except Exception as e:
        print("Error opening audio stream:", e)
        p.terminate()
        return
    
    print("Listening for music... Press Ctrl+C to exit.")
    
    try:
        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
            except Exception as e:
                print("Error reading from stream:", e)
                continue
            
            current_rms = compute_rms(data)
            
            if current_rms > THRESHOLD:
                print("Detected audio above threshold. Recording audio snippet...")
                audio_snippet = record_audio_chunk(p, stream, RECORD_SECONDS)
                if audio_snippet:
                    print("Sending audio snippet for recognition...")
                    recognition_result = call_recognition_api(audio_snippet)
                    if recognition_result:
                        print("Song recognized. Updating display...")
                        update_display(recognition_result)
                    else:
                        print("Could not recognize the song.")
                else:
                    print("Failed to record audio snippet.")
                
                time.sleep(2)
                
    except KeyboardInterrupt:
        print("Exiting program (KeyboardInterrupt).")
    except Exception as e:
        print("Unexpected error:", e)
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except Exception as e:
            print("Error closing stream:", e)
        p.terminate()

if __name__ == '__main__':
    main()