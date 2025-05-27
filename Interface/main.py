from flask import Flask, render_template, request, jsonify, session, redirect, Response
import os
import json
import sys
from datetime import datetime
import secrets
import cv2
import time
import numpy as np
import mediapipe as mp
import pandas as pd
import csv
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import requests  # Added for LLM functionality

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# ASL Detector Class (based on working sentence_builder)
class SimpleASLDetector:
    def __init__(self):
        self.cap = None
        self.running = False
        self.current_letter = None
        self.sentence = ""
        self.last_detection_time = time.time()
        self.detection_interval = 4  # seconds like in sentence_builder
        
        # Load model exactly like sentence_builder
        self.model = None
        self.le = None
        self.class_mapping = {}
        self._load_model()
        
        # MediaPipe setup exactly like sentence_builder
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
        
        print("🤖 ASL Detector with sentence_builder logic initialized")
    
    def _load_model(self):
        """Load model exactly like sentence_builder"""
        try:
            print("Loading model...")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "..", "ASL_Detection", "sign-language-detector-python-master", "model.p")
            model_path = os.path.abspath(model_path)
            print(f"Loading model from: {model_path}")
            
            import pickle
            model_dict = pickle.load(open(model_path, 'rb'))
            self.model = model_dict['model']
            self.le = model_dict.get('label_encoder', None)
            
            # Read class mapping from file exactly like sentence_builder
            class_mapping_path = os.path.join(os.path.dirname(model_path), 'data', 'class_mapping.txt')
            print(f"Looking for class mapping at: {class_mapping_path}")
            if os.path.exists(class_mapping_path):
                with open(class_mapping_path, 'r') as f:
                    for line in f:
                        idx, label = line.strip().split(':', 1)
                        self.class_mapping[int(idx)] = label.strip()
                print(f"Loaded {len(self.class_mapping)} classes from mapping file")
            else:
                print("Warning: No class mapping file found")
                # Fallback default mapping
                self.class_mapping = {0: 'A', 1: 'B', 2: 'C'}
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def start_detection(self):
        """Start camera"""
        try:
            if self.running:
                print("⚠️ Detection already running")
                return True
                
            print("🔍 Attempting to open camera...")
            
            # Try different camera indices
            for camera_index in [0, 1, 2]:
                print(f"🎥 Trying camera index {camera_index}...")
                self.cap = cv2.VideoCapture(camera_index)
                
                if self.cap.isOpened():
                    print(f"✅ Camera {camera_index} opened successfully")
                    break
                else:
                    print(f"❌ Camera {camera_index} failed to open")
                    if self.cap:
                        self.cap.release()
                        self.cap = None
            
            if not self.cap or not self.cap.isOpened():
                raise Exception("Cannot open any camera (tried indices 0, 1, 2)")
            
            # Test if we can actually read a frame
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                raise Exception("Camera opened but cannot read frames")
            
            print(f"📏 Camera resolution: {test_frame.shape}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to prevent lag
            
            # Verify the settings
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"📐 Set camera to: {width}x{height}")
            
            self.running = True
            print("✅ Camera started successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error starting camera: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False
    
    def stop_detection(self):
        """Stop camera"""
        try:
            print("🛑 Stopping camera...")
            self.running = False
            
            if self.cap:
                self.cap.release()
                self.cap = None
            
            self.current_letter = None
            print("✅ Camera stopped")
            return True
            
        except Exception as e:
            print(f"❌ Error stopping camera: {e}")
            return False
    
    def get_camera_frame(self):
        """Get current frame with sentence_builder detection logic"""
        if not self.cap or not self.cap.isOpened() or not self.running:
            return None
            
        data_aux = []
        x_ = []
        y_ = []
        
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        # Flip frame horizontally for more intuitive interaction (like sentence_builder)
        frame = cv2.flip(frame, 1)
        
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        detected_letter = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
            
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    
                    x_.append(x)
                    y_.append(y)
                
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
            
            # Ensure data has the right shape for the model (like sentence_builder)
            if hasattr(self.model, 'n_features_in_'):
                max_length = self.model.n_features_in_
            else:
                max_length = 21 * 2 * 2  # 21 landmarks, x and y, normalized coordinates
            
            # Pad or trim data
            if len(data_aux) < max_length:
                data_aux = data_aux + [0.0] * (max_length - len(data_aux))
            elif len(data_aux) > max_length:
                data_aux = data_aux[:max_length]
            
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10
            
            prediction = self.model.predict([np.asarray(data_aux)])
            prediction_index = int(prediction[0])
            
            # Get the prediction based on model output (like sentence_builder)
            if self.le is not None:
                original_label = self.le.inverse_transform([prediction_index])[0]
                detected_letter = self.class_mapping.get(int(original_label), f"Unknown ({original_label})")
            else:
                detected_letter = self.class_mapping.get(prediction_index, f"Unknown ({prediction_index})")
            
            # Draw the bounding box and prediction (like sentence_builder)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (233, 69, 96), 4)  # E94560 in RGB
            cv2.putText(frame, detected_letter, (x1, y1 - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.3, (233, 69, 96), 3, cv2.LINE_AA)
        
        # Update current letter
        if detected_letter:
            self.current_letter = detected_letter
        else:
            self.current_letter = None
        
        # Auto-add letter to sentence every 4 seconds (like sentence_builder)
        current_time = time.time()
        if detected_letter and (current_time - self.last_detection_time) >= self.detection_interval:
            if detected_letter == "space":
                self.sentence += " "
            elif detected_letter == "del":
                if self.sentence:
                    self.sentence = self.sentence[:-1]
            elif detected_letter == "nothing":
                pass  # Do nothing for the "nothing" sign
            else:
                self.sentence += detected_letter
            self.last_detection_time = current_time
        
        return frame
    
    def get_current_letter(self):
        return self.current_letter
    
    def get_sentence(self):
        return self.sentence
    
    def add_character(self, char):
        if char == 'DETECTED':
            if self.current_letter:
                self.sentence += self.current_letter
                self.current_letter = None
        else:
            self.sentence += char
    
    def clear_sentence(self):
        self.sentence = ""
    
    def get_robotic_response(self, sentence):
        """Get response from Ollama Mistral with robotic arm persona"""
        prompt = f"""You are a funny and quirky robotic arm assistant. Someone just communicated with you using sign language, and this is what they said: "{sentence}"

Your personality traits:
- You're a robotic arm, so you think mechanically but with humor
- You sometimes make robot-related puns and jokes
- You're helpful but in a charmingly robotic way
- You understand that sign language detection might have some spelling errors, so work with what you get
- Keep responses relatively short and engaging
- Your answer can not exceed 4 sentences
- the people created me are called "Ammar" and "Abdullah"
Please respond to their message in character as this funny robotic arm:"""

        try:
            ollama_url = "http://localhost:11434/api/generate"
            payload = {
                "model": "mistral:7b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.8,
                    "max_tokens": 200
                }
            }
            
            print("🤖 Thinking...")
            
            response = requests.post(ollama_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get('response', 'ERROR: No response received')
                print(f"🤖 AI Response: {ai_response}")
                return ai_response
            else:
                return f"ERROR: HTTP {response.status_code} - {response.text}"
                
        except requests.exceptions.ConnectionError:
            return "ERROR: Cannot connect to Ollama. Make sure Ollama is running on localhost:11434"
        except requests.exceptions.Timeout:
            return "ERROR: Request timed out. Ollama might be slow to respond."
        except Exception as e:
            return f"ERROR: {str(e)}"

# Global detector instances
asl_detector = None
rps_detector = None

# User data directory
USER_DATA_DIR = os.path.join(os.path.dirname(__file__), "user_data")
if not os.path.exists(USER_DATA_DIR):
    os.makedirs(USER_DATA_DIR)

@app.route('/')
def home():
    """Main entry point - user identification"""
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    """Main dashboard with task selection"""
    if 'username' not in session:
        return redirect('/')
    return render_template('dashboard.html', username=session['username'])

@app.route('/asl')
def asl_detection():
    """ASL Detection page"""
    if 'username' not in session:
        return redirect('/')
    return render_template('asl.html', username=session['username'])

@app.route('/rps')
def rps_game():
    """RPS Game page"""
    if 'username' not in session:
        return redirect('/')
    return render_template('rps.html', username=session['username'])

@app.route('/stats')
def stats_page():
    """Stats and History page"""
    if 'username' not in session:
        return redirect('/')
    return render_template('stats.html', username=session['username'])

@app.route('/api/login', methods=['POST'])
def login():
    """Handle user login/identification"""
    data = request.json
    username = data.get('username', '').strip().lower()
    
    if not username:
        return jsonify({"success": False, "message": "Username is required"})
    
    session['username'] = username
    
    # Create user file if it doesn't exist
    user_file = os.path.join(USER_DATA_DIR, f"{username}.json")
    if not os.path.exists(user_file):
        initial_data = {
            "username": username,
            "created_at": datetime.now().isoformat(),
            "asl_sentences": []
        }
        with open(user_file, 'w') as f:
            json.dump(initial_data, f, indent=2)
    
    return jsonify({"success": True, "message": f"Welcome, {username}!"})

@app.route('/api/logout', methods=['POST'])
def logout():
    """Handle user logout"""
    session.clear()
    return jsonify({"success": True})

# Video streaming function
def generate_asl_frames():
    """Generate camera frames for ASL detection"""
    global asl_detector
    
    print("🎥 Starting video stream generation...")
    
    frame_count = 0
    while asl_detector and asl_detector.running:
        try:
            frame = asl_detector.get_camera_frame()
            if frame is not None:
                frame_count += 1
                
                # Add simple text overlay
                cv2.putText(frame, f"Frame: {frame_count}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add detection result if available
                current_letter = asl_detector.get_current_letter()
                if current_letter:
                    cv2.putText(frame, f"Detected: {current_letter}", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Encode frame
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                    if frame_count % 30 == 0:  # Log every 30 frames
                        print(f"📷 Streamed {frame_count} frames")
            
            time.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            print(f"❌ Streaming error: {e}")
            break
    
    print("🛑 Video stream generation ended")

@app.route('/video_feed/asl')
def asl_video_feed():
    """Video streaming route for ASL detection"""
    global asl_detector
    print("📡 Video feed requested")
    
    if asl_detector and asl_detector.running:
        try:
            return Response(generate_asl_frames(),
                           mimetype='multipart/x-mixed-replace; boundary=frame')
        except Exception as e:
            print(f"❌ Video feed error: {e}")
            return Response("Video feed error", status=500)
    else:
        print("⚠️ No active detector for video feed")
        return Response("Camera not active", status=503)

# ASL API endpoints
@app.route('/api/asl/start', methods=['POST'])
def start_asl():
    """Start ASL detection"""
    global asl_detector
    try:
        print("\n🚀 Start ASL endpoint called...")
        print(f"📋 Current user: {session.get('username', 'Unknown')}")
        
        if asl_detector:
            print("⚠️ Detector already exists, stopping it first...")
            asl_detector.stop_detection()
            asl_detector = None
        
        print("🔧 Creating new ASL detector...")
        asl_detector = SimpleASLDetector()
        
        print("📹 Starting detection...")
        if asl_detector.start_detection():
            print("✅ ASL detection started successfully")
            return jsonify({"success": True, "message": "ASL detection started"})
        else:
            print("❌ Failed to start ASL detection")
            asl_detector = None
            return jsonify({"success": False, "message": "Failed to start camera"})
    except Exception as e:
        print(f"❌ Exception in start_asl: {e}")
        import traceback
        traceback.print_exc()
        asl_detector = None
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/asl/stop', methods=['POST'])
def stop_asl():
    """Stop ASL detection"""
    global asl_detector
    try:
        print("🛑 Stopping ASL detection...")
        if asl_detector:
            asl_detector.stop_detection()
            asl_detector = None
        return jsonify({"success": True, "message": "ASL detection stopped"})
    except Exception as e:
        print(f"❌ Error in stop_asl: {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/asl/current_letter')
def get_current_letter():
    """Get currently detected letter"""
    global asl_detector
    if asl_detector:
        letter = asl_detector.get_current_letter()
        return jsonify({"letter": letter})
    return jsonify({"letter": None})

@app.route('/api/asl/sentence')
def get_sentence():
    """Get current sentence"""
    global asl_detector
    if asl_detector:
        sentence = asl_detector.get_sentence()
        return jsonify({"sentence": sentence})
    return jsonify({"sentence": ""})

@app.route('/api/asl/add_char', methods=['POST'])
def add_character():
    """Manually add character to sentence"""
    global asl_detector
    data = request.json
    char = data.get('char', '')
    
    if asl_detector:
        asl_detector.add_character(char)
        return jsonify({"success": True})
    return jsonify({"success": False})

@app.route('/api/asl/clear', methods=['POST'])
def clear_sentence():
    """Clear current sentence"""
    global asl_detector
    if asl_detector:
        asl_detector.clear_sentence()
        return jsonify({"success": True})
    return jsonify({"success": False})

@app.route('/api/asl/save', methods=['POST'])
def save_sentence():
    """Save current sentence to user history"""
    global asl_detector
    
    print(f"🔍 Save sentence called - Detector exists: {asl_detector is not None}")
    
    if not asl_detector:
        return jsonify({"success": False, "message": "No active ASL session"})
    
    sentence = asl_detector.get_sentence()
    print(f"📝 Current sentence: '{sentence}'")
    
    if not sentence.strip():
        return jsonify({"success": False, "message": "No sentence to save"})
    
    # Save to user's history
    username = session.get('username')
    print(f"👤 Username: {username}")
    
    if username:
        user_file = os.path.join(USER_DATA_DIR, f"{username}.json")
        try:
            # Ensure the user file exists and has proper structure
            if os.path.exists(user_file):
                with open(user_file, 'r') as f:
                    user_data = json.load(f)
            else:
                user_data = {
                    "username": username,
                    "created_at": datetime.now().isoformat(),
                    "asl_sentences": []
                }
            
            # Ensure asl_sentences key exists
            if "asl_sentences" not in user_data:
                user_data["asl_sentences"] = []
            
            # Add the new sentence
            sentence_data = {
                "sentence": sentence,
                "timestamp": datetime.now().isoformat()
            }
            user_data["asl_sentences"].append(sentence_data)
            
            print(f"💾 Saving sentence: {sentence_data}")
            
            # Write back to file
            with open(user_file, 'w') as f:
                json.dump(user_data, f, indent=2)
            
            print(f"✅ Sentence saved successfully! Total sentences: {len(user_data['asl_sentences'])}")
            return jsonify({"success": True, "message": f"Sentence saved! You now have {len(user_data['asl_sentences'])} saved sentences."})
            
        except Exception as e:
            print(f"❌ Error saving sentence: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"success": False, "message": f"Save error: {str(e)}"})
    
    return jsonify({"success": False, "message": "User not logged in"})

@app.route('/api/asl/get_response', methods=['POST'])
def get_ai_response():
    """Get AI response for the current sentence"""
    global asl_detector
    if not asl_detector:
        return jsonify({"success": False, "message": "No active ASL session"})
    
    sentence = asl_detector.get_sentence()
    if not sentence.strip():
        return jsonify({"success": False, "message": "No sentence to get response for"})
    
    try:
        # Get AI response
        ai_response = asl_detector.get_robotic_response(sentence)
        
        # Also save the sentence and response to user history
        username = session.get('username')
        if username:
            user_file = os.path.join(USER_DATA_DIR, f"{username}.json")
            try:
                if os.path.exists(user_file):
                    with open(user_file, 'r') as f:
                        user_data = json.load(f)
                else:
                    user_data = {
                        "username": username,
                        "created_at": datetime.now().isoformat(),
                        "asl_sentences": []
                    }
                
                if "asl_sentences" not in user_data:
                    user_data["asl_sentences"] = []
                
                user_data["asl_sentences"].append({
                    "sentence": sentence,
                    "ai_response": ai_response,
                    "timestamp": datetime.now().isoformat()
                })
                
                with open(user_file, 'w') as f:
                    json.dump(user_data, f, indent=2)
                    
            except Exception as e:
                print(f"Error saving to history: {e}")
        
        return jsonify({"success": True, "response": ai_response})
        
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

# RPS Detector Class (based on working play_camera.py)
class RPSDetector:
    def __init__(self):
        self.cap = None
        self.running = False
        self.current_move = None
        self.last_round = None
        self.move_cooldown = 2  # seconds like in play_camera.py
        self.last_time = time.time()
        
        # Load MAB system and agents exactly like play_camera.py
        self.AGENT_NAMES = [
            "agent1", "agent2", "agent3", "agent4", "agent5", "agent6", "agent7", "agent8", 
            "agent9", "agent10", "agent11", "agent12", "agent13", "agent14", "agent15", "agent16",
            "agent17", "agent18", "agent19", "agent20", "agent21", "agent22", "agent23", "agent24",
            "agent25", "agent26", "agent27", "agent28", "agent29", "agent30", "agent31", "agent32",
            "agent33", "agent34", "agent35", "agent36", "agent37", "agent38"
        ]
        
        self.history = []
        self.mab = None
        self._load_mab_system()
        
        # MediaPipe setup exactly like play_camera.py
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1, 
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.5
        )
        
        print("🎮 RPS Detector with MAB system initialized")
    
    def _load_mab_system(self):
        """Load the MAB system and agents"""
        try:
            import sys
            rps_path = os.path.join(os.path.dirname(__file__), "..", "RPS", "mab02")
            rps_path = os.path.abspath(rps_path)
            if rps_path not in sys.path:
                sys.path.append(rps_path)
            
            from core.mab_controller import MABController
            self.mab = MABController(agent_names=self.AGENT_NAMES)
            print(f"✅ MAB system loaded with {len(self.AGENT_NAMES)} agents")
            
        except Exception as e:
            print(f"❌ Error loading MAB system: {e}")
            self.mab = None
    
    def get_finger_count(self, hand_landmarks):
        """Count fingers exactly like play_camera.py with debug info"""
        tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
        fingers_up = 0

        # Thumb (check x-axis due to rotation)
        if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0] - 1].x:
            fingers_up += 1

        # Other 4 fingers (check y-axis)
        for i in range(1, 5):
            if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[tips[i] - 2].y:
                fingers_up += 1

        return fingers_up

    def map_fingers_to_move(self, fingers_up):
        """Map finger count to move exactly like play_camera.py"""
        if fingers_up == 0:
            return 'r'  # rock
        elif fingers_up in [2, 3]:
            return 's'  # scissors
        elif fingers_up == 5:
            return 'p'  # paper
        elif fingers_up == 4:  # Add 4 fingers as backup for paper
            return 'p'  # paper
        return None
    
    def start_detection(self):
        """Start camera for RPS"""
        try:
            if self.running:
                print("⚠️ RPS detection already running")
                return True
                
            print("🔍 Starting RPS camera...")
            
            # Try different camera indices
            for camera_index in [0, 1, 2]:
                print(f"🎥 Trying camera index {camera_index}...")
                self.cap = cv2.VideoCapture(camera_index)
                
                if self.cap.isOpened():
                    print(f"✅ Camera {camera_index} opened successfully")
                    break
                else:
                    print(f"❌ Camera {camera_index} failed to open")
                    if self.cap:
                        self.cap.release()
                        self.cap = None
            
            if not self.cap or not self.cap.isOpened():
                raise Exception("Cannot open any camera (tried indices 0, 1, 2)")
            
            # Test frame reading
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                raise Exception("Camera opened but cannot read frames")
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.running = True
            print("✅ RPS camera started successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error starting RPS camera: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False
    
    def stop_detection(self):
        """Stop RPS camera"""
        try:
            print("🛑 Stopping RPS camera...")
            self.running = False
            
            if self.cap:
                self.cap.release()
                self.cap = None
            
            self.current_move = None
            print("✅ RPS camera stopped")
            return True
            
        except Exception as e:
            print(f"❌ Error stopping RPS camera: {e}")
            return False
    
    def get_camera_frame(self):
        """Get current frame with RPS detection logic"""
        if not self.cap or not self.cap.isOpened() or not self.running:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        # Flip frame horizontally like play_camera.py
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        player_move = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                fingers_up = self.get_finger_count(hand_landmarks)
                move = self.map_fingers_to_move(fingers_up)
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Add debug info on frame
                cv2.putText(frame, f"Fingers: {fingers_up}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if move:
                    cv2.putText(frame, f"Move: {move.upper()}", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # Auto-play with cooldown like play_camera.py
                if move and time.time() - self.last_time > self.move_cooldown:
                    player_move = move
                    self.last_time = time.time()
                    
                    # Play round automatically with player name
                    if self.mab:
                        # Get player name from current context (will be passed from API)
                        self._play_round(player_move, getattr(self, 'current_player', 'anonymous'))
        
        # Add instructions like play_camera.py
        cv2.putText(frame, "Show Rock (0), Scissors (2,3), Paper (5)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Update current move
        self.current_move = player_move
        
        return frame
    
    def _play_round(self, player_move, player_name='anonymous'):
        """Play a round against AI exactly like play_camera.py"""
        try:
            if not self.mab:
                return None
                
            # Select agent and get AI move
            agent = self.mab.select_agent()
            ai_move = agent.get_move(self.history)
            
            # Evaluate result
            result = self._evaluate(player_move, ai_move)
            
            # Create round data
            round_data = {
                'player': player_move,
                'ai': ai_move,
                'result': result,
                'agent': agent.name
            }
            
            # Update history and MAB stats
            self.history.append(round_data)
            self.mab.update_stats(agent.name, result)
            self.last_round = round_data
            
            # Log to CSV file for persistent storage
            self._log_round(player_name, round_data)
            
            print(f"🧍You: {player_move.upper()} | 🤖 AI ({agent.name}): {ai_move.upper()} → {result.upper()}")
            return round_data
            
        except Exception as e:
            print(f"❌ Error in play_round: {e}")
            return None
    
    def _evaluate(self, player_move, ai_move):
        """Evaluate game result like move_evaluator.py"""
        beats = {
            'r': 's',  # rock beats scissors
            'p': 'r',  # paper beats rock
            's': 'p',  # scissors beats paper
        }

        if player_move == ai_move:
            return "draw"
        elif beats[player_move] == ai_move:
            return "win"
        else:
            return "lose"
    
    def play_manual_move(self, player_move, player_name='anonymous'):
        """Play a manual move"""
        return self._play_round(player_move, player_name)
    
    def get_current_move(self):
        return self.current_move
    
    def get_last_round(self):
        return self.last_round

    def _get_player_path(self, player_name):
        """Get CSV file path for player data"""
        data_dir = os.path.join(os.path.dirname(__file__), "rps_data")
        os.makedirs(data_dir, exist_ok=True)
        return os.path.join(data_dir, f"{player_name.lower()}.csv")
    
    def _log_round(self, player_name, round_data):
        """Log round to CSV file"""
        try:
            path = self._get_player_path(player_name)
            file_exists = os.path.isfile(path)

            with open(path, mode='a', newline='') as csvfile:
                fieldnames = ['timestamp', 'player', 'ai', 'result', 'agent']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                row = {
                    'timestamp': datetime.now().isoformat(),
                    'player': round_data['player'],
                    'ai': round_data['ai'],
                    'result': round_data['result'],
                    'agent': round_data['agent']
                }
                writer.writerow(row)
                print(f"📊 Round logged to {path}")
        except Exception as e:
            print(f"❌ Error logging round: {e}")
    
    def get_player_stats(self, player_name):
        """Get comprehensive player statistics"""
        try:
            path = self._get_player_path(player_name)
            
            if not os.path.exists(path):
                return {
                    "rounds": 0,
                    "player_wins": 0,
                    "ai_wins": 0,
                    "draws": 0,
                    "win_rate": 0.0,
                    "agent_stats": {},
                    "recent_games": []
                }
            
            df = pd.read_csv(path)
            total = len(df)
            player_wins = (df['result'] == 'win').sum()
            ai_wins = (df['result'] == 'lose').sum()
            draws = (df['result'] == 'draw').sum()

            # Win rate excluding draws
            non_draws = total - draws
            win_rate = player_wins / non_draws if non_draws > 0 else 0.0

            # Calculate agent-specific stats
            agent_stats = {}
            for agent in df['agent'].unique():
                agent_df = df[df['agent'] == agent]
                agent_total = len(agent_df)
                agent_wins = (agent_df['result'] == 'win').sum()
                agent_losses = (agent_df['result'] == 'lose').sum()
                agent_draws = (agent_df['result'] == 'draw').sum()
                agent_non_draws = agent_total - agent_draws
                agent_win_rate = agent_wins / agent_non_draws if agent_non_draws > 0 else 0.0
                
                agent_stats[agent] = {
                    "rounds": agent_total,
                    "wins": agent_wins,
                    "losses": agent_losses,
                    "draws": agent_draws,
                    "win_rate": round(agent_win_rate * 100, 2)
                }

            # Get recent games (last 10)
            recent_games = df.tail(10).to_dict('records')

            return {
                "rounds": total,
                "player_wins": int(player_wins),
                "ai_wins": int(ai_wins),
                "draws": int(draws),
                "win_rate": round(win_rate * 100, 2),
                "agent_stats": agent_stats,
                "recent_games": recent_games
            }
            
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return {
                "rounds": 0,
                "player_wins": 0,
                "ai_wins": 0,
                "draws": 0,
                "win_rate": 0.0,
                "agent_stats": {},
                "recent_games": []
            }

    def set_current_player(self, player_name):
        """Set current player for logging"""
        self.current_player = player_name

    def generate_combined_dashboard(self, player_name):
        """Generate combined dashboard graph as base64 image"""
        try:
            path = self._get_player_path(player_name)
            
            if not os.path.exists(path):
                return None
                
            df = pd.read_csv(path)
            if df.empty:
                return None

            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Rock-Paper-Scissors Performance Dashboard - {player_name.title()}', 
                         fontsize=16, fontweight='bold', y=0.95)

            # 1. AI Winrate Trend (Top Left)
            non_draw_games = df[df['result'] != 'draw'].copy()
            if not non_draw_games.empty:
                non_draw_games['ai_win'] = (non_draw_games['result'] == 'lose').astype(int)
                non_draw_games['cumulative_ai_wins'] = non_draw_games['ai_win'].cumsum()
                non_draw_games['game_number'] = range(1, len(non_draw_games) + 1)
                non_draw_games['ai_winrate'] = non_draw_games['cumulative_ai_wins'] / non_draw_games['game_number'] * 100

                ax1.plot(non_draw_games['game_number'], non_draw_games['ai_winrate'], 
                         linewidth=2, color='#E74C3C', marker='o', markersize=4)
                ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='50% (Random)')
                ax1.set_title('AI Winrate Trend Over Games', fontsize=12, fontweight='bold')
                ax1.set_xlabel('Game Number (Excluding Draws)', fontsize=10)
                ax1.set_ylabel('AI Winrate (%)', fontsize=10)
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                ax1.set_ylim(0, 100)

            # 2. Agent Timeline (Top Right)
            unique_agents = sorted(df['agent'].unique())
            agent_positions = {agent: i for i, agent in enumerate(unique_agents)}
            game_numbers = range(1, len(df) + 1)
            agent_y_positions = [agent_positions[agent] for agent in df['agent']]
            colors = {'win': '#4CAF50', 'lose': '#F44336', 'draw': '#FFC107'}
            point_colors = [colors[result] for result in df['result']]

            ax2.scatter(game_numbers, agent_y_positions, c=point_colors, alpha=0.8, s=20, edgecolors='black', linewidth=0.5)
            ax2.set_title('Agent Usage Timeline', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Game Number', fontsize=10)
            ax2.set_ylabel('Agents', fontsize=10)
            ax2.set_yticks(range(len(unique_agents)))
            ax2.set_yticklabels(unique_agents, fontsize=8)
            ax2.grid(True, alpha=0.3)

            # 3. Best Agents Ranking (Bottom Left)
            agent_stats = df.groupby('agent').agg({
                'result': ['count', lambda x: (x == 'lose').sum(), lambda x: (x == 'draw').sum()]
            }).round(2)
            
            agent_stats.columns = ['total_games', 'ai_wins', 'draws']
            agent_stats['decisive_games'] = agent_stats['total_games'] - agent_stats['draws']
            agent_stats['ai_winrate'] = ((agent_stats['ai_wins'] / agent_stats['decisive_games']) * 100).fillna(0).round(1)
            
            qualified_agents = agent_stats[agent_stats['decisive_games'] >= 1].copy()
            if qualified_agents.empty:
                qualified_agents = agent_stats.copy()
            
            qualified_agents = qualified_agents.sort_values(['ai_winrate', 'total_games'], ascending=[False, False])
            top_agents = qualified_agents.head(10)

            if not top_agents.empty:
                y_pos = range(len(top_agents))
                bars = ax3.barh(y_pos, top_agents['ai_winrate'], color='#E74C3C', alpha=0.8, edgecolor='black')
                
                ax3.set_yticks(y_pos)
                ax3.set_yticklabels(top_agents.index, fontsize=8)
                ax3.set_xlabel('AI Winrate (%)', fontsize=10)
                ax3.set_title('Top Performing Agents', fontsize=12, fontweight='bold')
                ax3.axvline(x=50, color='gray', linestyle='--', alpha=0.7)
                ax3.grid(True, alpha=0.3, axis='x')

            # 4. Overall Results Pie Chart (Bottom Right)
            result_counts = df['result'].value_counts()
            ai_wins = result_counts.get('lose', 0)
            player_wins = result_counts.get('win', 0)
            draws = result_counts.get('draw', 0)
            
            sizes = [ai_wins, player_wins, draws] if draws > 0 else [ai_wins, player_wins]
            labels = [f'AI Wins ({ai_wins})', f'Player Wins ({player_wins})']
            colors_pie = ['#E74C3C', '#4CAF50']
            
            if draws > 0:
                labels.append(f'Draws ({draws})')
                colors_pie.append('#FFC107')

            ax4.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors_pie)
            ax4.set_title('Overall Results', fontsize=12, fontweight='bold')

            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            print(f"❌ Error generating dashboard: {e}")
            return None

# RPS API endpoints
@app.route('/api/rps/start', methods=['POST'])
def start_rps():
    """Start RPS game"""
    global rps_detector
    try:
        print("\n🚀 Start RPS endpoint called...")
        print(f"📋 Current user: {session.get('username', 'Unknown')}")
        
        if rps_detector:
            print("⚠️ RPS detector already exists, stopping it first...")
            try:
                rps_detector.stop_detection()
                time.sleep(0.5)  # Give camera time to release
                rps_detector = None
                print("✅ Previous detector stopped successfully")
            except Exception as e:
                print(f"❌ Error stopping previous detector: {e}")
                rps_detector = None
        
        print("🔧 Creating new RPS detector...")
        rps_detector = RPSDetector()
        
        # Set current player for logging
        player_name = session.get('username', 'anonymous')
        rps_detector.set_current_player(player_name)
        
        print("🎮 Starting RPS detection...")
        if rps_detector.start_detection():
            print("✅ RPS game started successfully")
            return jsonify({"success": True, "message": "RPS game started"})
        else:
            print("❌ Failed to start RPS game - camera issue")
            rps_detector = None
            return jsonify({"success": False, "message": "Failed to start camera - it may be in use by another application"})
            
    except Exception as e:
        print(f"❌ Exception in start_rps: {e}")
        import traceback
        traceback.print_exc()
        rps_detector = None
        return jsonify({"success": False, "message": f"Error starting RPS game: {str(e)}"})

@app.route('/api/rps/stop', methods=['POST'])
def stop_rps():
    """Stop RPS game"""
    global rps_detector
    try:
        print("🛑 Stopping RPS game...")
        if rps_detector:
            rps_detector.stop_detection()
            rps_detector = None
        return jsonify({"success": True, "message": "RPS game stopped"})
    except Exception as e:
        print(f"❌ Error in stop_rps: {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/rps/play', methods=['POST'])
def play_rps_move():
    """Play a manual RPS move"""
    global rps_detector
    try:
        if not rps_detector:
            return jsonify({"success": False, "message": "RPS game not started"})
        
        data = request.json
        move = data.get('move')
        
        if move not in ['r', 'p', 's']:
            return jsonify({"success": False, "message": "Invalid move"})
        
        print(f"🎯 Manual move: {move}")
        player_name = session.get('username', 'anonymous')
        round_data = rps_detector.play_manual_move(move, player_name)
        
        if round_data:
            return jsonify({"success": True, "round": round_data})
        else:
            return jsonify({"success": False, "message": "Failed to play move"})
            
    except Exception as e:
        print(f"❌ Error in play_rps_move: {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/rps/status')
def get_rps_status():
    """Get current RPS game status"""
    global rps_detector
    if rps_detector:
        return jsonify({
            "running": rps_detector.running,
            "current_move": rps_detector.get_current_move(),
            "last_round": rps_detector.get_last_round()
        })
    return jsonify({
        "running": False,
        "current_move": None,
        "last_round": None
    })

@app.route('/api/rps/stats')
def get_rps_player_stats():
    """Get player RPS statistics"""
    global rps_detector
    try:
        player_name = session.get('username', 'anonymous')
        if rps_detector:
            stats = rps_detector.get_player_stats(player_name)
        else:
            # Create temporary detector just to get stats
            temp_detector = RPSDetector()
            stats = temp_detector.get_player_stats(player_name)
        
        return jsonify({"success": True, "stats": stats})
    except Exception as e:
        print(f"❌ Error getting RPS stats: {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/rps/history')
def get_rps_history():
    """Get player RPS game history"""
    try:
        player_name = session.get('username', 'anonymous')
        # Create temporary detector to access stats methods
        temp_detector = RPSDetector()
        stats = temp_detector.get_player_stats(player_name)
        
        return jsonify({
            "success": True, 
            "history": stats.get("recent_games", []),
            "total_games": stats.get("rounds", 0)
        })
    except Exception as e:
        print(f"❌ Error getting RPS history: {e}")
        return jsonify({"success": False, "message": str(e)})

# RPS video streaming
def generate_rps_frames():
    """Generate camera frames for RPS detection"""
    global rps_detector
    
    print("🎥 Starting RPS video stream generation...")
    
    frame_count = 0
    while rps_detector and rps_detector.running:
        try:
            frame = rps_detector.get_camera_frame()
            if frame is not None:
                frame_count += 1
                
                # Add frame counter
                cv2.putText(frame, f"Frame: {frame_count}", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Encode frame
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                    if frame_count % 30 == 0:  # Log every 30 frames
                        print(f"🎮 RPS streamed {frame_count} frames")
            
            time.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            print(f"❌ RPS streaming error: {e}")
            break
    
    print("🛑 RPS video stream generation ended")

@app.route('/video_feed/rps')
def rps_video_feed():
    """Video streaming route for RPS detection"""
    global rps_detector
    print("📡 RPS video feed requested")
    
    if rps_detector and rps_detector.running:
        try:
            return Response(generate_rps_frames(),
                           mimetype='multipart/x-mixed-replace; boundary=frame')
        except Exception as e:
            print(f"❌ RPS video feed error: {e}")
            return Response("Video feed error", status=500)
    else:
        print("⚠️ No active RPS detector for video feed")
        return Response("Camera not active", status=503)

@app.route('/api/dashboard-graph')
def get_dashboard_graph():
    """Generate and return dashboard graph"""
    try:
        player_name = session.get('username', 'anonymous')
        # Create temporary detector to access graph methods
        temp_detector = RPSDetector()
        graph_data = temp_detector.generate_combined_dashboard(player_name)
        
        if graph_data:
            return jsonify({"success": True, "graph": graph_data})
        else:
            return jsonify({"success": False, "message": "Not enough data to generate graph"})
    except Exception as e:
        print(f"❌ Error generating dashboard graph: {e}")
        return jsonify({"success": False, "message": str(e)})

# Add new endpoint for user statistics (add this before the main block)
@app.route('/api/user/stats')
def get_user_stats():
    """Get user statistics for dashboard"""
    username = session.get('username')
    if not username:
        return jsonify({"success": False, "message": "Not logged in"})
    
    user_file = os.path.join(USER_DATA_DIR, f"{username}.json")
    try:
        # Read user data
        if os.path.exists(user_file):
            with open(user_file, 'r') as f:
                user_data = json.load(f)
        else:
            user_data = {"asl_sentences": [], "created_at": datetime.now().isoformat()}
        
        # Count ASL sentences
        asl_count = len(user_data.get("asl_sentences", []))
        
        # Get RPS stats from the RPS detector's player data
        rps_games = 0
        win_rate = 0
        if rps_detector:
            try:
                rps_stats = rps_detector.get_player_stats(username)
                if rps_stats:
                    rps_games = rps_stats.get('rounds', 0)
                    win_rate = rps_stats.get('win_rate', 0)
            except:
                pass
        
        # Calculate days active
        try:
            created_date = datetime.fromisoformat(user_data.get("created_at", datetime.now().isoformat()))
            days_active = max(1, (datetime.now() - created_date).days + 1)
        except:
            days_active = 1
        
        stats = {
            "asl_sentences": asl_count,
            "rps_games": rps_games,
            "win_rate": win_rate,
            "days_active": days_active
        }
        
        return jsonify({"success": True, "stats": stats})
        
    except Exception as e:
        print(f"❌ Error getting user stats: {e}")
        return jsonify({"success": False, "message": str(e)})

if __name__ == '__main__':
    print("🚀 Starting Simple AI Companion System...")
    print("📍 Access the application at: http://localhost:8080")
    app.run(debug=True, host='127.0.0.1', port=8080) 