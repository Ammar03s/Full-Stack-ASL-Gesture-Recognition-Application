import pickle
import os
import cv2
import mediapipe as mp
import numpy as np
import time
import tkinter as tk
from tkinter import scrolledtext, Button, Label, Frame, StringVar, messagebox
from threading import Thread
import datetime
import json
import requests
import threading
from PIL import Image, ImageTk
# Add missing imports 
# try:
#     from PIL import Image, ImageTk
# except ImportError:
#     messagebox.showerror("Missing Dependency") #pip install Pillow
#     exit(1)

class ASLSentenceBuilderWithLLM:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Sentence Builder with Robotic Arm Responder")
        self.root.geometry("900x900")
        self.root.configure(bg="#f0f0f0")
        
        # Load model
        print("Loading model...")
        model_dict = pickle.load(open('./model.p', 'rb'))
        self.model = model_dict['model']
        self.le = model_dict.get('label_encoder', None)
        
        # Read class mapping from file
        self.class_mapping = {}
        if os.path.exists('./data/class_mapping.txt'):
            with open('./data/class_mapping.txt', 'r') as f:
                for line in f:
                    idx, label = line.strip().split(':', 1)
                    self.class_mapping[int(idx)] = label.strip()
            print(f"Loaded {len(self.class_mapping)} classes from mapping file")
        else:
            print("Warning: No class mapping file found")
            # Fallback default mapping
            self.class_mapping = {0: 'A', 1: 'B', 2: 'C'}
        
        # Initialize ASL variables
        self.cap = None
        self.current_sentence = "Hey "
        self.last_detection_time = time.time()
        self.detection_interval = 4  #timer 
        self.running = False
        self.camera_thread = None
        self.last_detected_letter = None
        self.freeze_detection = False
        
        # MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
        
        # Ollama settings
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model_name = "mistral:7b"
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Camera frame at top
        self.camera_frame = Frame(self.root, bg="#2c3e50", width=640, height=350)# bigeer
        self.camera_frame.pack(pady=10)
        self.camera_frame.pack_propagate(False)
        
        self.camera_label = Label(self.camera_frame, bg="black")
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Middle frame for current detection and controls
        self.control_frame = Frame(self.root, bg="#f0f0f0")
        self.control_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Current letter detection
        self.detection_var = StringVar(value="Current: None")
        self.detection_label = Label(self.control_frame, textvariable=self.detection_var, 
                                   font=("Arial", 20, "bold"), bg="#f0f0f0")
        self.detection_label.pack(side=tk.LEFT, padx=10)
        
        # Control buttons
        self.start_button = Button(self.control_frame, text="Start Detection", 
                                 command=self.toggle_detection,
                                 font=("Arial", 12), bg="#27ae60", fg="white",
                                 width=15, height=2)
        self.start_button.pack(side=tk.RIGHT, padx=10)
        
        self.clear_button = Button(self.control_frame, text="Clear Sentence", 
                                 command=self.clear_sentence,
                                 font=("Arial", 12), bg="#e74c3c", fg="white",
                                 width=15, height=2)
        self.clear_button.pack(side=tk.RIGHT, padx=10)
        
        self.space_button = Button(self.control_frame, text="Add Space", 
                                 command=lambda: self.add_character(" "),
                                 font=("Arial", 12), bg="#3498db", fg="white",
                                 width=10, height=2)
        self.space_button.pack(side=tk.RIGHT, padx=10)
        
        self.backspace_button = Button(self.control_frame, text="Backspace", 
                                 command=self.remove_last_character,
                                 font=("Arial", 12), bg="#f39c12", fg="white", 
                                 width=10, height=2)
        self.backspace_button.pack(side=tk.RIGHT, padx=10)
        
        # Text frame for sentence
        self.text_frame = Frame(self.root, bg="#f0f0f0")
        self.text_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Label for the sentence
        self.sentence_label = Label(self.text_frame, text="Your Sentence:", 
                                  font=("Arial", 14), bg="#f0f0f0", anchor="w")
        self.sentence_label.pack(fill=tk.X)
        
        # Text area for displaying the sentence
        self.text_area = scrolledtext.ScrolledText(self.text_frame, 
                                                 font=("Arial", 16),
                                                 wrap=tk.WORD, 
                                                 height=3)
        self.text_area.pack(fill=tk.X, pady=(5, 0))
        
        # Bottom buttons
        self.bottom_frame = Frame(self.root, bg="#f0f0f0")
        self.bottom_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.save_button = Button(self.bottom_frame, text="Save & Get Response", 
                                command=self.save_and_respond,
                                font=("Arial", 12), bg="#9b59b6", fg="white",
                                width=20, height=2)
        self.save_button.pack(side=tk.RIGHT, padx=10)
        
        self.freeze_var = StringVar(value="Freeze OFF")
        self.freeze_button = Button(self.bottom_frame, textvariable=self.freeze_var,
                                  command=self.toggle_freeze, 
                                  font=("Arial", 12), bg="#34495e", fg="white",
                                  width=15, height=2)
        self.freeze_button.pack(side=tk.RIGHT, padx=10)
        
        # LLM Response frame (always visible but starts empty)
        self.response_frame = Frame(self.root, bg="black", relief=tk.RAISED, bd=2)
        self.response_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # LLM Response header
        self.response_header = Label(self.response_frame, text="Arm Response:", 
                                   font=("Arial", 14, "bold"), 
                                   bg="black", fg="#ecf0f1")
        self.response_header.pack(pady=10)
        
        # LLM Response area
        self.llm_response_area = scrolledtext.ScrolledText(self.response_frame, 
                                                          font=("Arial", 12),
                                                          wrap=tk.WORD, 
                                                          bg="#ecf0f1", fg="#2c3e50",
                                                          height=15)
        self.llm_response_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Initialize with placeholder text
        self.llm_response_area.insert(tk.END, "Wondering where am I? Talk to me first..")
        self.llm_response_area.config(state=tk.DISABLED)  # Make it read-only initially
        
        # Status bar
        self.status_var = StringVar(value="Ready")
        self.status_bar = Label(self.root, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W, bg="#ecf0f1")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def toggle_detection(self):
        if not self.running:
            self.start_detection()
            self.start_button.config(text="Stop Detection", bg="#c0392b")
        else:
            self.stop_detection()
            self.start_button.config(text="Start Detection", bg="#27ae60")
    
    def start_detection(self):
        self.running = True
        self.status_var.set("Starting camera...")
        
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
                
            self.camera_thread = Thread(target=self.camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            self.status_var.set("Detection running")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Camera Error", f"Could not start camera: {str(e)}")
            self.running = False
            self.start_button.config(text="Start Detection", bg="#27ae60")
    
    def stop_detection(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.camera_label.config(image="")
        self.status_var.set("Detection stopped")
        
    def camera_loop(self):
        while self.running:
            data_aux = []
            x_ = []
            y_ = []
            
            ret, frame = self.cap.read()
            if not ret:
                self.status_var.set("Failed to capture frame")
                break
                
            # Flip frame horizontally for more intuitive interaction
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
                
                # Ensure data has the right shape for the model
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
                
                # Get the prediction based on model output
                if self.le is not None:
                    original_label = self.le.inverse_transform([prediction_index])[0]
                    detected_letter = self.class_mapping.get(int(original_label), f"Unknown ({original_label})")
                else:
                    detected_letter = self.class_mapping.get(prediction_index, f"Unknown ({prediction_index})")
                
                # Draw the bounding box and prediction
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.putText(frame, detected_letter, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            
            # Add timer display and instructions
            current_time = time.time()
            time_left = max(0, self.detection_interval - (current_time - self.last_detection_time))
            
            if not self.freeze_detection:
                timer_text = f"Next letter in: {time_left:.1f}s"
                timer_color = (0, 255, 255)  # Yellow color for timer
            else:
                timer_text = "Detection paused"
                timer_color = (0, 0, 255)  # Red color for paused state
                
            cv2.putText(frame, timer_text, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, timer_color, 2, cv2.LINE_AA)
            
            # Show keyboard shortcuts on screen
            cv2.putText(frame, "Welcome :)", (10, H - 90), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "Press F to freeze", (10, H-30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                
            # Convert to format suitable for tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update GUI with camera frame and detection
            self.camera_label.config(image=imgtk)
            self.camera_label.image = imgtk  # Keep a reference
            
            if detected_letter:
                self.detection_var.set(f"Current: {detected_letter}")
                self.last_detected_letter = detected_letter
            else:
                self.detection_var.set("Current: None")
                self.last_detected_letter = None
            
            # Add letter to sentence every 4 seconds if detection is not frozen
            if not self.freeze_detection and detected_letter and (current_time - self.last_detection_time) >= self.detection_interval:
                if detected_letter == "space":
                    self.add_character(" ")
                elif detected_letter == "del":
                    self.remove_last_character()
                elif detected_letter == "nothing":
                    pass  # Do nothing for the "nothing" sign
                else:
                    self.add_character(detected_letter)
                self.last_detection_time = current_time
                
            # Required to display tkinter properly
            self.root.update()
    
    def add_character(self, char):
        self.current_sentence += char
        self.update_text_area()
        
    def remove_last_character(self):
        if self.current_sentence:
            self.current_sentence = self.current_sentence[:-1]
            self.update_text_area()
    
    def clear_sentence(self):
        self.current_sentence = ""
        self.update_text_area()
        
    def update_text_area(self):
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, self.current_sentence)
        
    def toggle_freeze(self):
        self.freeze_detection = not self.freeze_detection
        if self.freeze_detection:
            self.freeze_var.set("Freeze ON")
            self.freeze_button.config(bg="#e74c3c")
        else:
            self.freeze_var.set("Freeze OFF")
            self.freeze_button.config(bg="#34495e")
            # Reset the timer when detection is resumed
            self.last_detection_time = time.time()
    
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
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.8,
                    "max_tokens": 200
                }
            }
            
            self.status_var.set("🤖 Thinking...")
            self.root.update()
            
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'ERROR: No response received')
            else:
                return f"ERROR: HTTP {response.status_code} - {response.text}"
                
        except requests.exceptions.ConnectionError:
            return "ERROR: Cannot connect to Ollama. Make sure Ollama is running on localhost:11434"
        except requests.exceptions.Timeout:
            return "ERROR: Request timed out. Ollama might be slow to respond."
        except Exception as e:
            return f"ERROR: {str(e)}"
            
    def save_and_respond(self):
        """Save sentence and get LLM response"""
        if not self.current_sentence.strip():
            messagebox.showinfo("Info", "No text to save.")
            return
            
        # Save the sentence
        os.makedirs("saved_sentences", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"saved_sentences/sentence_{timestamp}.json"
        
        try:
            with open(filename, "w") as f:
                json.dump({"sentence": self.current_sentence}, f)
            self.status_var.set(f"Sentence saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save sentence: {str(e)}")
            return
        
        # Get LLM response in a separate thread
        threading.Thread(target=self.process_llm_response, 
                        args=(self.current_sentence,), daemon=True).start()
    
    def process_llm_response(self, sentence):
        """Process LLM response in separate thread"""
        # Enable the text area and show processing message
        self.llm_response_area.config(state=tk.NORMAL)
        self.llm_response_area.delete(1.0, tk.END)
        self.llm_response_area.insert(tk.END, "🤖 Processing your message...")
        self.llm_response_area.config(state=tk.DISABLED)
        self.llm_response_area.see(tk.END)
        self.root.update()
        
        # Get response from LLM
        response = self.get_robotic_response(sentence)
        
        # Update with actual response
        self.llm_response_area.config(state=tk.NORMAL)
        self.llm_response_area.delete(1.0, tk.END)
        self.llm_response_area.insert(tk.END, response)
        self.llm_response_area.config(state=tk.DISABLED)
        self.llm_response_area.see(tk.END)
        
        self.status_var.set("Response received! Ready for next sentence.")
    
    def on_closing(self):
        if self.running:
            self.stop_detection()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ASLSentenceBuilderWithLLM(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
