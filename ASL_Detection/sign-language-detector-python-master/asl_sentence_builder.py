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

class ASLSentenceBuilder:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Interpreter")
        self.root.geometry("960x720")
        self.root.configure(bg="#1E2A45")  # Deep blue-gray background (calming, professional)
        
        # Load model
        print("Loading model...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'model.p')
        print(f"Loading model from: {model_path}")
        model_dict = pickle.load(open(model_path, 'rb'))
        self.model = model_dict['model']
        self.le = model_dict.get('label_encoder', None)
        
        # Read class mapping from file
        self.class_mapping = {}
        class_mapping_path = os.path.join(current_dir, 'data', 'class_mapping.txt')
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
        
        # Initialize variables
        self.cap = None
        self.current_sentence = ""
        self.last_detection_time = time.time()
        self.detection_interval = 4  # seconds
        self.running = False
        self.camera_thread = None
        self.last_detected_letter = None
        self.freeze_detection = False
        
        # MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Main container
        self.main_container = Frame(self.root, bg="#1E2A45")  # Deep blue-gray
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Left column - for camera and sentence
        self.left_column = Frame(self.main_container, bg="#1E2A45", width=600)
        self.left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.left_column.pack_propagate(False)
        
        # Right column - for all controls
        self.right_column = Frame(self.main_container, bg="#2D3B55", width=300)  # Slightly lighter blue
        self.right_column.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        self.right_column.pack_propagate(False)
        
        # Title at the top right
        self.title_frame = Frame(self.right_column, bg="#2D3B55", pady=15)
        self.title_frame.pack(fill=tk.X)
        
        self.app_title = Label(self.title_frame, text="SIGN LANGUAGE\nINTERPRETER", 
                             font=("Helvetica", 18, "bold"), 
                             bg="#2D3B55", fg="#64DFDF",  # Teal - encourages communication
                             justify=tk.CENTER)
        self.app_title.pack()
        
        # 1. CAMERA - top left, very large
        self.camera_container = Frame(self.left_column, bg="#2D3B55", padx=8, pady=8, 
                                   highlightbackground="#64DFDF", highlightthickness=2)  # Teal border
        self.camera_container.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Make camera as large as possible
        self.camera_frame = Frame(self.camera_container, bg="#2D3B55", width=580, height=500)
        self.camera_frame.pack()
        self.camera_frame.pack_propagate(False)
        
        self.camera_label = Label(self.camera_frame, bg="#000000")  # Black background for camera
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # 2. SENTENCE - below camera on left side
        self.text_container = Frame(self.left_column, bg="#2D3B55", padx=8, pady=8,
                                  highlightbackground="#64DFDF", highlightthickness=1)  # Teal border
        self.text_container.pack(fill=tk.X, pady=(0, 10))
        
        self.text_frame = Frame(self.text_container, bg="#2D3B55")
        self.text_frame.pack(fill=tk.X)
        
        self.sentence_label = Label(self.text_frame, text="YOUR SENTENCE:", 
                                  font=("Helvetica", 14, "bold"), 
                                  bg="#2D3B55", fg="#64DFDF",  # Teal for communication
                                  anchor="w")
        self.sentence_label.pack(fill=tk.X, pady=(0, 5), padx=5)
        
        self.text_area = scrolledtext.ScrolledText(self.text_frame, 
                                                font=("Helvetica", 16),
                                                wrap=tk.WORD, 
                                                height=3,
                                                bg="#1E2A45",  # Darker for focus on text
                                                fg="#FFFFFF",  # White text for readability
                                                insertbackground="#64DFDF")  # Teal cursor
        self.text_area.pack(fill=tk.X, padx=5, pady=5)
        
        # 3. CURRENT DETECTION - below sentence on left
        self.detection_frame = Frame(self.left_column, bg="#2D3B55", padx=8, pady=8,
                                   highlightbackground="#64DFDF", highlightthickness=1)  # Teal border
        self.detection_frame.pack(fill=tk.X)
        
        self.detection_var = StringVar(value="Current: None")
        self.detection_label = Label(self.detection_frame, textvariable=self.detection_var, 
                                  font=("Helvetica", 22, "bold"), 
                                  bg="#2D3B55", fg="#FFA62B")  # Orange for attention
        self.detection_label.pack(fill=tk.X, pady=5)
        
        # 4. CONTROLS - stacked vertically on right side
        
        # Separator
        Frame(self.right_column, height=2, bg="#64DFDF").pack(fill=tk.X, pady=10, padx=15)  # Teal separator
        
        # Control panel label
        Label(self.right_column, text="CONTROLS", 
             font=("Helvetica", 16, "bold"), 
             bg="#2D3B55", fg="#64DFDF").pack(pady=(0, 15))  # Teal for organized action
        
        # Control buttons - all stacked vertically
        self.buttons_frame = Frame(self.right_column, bg="#2D3B55", padx=20)
        self.buttons_frame.pack(fill=tk.X, pady=5, expand=True)
        
        # Start/Stop button
        self.start_button = Button(self.buttons_frame, text="START DETECTION", 
                                command=self.toggle_detection,
                                font=("Helvetica", 12, "bold"), 
                                bg="#5FA8D3", fg="#FFFFFF",  # Blue - trust and reliability
                                height=2,
                                relief=tk.FLAT)
        self.start_button.pack(fill=tk.X, pady=10)
        
        # Freeze button
        self.freeze_var = StringVar(value="FREEZE OFF")
        self.freeze_button = Button(self.buttons_frame, textvariable=self.freeze_var,
                                 command=self.toggle_freeze, 
                                 font=("Helvetica", 12, "bold"), 
                                 bg="#2D3B55", fg="#FFFFFF",  # Dark blue - control
                                 height=2,
                                 relief=tk.FLAT)
        self.freeze_button.pack(fill=tk.X, pady=10)
        
        # Space button
        self.space_button = Button(self.buttons_frame, text="ADD SPACE", 
                                command=lambda: self.add_character(" "),
                                font=("Helvetica", 12, "bold"),
                                bg="#64DFDF", fg="#1E2A45",  # Teal - clarity and spaciousness
                                height=2,
                                relief=tk.FLAT)
        self.space_button.pack(fill=tk.X, pady=10)
        
        # Backspace button
        self.backspace_button = Button(self.buttons_frame, text="BACKSPACE", 
                                    command=self.remove_last_character,
                                    font=("Helvetica", 12, "bold"), 
                                    bg="#FFA62B", fg="#1E2A45",  # Orange - caution, adjusting
                                    height=2,
                                    relief=tk.FLAT)
        self.backspace_button.pack(fill=tk.X, pady=10)
        
        # Clear button
        self.clear_button = Button(self.buttons_frame, text="CLEAR SENTENCE", 
                                command=self.clear_sentence,
                                font=("Helvetica", 12, "bold"),
                                bg="#FF6B6B", fg="#FFFFFF",  # Red - clearing, fresh start
                                height=2,
                                relief=tk.FLAT)
        self.clear_button.pack(fill=tk.X, pady=10)
        
        # Save button
        self.save_button = Button(self.buttons_frame, text="SAVE SENTENCE", 
                               command=self.save_sentence,
                               font=("Helvetica", 12, "bold"),
                               bg="#80ED99", fg="#1E2A45",  # Green - success, completion
                               height=2,
                               relief=tk.FLAT)
        self.save_button.pack(fill=tk.X, pady=10)
        
        # Status bar at the very bottom
        self.status_var = StringVar(value="Ready")
        self.status_bar = Label(self.root, textvariable=self.status_var, 
                             relief=tk.FLAT, anchor=tk.W, 
                             bg="#1E2A45", fg="#80ED99",  # Green status indicates readiness
                             font=("Helvetica", 10), padx=10, pady=3)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def toggle_detection(self):
        if not self.running:
            self.start_detection()
            self.start_button.config(text="STOP DETECTION", bg="#FF6B6B")  # Red for stop
        else:
            self.stop_detection()
            self.start_button.config(text="START DETECTION", bg="#5FA8D3")  # Blue for start
    
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
            self.start_button.config(text="START DETECTION", bg="#5FA8D3")
    
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
            
            # Resize frame to ensure it fits in the larger display area
            H, W, _ = frame.shape
            frame = cv2.resize(frame, (int(580*0.95), int(500*0.95)), interpolation=cv2.INTER_AREA)
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
                
                # Draw the bounding box and prediction with updated styling
                cv2.rectangle(frame, (x1, y1), (x2, y2), (233, 69, 96), 4)  # E94560 in RGB
                cv2.putText(frame, detected_letter, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.3, (233, 69, 96), 3, cv2.LINE_AA)
            
            # Add timer display with semi-transparent text (no box)
            current_time = time.time()
            time_left = max(0, self.detection_interval - (current_time - self.last_detection_time))
            
            # Define timer text and color
            if not self.freeze_detection:
                timer_text = f"Next letter in: {time_left:.1f}s"
                base_color = (100, 223, 223)  # 64DFDF (teal) in RGB
            else:
                timer_text = "Detection paused"
                base_color = (255, 166, 43)  # FFA62B (orange) in RGB
            
            # Create semi-transparent text by drawing shadow with partial alpha
            # Create a copy for shadow effect
            text_overlay = frame.copy()
            cv2.putText(text_overlay, timer_text, (11, 31), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (45, 59, 85), 2, cv2.LINE_AA)  # Dark shadow
            # Blend with original frame for shadow
            cv2.addWeighted(text_overlay, 0.3, frame, 0.7, 0, frame)  # 30% opacity shadow
            
            # Draw main text (semi-transparent)
            text_overlay = frame.copy()
            cv2.putText(text_overlay, timer_text, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, base_color, 2, cv2.LINE_AA)
            # Blend with original frame for main text
            cv2.addWeighted(text_overlay, 0.7, frame, 0.3, 0, frame)  # 70% opacity text
            
            # Show welcome message instead of controls (semi-transparent)
            # Show semi-transparent welcome message at bottom without background box
            text_overlay = frame.copy()
            welcome_text = "Welcome to the Sign Language Interpreter :)"
            cv2.putText(text_overlay, welcome_text, (11, H-11), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (45, 59, 85), 1, cv2.LINE_AA)  # Shadow
            cv2.addWeighted(text_overlay, 0.3, frame, 0.7, 0, frame)  # 30% opacity shadow
            
            text_overlay = frame.copy()
            cv2.putText(text_overlay, welcome_text, (10, H-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)  # Main text
            cv2.addWeighted(text_overlay, 0.7, frame, 0.3, 0, frame)  # 70% opacity text
                
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
            
            # Add letter to sentence every 2 seconds if detection is not frozen
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
            self.freeze_var.set("FREEZE ON")
            self.freeze_button.config(bg="#FFA62B")  # Orange when frozen
        else:
            self.freeze_var.set("FREEZE OFF")
            self.freeze_button.config(bg="#2D3B55")  # Blue when not frozen
            # Reset the timer when detection is resumed
            self.last_detection_time = time.time()
            
    def save_sentence(self):
        """Save the current sentence to a JSON file, overwriting any previous file"""
        if not self.current_sentence.strip():
            messagebox.showinfo("Empty Sentence", "There is no sentence to save.")
            return
            
        # Create directory if it doesn't exist
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(current_dir, 'saved_sentences')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Use a fixed filename instead of timestamp
        filename = os.path.join(save_dir, "current_sentence.json")
        
        try:
            # Save as JSON
            data = {"sentence": self.current_sentence}
            with open(filename, "w") as f:
                json.dump(data, f, indent=4)
            messagebox.showinfo("Success", f"Sentence saved to {filename}")
            self.status_var.set(f"Sentence saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save sentence: {str(e)}")
    
    def on_closing(self):
        if self.running:
            self.stop_detection()
        self.root.destroy()

# Add missing imports for Image/ImageTk
try:
    from PIL import Image, ImageTk
except ImportError:
    messagebox.showerror("Missing Dependency", "Please install Pillow: pip install Pillow")
    exit(1)

#testing repo
if __name__ == "__main__":
    root = tk.Tk()
    app = ASLSentenceBuilder(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop() 