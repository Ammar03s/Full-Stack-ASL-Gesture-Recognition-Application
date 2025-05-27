# 🤖 Hello AI

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-red.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange.svg)](https://mediapipe.dev/)
[![Ollama](https://img.shields.io/badge/Ollama-Mistral-purple.svg)](https://ollama.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent multi-modal system that combines **ASL (American Sign Language) detection**, **AI-powered conversational responses**, and **adaptive Rock-Paper-Scissors gameplay**. Built with computer vision, machine learning, and large language models for seamless human-computer interaction.

![Hello AI Demo](Demo.mp4)

## 🚀 **Start**

### Prerequisites
- **Python 3.8+** with pip
- **Webcam** for gesture detection
- **Ollama** for AI responses

### Setup & Launch
```bash
# 1. Clone & Setup
git clone https://github.com/Ammar03s/Full-Stack-ASL-Gesture-Recognition-Application.git
cd Full-Stack-ASL-Gesture-Recognition-Application

# 2. Install Ollama & Mistral
# Install Ollama from https://ollama.ai
ollama pull mistral #adjustable

# 3. Install Dependencies
cd Interface
pip install -r requirements.txt

# 4. Launch the System
python main.py
```

### Access the Application
- **Web Interface**: `http://localhost:8080`
- **ASL Detection**: `http://localhost:8080/asl`
- **RPS Game**: `http://localhost:8080/rps`
- **Dashboard**: `http://localhost:8080/dashboard`

---

## 🎯 **Features**

### 🤟 **ASL Detection System**
- Real-time detection using MediaPipe hand tracking
- Machine learning classification with trained RandomForest model
- Automatic sentence building with 4-second intervals
- AI conversational responses via Ollama/Mistral LLM
- Manual controls for spaces, corrections, and quick letters

### 🎮 **Rock-Paper-Scissors AI**
- Computer vision gesture recognition (rock/paper/scissors)
- Multi-Armed Bandit learning algorithm
- 38 specialized AI agents with different strategies
- Pattern recognition that adapts to player behavior
- Real-time statistics and win rate tracking

### 🤖 **Hello AI**
- Funny robotic character with mechanical humor
- Context-aware responses that understand ASL detection errors
- Consistent personality across all interactions
- Local LLM processing via Ollama (privacy-focused)

---

## 🔧 **Alternative Usage**

### Standalone ASL Detection
```bash
cd ASL_Detection/sign-language-detector-python-master
pip install opencv-python mediapipe scikit-learn pillow requests
python asl_main.py
```

### Standalone RPS Game
```bash
cd RPS/mab02
pip install opencv-python mediapipe numpy pandas matplotlib
python main.py
```

**Perfect for:** Developers, custom integrations, and Testing

---

## ⚙️ **Configuration**

### Camera Settings
```python
# In main.py or asl_main.py
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
```

### AI Response Settings
```python
# Ollama configuration
ollama_url = "http://localhost:11434/api/generate"
model_name = "mistral"
temperature = 0.71  # Creativity level
max_tokens = 200   # Response length
```

### Detection Intervals
```python
# ASL detection timing
detection_interval = 4  # seconds between letter captures (adjustable)
```

---


---

## 📁 **Project Structure**

```
Hello-ai/
├── 📖 README.md                   # This file
├── 📖 PROJECT_STRUCTURE.md        # Detailed organization guide
├── 📄 LICENSE                     # MIT License
│
├── 🎯 Interface/                   # Complete Web Application
│   ├── 🐍 main.py                # Flask web app (all features)
│   ├── 📁 templates/             # HTML templates
│   ├── 👤 user_data/             # User profiles & history
│   ├── 🎲 rps_data/              # Game statistics
│   └── 📋 requirements.txt       # Python dependencies
│
├── 🤖 ASL_Detection/              # Standalone ASL System
│   └── sign-language-detector-python-master/
│       ├── 🧠 model.p            # Trained ML model
│       ├── 📊 data/              # Training data & mappings
│       ├── 🖥️ asl_main.py        # Tkinter GUI with AI
│       └── 📁 [training files]   # Model training scripts
│
└── 🎲 RPS/                        # Standalone RPS System
    └── mab02/                     # Multi-Armed Bandit AI
        ├── 🧠 core/              # MAB algorithm
        ├── 🤖 agents/            # 38 AI agents
        └── 📊 [game data]        # Statistics & learning data
```

---



## Development Setup
```bash
git clone https://github.com/Ammar03s/Full-Stack-ASL-Gesture-Recognition-Application.git
cd Full-Stack-ASL-Gesture-Recognition-Application
pip install -r Interface/requirements.txt
```

---

## 📄 **License**

This project is licensed under the MIT License.


<div align="center">

**Created by Ammar Salahie and Abdullah Mahmoud**

</div> 
