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
ollama pull mistral #or any model you want

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

### 🤖 **Hello AI**
- Funny robotic character with mechanical humor
- Context-aware responses that understand ASL detection errors
- Consistent personality across all interactions
- Local LLM processing via Ollama (privacy-focused)

### 🎮 **Rock-Paper-Scissors AI**
- Computer vision gesture recognition
- 38 specialized AI agents with different strategies
- Multi-Armed Bandit learning algorithm
- Pattern recognition that adapts to player behavior
- Historical Data Tracking
- Real-time statistics

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

**For** Developers, custom integrations, and testing.

---



---

## 📁 **Project Structure**

```
Hello-ai/
├── 📖 README.md                   # This file
|                     
│
├── 🎯 Interface/                # Complete Web Application
│   ├── 🐍 main.py                # Flask web app (all in)
│   ├── 📁 templates/             # frontend templates
│   ├── 👤 user_data/             # User profiles & history
│   ├── 🎲 rps_data/              # Game statistics
│   └── 📋 requirements.txt       # Dependencies
│
├── 🤖 ASL_Detection/              # The ASL System
│   └── sign-language-detector-python-master/
│       ├── 🧠 model.p            # Trained ML model
│       ├── 📊 data/              # Training data & mappings
│       ├── 🖥️ asl_main.py        # Tkinter GUI with AI
│       └── 📁 [training files]   # Model training scripts
│
└── 🎲 RPS/                        # The RPS System
    └── mab02/                     # Multi-Armed Bandit AI
        ├── 🧠 core/              # Algorithm
        ├── 🤖 agents/            # 38 AI agents
        └── 📊 [game data]        # Statistics
```




## Development Setup
```bash
git clone https://github.com/Ammar03s/Full-Stack-ASL-Gesture-Recognition-Application.git
cd Full-Stack-ASL-Gesture-Recognition-Application
pip install -r Interface/requirements.txt
```

---

## 📄 **License**

MIT License.


<div align="center">

**Created by Ammar Salahie and Abdullah Mahmoud**

</div> 
