# 📁 Enhanced AI Companion System - Project Structure

## 🎯 Clean & Organized Structure

```
asldraft/
├── 🚀 start_ai_companion.bat      # Main startup script (USE THIS!)
├── 🧹 cleanup_empty_dirs.bat      # Cleanup script for empty directories
├── 📖 README.md                   # Main project documentation
├── 📖 PROJECT_STRUCTURE.md        # This file
├── 📄 LICENSE                     # MIT License
├── 🗂️ .git/                       # Git repository data
│
├── 🎯 Interface/                   # MAIN APPLICATION (Working System)
│   ├── 🐍 app_simple.py          # Flask web app with all features
│   ├── 📋 requirements.txt       # Python dependencies
│   ├── 🚀 run_system.bat         # Original startup script (backup)
│   ├── 🐳 Dockerfile             # Docker configuration
│   ├── 🐳 docker-compose.yml     # Docker compose setup
│   ├── 📁 templates/             # HTML templates for web interface
│   │   ├── 🏠 base.html          # Base template
│   │   ├── 🔐 login.html         # User login page
│   │   ├── 📊 dashboard.html     # Main dashboard
│   │   ├── 🤟 asl.html           # ASL detection page (with AI responses)
│   │   ├── 🎮 rps.html           # Rock-Paper-Scissors game page
│   │   └── 📈 stats.html         # Statistics and history page
│   ├── 👤 user_data/             # User profiles and conversation history
│   ├── 🎲 rps_data/              # RPS game statistics and logs
│   ├── 🤖 asl_module/            # ASL detection module (Interface version)
│   ├── 🎮 rps_module/            # RPS game module (Interface version)
│   └── 🐍 venv/                  # Python virtual environment
│
├── 🤖 ASL_Detection/              # ASL Machine Learning Model
│   └── sign-language-detector-python-master/
│       ├── 🧠 model.p            # Trained ASL detection model
│       ├── 📊 data/              # Class mappings and training data
│       ├── 🖥️ detector_replier.py # Original tkinter implementation
│       └── 📁 [other ML files]   # Supporting files for model
│
├── 🎲 RPS/                        # Rock-Paper-Scissors AI System
│   └── mab02/                     # Multi-Armed Bandit implementation
│       ├── 🧠 core/              # MAB algorithm core
│       ├── 🤖 agents/            # 38 different AI agents
│       └── 📊 [data files]       # Game statistics and learning data
│
└── 🗑️ [Empty directories to be cleaned]
    ├── asl_module/               # Duplicate (use cleanup script)
    ├── rps_module/               # Duplicate (use cleanup script)
    └── user_data/                # Duplicate (use cleanup script)
```

## 🚀 How to Use

### Quick Start
```bash
# Just double-click this file:
start_ai_companion.bat
```

### Manual Start
```bash
cd Interface
python app_simple.py
```

### Clean Up Empty Directories
```bash
# Run when no Flask server is running:
cleanup_empty_dirs.bat
```

## 🎯 Key Files

| File | Purpose | Status |
|------|---------|--------|
| `start_ai_companion.bat` | **Main startup script** | ✅ Use this |
| `Interface/app_simple.py` | **Main Flask application** | ✅ Working system |
| `Interface/templates/asl.html` | **ASL page with AI responses** | ✅ Enhanced |
| `ASL_Detection/.../model.p` | **Trained ML model** | ✅ Required |
| `RPS/mab02/` | **AI game system** | ✅ Required |

## 🧹 Cleaned Up (Removed)

- ❌ `main.py` (duplicate, unused)
- ❌ `requirements.txt` (duplicate, use Interface version)
- ❌ `ENHANCED_ASL_README.md` (outdated)
- ❌ `test_imports.py` (no longer needed)
- ❌ Multiple startup scripts (consolidated to one)
- ❌ Duplicate modules (kept Interface versions)

## 🎯 Access Points

- **Web Interface**: `http://localhost:8080`
- **ASL Detection**: `http://localhost:8080/asl`
- **RPS Game**: `http://localhost:8080/rps`
- **Dashboard**: `http://localhost:8080/dashboard`
- **Statistics**: `http://localhost:8080/stats`

## 🔧 Dependencies

- **Python 3.8+** with packages in `Interface/requirements.txt`
- **Ollama** with Mistral model for AI responses
- **Webcam** for gesture detection

---

**Clean, organized, and ready to use! 🎉** 