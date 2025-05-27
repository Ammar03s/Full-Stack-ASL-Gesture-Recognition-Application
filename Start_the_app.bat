@echo off
title Enhanced AI Companion System
color 0A

echo.
echo ==========================================
echo    ENHANCED AI COMPANION SYSTEM
echo    ASL Detection + LLM Responses + RPS
echo ==========================================
echo.

:: Change to Interface directory where the working system is
cd /d "%~dp0Interface"

:: Clear any existing processes on port 8080
echo Clearing port 8080...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8080"') do (
    taskkill /f /pid %%a >nul 2>&1
)

:: Check if Ollama is running for LLM responses
echo Checking Ollama for AI responses...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo [INFO] Starting Ollama for AI responses...
    start /B ollama serve
    timeout /t 3 /nobreak >nul
) else (
    echo [OK] Ollama is running
)

:: Check if Mistral model is available
echo Checking Mistral model...
ollama list | find "mistral" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Downloading Mistral model...
    ollama pull mistral
) else (
    echo [OK] Mistral model ready
)

:: Start the enhanced application
echo Starting Enhanced AI Companion System...
echo.
echo Access at: http://localhost:8080
echo Features: ASL Detection + LLM Responses + RPS Game
echo NEW: AI Robotic Arm Responses powered by Mistral!
echo.

python app_simple.py

echo.
echo System stopped.
pause 