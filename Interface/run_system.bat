@echo off
title AI Companion System
color 0A

echo.
echo ==========================================
echo    AI COMPANION SYSTEM - ASL + RPS
echo ==========================================
echo.

:: Change to Interface directory
cd /d "%~dp0"

:: Clear any existing processes on port 8080
echo Clearing port 8080...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8080"') do (
    taskkill /f /pid %%a >nul 2>&1
)

:: Start the application
echo Starting AI Companion System...
echo.
echo ✓ Access at: http://localhost:8080
echo ✓ ASL Detection + RPS Game available
echo ✓ Press Ctrl+C to stop
echo.

python app_simple.py

echo.
echo System stopped.
pause 