@echo off
title AVC&C Demo Launcher
color 0A

echo ========================================
echo    AVC^&C - Vehicle Detection System
echo    University of Bradford - Group G25
echo ========================================
echo.
echo Starting Vehicle Counter...
start "Vehicle Counter" cmd /k "cd /d C:\Users\ayush\Documents\GitHub\avc-vehicle-classification && python vehicle_counter.py"

timeout /t 3

echo Starting Dashboard...
start "Dashboard" cmd /k "cd /d C:\Users\ayush\Documents\GitHub\avc-vehicle-classification && python -m streamlit run app.py"

echo.
echo Both windows are opening...
echo Vehicle counter + Dashboard will start automatically
echo.
pause