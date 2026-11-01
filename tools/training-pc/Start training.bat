@echo off
rem Starts the Minesweeper training and its dashboard, each only if it is not running already.
rem Windows asks to allow admin rights: the training task runs with them.
powershell -NoProfile -Command "Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File \"%~dp0start.ps1\"'"
