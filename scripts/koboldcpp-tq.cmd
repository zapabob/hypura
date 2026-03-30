@echo off
setlocal
set SCRIPT_DIR=%~dp0
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%run_koboldcpp_turboquant.ps1" %*
exit /b %ERRORLEVEL%
