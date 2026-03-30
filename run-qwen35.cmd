@echo off
setlocal

chcp 65001 >nul

set "MODEL=C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\KoboldCpp\Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"

if /I "%~1"=="--version" (
  llama-cli --version
  exit /b %errorlevel%
)

llama-cli -m "%MODEL%" -ngl 99 -c 8192 -i %*
exit /b %errorlevel%
