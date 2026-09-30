@echo off
rem ===================================================================
rem  Renwu koutu - drag an image, several images, or a folder onto
rem  this file. Output goes next to the source as  <name>_koutu.png
rem  Double-click with no file to see usage.  (ASCII only on purpose:
rem  cmd.exe mangles non-ASCII inside .bat files.)
rem ===================================================================
setlocal

rem Keep console and Python on the same encoding, or Chinese output garbles.
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"

set "PY=D:\py\python"
if not exist "%PY%" set "PY=python"

set "SCRIPT=%~dp0remove_bg.py"
if not exist "%SCRIPT%" (
    echo [ERROR] remove_bg.py not found next to this .bat
    pause
    exit /b 1
)

if "%~1"=="" goto noargs

"%PY%" "%SCRIPT%" %*
goto done

:noargs
"%PY%" "%SCRIPT%"

:done
echo.
pause
