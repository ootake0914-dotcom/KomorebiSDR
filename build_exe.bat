@echo off
rem Build a standalone executable with PyInstaller (ASCII only)
setlocal
pushd "%~dp0"

if not exist sdr_core.dll (
    echo [INFO] sdr_core.dll not found - building native core...
    call build_native.bat
    if errorlevel 1 goto :err
)

python -m PyInstaller --noconfirm --clean --windowed --name KomorebiSDR ^
    --add-binary "rtlsdr.dll;." ^
    --add-binary "sdr_core.dll;." ^
    --hidden-import sounddevice ^
    main.py
if errorlevel 1 goto :err

echo [OK] Built: dist\KomorebiSDR\KomorebiSDR.exe
popd
exit /b 0

:err
echo [ERROR] Build failed.
popd
exit /b 1
