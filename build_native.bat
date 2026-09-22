@echo off
rem KomorebiSDR native core build script (ASCII only)
setlocal
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSDIR="
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -property installationPath`) do set "VSDIR=%%i"
)
if "%VSDIR%"=="" (
    set "VSDIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
)
if not exist "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" (
    echo [ERROR] Visual Studio BuildTools not found.
    exit /b 1
)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
    echo [ERROR] vcvars64.bat failed.
    exit /b 1
)
pushd "%~dp0"
cl /nologo /O2 /W3 /LD sdr_core.c /Fe:sdr_core.dll /link /INCREMENTAL:NO
if errorlevel 1 (
    echo [ERROR] build failed.
    popd
    exit /b 1
)
del /q sdr_core.exp sdr_core.lib sdr_core.obj >nul 2>nul
echo [OK] sdr_core.dll built.
popd
endlocal
