# Antigravity SDR Radio

A full-scratch software-defined-radio receiver for **RTL-SDR** dongles.
Real-time **stereo FM**, autonomous gain/audio-quality optimization, frosted-glass
GUI, and an AI-friendly CLI/API — with no GNU Radio or SDR# required.

[日本語の説明は下にあります](#日本語)

---

## Features

- **Full-scratch DSP** (NumPy + native C core)
  - WFM **stereo** MPX decoder (19 kHz pilot PLL, 38 kHz synchronous detection,
    stereo blend to mono on weak signals)
  - NFM (12–16 kHz, Doppler AFC for ISS/amateur), AM / narrow AM (MW & HF)
  - 50 µs / 75 µs de-emphasis selectable by region
  - Adaptive drift resampler that actively keeps the audio buffer balanced —
    no clicks, no fast-forward artefacts
  - Click suppression, soft limiter, DC blocker
- **Native C core** (`sdr_core.dll`, built from `sdr_core.c`)
  - IIR filters, FM discriminator, complex mixer, click repair, stereo PLL
  - Called via ctypes so the GIL is released: GUI and audio never starve the DSP
- **Autonomous optimization (HyperController)**
  - Model-based gain search that avoids ADC clipping and locks a stable gain
  - Antenna environment profiling (compact / balanced / overload-risk)
  - Audio-quality-driven continuous filter morphing
- **Regions** (auto-detected from OS locale, override with `--country`)
  - JP 76–95 MHz, US/CA 87.5–108 MHz, CCIR 87.5–108 MHz, OIRT 65.8–74 MHz
  - Correct FM grid, de-emphasis and MW channel spacing per region
- **Frosted-glass GUI** (pygame, 30 fps)
  - Spectrum, waterfall, audio waveform, telemetry chips, stereo/mono indicator
  - Presets auto-generated from band scans and saved
  - Japanese / English UI (auto-detected, override with `--lang`)
- **Band scan & seek** — safe USB stop/restart, no receiver lock-ups
- **AI / automation tools**
  - `ai_sdr.py inspect|scan|listen` (JSON + PNG output)
  - `server.py` FastAPI endpoints (`/api/status`, `/api/tune`, `/api/spectrum.png`, …)

## Requirements

- Windows 10/11 (Linux/macOS may work; only Windows is tested)
- Python 3.10+ (for running from source)
- RTL-SDR dongle **with WinUSB driver installed** (see below)
- Bundled DLLs: `rtlsdr.dll`, `msvcr100.dll`, `pthreadVC2.dll`

### Driver setup (first time only)

1. Plug in the RTL-SDR dongle.
2. Download [Zadig](https://zadig.akeo.ie/).
3. In Zadig, select your dongle and install the **WinUSB** driver for
   `Bulk-In, Interface (Interface 0)`.
4. Start this application.

## Quick start

### Prebuilt executable

Download `AntigravitySDR.exe` from Releases and run it. Logs are written to
`%APPDATA%\AntigravitySDR\app.log`.

### From source

```bash
pip install -r requirements.txt
python main.py                 # auto-detects region and language
python main.py --country JP --lang ja
python main.py --freq 83.2 --mode WFM
python main.py --mono          # force monaural FM
```

### Controls

| Control | Action |
|---|---|
| Click spectrum / waterfall | Tune (snaps to detected stations) |
| `-1M … +1M` | Step tuning |
| `WFM / AM / NFM` | Demodulation mode |
| `固定/収束/Locked…` button | Toggle auto gain ⇄ hard lock |
| `G+ / G-` | Manual gain steps |
| `V+ / V-` | Volume |
| `Filter: Clean/Wide` | Audio bandwidth |
| `AFC / DX` | Frequency tracking / DX boost |
| `全帯域スキャン` | Full-band scan (presets are generated from results) |
| `<< Auto Seek >>` | Jump to next/previous detected station |
| Preset buttons | Auto-filled from the last scan |

## File layout

| File | Role |
|---|---|
| `main.py` | Application controller, threads, command queue |
| `rtlsdr_driver.py` | ctypes wrapper for `rtlsdr.dll` (sync + async, race-free cancel) |
| `dsp.py` | DSP pipeline (WFM stereo / NFM / AM, resampler, AFC) |
| `sdr_core.c` / `sdr_core.dll` | Native hot paths (GIL-free) |
| `hyper_controller.py` | Autonomous gain / filter / antenna optimization |
| `cascade_controller.py` | Legacy controller |
| `auto_tuner.py` | Band scan, peak detection, seek |
| `gui.py` | Frosted-glass pygame GUI |
| `audio_output.py` | sounddevice output (stereo, jitter buffer, underrun stats) |
| `config.py` | Region profiles and settings (`%APPDATA%\AntigravitySDR\config.json`) |
| `i18n.py` | Japanese / English strings |
| `ai_sdr.py`, `ai_sdr_core.py`, `server.py` | AI/automation interfaces |
| `build_native.bat` | Rebuild `sdr_core.dll` (MSVC) |
| `build_exe.bat` | Build a standalone executable (PyInstaller) |

## Building

### Native core

```bat
build_native.bat
```

Requires Visual Studio Build Tools (C++ workload). If `sdr_core.dll` is missing,
the DSP automatically falls back to pure Python (slower, more CPU).

### Executable

```bat
pip install pyinstaller
build_exe.bat
```

## Testing

```bash
python tests/run_all.py
```

Includes a hardware-free mock worker test, a native-vs-Python equivalence test,
and a synthetic stereo separation test (expects ≥ 20 dB).

## License

GPL-2.0-or-later. See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
`rtlsdr.dll` is the GPL-licensed RTL-SDR Blog fork of librtlsdr.

---

## 日本語

RTL-SDR 向けのフルスクラッチ SDR 受信ソフトです。**ステレオFM**、音質を目的関数にした
自律ゲイン最適化、ガラス調GUI、AI向けCLI/APIを搭載しています。

### 主な特徴

- **フルスクラッチDSP**: WFMステレオ (19kHzパイロットPLL＋38kHz同期検波＋弱電界ブレンド)、
  NFM (ISS等ドップラーAFC)、AM/短波狭帯域AM、地域別ディエンファシス (50/75µs)
- **ネイティブCコア** (`sdr_core.dll`): IIR・FM復調・ミキサー・クリック除去・ステレオPLL。
  ctypes経由でGILを解放するため、GUI描画中でも音飛びしません
- **Hyper自律制御**: ADCクリップを避けるモデルベースのゲイン探索、アンテナ環境判定、
  聴感品質に基づくフィルタの連続モーフィング
- **地域対応**: OSロケールから自動判定（日本 76–95MHz / 米国・欧州 87.5–108MHz /
  東欧OIRT 65.8–74MHz）。`--country JP` 等で上書き
- **ガラス調GUI**: スペクトラム、ウォーターフォール、音声波形、ステレオ/モノラル表示、
  スキャン結果から自動生成されるプリセット、日英自動切替 (`--lang`)
- **AI連携**: `ai_sdr.py` (inspect / scan / listen) と `server.py` (FastAPI)

### 起動方法

```bash
pip install -r requirements.txt
python main.py
```

初回のみ RTL-SDR に **WinUSB ドライバ**の導入が必要です（[Zadig](https://zadig.akeo.ie/) で
`Bulk-In, Interface (Interface 0)` に WinUSB をインストール）。

### ビルド

```bat
build_native.bat   :: sdr_core.dll を再ビルド (MSVC)
build_exe.bat      :: 配布用 exe を作成 (PyInstaller)
```

### ライセンス

GPL-2.0-or-later（`LICENSE` / `THIRD_PARTY_NOTICES.md` 参照）。
`rtlsdr.dll` は GPL の RTL-SDR Blog フォークです。
