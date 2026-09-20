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
  - **Stereo noise reduction**: estimates the (L−R) hiss floor and applies a
    per-frequency Wiener suppression (bins with programme stay stereo, noise
    dominated bins are attenuated; FM triangular-noise model), plus a smooth
    blend to mono for very weak stations. Clean stations keep full stereo
    (disable with `--no-stereo-nr`)
  - NFM (12–16 kHz, Doppler AFC for ISS/amateur), AM / narrow AM (MW & HF)
  - **AM synchronous detection (SAM)** — carrier-recovery PLL + coherent
    detector; kills selective-fading distortion on shortwave, with automatic
    fallback to envelope detection when the carrier is too weak
  - **SSB / CW** (USB / LSB) with a complex asymmetric band-pass for opposite
    sideband rejection (>20 dB) and a BFO fine-tune (±2 kHz)
  - **FM multipath suppression** — envelope-variation detector blends toward
    mono when reflections distort the signal
  - **Adaptive voice bandwidth** for AM/SSB: narrows 4 kHz → 2.5 kHz as hiss
    increases (keeps speech intelligible on weak HF)
  - 50 µs / 75 µs de-emphasis selectable by region
  - Adaptive drift resampler that actively keeps the audio buffer balanced —
    no clicks, no fast-forward artefacts
  - Click suppression, soft limiter, DC blocker
  - Native SIMD FIR (SSE2), sin/cos LUT with interpolation, FTZ/DAZ denormal
    flush — hot paths stay in C with the GIL released
  - Allocation-free audio callback (pre-allocated scratch, single copy)
  - RT deadline telemetry: DSP p50/p99, headroom and deadline-miss counters
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
- **Station info**
  - **RDS decoder** (PI / PS / PTY / RadioText / clock) on the native 57 kHz
    path — real station names on RDS-equipped FM (Europe/US; Japan has no RDS)
  - **EiBi shortwave schedule** auto-downloaded and cached: shows the real
    station, language and target area for HF frequencies + UTC time
  - **S-meter** (S-units, indicative) in the telemetry line
- **Frosted-glass GUI** (pygame, 30 fps)
  - Spectrum, waterfall, audio waveform, telemetry chips, stereo/mono indicator
  - Presets auto-generated from band scans and saved
  - Japanese / English UI (auto-detected, override with `--lang`)
- **Band scan & seek** — safe USB stop/restart, no receiver lock-ups
  - FM band scan with automatic presets
  - **Shortwave scan** (2–14.4 MHz, direct sampling): scans the international
    broadcast bands (120m…22m), finds carriers, snap-5 kHz, saves AM presets
    and auto-tunes the strongest one. AM uses a carrier-level AGC and a
    +150 kHz DC-spike offset so weak HF signals come out at full volume
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
python main.py --no-stereo-nr  # keep full stereo, no noise reduction
```

### Controls

| Control | Action |
|---|---|
| Click spectrum / waterfall | Tune (snaps to detected stations) |
| `-1M … +1M` | Step tuning |
| `WFM / AM / NFM / USB / LSB / CW` | Demodulation mode |
| `BFO- / BFO+` | Fine tune ±50 Hz per click (SSB/CW, ±2 kHz range) |
| `固定/収束/Locked…` button | Toggle auto gain ⇄ hard lock |
| `G+ / G-` | Manual gain steps |
| `V+ / V-` | Volume |
| `Filter: Clean/Wide` | Audio bandwidth |
| `AFC / DX` | Frequency tracking / DX boost |
| `FMスキャン / 短波スキャン` | FM band scan / shortwave (2–14.4 MHz) AM scan |
| `ステレオ / モノラル` | Force stereo or mono reception (saved) |
| `NR: ON/OFF` | Stereo noise reduction on/off (saved) |
| `<< Auto Seek >>` | Jump to next/previous detected station |
| Preset buttons | Auto-filled from the last scan (FM row / AM·SW row) |

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
- **AM同期検波 (SAM)**: 搬送波PLL＋同期検波で短波の選択性フェージング歪みを解消。
  キャリアが弱い時は包絡線検波へ自動フォールバック
- **SSB / CW (USB/LSB)**: 複素非対称バンドパスで反対側波帯を20dB以上除去。
  BFO微調整(±2kHz)付き
- **FMマルチパス抑圧**: IF包絡線の変動から反射波歪みを検出し、自動でブレンド
- **AM/SSB適応音声帯域**: ヒス量に応じて4kHz→2.5kHzへ連続的に狭めて了解度を確保
- **ネイティブ最適化**: SIMD FIR (SSE2)・sin/cos LUT補間・FTZ/DAZ denormal対策・
  オーディオコールバック無確保化。DSP処理時間のp50/p99と締切余裕をテレメトリ表示
- **ステレオノイズリダクション**: (L−R)のノイズフロアを常時推定し、周波数ごとの
  Wiener抑圧（番組のある帯域はステレオ維持、ノイズ支配の帯域のみ減衰。FM三角ノイズ
  モデル使用）＋極弱局はモノラルへブレンド。弱い局の「ステレオにすると出るヒス」を
  消しつつ、クリーンな局はフルステレオを維持 (`--no-stereo-nr` で無効化)
- **ネイティブCコア** (`sdr_core.dll`): IIR・FM復調・ミキサー・クリック除去・ステレオPLL。
  ctypes経由でGILを解放するため、GUI描画中でも音飛びしません
- **Hyper自律制御**: ADCクリップを避けるモデルベースのゲイン探索、アンテナ環境判定、
  聴感品質に基づくフィルタの連続モーフィング
- **地域対応**: OSロケールから自動判定（日本 76–95MHz / 米国・欧州 87.5–108MHz /
  東欧OIRT 65.8–74MHz）。`--country JP` 等で上書き
- **ガラス調GUI**: スペクトラム、ウォーターフォール、音声波形、ステレオ/モノラル表示、
  スキャン結果から自動生成されるプリセット、日英自動切替 (`--lang`)
  - **FMスキャン / 短波スキャン** ボタン、**ステレオ/モノラル切替**、**NR ON/OFF** ボタン
- **短波(HF)受信**: ダイレクトサンプリングで 2〜14.4MHz の国際放送バンド (120m〜22m) を
  スキャンし、キャリア検出→5kHzグリッド→AMプリセット保存→最強局へ自動選局。
  AMは搬送波レベルAGC＋DCスパイク回避(+150kHzオフセット)で弱いHFでも実用音量
- **局情報**: **RDSデコーダ** (PI/PS/PTY/曲名/時計 — 海外FM。日本はRDS未運用)、
  **EiBi短波番組表**自動取得 (周波数＋UTC時刻から実局名・言語・放送先を表示)、
  **Sメーター** (目安)
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
