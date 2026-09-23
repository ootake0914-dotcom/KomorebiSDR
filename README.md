# KomorebiSDR

> *Named after "Komorebi" (木漏れ日) — the sunlight filtering through trees.*  
> **An audiophile-grade, serene RTL-SDR receiver powered by deep-space and geophysical DSP algorithms.**

A calm, beautiful radio receiver for RTL-SDR. Plug in your dongle, launch the app, and enjoy crystal-clear broadcasts without complicated knobs, waterfall clutter, or radio jargon. 

Behind its minimalist, distraction-free interface lies an uncompromising digital signal processing engine engineered to audiophile standards.

![KomorebiSDR Screenshot](assets/screenshot.png)

---

## Highlights

### 🌿 Pure Simplicity on the Surface
- **Effortless Discovery**: One-touch automated scanning for local FM, Mediumwave / Shortwave AM, and Amateur Ham bands (80m / 40m / 20m).
- **Calm Interface**: Designed for focused work and background listening. Smooth, high-contrast spectrum display and responsive soft controls.
- **Natural Listening**: Balanced dynamics and continuous hiss reduction eliminate listener fatigue.

### 🔬 High-End DSP Under the Hood
- **Quadrature MPX Multipath Canceller**: Real-time NLMS decoupling of 38kHz orthogonal subcarrier interference, completely eliminating harsh sibilance and multipath distortion.
- **NASA DSN-Style Autonomous Kalman Pilot Tracker (AKCTL)**: Self-adjusting loop gains inspired by deep-space tracking, maintaining jitter-free 19kHz pilot lock even during severe fading.
- **Psychoacoustic Subband Wiener NR & Frequency-Dependent Blend**: Exploits Schroeder masking thresholds to selectively attenuate FM triangular noise ($f^2$) while preserving rich low-end stereo imaging.
- **FastICA BSS Stereo Separator**: Independent Component Analysis on Mid/Side subspaces to cleanly eliminate anti-phase hiss.
- **Minimum-Phase Apodizing Filters**: Cuts pre-ringing energy in half while preserving transient attack.
- **Active Zero-Phase DC Servo & R128 LUFS AGC**: Eliminates low-frequency phase smearing and prevents station-to-station volume jumps without unnatural pumping.

### 📐 Rigorous Measurement-Driven Engineering
- **Guaranteed Transparency**: Bit-identical output when enhancement stages are bypassed.
- **Dual-Metric Quality Gate**: Every optimization is verified against golden recordings using speech intelligibility (STOI = 1.000) and audible hiss metrics.
- **Real-Time Performance**: Hybrid C-core (`dsp_native`) and SIMD NumPy architecture delivering stable throughput:
  - **WFM stereo processing latency**: p50 = 24.2 ms / p99 = 39.4 ms (over 30% margin under the 57.3 ms budget).

---

## Supported Modes & Bands

| Mode | Bandwidth | Typical Use |
|:---:|:---:|:---|
| **WFM** | 200 kHz | FM Stereo Broadcasts (76.0 – 108.0 MHz) with RDS & Pilot Tracking |
| **AM** | 9 kHz / 6 kHz | Mediumwave (531 – 1602 kHz) & International Shortwave (2.3 – 26.1 MHz) |
| **NFM** | 12.5 kHz | VHF/UHF Communications & Airband |
| **USB / LSB** | 2.8 kHz | Amateur Radio HF SSB (80m, 40m, 20m) & Utility DX |
| **CW** | 500 Hz | Morse Code with 650 Hz automated pitch tracking |

---

## Quick Start

### Prerequisites
- Python 3.10 or higher
- An RTL-SDR USB dongle (RTL2832U based) with WinUSB / libusb drivers installed

### Installation

```bash
# Clone the repository
git clone https://github.com/ootake0914-dotcom/KomorebiSDR.git
cd KomorebiSDR

# Install Python dependencies
pip install -r requirements.txt

# Run KomorebiSDR
python main.py
```

*Note for Windows users: If running native acceleration, build tools are automated via `build_native.bat`.*

---

## Verification & Testing

KomorebiSDR includes a comprehensive automated test suite and measurement harness:

```bash
# Run all hardware-free DSP regression tests
python tests/run_all.py

# Profile DSP throughput and latency percentiles (p50 / p95 / p99)
python tools/profile_snapshot.py --blocks 30 --modes WFM --mem

# Evaluate A/B audio quality on golden broadcast recordings
python tools/ab_benchmark.py testdata/weak_775_10s.npy --bm all
```

---

## License

This project's source code is licensed under the [MIT License](LICENSE).  
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party notices and acknowledgements.
