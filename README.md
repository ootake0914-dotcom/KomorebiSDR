# KomorebiSDR

> *Named after "Komorebi" (木漏れ日) — the sunlight filtering through trees.*  
> A simple, distraction-free SDR receiver for RTL-SDR.

KomorebiSDR is a clean and lightweight software-defined radio receiver designed for RTL-SDR USB dongles. It provides an intuitive interface for broadcast and amateur radio reception without complicated controls or unnecessary jargon.

Under the hood, it implements measurement-driven signal processing focused on audio clarity, stereo stability, and resilience against overload.

![KomorebiSDR Screenshot](assets/screenshot.png)

---

## Key Features

### Simple and Focused Interface
- **Automated Scanning**: One-touch scanning for local FM, Mediumwave / Shortwave AM, and Amateur Ham bands.
- **Minimalist Controls**: A clear spectrum display and essential controls designed for comfortable background listening.
- **Station Logging**: Background telemetry and signal logging for monitoring reception conditions.

### Thoughtful Signal Processing
- **Stereo MPX Decoupling**: Adaptive subcarrier decoupling to reduce multipath distortion and sibilance on FM stereo.
- **Adaptive Pilot Tracking**: Phase-locked loop with adaptive tracking to maintain 19kHz stereo pilot lock during fading.
- **Frequency-Dependent Stereo Blend**: Progressively narrows high-frequency stereo separation under weak signals to mitigate triangular FM noise while retaining low-end imaging.
- **Receiver Resilience & Protection**: RF Health Governor monitors ADC clipping and coordinates with the tuner gain control to prevent overload; smooth 20ms crossfading eliminates clicks during mode switches.
- **Gentle Leveling**: Slow-acting AGC and DC servo to balance station-to-station loudness without unnatural dynamic pumping.

### Measurement-Driven Development
- **Strict Verification**: Every DSP component is evaluated against synthetic benchmarks and recorded RF data.
- **Bypass Transparency**: Processing stages are designed to be completely transparent or bit-identical when inactive.
- **Efficient Latency**: Built with NumPy and optional C extensions (`dsp_native`), maintaining real-time buffer safety within standard audio latency budgets.

---

## Supported Modes & Bands

| Mode | Bandwidth | Typical Use |
|:---:|:---:|:---|
| **WFM** | 200 kHz | FM Stereo Broadcasts (76.0 – 108.0 MHz) with RDS |
| **AM** | 9 kHz / 6 kHz | Mediumwave (531 – 1602 kHz) & International Shortwave (2.3 – 26.1 MHz) |
| **NFM** | 12.5 kHz | VHF/UHF Communications & Airband |
| **USB / LSB** | 2.8 kHz | Amateur Radio HF SSB (80m, 40m, 20m) & Utility DX |
| **CW** | 500 Hz | Morse Code with 650 Hz automated tone tracking |

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

*Note for Windows users: Optional native C acceleration can be compiled using `build_native.bat`.*

---

## Verification & Testing

KomorebiSDR includes automated regression and stress testing suites:

```bash
# Run all hardware-free regression tests
python tests/run_all.py

# Run RF stress and breaking point benchmarks
python tools/rf_stress_benchmark.py

# Profile DSP throughput and latency percentiles (p50 / p95 / p99)
python tools/profile_snapshot.py --blocks 30 --modes WFM --mem
```

---

## License

This project is licensed under the [MIT License](LICENSE).  
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party notices and acknowledgements.
