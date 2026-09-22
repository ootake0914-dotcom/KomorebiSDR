# Third-Party Notices

The source code of KomorebiSDR is licensed under the MIT License (see `LICENSE`).
This project uses and optionally bundles the following third-party components:

| Component | License | Notes |
|---|---|---|
| **librtlsdr** (`rtlsdr.dll`) | GPL-2.0 | RTL-SDR Blog fork (V4 support). Source: https://github.com/rtlsdrblog/rtl-sdr-blog |
| **pygame** | LGPL-2.1 | https://www.pygame.org |
| **NumPy** | BSD-3-Clause | https://numpy.org |
| **sounddevice** | MIT | https://python-sounddevice.readthedocs.io |
| **PortAudio** (via sounddevice) | MIT | https://portaudio.com |
| **Pillow** | HPND | https://python-pillow.org |
| **EiBi shortwave schedule** | Free use / redistribution | Data (c) Eike Bierwirth, http://www.eibispace.de — downloaded at runtime and cached locally, not bundled. "free to download, use, copy, or distribute these files or to use them within third-party software" (README.TXT) |
| **RDS / SSB / SAM / stereo NR DSP** | Project code | Implemented from the published standards (EN 50067 RDS, FM MPX), no third-party code |

## Distribution and Licensing Notes

- **KomorebiSDR Source Code**: Released under the [MIT License](LICENSE).
- **`rtlsdr.dll` (librtlsdr)**: Licensed under the GNU General Public License v2.0 (GPL-2.0).
  When distributing binaries or packages that bundle `rtlsdr.dll`, the combined distribution
  is subject to the GPL-2.0.
- If you wish to use the DSP/GUI code in a purely permissive (MIT) context, you can interface
  with alternative SDR hardware backends without linking or bundling `librtlsdr`.
