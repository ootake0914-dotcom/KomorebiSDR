# Third-Party Notices

This project is distributed under the GNU General Public License v2.0 or later
(see `LICENSE`). It uses / bundles the following third-party components:

| Component | License | Notes |
|---|---|---|
| **librtlsdr** (`rtlsdr.dll`) | GPL-2.0 | RTL-SDR Blog fork (V4 support). Source: https://github.com/rtlsdrblog/rtl-sdr-blog |
| **pygame** | LGPL-2.1 | https://www.pygame.org |
| **NumPy** | BSD-3-Clause | https://numpy.org |
| **sounddevice** | MIT | https://python-sounddevice.readthedocs.io |
| **PortAudio** (via sounddevice) | MIT | https://portaudio.com |
| **Pillow** | HPND | https://python-pillow.org |
| **MSVC runtime** (`msvcr100.dll`) | Microsoft EULA | Required by librtlsdr builds |
| **pthreads-win32** (`pthreadVC2.dll`) | LGPL-2.1 | Required by librtlsdr builds |

## Why GPL?

`rtlsdr.dll` (librtlsdr) is licensed under the GPL. Because this application
loads and bundles that library, the combined distribution is provided under
the GPL-2.0-or-later as well.

If you do not wish to use librtlsdr, the DSP/GUI code can be used with a
different SDR backend by re-implementing `rtlsdr_driver.py`.
