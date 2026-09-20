"""RDS decoder tests (synthetic, no hardware).

1) Decoder-only: build RDS groups (PI/PS/PTY), differential-encode,
   generate a 12 kHz biphase (Manchester) waveform and verify the decoder.
2) Full path: put the RDS subcarrier (57 kHz) + pilot on an FM-modulated
   carrier, run the real DSP pipeline and verify the PS name is recovered.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from rds import RdsDecoder, make_block, OFFSET_WORDS
from dsp import SdrDspPipeline

RF = 1152000
BLOCK = 132096
NBLK = 40
SPS = 12000.0 / 1187.5

PS_TEXT = "RDS TEST"
PI = 0x1234
PTY = 10


def rds_bitstream() -> list:
    bits = []

    def group0A(addr: int, ch2: str):
        b2 = (0 << 12) | (0 << 11) | (1 << 10) | ((PTY & 0x1F) << 5) | (addr & 3)
        chars = [ord(c) for c in ch2.ljust(2)]
        # ブロック3 = AF (ダミー), ブロック4 = PS 2文字
        return (make_block(PI, "A") + make_block(b2, "B")
                + make_block(0x0000, "C")
                + make_block((chars[0] << 8) | chars[1], "D"))

    bits += group0A(0, PS_TEXT[0:2])
    bits += group0A(1, PS_TEXT[2:4])
    bits += group0A(2, PS_TEXT[4:6])
    bits += group0A(3, PS_TEXT[6:8])
    return bits


def differential_encode(bits: list) -> list:
    out, prev = [], 0
    for b in bits:
        prev ^= b
        out.append(prev)
    return out


def biphase_waveform(enc: list, fs: float = 12000.0, amp: float = 1.0) -> np.ndarray:
    n = int(len(enc) * fs / 1187.5)
    t = np.arange(n) / fs
    sym_f = t * 1187.5
    sym = sym_f.astype(np.int64)
    frac = sym_f - sym
    sign = np.where(frac < 0.5, 1.0, -1.0)
    # Manchester: ビット0→-+ , ビット1→+- (±1に変換してから符号を掛ける)
    vals = (2.0 * np.asarray(enc, dtype=np.float64) - 1.0)[np.clip(sym, 0, len(enc) - 1)]
    return (vals * sign * amp).astype(np.float32)


def test_decoder() -> bool:
    # タイミング獲得に0.5秒必要なのでビット列を繰り返して十分な長さにする
    enc = differential_encode((rds_bitstream() * 4)[:832])
    wave = biphase_waveform(enc)
    dec = RdsDecoder(12000.0)
    for k in range(0, len(wave), 688):
        dec.feed(wave[k:k + 688])
    ok = (dec.pi == PI and dec.ps_name == PS_TEXT and dec.pty == PTY)
    print(f"[{'OK' if ok else 'FAIL'}] decoder: PI=0x{dec.pi:04X} PS='{dec.ps_name}' "
          f"PTY={dec.pty} groups={dec.groups}")
    return ok


def test_full_path() -> bool:
    blk_iq = BLOCK // 2
    total = blk_iq * NBLK
    t = np.arange(total) / RF
    enc = differential_encode(rds_bitstream() * 3)  # 繰り返し送信
    wave12 = biphase_waveform(enc)
    t12 = np.arange(len(wave12)) / 12000.0
    rds_bb = np.interp(t, t12, wave12, right=0.0)
    # BS.450/EN 50067準拠: パイロット・RDS副搬送波ともsin系 (ゼロクロス同相)
    mpx = (0.09 * np.sin(2 * np.pi * 19000.0 * t)
           + 0.06 * rds_bb * np.sin(2 * np.pi * 57000.0 * t))
    phase = 2 * np.pi * 30000.0 * np.cumsum(mpx) / RF
    iq = 0.6 * np.exp(1j * phase)
    raw = np.empty(2 * total, dtype=np.uint8)
    raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255).astype(np.uint8)
    raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255).astype(np.uint8)

    dsp = SdrDspPipeline(RF, 48000)
    dsp.set_offset_freq(0.0)
    dsp.afc_enabled = False
    dsp.cognitive_enabled = False
    for k in range(NBLK):
        dsp.process(raw[k * BLOCK:(k + 1) * BLOCK], mode="WFM")
    ok = dsp.rds_pi == PI and dsp.rds_ps == PS_TEXT
    print(f"[{'OK' if ok else 'FAIL'}] full path: PI=0x{dsp.rds_pi:04X} "
          f"PS='{dsp.rds_ps}' groups={dsp.rds_groups} lock={dsp.stereo_pilot_lock:.2f}")
    return ok


def test_ct_and_rt_cr() -> bool:
    """4A時計デコード (EN 50067レイアウト) とCR終端RTの検証 (デコーダ直接)。"""
    import rds as rds_mod

    # --- 4A時計: MJD=59945 (2023-01-01), 12:34 ---
    mjd = 59945
    hour, minute = 12, 34
    b2 = (4 << 12) | (0 << 11) | (1 << 10) | ((10 & 0x1F) << 5) | ((mjd >> 15) & 3)
    b3 = ((mjd & 0x7FFF) << 1) | ((hour >> 4) & 1)
    b4 = ((hour & 0xF) << 12) | (minute << 6)
    bits = (rds_mod.make_block(PI, "A") + rds_mod.make_block(b2, "B")
            + rds_mod.make_block(b3, "C") + rds_mod.make_block(b4, "D"))
    dec = RdsDecoder(12000.0)
    dec._sync = True  # 同期済み扱い
    dec._pending = list(bits)
    dec._decode_groups()
    ok = dec.clock == (2023, 1, 1, hour, minute)
    print(f"[{'OK' if ok else 'FAIL'}] 4A clock: {dec.clock} (want (2023,1,1,12,34))")
    # 全グループCRC OK (同期維持) であること
    ok2 = dec.groups >= 1 and dec._sync
    print(f"[{'OK' if ok2 else 'FAIL'}] 4A group CRC valid & sync kept")
    ok &= ok2

    # --- CR終端RT (短い文 "NEW" + CR、1グループ内) ---
    # 2A: type(0010) 0 TP PTY(5) AB(0) addr(0)
    seg = b"NEW\r"
    b2 = (2 << 12) | (0 << 11) | (1 << 10) | ((10 & 0x1F) << 5) | (0 << 4) | 0
    b3 = (seg[0] << 8) | seg[1]
    b4 = (seg[2] << 8) | seg[3]
    bits = (rds_mod.make_block(PI, "A") + rds_mod.make_block(b2, "B")
            + rds_mod.make_block(b3, "C") + rds_mod.make_block(b4, "D"))
    dec2 = RdsDecoder(12000.0)
    dec2._sync = True
    dec2._pending = list(bits)
    dec2._decode_groups()
    ok = dec2.radio_text == "NEW"
    print(f"[{'OK' if ok else 'FAIL'}] CR-terminated RT: {dec2.radio_text!r} (want 'NEW')")
    return ok


def main() -> int:
    ok = test_decoder()
    ok &= test_full_path()
    ok &= test_ct_and_rt_cr()
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
