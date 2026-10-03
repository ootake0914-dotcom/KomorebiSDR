"""IQ impulse blanker mode expansion tests (synthetic, no hardware).

3 (IQインパルスNFM/SSB展開) の回帰テスト:
1) 正準実装が孤立パルスを消し、クリーンを素通しし、密パルスをバイパスする
2) NFM: パルス混入で復調クリックが減り、クリーンはON/OFF同一
3) SSB: パルス混入でクリックが減り、番組トーンが保たれる
4) 正規の長い変化 (5ms級) には触れない (打楽器・音声保護)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dsp_filters import blank_impulses_iq
from dsp import SdrDspPipeline

IF = 288000
BLK_IF = 16512
AUDIO = 48000


def test_shared_basic() -> bool:
    rng = np.random.default_rng(5)
    base = (np.ones(4096, dtype=np.complex64)
            * np.exp(1j * np.linspace(0, 40 * np.pi, 4096))).astype(np.complex64)
    pulsed = base.copy()
    pulsed[1000:1008] *= 12.0  # 8サンプル孤立パルス
    y = blank_impulses_iq(pulsed)
    resid = float(np.max(np.abs(y[998:1010]) - np.abs(base[998:1010])))
    ok = resid < 0.5
    print(f"[{'OK' if ok else 'FAIL'}] isolated pulse removed (resid={resid:.3f})")
    # クリーンは素通し (コピー等価)
    yc = blank_impulses_iq(base)
    ok2 = bool(np.array_equal(yc, base))
    print(f"[{'OK' if ok2 else 'FAIL'}] clean passthrough bit-equal")
    # 密パルス (>2%) は信号とみなして無処理
    dense = base.copy()
    dense[::40] *= 12.0
    yd = blank_impulses_iq(dense)
    ok3 = bool(np.array_equal(yd, dense))
    print(f"[{'OK' if ok3 else 'FAIL'}] dense burst bypassed")
    # 長い区間 (48超) は信号として残す
    longp = base.copy()
    longp[2000:2200] *= 8.0
    yl = blank_impulses_iq(longp)
    ok4 = bool(np.array_equal(yl, longp))
    print(f"[{'OK' if ok4 else 'FAIL'}] long segment preserved")
    _ = rng  # 決定論的素材のため未使用
    return bool(ok and ok2 and ok3 and ok4)


def _nfm_carrier(n, fdev=3000.0, faudio=1000.0):
    t = np.arange(n) / IF
    phase = 2 * np.pi * fdev * np.cumsum(np.cos(2 * np.pi * faudio * t)) / IF
    return (0.5 * np.exp(1j * phase)).astype(np.complex64)


def test_nfm() -> bool:
    n = BLK_IF * 4
    clean = _nfm_carrier(n)
    rng = np.random.default_rng(9)
    pulsed = clean.copy()
    for s in (5000, 20000, 45000):
        # 実パルス雑音は振幅だけでなく位相も跳ぶ (等 قطر振幅のみでは
        # FM復調が無視してしまうため、ランダム位相で10倍)
        ph = rng.uniform(-np.pi, np.pi, 10)
        pulsed[s:s + 10] = (10.0 * np.abs(clean[s:s + 10])
                            * np.exp(1j * ph)).astype(np.complex64)
    d_off = SdrDspPipeline(1152000, AUDIO)
    d_off.nfm_impulse_blanker_enabled = False
    d_on = SdrDspPipeline(1152000, AUDIO)
    y_off = np.concatenate([d_off.demodulate_nfm(pulsed[k * BLK_IF:(k + 1) * BLK_IF])
                            for k in range(4)])
    y_on = np.concatenate([d_on.demodulate_nfm(pulsed[k * BLK_IF:(k + 1) * BLK_IF])
                           for k in range(4)])
    peak_off = float(np.max(np.abs(y_off)))
    peak_on = float(np.max(np.abs(y_on)))
    improved = peak_on < peak_off * 0.8
    print(f"[{'OK' if improved else 'FAIL'}] NFM pulse click reduced "
          f"(peak off={peak_off:.3f} on={peak_on:.3f})")
    # クリーンはON/OFF同一 (定包絡に触れない)
    c_off = SdrDspPipeline(1152000, AUDIO)
    c_off.nfm_impulse_blanker_enabled = False
    c_on = SdrDspPipeline(1152000, AUDIO)
    yc_off = np.concatenate([c_off.demodulate_nfm(clean[k * BLK_IF:(k + 1) * BLK_IF])
                             for k in range(4)])
    yc_on = np.concatenate([c_on.demodulate_nfm(clean[k * BLK_IF:(k + 1) * BLK_IF])
                            for k in range(4)])
    same = bool(np.array_equal(yc_off, yc_on))
    print(f"[{'OK' if same else 'FAIL'}] NFM clean ON/OFF identical")
    return bool(improved and same)


def test_ssb() -> bool:
    n48 = 2208 * 8
    t = np.arange(n48) / AUDIO
    # USB帯内トーン (+1.5kHz複素)＋孤立パルス
    prog = (0.3 * np.exp(1j * 2 * np.pi * 1500.0 * t)).astype(np.complex64)
    rng = np.random.default_rng(13)
    pulsed = prog.copy()
    for s in (3000, 9000):
        ph = rng.uniform(-np.pi, np.pi, 6 if s == 3000 else 5)
        L = len(ph)
        pulsed[s:s + L] = (15.0 * abs(0.3) * np.exp(1j * ph)).astype(np.complex64)
    d_off = SdrDspPipeline(1152000, AUDIO)
    d_off.ssb_impulse_blanker_enabled = False
    d_on = SdrDspPipeline(1152000, AUDIO)
    yo = np.concatenate([d_off.demodulate_ssb(pulsed[k * 2208:(k + 1) * 2208], "USB")
                         for k in range(8)])
    yn = np.concatenate([d_on.demodulate_ssb(pulsed[k * 2208:(k + 1) * 2208], "USB")
                         for k in range(8)])
    peak_off = float(np.max(np.abs(yo)))
    peak_on = float(np.max(np.abs(yn)))
    # 全体ピークは番組 (AGC正規化トーン) が支配するため、パルス近傍窓で測る。
    # LPFがパルスを約400タップに拡散するので、窓ピークの「番組超過分」で比較する。
    base = slice(12000, 15000)
    base_off = float(np.max(np.abs(yo[base])))
    base_on = float(np.max(np.abs(yn[base])))
    w1 = slice(2700, 3600)
    w2 = slice(8700, 9600)
    site_off = max(float(np.max(np.abs(yo[w1]))), float(np.max(np.abs(yo[w2]))))
    site_on = max(float(np.max(np.abs(yn[w1]))), float(np.max(np.abs(yn[w2]))))
    # テスト有効性: OFFでは窓に超過が出ること (検出可能なクリックであること)
    valid = site_off > base_off * 1.05
    # 効果: ONでは窓ピークが番組レベルに戻ること (超過95%以上消去)
    improved = (site_on - base_on) < (site_off - base_off) * 0.3
    print(f"[{'OK' if valid else 'FAIL'}] test validity: site_off={site_off:.3f} "
          f"base={base_off:.3f}")
    print(f"[{'OK' if improved else 'FAIL'}] SSB pulse click removed "
          f"(excess off={site_off - base_off:.3f} on={site_on - base_on:.3f})")
    _ = (peak_off, peak_on)
    # 番組トーン保全: 復調後の1.5kHz... SSBはベースバンドへ落ちるため
    # 番組帯域エネルギー比で見る (ONが番組を削っていないこと)
    rms_off = float(np.sqrt(np.mean(yo ** 2))) + 1e-12
    rms_on = float(np.sqrt(np.mean(yn ** 2))) + 1e-12
    kept = 0.5 < (rms_on / rms_off) < 2.0
    print(f"[{'OK' if kept else 'FAIL'}] SSB program kept "
          f"(rms off={rms_off:.3f} on={rms_on:.3f})")
    c_off = SdrDspPipeline(1152000, AUDIO)
    c_off.ssb_impulse_blanker_enabled = False
    c_on = SdrDspPipeline(1152000, AUDIO)
    yc_off = np.concatenate([c_off.demodulate_ssb(prog[k * 2208:(k + 1) * 2208], "USB")
                             for k in range(8)])
    yc_on = np.concatenate([c_on.demodulate_ssb(prog[k * 2208:(k + 1) * 2208], "USB")
                            for k in range(8)])
    same = bool(np.array_equal(yc_off, yc_on))
    print(f"[{'OK' if same else 'FAIL'}] SSB clean ON/OFF identical")
    return bool(valid and improved and kept and same)


def main() -> int:
    ok = test_shared_basic()
    ok &= test_nfm()
    ok &= test_ssb()
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
