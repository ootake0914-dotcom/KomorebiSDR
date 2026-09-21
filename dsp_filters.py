"""
FIR design helpers + click suppressor + de-emphasis sections (extracted from dsp.py).

正準の保持場所はこのファイル。`dsp.py` は後方互換のため同名を再エクスポートする。
"""

import numpy as np

from dsp_native import _NATIVE, _fptr


def design_fir_kaiser(num_taps: int, cutoff_norm: float, beta: float = 6.5) -> np.ndarray:
    """Kaiser窓を用いた高精度FIRローパスフィルタ設計"""
    if num_taps % 2 == 0:
        num_taps += 1
    m = (num_taps - 1) // 2
    n = np.arange(-m, m + 1)
    h = np.sinc(2 * cutoff_norm * n) * (2 * cutoff_norm)
    window = np.kaiser(num_taps, beta)
    h = h * window
    return (h / np.sum(h)).astype(np.float32)


def design_fir_highpass(num_taps: int, cutoff_norm: float, beta: float = 6.5) -> np.ndarray:
    """Kaiser窓LPFのスペクトル反転による相補ハイパスFIR (LPF+HPF=完全再構成)"""
    h = design_fir_kaiser(num_taps, cutoff_norm, beta)
    h = -h
    h[len(h) // 2] += 1.0
    return h.astype(np.float32)


design_fir_lowpass = design_fir_kaiser


def suppress_click_transients(audio: np.ndarray, threshold: float = 0.48) -> np.ndarray:
    """
    チューナーのゲイン切替やUSB過渡応答による単発インパルスノイズ（クリック・プチ音）を
    局所コサイン補間により原音のスペクトルや高域音感を損なわずピンポイント修復する
    高精度ディクリッカー。
    通常の音楽・音声の高域成分を誤検知しないよう、急峻な孤立跳躍（1〜4サンプル幅）のみを対象とする。
    """
    if len(audio) < 16:
        return audio

    if _NATIVE is not None:
        out = np.array(audio, dtype=np.float32, copy=True)
        _NATIVE.sdr_suppress_clicks(_fptr(out), len(out), float(threshold))
        return out

    diffs = np.abs(np.diff(audio))
    bad = np.where(diffs > threshold)[0]
    if len(bad) == 0:
        return audio

    out = audio.copy()
    mask = np.zeros(len(out), dtype=bool)

    # 差分が急峻で、かつ前後の信号推移から孤立して飛び出しているインパルスのみを検出
    for b in bad:
        # 真のインパルススパイク判定: 前後サンプルとの急峻な反転または孤立段差
        left_idx = max(0, b - 1)
        right_idx = min(len(audio) - 1, b + 2)
        local_span = abs(audio[right_idx] - audio[left_idx])
        # 差分に対して前後の接続が戻っている（孤立突起）のみ修復する。
        # 旧 `or diff > 0.65` は打楽器アタック等の正規過渡を誤って削るため撤去。
        if diffs[b] > threshold and diffs[b] > local_span * 1.5:
            mask[max(0, b - 1) : min(len(out), b + 3)] = True

    if not np.any(mask):
        return audio

    # 連続した異常区間（幅1〜4サンプルの短パルス）のみをコサインS字平滑補間
    in_bad = False
    start = 0
    for idx in range(len(out)):
        if mask[idx] and not in_bad:
            in_bad = True
            start = idx
        elif not mask[idx] and in_bad:
            in_bad = False
            end = idx - 1
            if 0 < start and end < len(out) - 1 and (end - start + 1) <= 4:
                v0 = out[start - 1]
                v1 = out[end + 1]
                L = (end + 1) - (start - 1)
                t = np.linspace(0.0, np.pi, L + 1, dtype=np.float32)[1:-1]
                w = 0.5 * (1.0 - np.cos(t))
                out[start : end + 1] = v0 + (v1 - v0) * w

    return out


def _deemph_sections(tau_us: float):
    """時定数に対応する2縦続1次IIR係数 [(b0,b1,minus_a1)×2] を返す。
    48kHz用に実測フィットした値で、アナログ1次LPF特性に帯域内±0.03dBで一致。
    未知の時定数には無プリワープ双一次1段＋素通しで近似する。"""
    fitted = {
        50.0: [(1.6231841965746954, -1.2710911965746954, 0.647907),
               (0.18622705854697402, 0.026085941453025934, 0.787687)],
        75.0: [(1.0283946, -0.5892977, 0.5609031),
               (0.2090227, 0.0301037, 0.7608736)],
    }
    for k, v in fitted.items():
        if abs(float(tau_us) - k) < 1e-6:
            return [tuple(s) for s in v]
    # フォールバック: 無プリワープ双一次1段＋素通し (50/75μs以外は未使用想定)
    T = 1.0 / 48000.0
    tau = float(tau_us) * 1e-6
    den = 2.0 * tau + T
    return [(T / den, T / den, (2.0 * tau - T) / den), (1.0, 0.0, 0.0)]
