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
    """Kaiser窓LPFのスペクトル反転による相補ハイパスFIR (LPF+HPF=再構成)"""
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


def blank_impulses_iq(iq_if: np.ndarray, thr_k: float = 6.0,
                       thr_ratio: float = 2.5, max_width: int = 48,
                       max_rate: float = 0.02, stride: int = 16) -> np.ndarray:
    """IQドメインのインパルスノイズブランカ (モード共通の正準実装)。
    振幅包絡の中央値/MAD基準で孤立パルスだけを検出し、端点コサイン補間で消去する。
    変調ピークや選択性フェージングの谷には触れない (長い区間は残す)。
    検出率がmax_rate超のブロックは信号とみなして無処理 (安全装置)。
    AM (288kHz, max_width=48) から純粋移動したもので、NFM (288kHz)・
    SSB (48kHz, max_width=8へスケール) でパラメータのみ変えて使う。
    """
    try:
        n = len(iq_if)
    except Exception:
        return iq_if
    if n < 64:
        return iq_if
    try:
        mag = np.abs(iq_if)
        # complex64入力ではabsが既にfloat32。astypeの二重コピーを避ける
        # (complex128入力時のみ変換)
        if mag.dtype != np.float32:
            mag = mag.astype(np.float32)
        # 統計は間引き＋partition直取り (np.medianはNaN検査経路で遅い。
        # 期待値同一のため検出性能不変。順序統計量単点で十分)
        sm = mag[::stride] if n > 256 else mag
        k = len(sm) // 2
        med = float(np.partition(sm, k)[k])
        if med < 1e-9:
            return iq_if
        dev = sm - med
        # 絶対値は二乗和ではなく符号付き偏差の順序統計なのでabsが要る
        # (madは絶対偏差の中央値。負値は中央値より下で無視される)
        np.abs(dev, out=dev)
        mad = float(np.partition(dev, k)[k]) + 1e-12
        thr = max(med + thr_k * mad, med * thr_ratio)
        mask = mag > thr
        if float(np.mean(mask)) > max_rate:
            return iq_if
        # 立ち上がり/立ち下がりをbool演算で直接抽出 (diff+astypeの2確保を排除)
        rise = mask[1:] & ~mask[:-1]
        fall = ~mask[1:] & mask[:-1]
        starts = list(np.flatnonzero(rise) + 1)
        ends = list(np.flatnonzero(fall) + 1)
        if mask[0]:
            starts.insert(0, 0)
        if mask[-1]:
            ends.append(n)
        if not starts:
            return iq_if
        out = iq_if.copy()
        # パルス補間を一括ベクトル化: パルス毎のnp.arange/np.cos確保と
        # Pythonループ (実測10パルスで0.4ms) を排除。数式は従来と同一
        # (各サンプル w=0.5*(1-cos(pi*(i+1)/(k+1)))、端点は線形→コサイン)。
        widths = np.asarray(ends[:512], dtype=np.int32) - \
            np.asarray(starts[:512], dtype=np.int32)
        starts_a = np.asarray(starts[:512], dtype=np.int32)
        ends_a = np.asarray(ends[:512], dtype=np.int32)
        valid = (widths > 0) & (widths <= max_width)
        if np.any(valid):
            wv = widths[valid]
            sv = starts_a[valid]
            ev = ends_a[valid]
            total = int(wv.sum())
            # 少数パルス (<=4) はベクトル化の固定費 (repeat/arange) が
            # 上回るため従来ループ。SSB (2208, 幅8, パルス1-2本) の実測で
            # ループ0.20ms < ベクトル0.29ms、多数パルス (NFM等) は逆転。
            if len(wv) <= 4:
                for s, e in zip(sv.tolist(), ev.tolist()):
                    l = out[s - 1] if s > 0 else out[e]
                    r = out[e] if e < n else l
                    kk = (e - s)
                    w = 0.5 * (1.0 - np.cos(
                        np.pi * np.arange(1, kk + 1, dtype=np.float32)
                        / (kk + 1.0)))
                    out[s:e] = (l * (1.0 - w) + r * w).astype(out.dtype)
            elif total > 0:
                starts_rep = np.repeat(sv, wv)
                ends_rep = np.repeat(ev, wv)
                cum = np.cumsum(wv) - wv
                pos = (np.arange(total, dtype=np.int64)
                       - np.repeat(cum, wv))
                kp1 = np.repeat(wv + 1.0, wv)
                w = (0.5 * (1.0 - np.cos(np.pi * (pos + 1) / kp1))
                     ).astype(np.float32)
                left_idx = np.where(sv > 0, sv - 1, ev)
                right_idx = np.where(ev < n, ev, left_idx)
                lv = out[np.repeat(left_idx, wv)]
                rv = out[np.repeat(right_idx, wv)]
                out[starts_rep + pos] = (lv * (1.0 - w) + rv * w)
        return out
    except Exception:
        return iq_if


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
