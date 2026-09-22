"""Owl direction finding: 2-antenna AoA + direct/reflection separation.

2アンテナによる到来方向推定 (AoA: Angle of Arrival) および直接波/反射波分離モジュール。
単一ドングルでは空間サンプル1点のため到来方向推定は不可ですが、
本モジュールは2ch合成IQでアルゴリズムを検証し、2台目ドングル (共通クロック) 接続に備えるための先行実装。
実機配線まではDSPパイプラインからは呼ばない (テストのみ)。

手法:
- 基本AoA: 2ch間相互相関の偏角 → 平面波位相差 → arcsin
- 較正追従: ブロック毎の相互相関偏角の緩慢ドリフトを線形回帰で除去
  (独立LOの位相ドリフト・固定ゲイン/位相誤差を吸収)
- 反射波分離: CLEAN逐次消去 (最強パスをヌル化→残差のAoA)
- コヒーレント対策: 周波数ビン毎AoAのパワー重み付き中央値
"""

import numpy as np


def _phase_to_aoa(dphi, d, wavelength):
    s = float(np.clip(dphi * wavelength / (2.0 * np.pi * d), -1.0, 1.0))
    return float(np.rad2deg(np.arcsin(s)))


def estimate_aoa(ch1, ch2, d, wavelength):
    """単一パス到来方向 [deg]。クリーン信号用。"""
    ch1 = np.asarray(ch1, dtype=np.complex128)
    ch2 = np.asarray(ch2, dtype=np.complex128)
    n = min(len(ch1), len(ch2))
    if n < 16:
        return 0.0
    r = np.vdot(ch1[:n], ch2[:n]) / float(n)  # conj(ch1)*ch2 の平均
    if abs(r) < 1e-18:
        return 0.0
    return _phase_to_aoa(float(np.angle(r)), d, wavelength)


def _block_phase_fit(ch1, ch2, fs, block_s=0.05):
    """ブロック相互相関偏角の線形回帰。戻り値 (傾き[rad/s], t=0位相[rad])。

    時刻はブロック中央で定義する (始端定義だと傾き/2のバイアスが残る)。
    """
    ch1 = np.asarray(ch1, dtype=np.complex128)
    ch2 = np.asarray(ch2, dtype=np.complex128)
    n = min(len(ch1), len(ch2))
    blk = max(256, int(block_s * fs))
    nb = n // blk
    if nb < 4:
        r = np.vdot(ch1[:n], ch2[:n]) / float(n)
        return 0.0, float(np.angle(r)) if abs(r) > 1e-18 else 0.0
    ph = np.empty(nb)
    for k in range(nb):
        a = ch1[k * blk:(k + 1) * blk]
        b = ch2[k * blk:(k + 1) * blk]
        r = np.vdot(a, b) / float(blk)
        ph[k] = float(np.angle(r)) if abs(r) > 1e-18 else 0.0
    phu = np.unwrap(ph)
    tc = (np.arange(nb, dtype=np.float64) + 0.5) * blk / fs
    A = np.stack([tc, np.ones(nb)], axis=1)
    slope, pha0 = np.linalg.lstsq(A, phu, rcond=None)[0]
    return float(slope), float(pha0)


def estimate_calibration(ch1, ch2, known_aoa_deg, d, wavelength, fs=200000.0):
    """既知方向ビーコンからch2の静的ゲイン/位相誤差を推定する。

    独立LOの緩慢ドリフトは回帰で分離し、静的オフセットのみ返す。
    戻り値 {'phase', 'gain'}: ch2_corrected = ch2 / (gain * e^{jphase})。
    """
    slope, pha0 = _block_phase_fit(ch1, ch2, fs)
    known_phi = 2.0 * np.pi * d * np.sin(np.deg2rad(known_aoa_deg)) / wavelength
    ch1 = np.asarray(ch1, dtype=np.complex128)
    ch2 = np.asarray(ch2, dtype=np.complex128)
    n = min(len(ch1), len(ch2))
    gain = float(np.mean(np.abs(ch2[:n])) / max(float(np.mean(np.abs(ch1[:n]))), 1e-18))
    return {"phase": float(pha0 - known_phi), "gain": gain, "drift": slope}


def estimate_aoa_calibrated(ch1, ch2, d, wavelength, fs=200000.0, cal=None, dt_since_cal=0.0):
    """較正＋ドリフト追従つきAoA [deg]。

    cal=None時はドリフトのみ除去 (静的オフセットは真値に畳み込まれるため、
    較正なし単一源では原理的に分離不能。較正には estimate_calibration
    で既知ビーコンを使うのが正規手順)。
    dt_since_cal: 較正窓開始から測定窓開始までの秒数。独立LOのドリフトが
    その間に進む分を cal['drift'] で外挿補正する (無視すると固着誤差)。
    """
    slope, pha0 = _block_phase_fit(ch1, ch2, fs)
    off = 0.0
    if cal:
        off = float(cal["phase"]) + float(cal.get("drift", 0.0)) * float(dt_since_cal)
    phi = pha0 - off
    # 正面±90°へ折り返し
    phi = float((phi + np.pi) % (2.0 * np.pi) - np.pi)
    return _phase_to_aoa(phi, d, wavelength)


def _subband_transfer(ch1, ch2, fs, n_sub=64, bw_hz=200000.0):
    """サブバンド毎の伝達比 H(f)=<Ch2·conj(Ch1)>/<|Ch1|^2> と中心周波数を返す。"""
    ch1 = np.asarray(ch1, dtype=np.complex128)
    ch2 = np.asarray(ch2, dtype=np.complex128)
    n = min(len(ch1), len(ch2))
    n_fft = 4096
    nfr = max(1, n // n_fft)
    num = None
    den = None
    win = np.hanning(n_fft)
    for k in range(nfr):
        a = ch1[k * n_fft:(k + 1) * n_fft] * win
        b = ch2[k * n_fft:(k + 1) * n_fft] * win
        A = np.fft.fft(a)
        B = np.fft.fft(b)
        X = B * np.conj(A)
        P = (np.abs(A) ** 2)
        num = X if num is None else num + X
        den = P if den is None else den + P
    freqs = np.fft.fftfreq(n_fft, 1.0 / fs)
    m = np.abs(freqs) <= bw_hz / 2.0
    # 占有帯域のみ使う (矩形窓の漏れ・無信号binのゴミ比を除外)。
    # デッドbinは重み0で適合から落とす。
    H = num[m] / np.maximum(den[m], 1e-18)
    f = freqs[m]
    occ = den[m] >= 0.05 * float(np.max(den[m]))
    idx = np.argsort(f)
    f, H = f[idx], H[idx]
    occ = occ[idx]
    edges = np.array_split(np.arange(len(f)), n_sub)
    fsb, Hsb, w = [], [], []
    for e in edges:
        if np.any(occ[e]):
            fsb.append(float(np.mean(f[e][occ[e]])))
            Hsb.append(np.mean(H[e][occ[e]]))
            w.append(float(np.mean(den[m][idx][e][occ[e]])))
    return np.array(fsb), np.array(Hsb), np.array(w)


def _transfer_raw(ch1, ch2, fs, bw_hz=200000.0):
    """等間隔bin上の伝達比 H(f) (リップルケプストラム用)。"""
    ch1 = np.asarray(ch1, dtype=np.complex128)
    ch2 = np.asarray(ch2, dtype=np.complex128)
    n = min(len(ch1), len(ch2))
    n_fft = 8192
    nfr = max(1, n // n_fft)
    num = None
    for k in range(nfr):
        a = ch1[k * n_fft:(k + 1) * n_fft]
        b = ch2[k * n_fft:(k + 1) * n_fft]
        A = np.fft.fft(a)
        B = np.fft.fft(b)
        X = B * np.conj(A)
        num = X if num is None else num + X
    freqs = np.fft.fftfreq(n_fft, 1.0 / fs)
    m = np.abs(freqs) <= bw_hz / 2.0
    idx = np.argsort(freqs[m])
    return freqs[m][idx], num[m][idx]


def _echo_delays_ripple(ch1, ch2, fs, max_us=12.0, min_us=0.8):
    """H(f)リップルのケプストラムからエコー遅延を検出する。

    2パス混合の伝達比は周期1/τでうねる。自己相関ではFMの自己相似に
    埋もれるため、周波数領域のうねりを見る (耳＋こだまの分離)。
    """
    f, H = _transfer_raw(ch1, ch2, fs)
    mag = np.abs(H)
    mag = mag / max(float(np.median(mag)), 1e-18)
    d = mag - float(np.mean(mag))
    df = float(f[1] - f[0])
    nq = int(1.0 / (df * min_us * 1e-6))
    C = np.abs(np.fft.ifft(d, n=4 * len(d)))
    lags = np.arange(len(C)) / (4 * len(d) * df)  # 秒
    lo = int(min_us * 1e-6 / (lags[1] - lags[0]))
    hi = min(len(C) // 2, int(max_us * 1e-6 / (lags[1] - lags[0])))
    if hi <= lo + 1:
        return []
    seg = C[lo:hi]
    base = float(np.median(C)) + 1e-18
    cand = []
    for i in range(1, len(seg) - 1):
        if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] > 2.0 * base:
            cand.append((float(seg[i]), float(lags[lo + i])))
    cand.sort(reverse=True)
    return [t for _, t in cand[:2]]


def _fit_two_path(fsb, Hsb, w, tau, d, wavelength, echo_gain=0.6):
    """遅延τ・エコー利得aの2パス有理モデルを適合する。

    測定 H(f) = [c1·e^{jφ1} + c2·e^{j(φ2−2πfτ)}] / [1 + a·e^{−j2πfτ}]
    の分母を払った T(f) = c1·e^{jφ1} + c2·e^{j(φ2−2πfτ)} を、
    基底 [1, e^{−j2πfτ}] の線形結合で適合する。角度位相は振幅に
    吸収されるためグリッド不要で、適合振幅の偏角から到来方向を
    読み取る (φ1=arg c1、φ2=arg c2)。残差はτ (とa) のみに依存し、
    遅延選択が well-posed になる。
    戻り値 (残差, φ1deg, φ2deg, c1, c2)。
    """
    f = fsb.astype(np.float64)
    D = 1.0 + echo_gain * np.exp(-2j * np.pi * f * tau)
    T = (Hsb * D).astype(np.complex128)
    W = np.asarray(w, dtype=np.float64)
    e1 = np.ones_like(f)
    e2 = np.exp(-2j * np.pi * f * tau)
    M = np.stack([e1, e2], axis=1) * np.sqrt(W)[:, None]
    Y = T * np.sqrt(W)
    c, *_ = np.linalg.lstsq(M, Y, rcond=None)
    c1, c2 = complex(c[0]), complex(c[1])
    pred = c1 * e1 + c2 * e2
    res = float(np.sum(W * np.abs(T - pred) ** 2) / max(float(np.sum(W * np.abs(T) ** 2)), 1e-30))
    p1 = _phase_to_aoa(float(np.angle(c1)), d, wavelength)
    p2 = _phase_to_aoa(float(np.angle(c2)), d, wavelength)
    return res, p1, p2, c1, c2


def estimate_aoa_robust(ch1, ch2, d, wavelength, fs=200000.0):
    """コヒーレント反射波下でも直接波に寄るAoA [deg] (2パス適合の直接波)。"""
    paths = resolve_multipath(ch1, ch2, d, wavelength, fs, n_paths=2)
    return float(paths[0]["aoa_deg"]) if paths else 0.0


def resolve_multipath(ch1, ch2, d, wavelength, fs=200000.0, n_paths=2):
    """エコー遅延＋2パス適合で [{aoa_deg, power}] を電力降順で返す。

    直接波=遅延0パス (先着原理) で同定する。電力は適合振幅の2乗。
    """
    ch1 = np.asarray(ch1, dtype=np.complex128)
    ch2 = np.asarray(ch2, dtype=np.complex128)
    n = min(len(ch1), len(ch2))
    a, b = ch1[:n], ch2[:n]
    fsb, Hsb, w = _subband_transfer(a, b, fs)
    detected = _echo_delays_ripple(a, b, fs)
    taus = [0.0] + detected + [1.5e-6, 3.0e-6, 6.0e-6]
    # 重複除去 (近接τは代表のみ)
    uniq = []
    for t in taus:
        if not any(abs(t - u) < 0.4e-6 for u in uniq):
            uniq.append(t)
    best = None
    for tau in uniq[:6]:
        for ag in (0.3, 0.5, 0.7):
            res, p1, p2, c1, c2 = _fit_two_path(fsb, Hsb, w, tau, d, wavelength, echo_gain=ag)
            if best is None or res < best[0]:
                best = (res, tau, p1, p2, c1, c2)
    _, tau, p1, p2, c1, c2 = best
    paths = [{"aoa_deg": float(p1), "power": float(abs(c1) ** 2)},
             {"aoa_deg": float(p2), "power": float(abs(c2) ** 2)}]
    paths.sort(key=lambda p: -p["power"])
    return paths[:n_paths]
