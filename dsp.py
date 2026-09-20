"""
Digital Signal Processing (DSP) Module for SDR - Ultra-Clear Hi-Fi Edition.
歪み・クリッピング・19kHzパイロットトーン・ヒスノイズを完全解消した高音質DSPパイプライン。

速度が要求される処理 (IIRフィルタ / FM復調 / 複素ミキサー / クリック除去) は
ネイティブCコア (sdr_core.dll) で実行する。ctypes呼び出し中はGILが解放されるため、
GUI描画やオーディオコールバックと競合せず、音飛び (バッファ枯渇) を防ぐ。
DLLが見つからない場合は純Python実装へ自動フォールバックする。
"""

import ctypes
import os
import numpy as np


# ================================================================
# ネイティブCコア (sdr_core.dll) ロード
# ================================================================
def _load_native_core():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sdr_core.dll")
    if not os.path.exists(path):
        return None
    try:
        lib = ctypes.CDLL(path)
        f = ctypes.c_float
        pf = ctypes.POINTER(f)
        lib.sdr_version.restype = ctypes.c_int
        lib.sdr_bilinear_deemphasis.argtypes = [pf, pf, ctypes.c_int, f, f, f, pf]
        lib.sdr_one_pole_highpass.argtypes = [pf, pf, ctypes.c_int, f, pf]
        lib.sdr_fm_demod.argtypes = [pf, pf, ctypes.c_int, pf]
        lib.sdr_mix_freq.argtypes = [pf, ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
        lib.sdr_suppress_clicks.argtypes = [pf, ctypes.c_int, f]
        lib.sdr_suppress_clicks.restype = ctypes.c_int
        lib.sdr_stereo_pll.argtypes = [pf, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_double,
                                       ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.c_double, pf, pf, pf]
        if lib.sdr_version() < 1:
            return None
        return lib
    except Exception:
        return None


_NATIVE = _load_native_core()
NATIVE_CORE_ENABLED = _NATIVE is not None


def _fptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


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
        # 差分に対して前後の接続が戻っている（孤立突起）、または極めて急峻なステップ
        if diffs[b] > threshold and (diffs[b] > local_span * 1.5 or diffs[b] > 0.65):
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


class AdaptiveDriftResampler:
    """
    RTL-SDRとDAC間の独立クロック偏差（ドリフト）を微小に伸縮補正し、
    サンプル欠落・重複・アンダーランを完全根絶する適応型分数リサンプラ。
    深層ジッターバッファ（約400ms）と広域不感帯により通常時は0ppm・完全ビットパーフェクト通過。
    """

    def __init__(self, target_chunks: float = 8.0, max_ppm: float = 2000.0):
        self.target_chunks = target_chunks
        self.deadband_low = 5.5   # 5.5〜10.5チャンク（約275ms〜525ms）の間は不感帯
        self.deadband_high = 10.5
        # ±2000ppm (0.2%, ≈3.5cent) までは聴感上ほぼ知覚不能。
        # クロック偏差に加え、GUI描画(GIL)による微小な処理落ちも吸収し、
        # バッファ枯渇(音飛び)を未然に防ぐ。
        self.max_ratio_offset = max_ppm * 1e-6
        self.integral_error = 0.0
        self.current_ratio = 1.0
        self.phase = 0.0
        self.last_sample = 0.0
        self.drift_ppm = 0.0

    def update_feedback(self, current_chunks: float, dt: float = 0.05):
        """
        オーディオバッファの残存チャンク数に応じた不感帯付き高精度PI制御。
        通常時（5.5〜10.5チャンク）は 0ppm・完全ビットパーフェクト通過。
        """
        # 不感帯（Deadband）判定: 健全領域
        if self.deadband_low <= current_chunks <= self.deadband_high:
            self.integral_error = 0.0
            # 無視できる補正量ならビットパーフェクト（1.0）へ戻す。
            # 一方、有意な補正は保持する。毎回1.0に戻すと恒常的な微小不足で
            # バッファが枯渇→アンダーランを繰り返すため。
            if abs(self.current_ratio - 1.0) < 200e-6:
                self.current_ratio = 1.0
                self.drift_ppm = 0.0
            return

        if current_chunks < self.deadband_low:
            error = current_chunks - self.deadband_low  # 負値（バッファ減）
        else:
            error = current_chunks - self.deadband_high  # 正値（バッファ増）

        # 積分器の更新（アンチワインドアップ付き）
        self.integral_error = float(np.clip(self.integral_error + error * dt, -3.0, 3.0))

        # PI制御ゲイン: 実際にバッファ水位を制御できる強さに再調整。
        # (旧値 kp=3e-5 は補正能力が実質ゼロで、枯渇を放置していた)
        kp = 8e-4
        ki = 5e-5
        adj = kp * error + ki * self.integral_error
        adj = float(np.clip(adj, -self.max_ratio_offset, self.max_ratio_offset))

        # 0.5ppm未満の微小な揺らぎは完全ゼロ（ビットパーフェクト）に丸める
        if abs(adj) < 0.5e-6:
            adj = 0.0

        self.current_ratio = 1.0 + adj
        self.drift_ppm = adj * 1e6

    def process(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) == 0:
            return audio

        # ドリフトがほぼゼロかつ位相ズレがないときはビットパーフェクトで通過 (補間ゼロ)
        if abs(self.current_ratio - 1.0) < 1e-6 and abs(self.phase) < 1e-5:
            self.last_sample = float(audio[-1])
            return audio

        # 境界連続性のために前回の最終サンプル(t=-1)と今回の末尾(t=N)を付加
        ext_audio = np.concatenate(([self.last_sample], audio, [audio[-1]]))
        n_in = len(audio)

        indices = np.arange(self.phase, n_in, self.current_ratio)
        if len(indices) == 0:
            self.phase -= n_in
            self.last_sample = float(audio[-1])
            return np.zeros(0, dtype=np.float32)

        # ext_audio 内での正確なインデックス (時刻 t=0 は ext_audio[1] に対応)
        ext_indices = indices + 1.0
        idx_floor = ext_indices.astype(np.int32)
        idx_frac = (ext_indices - idx_floor).astype(np.float32)

        s0 = ext_audio[idx_floor]
        s1 = ext_audio[idx_floor + 1]
        out = s0 + idx_frac * (s1 - s0)

        last_idx = indices[-1] + self.current_ratio
        self.phase = float(last_idx - n_in)
        self.last_sample = float(audio[-1])

        return out.astype(np.float32)


class SdrDspPipeline:
    """超低ノイズ・超高音質 SDR 信号処理パイプライン"""

    def __init__(self, sample_rate: int = 1152000, audio_rate: int = 48000):
        self.rf_rate = sample_rate
        self.audio_rate = audio_rate

        self.if_decim = 4
        self.if_rate = self.rf_rate // self.if_decim  # 288 kHz
        self.audio_decim = self.if_rate // self.audio_rate  # 6
        self.total_decim = self.if_decim * self.audio_decim  # 24

        # 端数IQサンプル持ち越し用バッファ (時間軸断絶・クリック音の完全根絶)
        self.raw_leftover = np.empty(0, dtype=np.uint8)

        self.offset_freq = 0.0
        self.mixer_phase = 0.0

        # IF用ローパス (±95kHz Carson標準帯域幅)
        cutoff_if = 95000.0 / self.rf_rate
        self.fir_if = design_fir_kaiser(num_taps=65, cutoff_norm=cutoff_if, beta=6.0)

        # DX微弱局専用 IF狭帯域ローパス (±60kHz: Carson狭窄でホワイトノイズパワーを大幅低減)
        cutoff_if_narrow = 60000.0 / self.rf_rate
        self.fir_if_narrow = design_fir_kaiser(num_taps=65, cutoff_norm=cutoff_if_narrow, beta=6.5)

        # AM用ローパス (±6kHz: 中波放送用)
        cutoff_am = 6000.0 / self.rf_rate
        self.fir_am = design_fir_kaiser(num_taps=81, cutoff_norm=cutoff_am, beta=6.5)

        # 短波放送用 狭帯域AMローパス (±3.5kHz: 短波HF帯の混信をカット)
        cutoff_am_narrow = 3500.0 / self.rf_rate
        self.fir_am_narrow = design_fir_kaiser(num_taps=97, cutoff_norm=cutoff_am_narrow, beta=7.0)

        # ISS / アマチュア無線専用 NFM (ナローバンドFM) IFローパス (±8kHz Carson帯域幅)
        cutoff_nfm_if = 8000.0 / self.rf_rate
        self.fir_nfm = design_fir_kaiser(num_taps=97, cutoff_norm=cutoff_nfm_if, beta=7.0)

        # NFM用 通信音声帯域ハイカットフィルタ (3.0kHz, 48kHzレート)
        cutoff_nfm_audio = 3000.0 / self.audio_rate
        self.fir_nfm_audio = design_fir_kaiser(num_taps=81, cutoff_norm=cutoff_nfm_audio, beta=7.0)

        # 19kHzパイロットトーンを完全阻止するIF段オーディオフィルタ (288kHzレート)
        # カットオフ 15kHz, 19kHzで -60dB以上の超急峻減衰 (97タップ)
        cutoff_if_audio = 15000.0 / self.if_rate
        self.fir_if_audio = design_fir_kaiser(num_taps=97, cutoff_norm=cutoff_if_audio, beta=7.0)

        # 48kHzオーディオ段のアンチエイリアス・ハイカットフィルタ (48kHzレート)
        # 14kHz: 音楽用Hi-Fiワイド (51タップ)
        cutoff_audio_wide = 14000.0 / self.audio_rate
        self.fir_audio_wide = design_fir_kaiser(num_taps=51, cutoff_norm=cutoff_audio_wide, beta=6.0)

        # 8.5kHz: 強力ノイズクリーナー (ヒスノイズ「サー」を消滅させ人の声を鮮明化, 65タップ)
        cutoff_audio_clean = 8500.0 / self.audio_rate
        self.fir_audio_clean = design_fir_kaiser(num_taps=65, cutoff_norm=cutoff_audio_clean, beta=7.0)

        # 5.5kHz: DX超高感度ボイスフィルタ (極微弱局のノイズフロアを徹底抑圧し声の明瞭度を最大化, 65タップ)
        cutoff_audio_narrow = 5500.0 / self.audio_rate
        self.fir_audio_narrow = design_fir_kaiser(num_taps=65, cutoff_norm=cutoff_audio_narrow, beta=7.0)

        # 適応型分数リサンプラ (SDRとサウンドカードのクロックドリフトを微小補正し音飛び根絶)
        self.resampler = AdaptiveDriftResampler(target_chunks=8.0, max_ppm=120.0)

        # ノイズフィルターモード ("clean", "wide", または "narrow")
        self.filter_mode = "clean"

        # ===== Hyper連続認知制御パラメータ (Cascadeの離散切替を無段階モーフィングへ) =====
        self.cognitive_enabled = False
        self.target_cutoff_hz = 8500.0
        self.applied_cutoff_hz = 8500.0
        self.target_if_bw_hz = 190000.0
        self.applied_if_bw_hz = 190000.0
        self.target_hf_gain = 1.0
        self.hf_gain_applied = 1.0
        self.cognitive_alpha = 0.18  # 1フレームあたりの平滑追従率 (ポップノイズ根絶)
        self._fir_cache = {}

        # 4.5kHzクロスオーバーによる心理音響ハイシェルフ (FM三角雑音のみ連続減衰)
        shelf_cut = 4500.0 / self.audio_rate
        self.fir_shelf_lp = design_fir_kaiser(num_taps=81, cutoff_norm=shelf_cut, beta=6.5)
        self.fir_shelf_hp = design_fir_highpass(num_taps=81, cutoff_norm=shelf_cut, beta=6.5)
        self.history_shelf_lp = np.zeros(len(self.fir_shelf_lp) - 1, dtype=np.float32)
        self.history_shelf_hp = np.zeros(len(self.fir_shelf_hp) - 1, dtype=np.float32)

        # AFC (Automatic Frequency Control: 100Hz精度の自動搬送波追従)
        self.afc_enabled = True
        self.afc_offset_hz = 0.0
        self.afc_alpha = 0.05  # 滑らかな追従時定数

        self.fm_last_sample = 0.0 + 0.0j
        self.nfm_last_sample = 0.0 + 0.0j
        self.nfm_afc_offset_hz = 0.0
        self.nfm_afc_alpha = 0.08  # ISSドップラー追従用時定数

        # FIRフィルタの境界連続性保持用バッファ
        self.history_if = np.zeros(len(self.fir_if) - 1, dtype=np.complex64)
        self.history_am = np.zeros(len(self.fir_am) - 1, dtype=np.complex64)
        self.history_am_narrow = np.zeros(len(self.fir_am_narrow) - 1, dtype=np.complex64)
        self.history_nfm = np.zeros(len(self.fir_nfm) - 1, dtype=np.complex64)
        self.history_if_audio = np.zeros(len(self.fir_if_audio) - 1, dtype=np.float32)
        self.history_nfm_audio = np.zeros(len(self.fir_nfm_audio) - 1, dtype=np.float32)
        self.history_final = np.zeros(len(self.fir_audio_clean) - 1, dtype=np.float32)

        # 双一次変換ディエンファシス (地域設定: 日本/欧州=50μs, 米国/韓国=75μs)
        self.deemph_tau_us = 50.0
        T = 1.0 / self.audio_rate
        tau = 50e-6
        denom = 2.0 * tau + T
        self.deemph_b0 = T / denom
        self.deemph_b1 = T / denom
        self.deemph_a1 = -(2.0 * tau - T) / denom
        self.deemph_x1 = 0.0
        self.deemph_y1 = 0.0
        # ネイティブCコア用フィルタ状態 (x1, y1)
        self._deemph_state = np.zeros(2, dtype=np.float32)
        self._dc_hp_state = np.zeros(2, dtype=np.float32)
        self._voice_hp_state = np.zeros(2, dtype=np.float32)
        self._deemph_state_l = np.zeros(2, dtype=np.float32)
        self._deemph_state_r = np.zeros(2, dtype=np.float32)
        self._dc_hp_state_l = np.zeros(2, dtype=np.float32)
        self._dc_hp_state_r = np.zeros(2, dtype=np.float32)
        self.deemph_x1_l = self.deemph_y1_l = 0.0
        self.deemph_x1_r = self.deemph_y1_r = 0.0
        self.dc_hp_x1_l = self.dc_hp_y1_l = 0.0
        self.dc_hp_x1_r = self.dc_hp_y1_r = 0.0

        # ===== FMステレオMPXデコーダ (19kHzパイロットPLL + 38kHz同期検波) =====
        self.stereo_enabled = True
        self.is_stereo = False
        self.stereo_blend = 0.0
        cutoff_pilot_lp = 20500.0 / self.if_rate
        cutoff_pilot_hp = 17000.0 / self.if_rate
        self.fir_pilot_lp = design_fir_kaiser(num_taps=97, cutoff_norm=cutoff_pilot_lp, beta=6.5)
        self.fir_pilot_hp = design_fir_highpass(num_taps=97, cutoff_norm=cutoff_pilot_hp, beta=6.5)
        self.history_pilot_lp = np.zeros(len(self.fir_pilot_lp) - 1, dtype=np.float32)
        self.history_pilot_hp = np.zeros(len(self.fir_pilot_hp) - 1, dtype=np.float32)
        self.history_lpr = np.zeros(len(self.fir_if_audio) - 1, dtype=np.float32)
        self.history_final_l = np.zeros(len(self.fir_audio_clean) - 1, dtype=np.float32)
        self.history_final_r = np.zeros(len(self.fir_audio_clean) - 1, dtype=np.float32)
        self.history_shelf_lp_l = np.zeros(len(self.fir_shelf_lp) - 1, dtype=np.float32)
        self.history_shelf_hp_l = np.zeros(len(self.fir_shelf_hp) - 1, dtype=np.float32)
        self.history_shelf_lp_r = np.zeros(len(self.fir_shelf_lp) - 1, dtype=np.float32)
        self.history_shelf_hp_r = np.zeros(len(self.fir_shelf_hp) - 1, dtype=np.float32)
        self._pll_theta = 0.0
        self._pll_integ = 0.0
        self._pll_w0 = 2.0 * np.pi * 19000.0 / self.if_rate
        self._pll_kp = 0.01
        self._pll_ki = 2e-5
        self._pll_alpha = 2.0 * np.pi * 100.0 / self.if_rate  # ループフィルタ遮断 ~100Hz
        self._pll_ef = 0.0
        self._last_cos2 = None
        self._last_sin2 = None
        self._stereo_blend = 0.0
        # 38kHz副搬送波の位相補正 (DSP経路の遅延を実測校正した値。DSBの180°曖昧性は
        # パイロットとの2:1位相関係に基づきL/Rが正しくなる側を採用)
        self.stereo_phase_offset = np.deg2rad(-47.0)
        # ステレオRch用 追加リサンプラ (Lchは既存resamplerを共用)
        self.resampler_r = AdaptiveDriftResampler(target_chunks=8.0, max_ppm=2000.0)

        # DCカット用ハイパスフィルタ状態 (30Hzカットオフ)
        # y[n] = x[n] - x[n-1] + R * y[n-1]
        self.dc_hp_r = float(1.0 - (2.0 * np.pi * 30.0 / self.audio_rate))
        self.dc_hp_x1 = 0.0
        self.dc_hp_y1 = 0.0

        # 通信用300Hzハイパスフィルタ状態 (音声通信用)
        self.voice_hp_r = float(1.0 - (2.0 * np.pi * 300.0 / self.audio_rate))
        self.voice_hp_x1 = 0.0
        self.voice_hp_y1 = 0.0

        # スケルチ (放送受信中のバタつき・ブツブツ音防止のためデフォルトOFF)
        self.squelch_threshold = -85.0
        self.squelch_enabled = False

        # スペクトラム表示設定
        self.fft_size = 1024
        self.fft_window = np.hamming(self.fft_size).astype(np.float32)
        self.fft_smooth = None
        self.smooth_alpha = 0.25

    def set_offset_freq(self, offset_hz: float):
        self.offset_freq = offset_hz
        self.afc_offset_hz = 0.0
        self.nfm_afc_offset_hz = 0.0

    def update_resampler_feedback(self, current_chunks: float, dt: float = 0.05):
        """オーディオバッファの残存チャンク数をリサンプラにフィードバック (クロック自動同期)"""
        self.resampler.update_feedback(current_chunks, dt=dt)
        self.resampler_r.update_feedback(current_chunks, dt=dt)

    def set_stereo_enabled(self, enabled: bool):
        """FMステレオMPXデコードの有効/無効 (モノラル強制)"""
        self.stereo_enabled = bool(enabled)
        if not self.stereo_enabled:
            self.is_stereo = False
            self.stereo_blend = 0.0
            self._stereo_blend = 0.0
            self._last_cos2 = None

    def set_deemphasis(self, tau_us: float):
        """ディエンファシス時定数を設定 (日本/欧州=50μs, 米国/韓国=75μs)"""
        T = 1.0 / self.audio_rate
        tau = float(tau_us) * 1e-6
        denom = 2.0 * tau + T
        self.deemph_tau_us = float(tau_us)
        self.deemph_b0 = T / denom
        self.deemph_b1 = T / denom
        self.deemph_a1 = -(2.0 * tau - T) / denom
        self.deemph_x1 = self.deemph_y1 = 0.0
        self._deemph_state[:] = 0.0
        self._deemph_state_l[:] = 0.0
        self._deemph_state_r[:] = 0.0

    def set_squelch(self, enabled: bool, threshold_db: float = -68.0):
        self.squelch_enabled = enabled
        self.squelch_threshold = threshold_db

    def set_cognitive_parameters(
        self,
        cutoff_hz: float = None,
        hf_gain: float = None,
        if_bw_hz: float = None,
        enabled: bool = True,
    ):
        """
        HyperControllerからの連続制御パラメータ受け口。
        :param cutoff_hz: オーディオ段カットオフ (4200〜16000Hz, 無段階)
        :param hf_gain: 高域(4.5kHz超)ノイズ抑圧ゲイン (0.0〜1.0, 無段階)
        :param if_bw_hz: IF実効帯域幅 (110000〜200000Hz, 無段階)
        """
        self.cognitive_enabled = enabled
        if cutoff_hz is not None:
            self.target_cutoff_hz = float(np.clip(cutoff_hz, 4200.0, 16000.0))
        if hf_gain is not None:
            self.target_hf_gain = float(np.clip(hf_gain, 0.0, 1.0))
        if if_bw_hz is not None:
            self.target_if_bw_hz = float(np.clip(if_bw_hz, 110000.0, 200000.0))

    def _update_cognitive_morph(self):
        """離散切替ではなくサンプル単位で滑らかにフィルタを変形 (クリック・ポップ根絶)"""
        if not self.cognitive_enabled:
            return
        a = self.cognitive_alpha
        self.applied_cutoff_hz += a * (self.target_cutoff_hz - self.applied_cutoff_hz)
        self.applied_if_bw_hz += a * (self.target_if_bw_hz - self.applied_if_bw_hz)
        self.hf_gain_applied += a * (self.target_hf_gain - self.hf_gain_applied)

    def _get_dynamic_filter(self, kind: str, cutoff_hz: float) -> np.ndarray:
        """カットオフ毎にFIRをその場設計 (100Hz量子化キャッシュで実時間コスト極小)"""
        quant = max(200, int(round(cutoff_hz / 100.0)) * 100)
        key = (kind, quant)
        taps = self._fir_cache.get(key)
        if taps is None:
            if kind == "if":
                taps = design_fir_kaiser(num_taps=65, cutoff_norm=quant / self.rf_rate, beta=6.2)
            else:
                taps = design_fir_kaiser(num_taps=81, cutoff_norm=quant / self.audio_rate, beta=6.8)
            if len(self._fir_cache) > 128:
                self._fir_cache.clear()
            self._fir_cache[key] = taps
        return taps

    def _apply_hf_shelf(self, audio: np.ndarray, ch: str = "") -> np.ndarray:
        """線形位相クロスオーバーにより高域ヒスノイズのみを連続可変減衰 (低域は完全素通し)"""
        if len(audio) == 0:
            return audio
        low = self.decimate_with_history(audio, self.fir_shelf_lp, 1, f"history_shelf_lp{ch}")
        high = self.decimate_with_history(audio, self.fir_shelf_hp, 1, f"history_shelf_hp{ch}")
        g = self.hf_gain_applied
        if g >= 0.999:
            return (low + high).astype(np.float32)
        return (low + g * high).astype(np.float32)

    def raw_to_iq(self, raw_bytes: np.ndarray) -> np.ndarray:
        if len(raw_bytes) < 2:
            return np.empty(0, dtype=np.complex64)
        n = (len(raw_bytes) // 2) * 2
        raw_f = raw_bytes[:n].astype(np.float32)
        i_comp = (raw_f[0::2] - 127.5) * (1.0 / 128.0)
        q_comp = (raw_f[1::2] - 127.5) * (1.0 / 128.0)
        return (i_comp + 1j * q_comp).astype(np.complex64)

    def mix_frequency(self, iq: np.ndarray, mode: str = "WFM") -> np.ndarray:
        afc = self.nfm_afc_offset_hz if mode == "NFM" else (self.afc_offset_hz if self.afc_enabled else 0.0)
        effective_offset = self.offset_freq + afc
        if abs(effective_offset) < 1.0 or len(iq) == 0:
            return iq
        n = len(iq)
        phase_step = (2.0 * np.pi * effective_offset) / self.rf_rate
        if _NATIVE is not None:
            # in-place複素ミキサー (Cコア: 一時配列なし・GIL解放)
            work = np.ascontiguousarray(iq, dtype=np.complex64)
            ph = ctypes.c_double(float(self.mixer_phase))
            _NATIVE.sdr_mix_freq(_fptr(work), n, phase_step, ctypes.byref(ph))
            self.mixer_phase = float(ph.value % (2.0 * np.pi))
            return work
        phases = self.mixer_phase + phase_step * np.arange(n, dtype=np.float64)
        lo = np.exp(1j * phases).astype(np.complex64)
        self.mixer_phase = float((self.mixer_phase + phase_step * n) % (2.0 * np.pi))
        return iq * lo

    def decimate_with_history(self, x: np.ndarray, fir_taps: np.ndarray, factor: int, history_attr: str) -> np.ndarray:
        """過去サンプルを保持したシームレスなFIR畳み込み & デシメーション (任意のフィルタ長に動的完全同期)"""
        if len(x) == 0:
            return x
        req_hist = len(fir_taps) - 1
        hist = getattr(self, history_attr)
        if len(hist) != req_hist:
            if len(hist) < req_hist:
                pad = np.zeros(req_hist - len(hist), dtype=hist.dtype)
                hist = np.concatenate((pad, hist))
            else:
                hist = hist[-req_hist:]
            setattr(self, history_attr, hist)

        x_ext = np.concatenate((hist, x))
        setattr(self, history_attr, x[-req_hist:] if len(x) >= req_hist else x_ext[-req_hist:])

        # 畳み込み (validモードで境界アーティファクトを完全排除)
        if np.iscomplexobj(x_ext):
            r = np.convolve(x_ext.real, fir_taps, mode="valid")
            i = np.convolve(x_ext.imag, fir_taps, mode="valid")
            filtered = r + 1j * i
        else:
            filtered = np.convolve(x_ext, fir_taps, mode="valid")

        return filtered[::factor]

    def decimate(self, x: np.ndarray, fir_taps: np.ndarray, factor: int) -> np.ndarray:
        return self.decimate_with_history(x, fir_taps, factor, "history_if")

    def _apply_hard_limiter(self, iq_if: np.ndarray) -> np.ndarray:
        mag = np.abs(iq_if) + 1e-12
        return iq_if / mag

    def _apply_dc_highpass(self, audio: np.ndarray, ch: str = "") -> np.ndarray:
        """30Hz以下の不要な直流・ボコボコ音を完全カット (Cコア: GIL解放で並行実行)"""
        if len(audio) == 0:
            return audio
        if _NATIVE is not None:
            x = np.ascontiguousarray(audio, dtype=np.float32)
            y = np.empty_like(x)
            state = getattr(self, f"_dc_hp_state{ch}")
            _NATIVE.sdr_one_pole_highpass(_fptr(x), _fptr(y), len(x),
                                          float(self.dc_hp_r), _fptr(state))
            return y
        y = np.empty_like(audio)
        r = self.dc_hp_r
        x1 = getattr(self, f"dc_hp_x1{ch}")
        y1 = getattr(self, f"dc_hp_y1{ch}")
        for i in range(len(audio)):
            x = audio[i]
            curr_y = x - x1 + r * y1
            y[i] = curr_y
            x1 = x
            y1 = curr_y
        setattr(self, f"dc_hp_x1{ch}", x1)
        setattr(self, f"dc_hp_y1{ch}", y1)
        return y

    def demodulate_wfm(self, iq_if: np.ndarray) -> np.ndarray:
        if len(iq_if) < 2:
            return np.zeros(0, dtype=np.float32)

        # スケルチ判定
        power_db = 10.0 * np.log10(np.mean(np.abs(iq_if) ** 2) + 1e-12)
        if self.squelch_enabled and power_db < self.squelch_threshold:
            return np.zeros(len(iq_if) // self.audio_decim, dtype=np.float32)

        # 1. ハードリミッター適用
        limited = self._apply_hard_limiter(iq_if)

        # 2. 瞬時位相差分法 (FM復調)
        if _NATIVE is not None:
            work = np.ascontiguousarray(limited, dtype=np.complex64)
            demod = np.empty(len(work), dtype=np.float32)
            last = np.array([self.fm_last_sample.real, self.fm_last_sample.imag], dtype=np.float32)
            _NATIVE.sdr_fm_demod(_fptr(work), _fptr(demod), len(work), _fptr(last))
            self.fm_last_sample = complex(float(last[0]), float(last[1]))
        else:
            s = np.concatenate(([self.fm_last_sample], limited))
            self.fm_last_sample = limited[-1]
            diff = s[1:] * np.conj(s[:-1])
            demod = np.angle(diff)

        # AFC (Automatic Frequency Control): 復調信号のDCバイアスから周波数偏差を推定してフィードバック
        if self.afc_enabled and len(demod) > 0:
            mean_dc = float(np.mean(demod))
            freq_error_hz = mean_dc * (self.if_rate / (2.0 * np.pi))
            if abs(freq_error_hz) < 20000.0:  # ±20kHz以内の偏差に自動追従
                # 20Hz未満の微小ジッターは補正を休止し完全ロックを保持
                if abs(freq_error_hz) > 20.0:
                    self.afc_offset_hz = float(np.clip(
                        self.afc_offset_hz - self.afc_alpha * freq_error_hz,
                        -20000.0,
                        20000.0
                    ))

        # 3. 適切な名目オーディオゲインにスケーリング
        # 日本規格の最大周波数偏移(±75kHz)でも振幅0.95以内に収め、過変調時のソフトリミッターポンピング歪みを根絶
        demod_scaled = demod * 0.58

        # 4. 19kHzパイロットトーンを完全阻止する急峻なIFオーディオFIRフィルタ & デシメーション
        audio = self.decimate_with_history(demod_scaled, self.fir_if_audio, self.audio_decim, "history_if_audio")

        # 5. モノラル (L+R) を48kHzへデシメーション
        #    (19kHzパイロット・38kHz副搬送波はアンチエイリアスLPFで除去)
        mono = self.decimate_with_history(demod_scaled, self.fir_if_audio,
                                          self.audio_decim, "history_if_audio")

        # 5b. ステレオMPXデコード (19kHzパイロットPLL + 38kHz同期検波)
        self._update_stereo_pilot(demod_scaled)
        stereo_diff = None
        if self._last_cos2 is not None and self._stereo_blend > 0.02:
            carrier = self._last_cos2
            if abs(self.stereo_phase_offset) > 1e-6 and self._last_sin2 is not None:
                co = np.cos(self.stereo_phase_offset)
                si = np.sin(self.stereo_phase_offset)
                carrier = self._last_cos2 * co - self._last_sin2 * si
            lpr = demod_scaled * carrier
            diff = self.decimate_with_history(lpr, self.fir_if_audio,
                                              self.audio_decim, "history_lpr") * 2.0
            if len(diff) == len(mono):
                stereo_diff = diff * self._stereo_blend

        # 6. チャンネル別ポスト処理 (ディエンファシス・ハイカット・DCカット・シェルフ)
        if stereo_diff is not None:
            left = self._post_process_wfm(mono + stereo_diff, "_l")
            right = self._post_process_wfm(mono - stereo_diff, "_r")
            self.is_stereo = self._stereo_blend > 0.5
            return np.stack([left, right], axis=1).astype(np.float32)

        self.is_stereo = False
        return self._post_process_wfm(mono, "")

    def _post_process_wfm(self, audio: np.ndarray, ch: str = "") -> np.ndarray:
        """WFM音声のチャンネル別仕上げ (ch: ''=モノ, '_l'/'_r'=ステレオ各ch)"""
        # ディエンファシス (50/75μs)
        audio = self._apply_bilinear_deemphasis(audio, ch=ch)

        # オーディオ段ハイカットフィルタ (Hyper時は無段階モーフィング)
        if self.cognitive_enabled:
            fir_final = self._get_dynamic_filter("audio", self.applied_cutoff_hz)
        elif self.filter_mode == "wide":
            fir_final = self.fir_audio_wide
        elif self.filter_mode == "narrow":
            fir_final = self.fir_audio_narrow
        else:
            fir_final = self.fir_audio_clean

        # decimate_with_history (factor=1) を用いることで、フィルタ長変更時にもサンプル数の一致を保証
        audio = self.decimate_with_history(audio, fir_final, 1, f"history_final{ch}")

        # DCハイパスフィルタ
        audio = self._apply_dc_highpass(audio, ch=ch)

        # Hyper心理音響ハイシェルフ (FM三角雑音を連続減衰) / Cascade離散エキスパンダー
        if self.cognitive_enabled:
            audio = self._apply_hf_shelf(audio, ch=ch)
        elif self.filter_mode == "narrow":
            audio = self._apply_noise_expander(audio, threshold=0.09)

        return audio.astype(np.float32)

    def _update_stereo_pilot(self, mpx: np.ndarray):
        """19kHzパイロットPLLを更新し、ステレオブレンド係数を決定する"""
        self._last_cos2 = None
        if not self.stereo_enabled or _NATIVE is None or len(mpx) < 64:
            self._stereo_blend *= 0.9
            self.stereo_blend = self._stereo_blend
            return

        try:
            pilot = self.decimate_with_history(mpx, self.fir_pilot_lp, 1, "history_pilot_lp")
            pilot = self.decimate_with_history(pilot, self.fir_pilot_hp, 1, "history_pilot_hp")
            n = len(pilot)
            if n < 32:
                return
            # パイロット振幅で正規化した生MPXをPLLへ入力 (帯域制限による群遅延を回避し、
            # 搬送波位相をMPX本来のタイムラインに一致させる)
            pilot_rms = float(np.sqrt(np.mean(pilot.astype(np.float64) ** 2)) + 1e-12)
            sig = np.ascontiguousarray(np.asarray(mpx, dtype=np.float32) / pilot_rms, dtype=np.float32)
            cos2 = np.empty(n, dtype=np.float32)
            sin2 = np.empty(n, dtype=np.float32)
            quality = ctypes.c_float(0.0)
            th = ctypes.c_double(self._pll_theta)
            ig = ctypes.c_double(self._pll_integ)
            ef = ctypes.c_double(self._pll_ef)
            _NATIVE.sdr_stereo_pll(_fptr(sig), n, ctypes.byref(th), self._pll_w0,
                                   self._pll_kp, self._pll_ki, ctypes.byref(ig),
                                   ctypes.byref(ef), self._pll_alpha,
                                   _fptr(cos2), _fptr(sin2), ctypes.byref(quality))
            self._pll_theta = th.value
            self._pll_integ = ig.value
            self._pll_ef = ef.value

            mpx_rms = float(np.sqrt(np.mean(np.asarray(mpx, dtype=np.float64) ** 2)) + 1e-12)
            ratio = pilot_rms / mpx_rms
            lock = float(quality.value)  # 正規化パイロット基準: ロック時 ~0.5-0.7

            target = 0.0
            if lock > 0.25 and 0.02 < ratio < 0.8:
                target = min(1.0, (ratio - 0.02) / 0.05)

            if target > self._stereo_blend:
                self._stereo_blend = min(target, self._stereo_blend + 0.25)
            else:
                self._stereo_blend = max(target, self._stereo_blend * 0.97)

            self.stereo_blend = self._stereo_blend
            if self._stereo_blend > 0.02:
                self._last_cos2 = cos2
                self._last_sin2 = sin2
        except Exception:
            self._last_cos2 = None
            self._last_sin2 = None
            self._stereo_blend *= 0.9

    def _apply_voice_highpass(self, audio: np.ndarray) -> np.ndarray:
        """通信音声用の300Hzハイパスフィルタ (Cコア: GIL解放で並行実行)"""
        if len(audio) == 0:
            return audio
        if _NATIVE is not None:
            x = np.ascontiguousarray(audio, dtype=np.float32)
            y = np.empty_like(x)
            _NATIVE.sdr_one_pole_highpass(_fptr(x), _fptr(y), len(x),
                                          float(self.voice_hp_r), _fptr(self._voice_hp_state))
            return y
        y = np.empty_like(audio)
        r = self.voice_hp_r
        x1 = self.voice_hp_x1
        y1 = self.voice_hp_y1
        for i in range(len(audio)):
            x = audio[i]
            curr_y = x - x1 + r * y1
            y[i] = curr_y
            x1 = x
            y1 = curr_y
        self.voice_hp_x1 = x1
        self.voice_hp_y1 = y1
        return y

    def demodulate_nfm(self, iq_if: np.ndarray) -> np.ndarray:
        """
        ISS（国際宇宙ステーション）やアマチュア無線用ナローバンドFM（NFM）復調器。
        - Carson帯域幅 12〜16kHz
        - ドップラーシフト（±3.5kHz）追従AFC
        - 通信音声用帯域通過フィルタ（300Hz〜3000Hz）
        - 最適化された通信オーディオゲイン補償
        """
        if len(iq_if) < 2:
            return np.zeros(0, dtype=np.float32)

        # スケルチ判定 (通信用NFMは無信号時の突発ノイズを防止するため、スケルチ必須)
        power_db = 10.0 * np.log10(np.mean(np.abs(iq_if) ** 2) + 1e-12)
        effective_threshold = self.squelch_threshold if self.squelch_enabled else -74.0
        if power_db < effective_threshold:
            return np.zeros(len(iq_if) // self.audio_decim, dtype=np.float32)

        # 1. ハードリミッター適用
        limited = self._apply_hard_limiter(iq_if)

        # 2. 瞬時位相差分法 (FM復調)
        if _NATIVE is not None:
            work = np.ascontiguousarray(limited, dtype=np.complex64)
            demod = np.empty(len(work), dtype=np.float32)
            last = np.array([self.nfm_last_sample.real, self.nfm_last_sample.imag], dtype=np.float32)
            _NATIVE.sdr_fm_demod(_fptr(work), _fptr(demod), len(work), _fptr(last))
            self.nfm_last_sample = complex(float(last[0]), float(last[1]))
        else:
            s = np.concatenate(([self.nfm_last_sample], limited))
            self.nfm_last_sample = limited[-1]
            diff = s[1:] * np.conj(s[:-1])
            demod = np.angle(diff)

        # 3. NFMドップラー自動周波数追従 (ISSが飛翔する際の ±3.5kHz 移動に自動ロック)
        if self.afc_enabled and len(demod) > 0:
            mean_dc = float(np.mean(demod))
            freq_error_hz = mean_dc * (self.if_rate / (2.0 * np.pi))
            if abs(freq_error_hz) < 8000.0:  # ±8kHz以内のドップラー偏移に追従
                # 15Hz未満の微小ジッターは補正を休止し完全ロックを保持
                if abs(freq_error_hz) > 15.0:
                    self.nfm_afc_offset_hz = float(np.clip(
                        self.nfm_afc_offset_hz - self.nfm_afc_alpha * freq_error_hz,
                        -8000.0,
                        8000.0
                    ))

        # 4. オーディオゲイン補償 (NFMの微小周波数偏移を標準通信音量に最適化)
        demod_scaled = demod * 8.5

        # 5. IFオーディオデシメーション (288kHz -> 48kHz)
        audio = self.decimate_with_history(demod_scaled, self.fir_if_audio, self.audio_decim, "history_if_audio")

        # 6. 通信用3.0kHzハイカットフィルタ
        audio = self.decimate_with_history(audio, self.fir_nfm_audio, 1, "history_nfm_audio")

        # 7. 通信用300Hz音声ハイパスフィルタ
        audio = self._apply_voice_highpass(audio)

        return audio.astype(np.float32)

    def _apply_noise_expander(self, audio: np.ndarray, threshold: float = 0.09) -> np.ndarray:
        """弱電界時のFM三角雑音（高域ヒスノイズ）を抑え込み人の声を浮き彫りにするソフトエキスパンダー"""
        if len(audio) == 0:
            return audio
        mag = np.abs(audio)
        gain = np.where(mag < threshold, (mag / threshold) ** 0.5, 1.0)
        return (audio * gain).astype(np.float32)

    def demodulate_am(self, iq_if: np.ndarray) -> np.ndarray:
        if len(iq_if) == 0:
            return np.zeros(0, dtype=np.float32)

        env = np.abs(iq_if)
        power_db = 10.0 * np.log10(np.mean(env**2) + 1e-12)
        if self.squelch_enabled and power_db < self.squelch_threshold:
            return np.zeros(len(iq_if) // self.audio_decim, dtype=np.float32)

        audio_raw = env * 1.5
        if self.cognitive_enabled:
            fir_final = self._get_dynamic_filter("audio", min(self.applied_cutoff_hz, 8000.0))
        else:
            fir_final = self.fir_audio_clean if self.filter_mode == "clean" else self.fir_audio_wide
        audio = self.decimate_with_history(audio_raw, self.fir_if_audio, self.audio_decim, "history_if_audio")

        audio = self.decimate_with_history(audio, fir_final, 1, "history_final")
        audio = self._apply_dc_highpass(audio)
        return audio.astype(np.float32)

    def _apply_bilinear_deemphasis(self, x: np.ndarray, ch: str = "") -> np.ndarray:
        if len(x) == 0:
            return x
        if _NATIVE is not None:
            xin = np.ascontiguousarray(x, dtype=np.float32)
            y = np.empty_like(xin)
            state = getattr(self, f"_deemph_state{ch}")
            _NATIVE.sdr_bilinear_deemphasis(
                _fptr(xin), _fptr(y), len(xin),
                float(self.deemph_b0), float(self.deemph_b1), float(-self.deemph_a1),
                _fptr(state))
            return y
        y = np.empty_like(x)
        b0 = self.deemph_b0
        b1 = self.deemph_b1
        minus_a1 = -self.deemph_a1
        x1 = getattr(self, f"deemph_x1{ch}")
        y1 = getattr(self, f"deemph_y1{ch}")
        for i in range(len(x)):
            curr_x = x[i]
            curr_y = b0 * curr_x + b1 * x1 + minus_a1 * y1
            y[i] = curr_y
            x1 = curr_x
            y1 = curr_y
        setattr(self, f"deemph_x1{ch}", x1)
        setattr(self, f"deemph_y1{ch}", y1)
        return y

    def compute_spectrum(self, iq: np.ndarray) -> np.ndarray:
        if len(iq) < self.fft_size:
            return np.zeros(self.fft_size, dtype=np.float32)
        chunk = iq[-self.fft_size :] * self.fft_window
        fft_data = np.fft.fftshift(np.fft.fft(chunk)) / self.fft_size
        power = (np.abs(fft_data) ** 2) / (np.sum(self.fft_window**2) / self.fft_size) + 1e-12
        power_db = 10.0 * np.log10(power)
        if self.fft_smooth is None:
            self.fft_smooth = power_db
        else:
            self.fft_smooth = self.smooth_alpha * power_db + (1.0 - self.smooth_alpha) * self.fft_smooth
        return self.fft_smooth.astype(np.float32)

    def process(self, raw_bytes: np.ndarray, mode: str = "WFM") -> tuple[np.ndarray, np.ndarray]:
        # 前回余った端数バイトと結合
        if len(self.raw_leftover) > 0:
            raw_bytes = np.concatenate((self.raw_leftover, raw_bytes))

        # 48バイト (24 IQサンプル = IFデシメーション4 × オーディオデシメーション6) の完全な整数倍に切り分ける
        unit_bytes = self.total_decim * 2  # 24 * 2 = 48
        usable_len = (len(raw_bytes) // unit_bytes) * unit_bytes
        self.raw_leftover = raw_bytes[usable_len:]
        raw_work = raw_bytes[:usable_len]

        if len(raw_work) == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(self.fft_size, dtype=np.float32)

        iq = self.raw_to_iq(raw_work)
        iq_shifted = self.mix_frequency(iq, mode=mode)
        spectrum_db = self.compute_spectrum(iq_shifted)

        # Hyper連続認知制御: フィルタを離散切替ではなく無段階モーフィング
        self._update_cognitive_morph()

        if mode == "NFM":
            # ISS / アマチュア無線用ナローバンドFM
            iq_if = self.decimate_with_history(iq_shifted, self.fir_nfm, self.if_decim, "history_nfm")
            audio = self.demodulate_nfm(iq_if)
        elif mode in ("AM", "AM_NARROW"):
            # 中波・短波放送用AM (狭帯域AMフィルタ対応)
            fir_target = self.fir_am_narrow if (mode == "AM_NARROW" or self.filter_mode == "narrow") else self.fir_am
            hist_attr = "history_am_narrow" if fir_target is self.fir_am_narrow else "history_am"
            iq_if = self.decimate_with_history(iq_shifted, fir_target, self.if_decim, hist_attr)
            audio = self.demodulate_am(iq_if)
        else:
            # ワイドFM (WFM)
            if self.cognitive_enabled:
                fir_if_target = self._get_dynamic_filter("if", self.applied_if_bw_hz / 2.0)
            elif self.filter_mode == "narrow":
                fir_if_target = self.fir_if_narrow
            else:
                fir_if_target = self.fir_if
            iq_if = self.decimate(iq_shifted, fir_if_target, self.if_decim)
            audio = self.demodulate_wfm(iq_if)

        # 適応型分数リサンプラ (独立クロック間のドリフトを微小補正し完全連続再生)
        if audio.ndim == 2:
            left = self.resampler.process(audio[:, 0])
            right = self.resampler_r.process(audio[:, 1])
            m = min(len(left), len(right))
            audio_synced = np.stack([left[:m], right[:m]], axis=1)
            # 過渡クリックサプレッサー (チャンネル別)
            audio_clean = np.stack([
                suppress_click_transients(audio_synced[:, 0]),
                suppress_click_transients(audio_synced[:, 1]),
            ], axis=1).astype(np.float32)
        else:
            audio_synced = self.resampler.process(audio)
            # 過渡クリックサプレッサー (チューナー切替過渡ノイズを完全消滅)
            audio_clean = suppress_click_transients(audio_synced)

        return audio_clean, spectrum_db
