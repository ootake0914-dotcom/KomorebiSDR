"""
Digital Signal Processing (DSP) Module for SDR - Ultra-Clear Hi-Fi Edition.
歪み・クリッピング・19kHzパイロットトーン・ヒスノイズを完全解消した高音質DSPパイプライン。

速度が要求される処理 (IIRフィルタ / FM復調 / 複素ミキサー / クリック除去) は
ネイティブCコア (sdr_core.dll) で実行する。ctypes呼び出し中はGILが解放されるため、
GUI描画やオーディオコールバックと競合せず、音飛び (バッファ枯渇) を防ぐ。
DLLが見つからない場合、純Python代替のある処理 (AM同期検波・FIR畳み込み等) は
自動フォールバックする。ステレオMPX-PLLとRDS-57kHz搬送波生成はCコア必須の
ため、DLL不在時はモノラル受信となる (ブレンドは0へ減衰)。
"""

import ctypes
import os
from collections import deque
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
        # 追加関数は無くてもネイティブコア全体を無効化しない (旧DLLとの後方互換)
        global NATIVE_AM_SYNC, NATIVE_PLL3, NATIVE_FIR
        if hasattr(lib, "sdr_stereo_pll3"):
            lib.sdr_stereo_pll3.argtypes = [pf, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
                                            ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                            ctypes.c_double, pf, pf, pf, pf, pf]
            NATIVE_PLL3 = True
        if hasattr(lib, "sdr_am_sync"):
            lib.sdr_am_sync.argtypes = [pf, pf, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
                                        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                        ctypes.c_double, ctypes.c_double, ctypes.c_double, pf]
            NATIVE_AM_SYNC = True
        if hasattr(lib, "sdr_fir_real"):
            lib.sdr_fir_real.argtypes = [pf, pf, pf, ctypes.c_int, ctypes.c_int]
            NATIVE_FIR = True
        if hasattr(lib, "sdr_fast_fpu"):
            try:
                lib.sdr_fast_fpu()  # FTZ/DAZ有効化 (denormalジッタ対策)
            except Exception:
                pass
        if lib.sdr_version() < 1:
            return None
        return lib
    except Exception:
        return None


NATIVE_AM_SYNC = False
NATIVE_PLL3 = False
NATIVE_FIR = False
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

        # SSB用 複素バンドパスを構成する実LPF (±1.5kHz通過, 48kHzレート)
        # シフト→LPF→逆シフトで非対称バンドパスを作り、反対側波帯を除去する
        # 反対側波帯は±1.8kHz以遠にあるため、急峻な401タップで十分な阻止特性を確保
        cutoff_ssb_lp = 1350.0 / self.audio_rate
        self.fir_ssb_lp = design_fir_kaiser(num_taps=401, cutoff_norm=cutoff_ssb_lp, beta=7.5)

        # CW用 狭帯域LPF (±350Hz)
        cutoff_cw_lp = 350.0 / self.audio_rate
        self.fir_cw_lp = design_fir_kaiser(num_taps=481, cutoff_norm=cutoff_cw_lp, beta=8.0)

        # AM/SSB通信用 4kHzローパス (8.5kHzでは短波のヒスを通しすぎるため)
        cutoff_am_audio = 4000.0 / self.audio_rate
        self.fir_am_audio = design_fir_kaiser(num_taps=81, cutoff_norm=cutoff_am_audio, beta=7.0)

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
        self.history_ssb = np.zeros(len(self.fir_am_narrow) - 1, dtype=np.complex64)
        self.history_ssb2 = np.zeros(len(self.fir_if_audio) - 1, dtype=np.complex64)
        self.history_ssb_lp = np.zeros(len(self.fir_ssb_lp) - 1, dtype=np.complex64)
        self.history_cw_lp = np.zeros(len(self.fir_cw_lp) - 1, dtype=np.complex64)
        self.history_ssb_audio = np.zeros(len(self.fir_nfm_audio) - 1, dtype=np.float32)
        self.history_am_audio = np.zeros(len(self.fir_am_audio) - 1, dtype=np.float32)
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
        self.stereo_pilot_lock = 0.0
        self.stereo_pilot_ratio = 0.0
        cutoff_pilot_lp = 20500.0 / self.if_rate
        cutoff_pilot_hp = 17000.0 / self.if_rate
        self.fir_pilot_lp = design_fir_kaiser(num_taps=97, cutoff_norm=cutoff_pilot_lp, beta=6.5)
        self.fir_pilot_hp = design_fir_highpass(num_taps=97, cutoff_norm=cutoff_pilot_hp, beta=6.5)
        self.history_pilot_lp = np.zeros(len(self.fir_pilot_lp) - 1, dtype=np.float32)
        self.history_pilot_hp = np.zeros(len(self.fir_pilot_hp) - 1, dtype=np.float32)
        self.history_lpr = np.zeros(len(self.fir_if_audio) - 1, dtype=np.float32)
        self.history_rds = np.zeros(len(self.fir_am_narrow) - 1, dtype=np.float32)
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
        self._last_cos3 = None
        self._last_sin3 = None

        # ===== RDS (57kHz) =====
        self.rds_enabled = True
        self.rds = None            # 遅延生成 (rds.RdsDecoder)
        self.rds_ps = ""
        self.rds_rt = ""
        self.rds_pi = 0
        self.rds_pty = None
        self.rds_groups = 0
        # パイロットPLLは90°進んだ位相でロックするため、3逓倍(57kHz)は直交になる。
        # -90°回転で搬送波を同相へ戻す (差動復号なので±符号は不問)。
        self.rds_phase_offset = np.deg2rad(-90.0)
        self._stereo_blend = 0.0
        # 38kHz副搬送波の位相補正 (DSP経路の遅延を実測校正した値。DSBの180°曖昧性は
        # パイロットとの2:1位相関係に基づきL/Rが正しくなる側を採用)
        self.stereo_phase_offset = np.deg2rad(-47.0)
        # ステレオRch用 追加リサンプラ (Lchは既存resamplerを共用。
        # L/Rでmax_ppmを変えると逼迫時に出力長が乖離し片ch切り捨てが起きるため、
        # 必ず同一パラメータにする)
        self.resampler_r = AdaptiveDriftResampler(target_chunks=8.0, max_ppm=120.0)

        # ===== ステレオノイズリダクション =====
        # 弱電界でステレオ化すると増えるヒスノイズを、(L-R)差信号の高域/中域パワー比から
        # 検出し、ノイズ量に応じて 1) サブバンドWiener抑圧 2) 可変ローパス
        # 3) ブレンドでモノラルへ寄せる。実測はNR適用前の生差信号で行うため発振しない。
        self.stereo_nr_enabled = True
        self.stereo_status = "MONO"       # "STEREO" / "BLEND" / "MONO"
        self.stereo_nr_gain = 1.0         # ノイズ由来ブレンド (1=フルステレオ, 0=モノラル)
        self.stereo_cut_hz = 15000.0      # 差信号ローパス遮断周波数 (平滑)
        self.stereo_hiss_db = -60.0       # (L-R)ヒス指標 (初期値=クリーン, NR不発動)
        self._nr_cut_max_hz = 15000.0
        self._nr_cut_min_hz = 5000.0
        # ブレンド量 (極端に弱い局のみモノラル化。通常はWienerが周波数別に処理)
        self._nr_lo_db = -18.0
        self._nr_hi_db = -4.0
        # Wiener適用量 (これより上のノイズで段階的にサブバンド抑圧)
        self._nr_wiener_lo_db = -40.0
        self._nr_wiener_hi_db = -18.0
        self._nr_primed = False
        self._nr_s_w = 0.0                # 平滑化されたWiener適用度 (0=off, 1=full)
        self._nr_s = 0.0                  # 平滑化されたノイズ度 (0=クリーン, 1=ノイズ)
        self._nr_hist = deque(maxlen=100)  # 差分HFパワー履歴 (下位10%をノイズフロア推定に使用)
        self._nr_cut_levels = np.array([2500.0, 4000.0, 6500.0, 10000.0, 15000.0])
        self._nr_filters = [
            design_fir_kaiser(num_taps=65, cutoff_norm=float(c) / self.audio_rate, beta=6.5)
            for c in self._nr_cut_levels
        ]
        self.history_nr_lp = np.zeros(64, dtype=np.float32)
        self._nr_delay = (len(self._nr_filters[0]) - 1) // 2  # 線形位相FIRの群遅延

        # ===== サブバンドWiener NR (STFT 256pt / hop 128 / Hann 50%オーバーラップ) =====
        # 差信号を周波数ごとにWiener抑圧。低域(ノイズが少なく音が濃い)はステレオのまま、
        # ノイズに埋もれた高域のみを選択的に落とすため、単一ローパスより音場が広い。
        self._wf_n = 128
        self._wf_hop = 64
        self._wf_win = np.hanning(self._wf_n).astype(np.float32)
        cola = np.zeros(self._wf_n, dtype=np.float32)
        for m in range(-2, 3):
            s = m * self._wf_hop
            if s >= 0:
                cola[s:] += self._wf_win[: self._wf_n - s] ** 2
            else:
                cola[:s] += self._wf_win[-s:] ** 2
        self._wf_cola = np.maximum(cola, 1e-6)
        self._wf_in = np.zeros(0, dtype=np.float32)
        self._wf_out = np.zeros(self._wf_hop, dtype=np.float32)  # 初期プリフィル=固定遅延
        self._wf_ola = np.zeros(self._wf_n, dtype=np.float32)
        self._wf_p = None                  # 番組パワーの時間平滑
        self._wf_g = None
        self._wf_f2 = np.fft.rfftfreq(self._wf_n, 1.0 / self.audio_rate) ** 2
        hf_mask = (np.fft.rfftfreq(self._wf_n, 1.0 / self.audio_rate) >= 6000.0) & \
                  (np.fft.rfftfreq(self._wf_n, 1.0 / self.audio_rate) <= 15000.0)
        self._wf_hf_f2_mean = float(np.mean(self._wf_f2[hf_mask]) + 1e-12)
        # 1024点指標→STFTドメインへのパワースケール補正 (E|X|²=σ²Σw²)
        self._wf_scale = float(np.sum(self._wf_win ** 2) / np.sum(np.hanning(1024) ** 2))
        self._nr_floor_pow = 0.0           # ブロードバンド指標のノイズ床 (HF帯)
        self._nr_floor_bias = 2.0          # 下位タイル→平均ノイズへの補正
        self._nr_gmin = 0.05               # 最大抑圧 (-26dB)
        self.stereo_wiener_gain = 1.0
        self._nr_delay += self._wf_hop     # Wiener経路の遅延をmono側で補償
        self.history_mono_delay = np.zeros(self._nr_delay, dtype=np.float32)
        self._nr_window = np.hanning(1024).astype(np.float32)

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
        self.am_agc_level = 0.0  # AM搬送波レベルAGC状態

        # ===== AM同期検波 (キャリア再生PLL) =====
        # 選択性フェージング時のひずみを避けるため、包絡線検波ではなく
        # キャリアに同期した同相検波を使う。ロックできない時は包絡線へ自動復帰。
        self.am_sync_enabled = True
        self.am_sync_lock = 0.0
        # ===== SSB / CW =====
        self.bfo_offset_hz = 0.0     # BFO微調整 (SSB/CWのみ)
        self.ssb_agc_level = 0.0
        self._ssb_bp_phase = 0.0

        # ===== FMマルチパス検出 =====
        # 反射波(マルチパス)はFM波に振幅変動(PM→AM変換)を与える。IF信号の包絡線変動を
        # 検出し、強い時はステレオ/帯域を絞って耳障りな歪みを抑える。
        self.multipath_enabled = True
        self.multipath_amount = 0.0
        self.multipath_gain = 1.0
        self._mp_var = 0.0
        self.mp_lo = 0.10
        self.mp_hi = 0.35
        self.mp_depth = 0.7

        # ===== AM/SSB 適応音声帯域 =====
        # ヒスが多い時は音声帯域を狭めて了解度を上げる (自動トーンコントロール)
        self.voice_auto_bw = True
        self.voice_cut_hz = 4000.0
        self._vc_ratio_db = -40.0

        # ===== Sメーター (チャンネル内電力。絶対校正はないため目安) =====
        self.s_meter_dbfs = -90.0
        self.s_units = 0.0
        self._am_th = 0.0
        self._am_ig = 0.0
        self._am_ef = 0.0
        self._am_sync_mix = 0.0
        # ループ帯域 ~20Hz (搬送波のドリフトに追従しつつ変調側波帯は追わない)
        self.am_kp = 6.0e-4
        self.am_ki = 1.9e-7
        self._am_alpha = 2.0 * np.pi * 100.0 / self.if_rate

        # スペクトラム表示設定
        self.fft_size = 1024
        self.fft_window = np.hamming(self.fft_size).astype(np.float32)
        self.fft_smooth = None
        self.smooth_alpha = 0.25

    def set_offset_freq(self, offset_hz: float):
        self.offset_freq = offset_hz
        self.afc_offset_hz = 0.0
        self.nfm_afc_offset_hz = 0.0
        # 選局でAM同期PLLを初期化 (再ロック)
        self._am_th = 0.0
        self._am_ig = 0.0
        self._am_ef = 0.0
        self._am_sync_mix = 0.0
        # history_* はWFM/NFM/AM/SSBで共有しているため、選局・モード切替で
        # 全FIR履歴をゼロ化 (前局の残響・タップ数違いの過渡ポップを防止)。
        # AGCレベルは維持し音量ポンピングを避ける。
        for k, v in list(self.__dict__.items()):
            if k.startswith("history_") and isinstance(v, np.ndarray):
                v.fill(0)
        self._pll_theta = 0.0
        self._pll_integ = 0.0
        self._pll_ef = 0.0
        self.fm_last_sample = 0.0 + 0.0j
        self.nfm_last_sample = 0.0 + 0.0j
        self.reset_stereo_nr()
        self.am_sync_lock = 0.0

    def update_resampler_feedback(self, current_chunks: float, dt: float = 0.05):
        """オーディオバッファの残存チャンク数をリサンプラにフィードバック (クロック自動同期)"""
        self.resampler.update_feedback(current_chunks, dt=dt)
        self.resampler_r.update_feedback(current_chunks, dt=dt)

    def set_stereo_enabled(self, enabled: bool):
        """FMステレオMPXデコードの有効/無効 (モノラル強制)"""
        self.stereo_enabled = bool(enabled)
        if not self.stereo_enabled:
            self.is_stereo = False
            self.stereo_status = "MONO"
            self.stereo_blend = 0.0
            self._stereo_blend = 0.0
            self._last_cos2 = None

    def reset_stereo_nr(self):
        """選局・モード変更時にノイズ推定履歴をリセットして素早く追従させる"""
        self._nr_hist.clear()
        self._nr_primed = False
        self._wf_p = None
        self._wf_g = None
        self._nr_floor_pow = 0.0

    def set_stereo_nr(self, enabled: bool):
        """ステレオノイズリダクションの有効/無効 (無効時はフルステレオ固定)"""
        self.stereo_nr_enabled = bool(enabled)
        self.reset_stereo_nr()
        if not self.stereo_nr_enabled:
            self.stereo_nr_gain = 1.0
            self.stereo_cut_hz = self._nr_cut_max_hz
            self._nr_s = 0.0
            self._nr_s_w = 0.0
            self.stereo_wiener_gain = 1.0

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
        if mode in ("USB", "LSB", "CW"):
            # BFO: 復調音声を微調整 (正で音声が高くなる方向)
            effective_offset += self.bfo_offset_hz
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

    @staticmethod
    def _convolve_valid(x: np.ndarray, h: np.ndarray) -> np.ndarray:
        """valid畳み込み。ネイティブCコア(SSE2 4並列)があれば使用 (np.convolve比 5-10倍)"""
        if _NATIVE is not None and NATIVE_FIR:
            n_out = len(x) - len(h) + 1
            if n_out > 0:
                xa = np.ascontiguousarray(x, dtype=np.float32)
                ha = np.ascontiguousarray(h, dtype=np.float32)
                y = np.empty(n_out, dtype=np.float32)
                _NATIVE.sdr_fir_real(_fptr(xa), _fptr(ha), _fptr(y), n_out, len(ha))
                return y
        return np.convolve(x, h, mode="valid")

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
            r = self._convolve_valid(x_ext.real, fir_taps)
            i = self._convolve_valid(x_ext.imag, fir_taps)
            filtered = r + 1j * i
        else:
            filtered = self._convolve_valid(x_ext, fir_taps)

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

        # 0. マルチパス検出 (包絡線の変動 = PM→AM変換量)
        if self.multipath_enabled:
            env = np.abs(iq_if)
            var = float(np.std(env) / (np.mean(env) + 1e-12))
            dt_mp = len(iq_if) / self.if_rate
            self._mp_var += (1.0 - np.exp(-dt_mp / 0.5)) * (var - self._mp_var)
            x = float(np.clip((self._mp_var - self.mp_lo) / (self.mp_hi - self.mp_lo), 0.0, 1.0))
            s = x * x * (3.0 - 2.0 * x)
            tau = 0.5 if s > self.multipath_amount else 2.5
            self.multipath_amount += (1.0 - np.exp(-dt_mp / tau)) * (s - self.multipath_amount)
            self.multipath_gain = 1.0 - self.mp_depth * self.multipath_amount

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

        # 4. モノラル (L+R) を48kHzへデシメーション
        #    (19kHzパイロット・38kHz副搬送波はアンチエイリアスLPFで除去)
        #    ※同一 history を2回呼ぶと mono だけ履歴が二重送りになり、差信号との
        #      位相がずれて分離度・モノラル特性が劣化するため、呼び出しは1回のみ。
        mono = self.decimate_with_history(demod_scaled, self.fir_if_audio,
                                          self.audio_decim, "history_if_audio")

        # 5b. ステレオMPXデコード (19kHzパイロットPLL + 38kHz同期検波)
        self._update_stereo_pilot(demod_scaled)

        # 5c. RDS復調 (57kHz = 3θ, ステレオ状態と独立して常時動作)
        if self.rds_enabled and self._last_cos3 is not None:
            try:
                if self.rds is None:
                    import rds as rds_mod
                    self.rds = rds_mod.RdsDecoder(12000.0)
                carrier57 = self._last_cos3
                if abs(self.rds_phase_offset) > 1e-6 and self._last_sin3 is not None:
                    co = np.cos(self.rds_phase_offset)
                    si = np.sin(self.rds_phase_offset)
                    carrier57 = self._last_cos3 * co - self._last_sin3 * si
                rds_mix = demod_scaled * carrier57
                rds_base = self.decimate_with_history(rds_mix, self.fir_am_narrow, 24, "history_rds")
                self.rds.feed(rds_base)
                self.rds_ps = self.rds.ps_name
                self.rds_rt = self.rds.radio_text
                self.rds_pi = self.rds.pi
                self.rds_pty = self.rds.pty
                self.rds_groups = self.rds.groups
            except Exception:
                pass

        stereo_diff = None
        if self._last_cos2 is not None and self._stereo_blend > 0.02:
            carrier = self._last_cos2
            if abs(self.stereo_phase_offset) > 1e-6 and self._last_sin2 is not None:
                co = np.cos(self.stereo_phase_offset)
                si = np.sin(self.stereo_phase_offset)
                carrier = self._last_cos2 * co - self._last_sin2 * si
            lpr = demod_scaled * carrier
            diff_raw = self.decimate_with_history(lpr, self.fir_if_audio,
                                                  self.audio_decim, "history_lpr") * 2.0
            if len(diff_raw) == len(mono):
                # 常に実測 (診断・A/B用)。NR無効時はフィルタ素通し・フルステレオ固定
                self._update_stereo_nr(diff_raw, mono)
                if self.stereo_nr_enabled:
                    cut = self.stereo_cut_hz
                    blend = self._stereo_blend * self.stereo_nr_gain
                    blend *= self.multipath_gain
                    diff = self._diff_lowpass(diff_raw, cut)
                    diff = self._wiener_diff(diff)
                    stereo_diff = diff * blend
                    # 差信号FIRの群遅延を補償 (mono/diffの位相ズレによる分離度劣化を防止)
                    mono = self._delay_mono(mono)
                else:
                    # NR無効時はLPF/STFT往復をせず生差信号へブレンドのみ
                    # (15kHz LPF＋COLAリップルが可聴域を変える問題とCPU浪費を回避)。
                    # mono遅延履歴だけは更新し、再有効時の継ぎ目を無くす。
                    stereo_diff = (diff_raw
                                   * (self._stereo_blend * self.multipath_gain)).astype(np.float32)
                    self._delay_mono(mono)
                    self.stereo_wiener_gain = 1.0

        # 6. チャンネル別ポスト処理 (ディエンファシス・ハイカット・DCカット・シェルフ)
        if stereo_diff is not None:
            # 注: L/Rのスレッド並列化は実測で逆効果 (CPython GIL + C呼び出しが短く
            # オーバーヘッドが上回る)。逐次実行が最速。
            left = self._post_process_wfm(mono + stereo_diff, "_l")
            right = self._post_process_wfm(mono - stereo_diff, "_r")
            blend = self._stereo_blend * (self.stereo_nr_gain if self.stereo_nr_enabled else 1.0) \
                * self.multipath_gain
            self.is_stereo = blend > 0.5
            if blend > 0.85:
                self.stereo_status = "STEREO"
            elif blend > 0.02:
                self.stereo_status = "BLEND"
            else:
                self.stereo_status = "MONO"
            # WFM経路も±1.0へクリップ (AM/SSBと統一。過偏移・弱電界ノイズで
            # ±1.82超→後段int16変換でのラップ歪みを防止)
            return np.clip(np.stack([left, right], axis=1), -1.0, 1.0).astype(np.float32)

        self.is_stereo = False
        self.stereo_status = "MONO"
        return np.clip(self._post_process_wfm(mono, ""), -1.0, 1.0)

    def _delay_mono(self, mono: np.ndarray) -> np.ndarray:
        """monoを群遅延分だけ遅延させ、NRフィルタ通過後の差信号と時間整合を取る"""
        d = self._nr_delay
        if d <= 0 or len(mono) == 0:
            return mono
        y = np.concatenate((self.history_mono_delay, mono))[:len(mono)]
        if len(mono) >= d:
            self.history_mono_delay = mono[-d:].copy()
        else:
            self.history_mono_delay = np.concatenate((self.history_mono_delay, mono))[-d:]
        return y.astype(np.float32)

    def _update_stereo_nr(self, diff: np.ndarray, mono: np.ndarray):
        """(L-R)高域ノイズを(L+R)中域プログラムレベルで正規化してヒス量を推定する"""
        n = 1024
        if len(diff) < 128 or len(mono) < 128:
            return
        bin_hz = self.audio_rate / n

        def band_power(sig: np.ndarray, lo: float, hi: float) -> float:
            if len(sig) >= n:
                seg = sig[-n:].astype(np.float32)
            else:
                seg = np.zeros(n, dtype=np.float32)
                seg[-len(sig):] = sig
            seg = seg - float(np.mean(seg))
            power = np.abs(np.fft.rfft(seg * self._nr_window)) ** 2 + 1e-12
            i0 = max(1, int(lo / bin_hz))
            i1 = min(len(power) - 1, int(hi / bin_hz))
            if i1 <= i0:
                return 1e-12
            return float(np.mean(power[i0:i1 + 1]))

        hf = band_power(diff, 6000.0, 15000.0)
        mf = band_power(mono, 300.0, 3000.0)
        # ノイズフロア推定: ~10秒履歴の下位10%を使い、番組自身の高域成分ではなく
        # 定常的に存在するとヒス成分のみを検出する (明るい音楽での過剰なNRを防止)
        self._nr_hist.append(hf)
        # 下位10%タイル: percentile(ソート)よりpartition(O(n))で高速化
        if len(self._nr_hist) >= 4:
            arr = np.asarray(self._nr_hist, dtype=np.float32)
            k = int(0.10 * (len(arr) - 1))
            floor = float(np.partition(arr, k)[k])
        else:
            floor = hf
        self._nr_floor_pow = floor
        ratio_db = 10.0 * np.log10((floor + 1e-12) / (mf + 1e-12))

        dt = len(diff) / self.audio_rate
        if not self._nr_primed:
            # 初回は実測値で即座に初期化 (起動直後のランプを排除)
            self._nr_primed = True
            self.stereo_hiss_db = ratio_db
        else:
            a = 1.0 - np.exp(-dt / 0.35)
            self.stereo_hiss_db += a * (ratio_db - self.stereo_hiss_db)

        def amount(value_db: float, lo: float, hi: float) -> float:
            x = float(np.clip((value_db - lo) / (hi - lo), 0.0, 1.0))
            return x * x * (3.0 - 2.0 * x)  # smoothstep

        s_b = amount(self.stereo_hiss_db, self._nr_lo_db, self._nr_hi_db)
        s_w = amount(self.stereo_hiss_db, self._nr_wiener_lo_db, self._nr_wiener_hi_db)
        # 非対称スムージング: ノイズ増加時は速く、回復はゆっくり
        tau = 0.3 if s_b > self._nr_s else 1.5
        self._nr_s += (1.0 - np.exp(-dt / tau)) * (s_b - self._nr_s)
        tau_w = 0.25 if s_w > self._nr_s_w else 2.5
        self._nr_s_w += (1.0 - np.exp(-dt / tau_w)) * (s_w - self._nr_s_w)
        self.stereo_nr_gain = 1.0 - self._nr_s
        self.stereo_cut_hz = self._nr_cut_max_hz * (
            (self._nr_cut_min_hz / self._nr_cut_max_hz) ** self._nr_s
        )

    def _diff_lowpass(self, x: np.ndarray, cutoff_hz: float) -> np.ndarray:
        """差信号用の可変ローパス。同一長の線形位相FIRを2本クロスフェードし、
        群遅延を変えずに遮断周波数を連続変化させる (スイッチングノイズなし)。"""
        if len(x) == 0:
            return x
        req = len(self._nr_filters[0]) - 1
        hist = self.history_nr_lp
        if len(hist) != req:
            hist = np.zeros(req, dtype=np.float32)
        x_ext = np.concatenate((hist, x))
        self.history_nr_lp = x[-req:].copy() if len(x) >= req else x_ext[-req:].astype(np.float32)

        levels = self._nr_cut_levels
        cut = float(np.clip(cutoff_hz, levels[0], levels[-1]))
        # 非等間隔レベル間の区分線形補間 (等間隔仮定の線形posでは中間cutがずれる)
        i0 = int(np.clip(np.searchsorted(levels, cut, side="right") - 1, 0, len(levels) - 2))
        span = float(levels[i0 + 1] - levels[i0])
        w = float(np.clip((cut - levels[i0]) / (span if span > 0 else 1.0), 0.0, 1.0))
        y = np.convolve(x_ext, self._nr_filters[i0], mode="valid")
        if w > 1e-3:
            y2 = np.convolve(x_ext, self._nr_filters[i0 + 1], mode="valid")
            y = y * (1.0 - w) + y2 * w
        return y.astype(np.float32)

    def _wiener_diff(self, x: np.ndarray) -> np.ndarray:
        """差信号のサブバンドWiener抑圧 (STFT 256/hop 128, Hann 50%オーバーラップ)。

        周波数ごとに 信号/(信号+ノイズ) の最適重みを掛けるため、ノイズに埋もれた
        高域だけが落ち、SNRの良い低域のステレオ感はそのまま残る。
        入出力のサンプル数は厳密に一致させ、mono側は_nr_delayで遅延補償する。
        """
        n = len(x)
        if n == 0:
            return x
        buf = np.concatenate((self._wf_in, x))
        nfft = self._wf_n
        hop = self._wf_hop
        chunks = []
        if self._wf_p is None:
            self._wf_p = np.zeros(nfft // 2 + 1, dtype=np.float32)
        while len(buf) >= nfft:
            spec = np.fft.rfft(buf[:nfft] * self._wf_win)
            power = np.abs(spec) ** 2 + 1e-12

            # 番組パワーの時間平滑 (Wienerの分子・音楽性ノイズ抑制)
            self._wf_p = 0.5 * power + 0.5 * self._wf_p
            # ノイズモデル: FM三角ノイズ (∝f²) をブロードバンド指標のHF床でスケール。
            # ビン別最小値だと持続音をノイズと誤認するため、形状は物理モデルで与える。
            c_noise = (self._nr_floor_pow * self._nr_floor_bias * self._wf_scale) / self._wf_hf_f2_mean
            g_w = np.maximum(1.0 - (c_noise * self._wf_f2) / (self._wf_p + 1e-12),
                             self._nr_gmin).astype(np.float32)
            g_w[:3] = 1.0  # DC〜低域は保護
            if self._wf_g is None or len(self._wf_g) != len(g_w):
                self._wf_g = g_w
            else:
                a = np.where(g_w < self._wf_g, 0.7, 0.1)
                self._wf_g = self._wf_g + a * (g_w - self._wf_g)
            sw = self._nr_s_w if self.stereo_nr_enabled else 0.0
            g_mix = 1.0 - sw * (1.0 - self._wf_g)
            self.stereo_wiener_gain = float(np.mean(g_mix))

            y = np.fft.irfft(spec * g_mix, n=nfft)
            self._wf_ola[:nfft] += y * self._wf_win
            chunks.append((self._wf_ola[:hop] / self._wf_cola[:hop]).copy())
            self._wf_ola = np.concatenate(
                (self._wf_ola[hop:], np.zeros(hop, dtype=np.float32)))
            buf = buf[hop:]
        self._wf_in = buf

        if chunks:
            self._wf_out = np.concatenate(
                (self._wf_out, np.concatenate(chunks).astype(np.float32)))
        if len(self._wf_out) >= n:
            y = self._wf_out[:n].copy()
            self._wf_out = self._wf_out[n:]
        else:
            y = np.concatenate((self._wf_out, np.zeros(n - len(self._wf_out), dtype=np.float32)))
            self._wf_out = np.zeros(0, dtype=np.float32)
        if len(self._wf_out) > 8 * n:
            self._wf_out = self._wf_out[-2 * n:]
        return y

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
        """19kHzパイロットPLLを更新し、ステレオブレンド係数とRDS用57kHz搬送波を生成する"""
        self._last_cos2 = None
        self._last_cos3 = None
        if not (self.stereo_enabled or self.rds_enabled) or _NATIVE is None or len(mpx) < 64:
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
            mpx_rms_pre = float(np.sqrt(np.mean(np.asarray(mpx, dtype=np.float64) ** 2)) + 1e-12)
            if pilot_rms < 1e-4 or pilot_rms < 0.015 * mpx_rms_pre:
                # パイロット不在ゲート: 19kHz帯が無音・微小のまま正規化PLLへ渡すと
                # 入力が1e11級に膨張→PLL発散→C側の位相正規化が爆発し復帰不能
                # (モノラル局・無信号でDSPスレッドがハングする)。ここで打ち切り、
                # ブレンドを減衰させてモノラルへ落とす。弱電界ステレオの瞬断は
                # 次ブロックで回復するため実害なし。
                self._stereo_blend *= 0.9
                self.stereo_blend = self._stereo_blend
                self.stereo_pilot_lock = 0.0
                self.stereo_pilot_ratio = 0.0
                return
            sig = np.ascontiguousarray(np.asarray(mpx, dtype=np.float32) / pilot_rms, dtype=np.float32)
            cos2 = np.empty(n, dtype=np.float32)
            sin2 = np.empty(n, dtype=np.float32)
            quality = ctypes.c_float(0.0)
            th = ctypes.c_double(self._pll_theta)
            ig = ctypes.c_double(self._pll_integ)
            ef = ctypes.c_double(self._pll_ef)
            cos3 = sin3 = None
            if NATIVE_PLL3:
                cos3 = np.empty(n, dtype=np.float32)
                sin3 = np.empty(n, dtype=np.float32)
                _NATIVE.sdr_stereo_pll3(_fptr(sig), n, ctypes.byref(th), self._pll_w0,
                                        self._pll_kp, self._pll_ki, ctypes.byref(ig),
                                        ctypes.byref(ef), self._pll_alpha,
                                        _fptr(cos2), _fptr(sin2),
                                        _fptr(cos3), _fptr(sin3), ctypes.byref(quality))
            else:
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
            self.stereo_pilot_lock = lock
            self.stereo_pilot_ratio = ratio

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
            if cos3 is not None:
                self._last_cos3 = cos3
                self._last_sin3 = sin3
        except Exception:
            self._last_cos2 = None
            self._last_sin2 = None
            self._last_cos3 = None
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

        # 同期検波と包絡線検波をロック状態に応じて混合 (未ロック時は包絡線へ自動復帰)
        sig = env
        if self.am_sync_enabled:
            coherent = self._am_sync_detect(iq_if)
            w = self._am_sync_mix
            if w > 0.01:
                sig = w * coherent + (1.0 - w) * env

        # 搬送波レベルAGC: 信号強度やダイレクトサンプリングの低入力でも一定音量にする
        # (時定数 ~0.2sアタック / ~1sリリース。無信号時の過剰増幅は3000倍で制限)
        # 無信号フロア: レベル極小でAGC=0張り付き→gain3000→AMは-0.6のDC定数出力や
        # ノイズ爆音になるため、フロア以下では出力を滑らかにミュートする。
        level = float(np.mean(np.abs(sig)))
        if self.am_agc_level <= 0.0:
            self.am_agc_level = max(level, 2e-4)
        else:
            alpha = 0.25 if level > self.am_agc_level else 0.05
            self.am_agc_level += alpha * (level - self.am_agc_level)
        gain = min(1.0 / (self.am_agc_level + 1e-9), 3000.0)
        fade = min(1.0, level / 2e-4)
        audio_raw = np.clip((sig * gain - 1.0) * 0.6, -1.0, 1.0) * fade
        if self.cognitive_enabled:
            fir_final = self._get_dynamic_filter("audio", min(self.applied_cutoff_hz, 8000.0))
        elif self.filter_mode == "wide":
            fir_final = self.fir_audio_wide
        else:
            fir_final = self.fir_am_audio  # AMは4kHz (8.5kHzでは短波のヒスが酷い)
        audio = self.decimate_with_history(audio_raw, self.fir_if_audio, self.audio_decim, "history_if_audio")

        audio = self.decimate_with_history(audio, fir_final, 1, "history_final")
        audio = self._apply_dc_highpass(audio)
        audio = self._voice_bandwidth(audio, 4000.0, 2500.0)
        return audio.astype(np.float32)

    def _voice_bandwidth(self, audio: np.ndarray, f_max: float = 4000.0,
                         f_min: float = 2500.0) -> np.ndarray:
        """AM/SSB適応帯域: 3.2-4.5kHzのヒス量に応じて音声帯域を連続的に狭める"""
        if not self.voice_auto_bw or len(audio) < 256:
            return audio
        seg = audio[-1024:].astype(np.float32)
        seg = seg - float(np.mean(seg))
        power = np.abs(np.fft.rfft(seg * self._nr_window)) ** 2 + 1e-12
        bin_hz = self.audio_rate / len(seg)

        def band(lo: float, hi: float) -> float:
            i0 = max(1, int(lo / bin_hz))
            i1 = min(len(power) - 1, int(hi / bin_hz))
            return float(np.mean(power[i0:i1 + 1])) if i1 > i0 else 1e-12

        ratio_db = 10.0 * np.log10(band(3200.0, 4500.0) / band(300.0, 3000.0))
        dt = len(audio) / self.audio_rate
        self._vc_ratio_db += (1.0 - np.exp(-dt / 0.5)) * (ratio_db - self._vc_ratio_db)
        x = float(np.clip((self._vc_ratio_db + 32.0) / 20.0, 0.0, 1.0))
        s = x * x * (3.0 - 2.0 * x)
        self.voice_cut_hz = f_max * (f_min / f_max) ** s
        fir = self._get_dynamic_filter("audio", self.voice_cut_hz)
        return self.decimate_with_history(audio, fir, 1, "history_am_audio")

    def _am_sync_detect(self, iq_if: np.ndarray) -> np.ndarray:
        """キャリア再生PLLによるAM同期検波 (Cコア) とロック度の平滑化"""
        if len(iq_if) == 0:
            return np.zeros(0, dtype=np.float32)
        if _NATIVE is not None and NATIVE_AM_SYNC:
            # PLLゲインは振幅に比例するため、先に正規化 (AGC) して感度を一定化。
            # これが無いとダイレクトサンプリングの微小入力 (1e-4) でPLLが実質停止する。
            scale = float(np.mean(np.abs(iq_if))) + 1e-12
            work = np.ascontiguousarray(iq_if / scale, dtype=np.complex64)
            out = np.empty(len(work), dtype=np.float32)
            th = ctypes.c_double(self._am_th)
            ig = ctypes.c_double(self._am_ig)
            ef = ctypes.c_double(self._am_ef)
            lock = ctypes.c_float(0.0)
            _NATIVE.sdr_am_sync(_fptr(work), _fptr(out), len(work),
                                ctypes.byref(th), ctypes.byref(ig), ctypes.byref(ef),
                                self.am_kp, self.am_ki, self._am_alpha, ctypes.byref(lock))
            self._am_th = th.value
            self._am_ig = ig.value
            self._am_ef = ef.value
            out *= scale  # 正規化を戻し包絡線とスケールを一致させる
            lock_v = float(lock.value)
        else:
            # フォールバック: ブロック平均位相によるコヒーレント検波
            m = complex(np.mean(iq_if))
            theta = float(np.angle(m))
            out = np.real(iq_if * np.exp(-1j * theta)).astype(np.float32)
            lock_v = float(abs(m) / (np.mean(np.abs(iq_if)) + 1e-12))

        self.am_sync_lock = lock_v
        x = float(np.clip((lock_v - 0.35) / 0.30, 0.0, 1.0))
        target = x * x * (3.0 - 2.0 * x)
        dt = len(iq_if) / self.if_rate
        tau = 0.5 if target > self._am_sync_mix else 2.5
        self._am_sync_mix += (1.0 - np.exp(-dt / tau)) * (target - self._am_sync_mix)
        return out

    def demodulate_ssb(self, iq_48: np.ndarray, mode: str) -> np.ndarray:
        """SSB/CW復調 (48kHz複素IF)。
        シフト→実LPF→逆シフトで非対称バンドパスを構成し、選択側波帯のみを取り出す。
        USB=+1.5kHz帯, LSB=-1.5kHz帯, CW=+650Hz±350Hz (BFOで微調整可能)。"""
        if len(iq_48) == 0:
            return np.zeros(0, dtype=np.float32)

        if mode == "USB":
            center, taps, attr = 1500.0, self.fir_ssb_lp, "history_ssb_lp"
        elif mode == "LSB":
            center, taps, attr = -1500.0, self.fir_ssb_lp, "history_ssb_lp"
        else:  # CW
            center, taps, attr = 650.0, self.fir_cw_lp, "history_cw_lp"

        n = len(iq_48)
        omega = 2.0 * np.pi * center / self.audio_rate
        ph = self._ssb_bp_phase + omega * np.arange(n, dtype=np.float64)
        self._ssb_bp_phase = float((ph[-1] + omega) % (2.0 * np.pi))
        shifted = (iq_48 * np.exp(-1j * ph)).astype(np.complex64)
        lp = self.decimate_with_history(shifted, taps, 1, attr)
        # 再シフト位相はFIR群遅延Dサンプル分だけ遅らせる (線形位相FIRの遅延補償。
        # 未補償だとUSB 1500Hz×D200で12.5π→0.5πの定数回転が残る。可聴差は
        # 微小だがコヒーレント復調として正しい位相に戻す)
        dly = (len(taps) - 1) // 2
        audio = np.real(lp * np.exp(1j * (ph - omega * dly))).astype(np.float32)

        # AGC (SSBは搬送波が無いため平均振幅で正規化。無信号時の過剰増幅は3000倍で制限)
        # 無信号フロア (AM側と同型): ノイズ2400倍の爆音化を防ぐため滑らかにミュート。
        level = float(np.mean(np.abs(audio)))
        if self.ssb_agc_level <= 0.0:
            self.ssb_agc_level = max(level, 2e-4)
        else:
            alpha = 0.25 if level > self.ssb_agc_level else 0.05
            self.ssb_agc_level += alpha * (level - self.ssb_agc_level)
        gain = min(1.0 / (self.ssb_agc_level + 1e-9), 3000.0)
        fade = min(1.0, level / 2e-4)
        audio = np.clip(audio * gain * 0.8, -1.0, 1.0) * fade

        # 音声帯域整形 (3kHz LPF + 300Hz HP)
        if self.cognitive_enabled:
            fir_final = self._get_dynamic_filter("audio", min(self.applied_cutoff_hz, 3500.0))
        else:
            fir_final = self.fir_audio_narrow if self.filter_mode == "narrow" else self.fir_nfm_audio
        audio = self.decimate_with_history(audio, fir_final, 1, "history_ssb_audio")
        audio = self._apply_voice_highpass(audio)
        audio = self._voice_bandwidth(audio, 3000.0, 2200.0)
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
        elif mode in ("USB", "LSB", "CW"):
            # SSB / CW (HF用): 複素非対称バンドパスで側波帯を選択
            iq_if = self.decimate_with_history(iq_shifted, self.fir_am_narrow, self.if_decim, "history_ssb")
            iq_48 = self.decimate_with_history(iq_if, self.fir_if_audio, self.audio_decim, "history_ssb2")
            audio = self.demodulate_ssb(iq_48, mode)
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

        # Sメーター: チャンネル通過後の電力を平滑化 (S9=-30dBFS, 6dB/S-unitの目安)
        try:
            p_ch = float(np.mean(np.abs(iq_if) ** 2)) + 1e-18
            dbfs = 10.0 * np.log10(p_ch)
            dt_sm = len(iq_if) / self.if_rate
            self.s_meter_dbfs += (1.0 - np.exp(-dt_sm / 0.3)) * (dbfs - self.s_meter_dbfs)
            self.s_units = 9.0 + (self.s_meter_dbfs + 30.0) / 6.0
        except Exception:
            pass

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
