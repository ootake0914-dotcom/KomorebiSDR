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
import math
import threading
import time
from collections import deque, OrderedDict
import numpy as np

from adaptive_dsp import (
    AdaptiveIqCorrector,
    CognitiveSpeechMusicTracker,
    UltrasonicSquelchTracker,
    QuadratureMpxCanceller,
    DeepSpaceEkfDemodulator,
    KalmanPilotTracker,
    HolographicAudioEnhancer,
    RiemannianTopologicalDemodulator,
    SuperSpatialBssStereoSeparator,
    RmtHankelDenoiser,
    MonoNoiseSuppressor,
    DigitalSelfInterferenceCanceller,
)
from audiophile_dsp import (
    ActiveDcServo,
    MinimumPhaseApodizer,
    TpdfDitherNoiseShaper,
)


# ================================================================
# ネイティブCコア (sdr_core.dll) ロード
# 正準の実装は dsp_native.py。本モジュールは後方互換のため同名を再エクスポートする。
# (NATIVE_* フラグはimport時に確定する定数のためスナップショットで問題ない)
# ================================================================
from dsp_native import (
    _load_native_core,
    _NATIVE,
    NATIVE_CORE_ENABLED,
    NATIVE_AM_SYNC,
    NATIVE_PLL3,
    NATIVE_FIR,
    NATIVE_POLY,
    NATIVE_PLLFM,
    NATIVE_CMA,
    enable_fast_fpu,
    _fptr,
)



# FIR設計・クリック抑圧・ディエンファシス係数は dsp_filters.py が正準。
# 後方互換のため同名を再エクスポートする。
from dsp_filters import (
    design_fir_kaiser,
    design_fir_highpass,
    design_fir_lowpass,
    suppress_click_transients,
    _deemph_sections,
)


# 適応ドリフトリサンプラの正準は dsp_resampler.py。後方互換のため再エクスポート。
from dsp_resampler import AdaptiveDriftResampler


# 黒魔法三点セット (弱電界検出補助)。既定ではdsp経路に接続しない。
# import失敗時はNoneとなり、配線側のhasattr/Noneガードで既存経路のみ動作する。
try:
    from cyclostationary_detector import CyclostationaryPilotDetector
except ImportError:
    CyclostationaryPilotDetector = None
try:
    from rmt_denoiser import SafeRmtDenoiser
except ImportError:
    SafeRmtDenoiser = None
try:
    from stochastic_resonance import StochasticResonanceDetector
except ImportError:
    StochasticResonanceDetector = None
try:
    from black_magic import BlackMagicController
except ImportError:
    BlackMagicController = None
try:
    from adaptive_notch import AdaptiveNotchCanceller
except ImportError:
    AdaptiveNotchCanceller = None


class SdrDspPipeline:
    """超低ノイズ・超高音質 SDR 信号処理パイプライン"""

    def __init__(self, sample_rate: int = 1152000, audio_rate: int = 48000):
        self.rf_rate = sample_rate
        self.audio_rate = audio_rate
        # 選局リセットとDSP処理の同時実行によるtorn history防止用
        self._state_lock = threading.Lock()

        self.if_decim = 4
        self.if_rate = self.rf_rate // self.if_decim  # 288 kHz
        self.audio_decim = self.if_rate // self.audio_rate  # 6
        self.total_decim = self.if_decim * self.audio_decim  # 24

        # 端数IQサンプル持ち越し用バッファ (時間軸断絶・クリック音の完全根絶)
        self.raw_leftover = np.empty(0, dtype=np.uint8)

        self.offset_freq = 0.0
        self.mixer_phase = 0.0
        self._tune_monotonic = time.monotonic()

        # グラム・シュミット直交化によるリアルタイム適応IQインバランス補正器 (鏡像ゴースト自動消去)
        self.iq_corrector = AdaptiveIqCorrector(sample_rate=self.rf_rate, time_constant_sec=3.0)

        # 全二重通信理論 (IBFD) デジタル自己干渉消去器 (SIC: PC直挿し時のクロック・スイッチングビート消去)
        # IF段 (288kHz) でスプリアスを自動検出し、NLMS直交基底追従で逆位相ノッチ消去。
        # 延長ケーブル使用時などスプリアスが存在しない場合は完全素通し (相関0.999+) となり副作用ゼロ。
        self.sic_canceller = DigitalSelfInterferenceCanceller(
            sample_rate=self.if_rate,
            mu=0.08,
            max_tones=4,
        )
        self.sic_enabled = True
        self._sic_detect_counter = 0

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
        # カットオフ 15kHz, 19kHzで -60dB以上の超急峻減衰 (257タップ。
        # 97タップでは19kHzで-25.8dBしかなく超音波漏洩していた)
        cutoff_if_audio = 15000.0 / self.if_rate
        self.fir_if_audio = design_fir_kaiser(num_taps=257, cutoff_norm=cutoff_if_audio, beta=7.0)

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
        self._if_snr_db = 10.0  # WFM復調前の局所チャンネルSNR推定（IFモーフィング用）
        self._fir_cache = OrderedDict()

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
        # PLL-FM復調状態 (fn=25kHz, ζ=1.0 の実測勝ち値。w=2πfn/fsで正規化設計)
        _w = 2.0 * np.pi * 25000.0 / 288000.0
        _den = 1.0 + _w + 0.25 * _w * _w
        self._fm_pll_kp = 2.0 * _w / _den
        self._fm_pll_ki = _w * _w / _den
        self._fm_pll_state = np.zeros(2, dtype=np.float64)
        self.fm_pll_enabled = False  # ワイドFMの過変調歪み・脱調防止のため通常は差分検波を標準使用
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

        # ディエンファシス (地域設定: 日本/欧州=50μs, 米国/韓国=75μs)
        # 1次双一次では高域ワープ歪み (15kHzで-3.55dB) が避けられないため、
        # 実測フィットした2縦続1次IIRでアナログ特性に±0.03dBで一致させる。
        # 各段は既存ネイティブ1次IIR (b0,b1,minus_a1) そのまま実行できる。
        self.deemph_tau_us = 50.0
        self.deemph_sections = _deemph_sections(50.0)
        self.deemph_x1 = self.deemph_y1 = 0.0
        self.deemph2_x1 = self.deemph2_y1 = 0.0
        # ネイティブCコア用フィルタ状態 (x1, y1) ×2段
        self._deemph_state = np.zeros(2, dtype=np.float32)
        self._deemph2_state = np.zeros(2, dtype=np.float32)
        self._dc_hp_state = np.zeros(2, dtype=np.float32)
        self._voice_hp_state = np.zeros(2, dtype=np.float32)
        self._deemph_state_l = np.zeros(2, dtype=np.float32)
        self._deemph_state_r = np.zeros(2, dtype=np.float32)
        self._deemph2_state_l = np.zeros(2, dtype=np.float32)
        self._deemph2_state_r = np.zeros(2, dtype=np.float32)
        self._dc_hp_state_l = np.zeros(2, dtype=np.float32)
        self._dc_hp_state_r = np.zeros(2, dtype=np.float32)
        self.deemph_x1_l = self.deemph_y1_l = 0.0
        self.deemph_x1_r = self.deemph_y1_r = 0.0
        self.deemph2_x1_l = self.deemph2_y1_l = 0.0
        self.deemph2_x1_r = self.deemph2_y1_r = 0.0
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
        self.history_lpr_q = np.zeros(len(self.fir_if_audio) - 1, dtype=np.float32)
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
        # 高級FMチューナー基準 超低ジッターPLL設計 (fn=16Hz, ζ=0.85, ループフィルタ遮断 20Hz)
        # 従来の過大帯域(205Hz)による低音変調漏れ・位相揺らぎ・定位のあるノイズを根絶
        _fn_pll = 16.0
        _wn_pll = 2.0 * np.pi * _fn_pll
        self._pll_kp = float(2.0 * 0.85 * _wn_pll / self.if_rate)
        self._pll_ki = float((_wn_pll / self.if_rate) ** 2)
        self._pll_alpha = float(2.0 * np.pi * 20.0 / self.if_rate)
        self._pll_ef = 0.0
        self._last_cos2 = None
        self._last_sin2 = None
        self._last_cos3 = None
        self._last_sin3 = None

        # 適応カルマン・パイロット搬送波トラッカー (19kHz追従)
        self.pilot_tracker = KalmanPilotTracker(sample_rate=self.if_rate)

        # 高域高調波補完エキサイター (Harmonic Exciter: 15kHz以上の高域倍音付加)
        # NOTE: 実機実測(ラッキーFM 94.6MHz 強電界)で無音時の12-15kHzを+18.7dB
        # 持ち上げ、静かな場面に合成ヒスが乗ることを確認したため既定OFF。
        # 再有効化は .enabled=True (弱局で空気感を出したい場合のみ推奨)。
        self.holographic_enhancer = HolographicAudioEnhancer(sample_rate=self.audio_rate, air_gain=0.08)
        self.holographic_enhancer.enabled = False

        # 位相スリップ抑制型FM復調器 (特異点クリック防止)
        self.riemann_demodulator = RiemannianTopologicalDemodulator(sample_rate=self.if_rate)

        # 独立成分分析ステレオ復調器 (BSS / FastICA によるヒス低減)
        self.bss_separator = SuperSpatialBssStereoSeparator(sample_rate=self.audio_rate)

        # ハンケル行列SVD部分空間ノイズフィルター (SVD特異値しきい値処理)
        # NOTE: 実機実測で番組の12-15kHzを+7.1dB変形 (入出力相関0.9916=非透明) し、
        # 弱局でのノイズ低減効果も確認できなかったため既定OFF (CPUも節約)。
        self.rmt_denoiser = RmtHankelDenoiser(sample_rate=self.audio_rate, embed_dim=24)
        self.rmt_denoiser.enabled = False

        # 単一ch スペクトル抑圧NR (帯域内ノイズの最小統計Wiener抑圧。
        # 弱電界FMでハイカットでは消せない番組帯ノイズを低減。クリーン時は透明)
        self.mono_nr = MonoNoiseSuppressor(sample_rate=self.audio_rate)
        self.mono_nr_enabled = True

        # ===== 黒魔法三点セット (弱電界検出補助。既定は全て無効) =====
        # master=self.black_magic_enabled がFalseの間は一切動作せず、
        # 既存経路とビット同一の出力を保つ (tests/test_black_magic.pyで検証)。
        # 各インスタンスは遅延生成 (有効化時のみ) し、失敗時はNoneのまま
        # 既存経路へフォールバックする。
        self.black_magic_enabled = False
        self.bm_cyclo_enabled = False
        self.bm_rmt_enabled = False
        self.bm_sr_enabled = False
        self.bm_notch_enabled = False
        self.bm_sq_assist_enabled = False
        self.bm_sq_open_conf = 0.75
        self.bm_sq_close_conf = 0.55
        self.bm_sq_close_smeter_db = -25.0
        self.bm_sq_open_smeter_db = -40.0
        self._bm_sq_open = True  # 起動時は開 (いきなりミュートしない)
        self._bm_sq_gain = 1.0
        self.cyclo_detector = None
        self.bm_rmt = None
        self.bm_sr = None
        self.bm_notch = None
        self.bm_controller = None
        self.bm_cyclo_min_conf = 0.55
        self.bm_cyclo_confidence = 0.0
        self.bm_sr_confidence = 0.0
        self.bm_rmt_cap = 0.65
        # flutter検出用lock履歴 (直近32ブロック) と判定閾値
        self._bm_lock_hist = deque(maxlen=32)
        self.bm_flutter_std = 0.15
        self._bm_params = None
        self._bm_last_rmt_info = None
        self._bm_sr_tw = None  # SR用19kHz単一ビンtwiddleキャッシュ

        # ===== RDS (57kHz) =====
        self.rds_enabled = True
        self.rds = None            # 遅延生成 (rds.RdsDecoder)
        self.rds_ps = ""
        self.rds_rt = ""
        self.rds_pi = 0
        self.rds_pty = None
        self.rds_groups = 0
        # BS.450/EN 50067準拠キャリア生成のため追加回転は不要 (0.0)
        self.rds_phase_offset = 0.0
        self._stereo_blend = 0.0
        # パイロット瞬断用フライホイール: ロック喪失直後はブレンドを凍結し、
        # 短い発作 (1.4秒/25ブロックまで) ではステレオ像を維持する
        self._pilot_hold_max = 25
        self._pilot_hold_n = 0
        # CコアのPLL 1サンプル進みが解消されたため、副搬送波オフセットは 0.0
        self.stereo_phase_offset = 0.0
        # 直交復調の残差 (位相誤差の符号付き観測用。MPXキャンセラ経路でのみ有効)
        self._last_diff_q = None
        # 38kHz再生位相オートトリム (ドングル個体差・温度ドリフトによる分離度劣化を
        # L-R電力最大化サーボで吸収。stereo_phase_offsetをアクチュエータに使う)
        self.stereo_trim_enabled = True
        self._trim_dir = 1.0
        self._trim_step_rad = float(np.deg2rad(0.5))
        self._trim_max_rad = float(np.deg2rad(15.0))
        self._trim_block = 0
        self._trim_m_smooth = 0.0
        self._trim_prev_m = 0.0
        self._trim_primed = False
        self._trim_err_ema = 0.0
        # ステレオ2ch完全同期 適応ドリフトリサンプラ (後方互換参照)
        self.resampler_r = self.resampler

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
        # 固定高域ブレンド上限: FMステレオ副搬送波(38kHz DSB)の三角雑音は
        # 高域ほど大きく、強局でも12-15kHzで番組と同程度まで残る (実測: ラッキーFM
        # 94.6MHz 強電界で S高域ノイズが番組-5dB)。ヒス指標に依らず常時S側を
        # 13kHzで緩く減衰させる (カーラジオ標準の高域ブレンド。低域のステレオ感は不変)。
        self._nr_cut_fixed_hz = 13000.0
        # ブレンド量 (極端に弱い局のみモノラル化。通常はWienerが周波数別に処理)
        self._nr_lo_db = -18.0
        self._nr_hi_db = -4.0
        # Wiener適用量 (これより上のノイズで段階的にサブバンド抑圧)
        # 実測: 強局(ラッキーFM 94.6)でも副搬送波ヒスは-36dBあり、旧-40/-18では
        # 適用度0.1しか立たず12-15kHzのヒスが残った。サブバンドWienerは
        # 知覚マスキングゲート内蔵で番組高域を保護するため、適用域を下げて
        # 「聞こえるヒス」を抑える (ブレンド側しきい値は据え置き=高域ブレンド不要)。
        self._nr_wiener_lo_db = -46.0
        self._nr_wiener_hi_db = -26.0
        self._nr_primed = False
        self._nr_s_w = 0.0                # 平滑化されたWiener適用度 (0=off, 1=full)
        self._nr_s = 0.0                  # 平滑化されたノイズ度 (0=クリーン, 1=ノイズ)
        # モノラル番組検出 (M-S相関): 真のステレオでは直交するため、相関が高い=
        # S成分が分離漏れ+ノイズ。モノラル番組ではS側を積極抑圧してヒスを消す
        self._nr_mono_rho = 0.0
        self._nr_mono_w = 0.0
        self._nr_mono_primed = False
        self._nr_sw_eff = 0.0             # 有効Wiener適用度 (モノラル判定反映)
        self._nr_cut_eff = 15000.0        # 有効S側カットオフ (モノラル判定反映)
        self._nr_hist = deque(maxlen=100)  # 差分HFパワー履歴 (下位10%をノイズフロア推定に使用)
        self._nr_mf_smooth = 0.0  # 番組パワー平滑値 (未初期化=0で初回に即時セット)
        self._nr_cut_levels = np.array([2500.0, 4000.0, 6500.0, 10000.0, 15000.0])
        self._nr_filters = [
            design_fir_kaiser(num_taps=65, cutoff_norm=float(c) / self.audio_rate, beta=6.5)
            for c in self._nr_cut_levels
        ]
        self.history_nr_lp = np.zeros(64, dtype=np.float32)
        self._nr_delay = (len(self._nr_filters[0]) - 1) // 2  # 線形位相FIRの群遅延

        # ===== サブバンドWiener NR (STFT 128pt / hop 64 / 平方根Hann(Sine窓) 50%オーバーラップ) =====
        # 差信号を周波数ごとにWiener抑圧。低域(ノイズが少なく音が濃い)はステレオのまま、
        # ノイズに埋もれた高域のみを選択的に落とすため、単一ローパスより音場が広い。
        # 平方根Hann窓 (Sine窓: sin(pi*(n+0.5)/N)) を分析・合成の両面で適用することで、
        # 50% OLAの二乗和が sin² + cos² ≡ 1.0 となり、再構成時の振幅変調リップル(750Hzとその倍音)が
        # 数学的に完全ゼロ(0.000000dB)に消滅する。
        self._wf_n = 128
        self._wf_hop = 64
        self._wf_win = np.sin(np.pi * (np.arange(self._wf_n) + 0.5) / self._wf_n).astype(np.float32)
        self._wf_cola = np.ones(self._wf_n, dtype=np.float32)
        self._wf_in = np.zeros(0, dtype=np.float32)
        self._wf_out = np.zeros(self._wf_hop, dtype=np.float32)  # 初期プリフィル=固定遅延
        # STFTがゼロ埋めで生じた追加遅延。mono側を同量遅らせて時間整合を保つ
        # (ブロック長がホップの倍数でない場合に分離度が崩壊するのを防ぐ)
        self._wf_extra_delay = 0
        self._wf_ola = np.zeros(self._wf_n, dtype=np.float32)
        self._wf_p = None                  # 番組パワーの時間平滑
        self._wf_g = None
        self._wf_f2 = np.fft.rfftfreq(self._wf_n, 1.0 / self.audio_rate) ** 2
        # 知覚マスキング行列 (Bark拡散・Schroeder): T = P @ S でビン別マスキング閾値。
        # マスクされるノイズは抑圧不要 (g=1) とし、音楽性ノイズを設計上出さない。
        _bf = np.fft.rfftfreq(self._wf_n, 1.0 / self.audio_rate)
        _bk = 13.0 * np.arctan(0.76 * _bf / 1000.0) + 3.5 * np.arctan((_bf / 7500.0) ** 2)
        _dz = _bk[:, None] - _bk[None, :]
        _sp = 15.81 + 7.5 * (_dz + 0.474) - 17.5 * np.sqrt(1.0 + (_dz + 0.474) ** 2)
        self._wf_spread = (10.0 ** (_sp / 10.0)).astype(np.float32)
        self._wf_mask_offset = 0.1  # マスキング閾値オフセット (同時マスキング-10dB相当)
        hf_mask = (np.fft.rfftfreq(self._wf_n, 1.0 / self.audio_rate) >= 6000.0) & \
                  (np.fft.rfftfreq(self._wf_n, 1.0 / self.audio_rate) <= 15000.0)
        self._wf_hf_f2_mean = float(np.mean(self._wf_f2[hf_mask]) + 1e-12)
        # 1024点指標→STFTドメインへのパワースケール補正 (E|X|²=σ²Σw²)
        self._wf_scale = float(np.sum(self._wf_win ** 2) / np.sum(np.hanning(1024) ** 2))
        self._nr_floor_pow = 0.0           # ブロードバンド指標のノイズ床 (HF帯)
        self._nr_floor_bias = 2.3          # 下位タイル→平均ノイズへの補正 (Sine窓特性に最適化)
        self._nr_gmin = 0.05               # 最大抑圧 (-26dB)
        self.stereo_wiener_gain = 1.0
        self._nr_delay += self._wf_hop     # Wiener経路の遅延をmono側で補償
        self.history_mono_delay = np.zeros(self._nr_delay, dtype=np.float32)
        self._nr_window = np.hanning(1024).astype(np.float32)
        # 周波数依存ブレンド用クロスオーバー状態 (1次相補: lo + hi = diff で完全再構成。
        # blend=1時は_lo/_hiとも1.0で旧スカラー動作とビット一致)
        self.freq_blend_enabled = True
        self.freq_blend_xo_hz = 3500.0
        self._blend_xo_y1 = 0.0

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
        self._am_agc_hang = 0  # AGCハングタイマ (残ブロック数)
        self.impulse_blanker_enabled = True  # AMインパルスノイズブランカ
        # 局間音量レベリング用スローAGC (全モード共通・L/R連動で音像保存。
        # 番組の緩急 (バース/サビ) には追従させず、局替わり等の持続的な
        # レベル差だけを均す: 時定数は秒〜十秒オーダー、範囲は±6dBに制限。
        # 短い時定数・広い範囲にすると番組ダイナミクスを潰してパンピング
        # (サビが小さく・出頭が爆音に) するため厳禁)
        self.slow_agc_enabled = True
        self.slow_agc_target = 0.10      # 目標RMS (-20dBFS)
        self.slow_agc_min = 0.5          # -6dB (過大局の絞り)
        self.slow_agc_max = 2.0          # +6dB (微弱局の持ち上げ上限)
        self.slow_agc_attack = 2.0       # 絞り方向の時定数 (秒)
        self.slow_agc_release = 10.0     # 持ち上げ方向の時定数 (秒)
        self.slow_agc_gain = 1.0
        self._slow_agc_floor = 1e-4      # -80dBFS未満は無音とみなし凍結

        # 超音波ノイズ比追従型 コグニティブ・オートスケルチ (FM三角ノイズクワイエティング追従)
        self.ultra_squelch = UltrasonicSquelchTracker(sample_rate=self.if_rate)
        self.ultra_squelch.enabled = False  # squelch_enabled と連動

        # 音声/音楽 認知型オートチルトEQ (トーク了解度 / 音楽フラットHi-Fi 自動追従)
        self.cognitive_eq = CognitiveSpeechMusicTracker(sample_rate=self.audio_rate)
        self._cog_wide = None  # トラッカー用広帯域タップ (_post_process_wfmが更新)
        # 38kHz 直交副搬送波マルチパス適応キャンセラ (サ行シピシピ歪み・混濁の逆位相相殺)
        self.mpx_canceller = QuadratureMpxCanceller(sample_rate=self.audio_rate)
        # 深宇宙通信級 拡張カルマンフィルタ (Deep-Space EKF) FM復調エンジン
        self.ekf_demod = DeepSpaceEkfDemodulator(sample_rate=self.if_rate)
        self.ekf_enabled = True

        # ===== 高級オーディオ (Accuphase / dCS 理論) 統合モジュール =====
        # 低域位相回転ゼロ・アクティブDCサーボ (20Hz〜300Hzの低音位相進み歪みを根絶)
        self.dc_servo = ActiveDcServo(sample_rate=self.audio_rate, time_constant_sec=3.5)
        # TPDFディザー & 音響心理ノイズシェーピング (16bit量子化歪み排除 & 微小残響保持)
        self.dither = TpdfDitherNoiseShaper(sample_rate=self.audio_rate)
        self.dither.enabled = False  # 内部DSPの数学的等価性維持のためデフォルトOFF (set_audiophile_modeで切替)
        self.apodizing_enabled = False
        self._linear_fir_audio_clean = self.fir_audio_clean.copy()
        self._linear_fir_audio_narrow = self.fir_audio_narrow.copy()
        self._linear_fir_am_audio = self.fir_am_audio.copy()

        # ===== AM同期検波 (キャリア再生PLL) =====
        # 選択性フェージング時のひずみを避けるため、包絡線検波ではなく
        # キャリアに同期した同相検波を使う。ロックできない時は包絡線へ自動復帰。
        self.am_sync_enabled = True
        self.am_sync_lock = 0.0
        # ===== SSB / CW =====
        self.bfo_offset_hz = 0.0     # BFO微調整 (SSB/CWのみ)
        self.ssb_agc_level = 0.0
        self._ssb_agc_hang = 0
        self._ssb_bp_phase = 0.0
        self._ssb_bfo_phase = 0.0

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
        # CMAブラインド等化器 (マルチパス・キャンセル)。手動は実機アンテナの安定性のためデフォルトOFF。
        # cognitive時の強い反射波には、multipath量ヒステリシス＋信号存在ゲートで自動介入する。
        self.multipath_cancel_enabled = False
        self.multipath_auto_cancel = True
        self._cma_auto = False
        self._cma_taps = 33
        # μは0.03→0.02へ (長遅延強エコー d=40/g=1.2で0.03はlock 0.18・
        # 0.02は0.39・0.015は0.56。短エコー・flutterは同等以上。
        # 0.015が最良だが実フラッターの追従速度を残すため0.02を採用)。
        self._cma_mu = 0.02
        self._cma_w = np.zeros(2 * self._cma_taps, dtype=np.float32)
        self._cma_w[2 * (self._cma_taps // 2)] = 1.0  # 中央タップ=デルタ初期化
        self._cma_hist = np.zeros(self._cma_taps - 1, dtype=np.complex64)
        self.cma_active = False

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
        self.am_kp = 3.0e-4
        self.am_ki = 5.0e-8
        # ループ帯域を約10Hz級へ狭帯域化 (旧100Hzでは低音音声がVCOを変調し
        # 混変調歪みを誘発)。ロックが遅くなってもmixが包絡線へ自動復帰する。
        self._am_alpha = 2.0 * np.pi * 30.0 / self.if_rate

        # スペクトラム表示設定
        self.fft_size = 1024
        self.fft_window = np.hamming(self.fft_size).astype(np.float32)
        self.fft_smooth = None
        self.smooth_alpha = 0.25

    def set_offset_freq(self, offset_hz: float):
        with self._state_lock:
            self._set_offset_freq_locked(offset_hz)

    def _set_offset_freq_locked(self, offset_hz: float):
        self.offset_freq = offset_hz
        self._tune_monotonic = time.monotonic()
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
        # 38kHz位相トリムも局替わりでリセット (前局の multipath 位相を持ち越さない)
        self.stereo_phase_offset = 0.0
        # ブレンドも局替わりでリセット (前局のステレオ像・フライホイールを持ち越さない。
        # しないと新局の冒頭1秒が前局ブレンドのまま誤ステレオ化する)
        self._stereo_blend = 0.0
        self.stereo_blend = 0.0
        self._pilot_hold_n = 0
        # スケルチ統合も開状態から (前局のミュートを持ち越さない。
        # AM等WFM外ではhelperが呼ばれないためここで戻す)
        self._bm_sq_open = True
        self._bm_sq_gain = 1.0
        self._trim_dir = 1.0
        self._trim_block = 0
        self._trim_primed = False
        self._trim_prev_m = 0.0
        self._trim_m_smooth = 0.0
        self._trim_err_ema = 0.0
        self.fm_last_sample = 0.0 + 0.0j
        self.nfm_last_sample = 0.0 + 0.0j
        self._fm_pll_state[:] = 0.0
        # CMA等化器も再初期化 (前局のチャネル推定を持ち越さない)
        self._cma_w[:] = 0.0
        self._cma_w[2 * (self._cma_taps // 2)] = 1.0
        self._cma_hist[:] = 0.0
        self.cma_active = False
        self._cma_auto = False
        self.reset_stereo_nr()
        self.am_sync_lock = 0.0
        if hasattr(self, "ultra_squelch"):
            self.ultra_squelch.reset()
        if hasattr(self, "cognitive_eq"):
            self.cognitive_eq.reset()
        self._cog_wide = None
        if hasattr(self, "dc_servo"):
            self.dc_servo.reset()
        if hasattr(self, "dither"):
            self.dither.reset()
        if hasattr(self, "bss_separator"):
            self.bss_separator.reset()
        if hasattr(self, "riemann_demodulator"):
            self.riemann_demodulator.reset()
        if hasattr(self, "rmt_denoiser"):
            self.rmt_denoiser.reset()
        if hasattr(self, "mono_nr"):
            self.mono_nr.reset()
        if hasattr(self, "sic_canceller"):
            self.sic_canceller.reset()
        # 黒魔法状態も選局でリセット (前局のconfidence・強度を持ち越さない)
        try:
            if getattr(self, "cyclo_detector", None) is not None:
                self.cyclo_detector.reset()
            if getattr(self, "bm_rmt", None) is not None:
                self.bm_rmt.reset()
            if getattr(self, "bm_sr", None) is not None:
                self.bm_sr.reset()
            if getattr(self, "bm_notch", None) is not None:
                self.bm_notch.reset()
            if getattr(self, "bm_controller", None) is not None:
                self.bm_controller.reset()
        except Exception:
            pass
        self.bm_cyclo_confidence = 0.0
        self.bm_sr_confidence = 0.0
        self._bm_sq_open = True
        self._bm_sq_gain = 1.0
        try:
            if getattr(self, "_bm_lock_hist", None) is not None:
                self._bm_lock_hist.clear()
        except Exception:
            pass
        self._sic_detect_counter = 0
        # RDS状態も選局でリセット (前局のPS/PI/RTを次局へ持ち越さない)
        if self.rds is not None:
            try:
                self.rds.reset()
            except Exception:
                pass
        self.rds_ps = ""
        self.rds_rt = ""
        self.rds_pi = 0
        self.rds_pty = None
        self.rds_groups = 0

    def update_resampler_feedback(self, current_chunks: float, dt: float = 0.05):
        """オーディオバッファの残存チャンク数をリサンプラにフィードバック (クロック自動同期)"""
        self.resampler.update_feedback(current_chunks, dt=dt)

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
        self._nr_mf_smooth = 0.0
        self._nr_primed = False
        self._nr_mono_rho = 0.0
        self._nr_mono_w = 0.0
        self._nr_mono_primed = False
        self._nr_sw_eff = 0.0
        self._nr_cut_eff = 15000.0
        self._wf_p = None
        self._wf_g = None
        self._nr_floor_pow = 0.0
        # STFT内部バッファもクリア (前局のL-R残響が新局の差信号へ混入するのを防ぐ)
        self._wf_in = np.zeros(0, dtype=np.float32)
        self._wf_out = np.zeros(self._wf_hop, dtype=np.float32)  # 固定遅延の初期プリフィル
        self._wf_ola = np.zeros(self._wf_n, dtype=np.float32)
        self._wf_extra_delay = 0
        # 周波数依存ブレンドのクロスオーバー状態もクリア (前局の低域残響防止)
        self._blend_xo_y1 = 0.0
        # 直交残差・トリム方向もクリア
        self._last_diff_q = None

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
        self.deemph_tau_us = float(tau_us)
        self.deemph_sections = _deemph_sections(self.deemph_tau_us)
        self.deemph_x1 = self.deemph_y1 = 0.0
        self.deemph2_x1 = self.deemph2_y1 = 0.0
        # Pythonフォールバックのチャンネル別状態もリセット (旧時定数の残留防止)
        self.deemph_x1_l = self.deemph_y1_l = 0.0
        self.deemph_x1_r = self.deemph_y1_r = 0.0
        self.deemph2_x1_l = self.deemph2_y1_l = 0.0
        self.deemph2_x1_r = self.deemph2_y1_r = 0.0
        self._deemph_state[:] = 0.0
        self._deemph2_state[:] = 0.0
        self._deemph_state_l[:] = 0.0
        self._deemph_state_r[:] = 0.0
        self._deemph2_state_l[:] = 0.0
        self._deemph2_state_r[:] = 0.0

    def set_squelch(self, enabled: bool, threshold_db: float = -68.0):
        self.squelch_enabled = enabled
        self.squelch_threshold = threshold_db
        if hasattr(self, "ultra_squelch"):
            self.ultra_squelch.enabled = enabled

    def set_audiophile_mode(
        self,
        apodizing: bool = None,
        dc_servo: bool = None,
        dither: bool = None,
    ):
        """
        高級オーディオ処理 (Accuphase / dCS / Esoteric 理論) の動的設定。
        :param apodizing: 最小位相アポダイジングフィルタ (インパルス応答のプリリンギング完全ゼロ化)
        :param dc_servo: 超低域位相回転ゼロ・アクティブDCサーボ (20Hz〜300Hzの低域位相歪み根絶)
        :param dither: TPDFディザー & 音響心理ノイズシェーピング (16bit量子化歪み・階段歪み排除)
        """
        if dc_servo is not None:
            self.dc_servo.enabled = bool(dc_servo)
        if dither is not None:
            self.dither.enabled = bool(dither)
        if apodizing is not None and bool(apodizing) != self.apodizing_enabled:
            self.apodizing_enabled = bool(apodizing)
            self._update_apodizing_filters()

    def _update_apodizing_filters(self):
        """アポダイジング (最小位相) と直線位相フィルタの動的切り替え"""
        if self.apodizing_enabled:
            self.fir_audio_clean = MinimumPhaseApodizer.convert_fir_to_minimum_phase(self._linear_fir_audio_clean)
            self.fir_audio_narrow = MinimumPhaseApodizer.convert_fir_to_minimum_phase(self._linear_fir_audio_narrow)
            self.fir_am_audio = MinimumPhaseApodizer.convert_fir_to_minimum_phase(self._linear_fir_am_audio)
        else:
            self.fir_audio_clean = self._linear_fir_audio_clean.copy()
            self.fir_audio_narrow = self._linear_fir_audio_narrow.copy()
            self.fir_am_audio = self._linear_fir_am_audio.copy()

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

    @staticmethod
    def _wfm_if_snr_db(spectrum_db: np.ndarray, rf_rate: float):
        """WFM復調前の局所チャンネル内SNR推定（IFダイナミクス用）。

        Hyperの帯域内定義と同一: ±85kHz信号帯域とスペクトル下位25%タイル床を比較し、
        弱電界では尖頭寄りで融合する。表示スムージング済みスペクトルでも動作する。
        """
        if spectrum_db is None:
            return None
        spec = np.asarray(spectrum_db, dtype=np.float64).reshape(-1)
        if len(spec) < 128:
            return None
        spec = np.nan_to_num(spec, nan=-120.0, posinf=0.0, neginf=-120.0)
        lin = np.power(10.0, spec / 10.0)
        n = len(lin)
        c = n // 2
        bin_hz = float(rf_rate) / float(n)
        ch_half = max(4, int(85000.0 / bin_hz))
        g_in = max(ch_half + 2, int(115000.0 / bin_hz))
        g_out = int(250000.0 / bin_hz)
        g_out = min(g_out, c - 2)
        g_in = min(g_in, g_out - 1)
        if g_out <= g_in or c - g_out < 0:
            return None
        ch = lin[c - ch_half: c + ch_half + 1]
        if len(ch) == 0:
            return None
        sig_mean = float(np.mean(ch))
        sig_peak = float(np.max(ch))
        noise_p = float(np.percentile(lin, 25))
        mean_snr = 10.0 * np.log10((sig_mean + 1e-12) / (noise_p + 1e-12))
        peak_snr = 10.0 * np.log10((sig_peak + 1e-12) / (noise_p + 1e-12))
        snr_db = max(mean_snr, 0.35 * mean_snr + 0.65 * peak_snr)
        if not np.isfinite(snr_db):
            return None
        return float(snr_db)

    def _update_cognitive_morph(self, if_snr_db=None, mode="WFM"):
        """離散切替ではなくサンプル単位で滑らかにフィルタを変形 (クリック・ポップ根絶)

        WFMでは復調前の局所SNRでIF帯域だけに非対称スルーを掛ける。
        低SNRでは狭窄を低速化し、回復時は開放を優先する。最終到達点は変えない。
        """
        if not self.cognitive_enabled:
            return
        a = self.cognitive_alpha
        self.applied_cutoff_hz += a * (self.target_cutoff_hz - self.applied_cutoff_hz)
        if mode == "WFM" and if_snr_db is not None and np.isfinite(if_snr_db):
            q = float(if_snr_db)
            self._if_snr_db = q
            target = float(np.clip(self.target_if_bw_hz, 110000.0, 200000.0))
            delta = target - self.applied_if_bw_hz
            # 低SNRほど1ブロックあたりの変化量を絞り、履歴FIRの急変ショックを防ぐ。
            limit = 2500.0 if q < 4.0 else (4000.0 if q < 8.0 else 8000.0)
            self.applied_if_bw_hz += max(-limit, min(limit, delta))
        else:
            self.applied_if_bw_hz += a * (self.target_if_bw_hz - self.applied_if_bw_hz)
        self.hf_gain_applied += a * (self.target_hf_gain - self.hf_gain_applied)

    def _get_dynamic_filter(self, kind: str, cutoff_hz: float) -> np.ndarray:
        """カットオフ毎にFIRをその場設計 (100Hz量子化LRUキャッシュで実時間コスト極小・全消去スパイクゼロ)"""
        quant = max(200, int(round(cutoff_hz / 100.0)) * 100)
        key = (kind, quant)
        taps = self._fir_cache.get(key)
        if taps is not None:
            self._fir_cache.move_to_end(key)
            return taps
        if kind == "if":
            taps = design_fir_kaiser(num_taps=65, cutoff_norm=quant / self.rf_rate, beta=6.2)
        else:
            taps = design_fir_kaiser(num_taps=81, cutoff_norm=quant / self.audio_rate, beta=6.8)
        self._fir_cache[key] = taps
        if len(self._fir_cache) > 128:
            self._fir_cache.popitem(last=False)  # 最も長く使われていない最古エントリのみ破棄
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

        # ポリフェーズ間引き (valid畳み込み[::factor]と完全等価、計算量1/factor)
        n_valid = len(x_ext) - len(fir_taps) + 1
        if _NATIVE is not None and NATIVE_POLY and n_valid > 0 and factor >= 1:
            n_out = (n_valid + factor - 1) // factor
            if np.iscomplexobj(x_ext):
                yr = np.empty(n_out, dtype=np.float32)
                yi = np.empty(n_out, dtype=np.float32)
                xr = np.ascontiguousarray(x_ext.real, dtype=np.float32)
                xi = np.ascontiguousarray(x_ext.imag, dtype=np.float32)
                ha = np.ascontiguousarray(fir_taps, dtype=np.float32)
                _NATIVE.sdr_polyphase_decim(_fptr(xr), _fptr(ha), _fptr(yr),
                                            n_out, len(ha), factor)
                _NATIVE.sdr_polyphase_decim(_fptr(xi), _fptr(ha), _fptr(yi),
                                            n_out, len(ha), factor)
                return yr + 1j * yi
            xa = np.ascontiguousarray(x_ext, dtype=np.float32)
            ha = np.ascontiguousarray(fir_taps, dtype=np.float32)
            y = np.empty(n_out, dtype=np.float32)
            _NATIVE.sdr_polyphase_decim(_fptr(xa), _fptr(ha), _fptr(y),
                                        n_out, len(ha), factor)
            return y

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

    def _apply_cma(self, iq_if: np.ndarray) -> np.ndarray:
        """CMAブラインド等化器 (history前置でブロック連続性を保つ)。"""
        n_out = len(iq_if)
        if n_out == 0:
            return iq_if
        taps = self._cma_taps
        if len(self._cma_hist) != taps - 1:
            self._cma_hist = np.zeros(taps - 1, dtype=np.complex64)
        x_ext = np.concatenate((self._cma_hist, np.ascontiguousarray(iq_if)))
        self._cma_hist = x_ext[-(taps - 1):].copy()

        xa = np.ascontiguousarray(x_ext, dtype=np.complex64)
        y = np.empty(n_out, dtype=np.complex64)
        _NATIVE.sdr_cma_equalize(_fptr(xa), _fptr(y), n_out,
                                 _fptr(self._cma_w), taps, float(self._cma_mu))
        # フェイルセーフ: 万一 NaN/Inf が出力されたら重みと履歴を即座にリセットし
        # 原信号をサニタイズして通過 (NaN履歴が数ブロック再発するのを防ぐ)
        if not np.all(np.isfinite(y)):
            self._cma_w[:] = 0.0
            self._cma_w[2 * (taps // 2)] = 1.0
            self._cma_hist[:] = 0.0
            return np.nan_to_num(iq_if, nan=0.0, posinf=1.0, neginf=-1.0)
        return y

    def _update_cma_auto_gate(self) -> bool:
        """CMA自動介入ゲート（ヒステリシス＋信号存在条件）。

        強い反射波でのみ作動し、弱まったら速やかに戻す。手動設定は常に尊重する。
        cognitive無効時は自動介入しない（従来テスト/非認知経路の挙動を保護）。
        介入条件はパイロットlock>0.2が必須。S-meterだけでの作動
        (旧OR条件) は、lock≈0の深フェードでCMAが入りっぱなしになり
        blendを下げる (合成deep-fadeでblend 1.00→0.73を実測) ため廃止。
        S-meterは-60dBFSのノイズ床 veto としてのみ使う。
        """
        if not self.multipath_auto_cancel or not self.cognitive_enabled:
            self._cma_auto = bool(self.multipath_cancel_enabled)
            return self._cma_auto
        try:
            lock = abs(float(self.stereo_pilot_lock))
            s_db = float(self.s_meter_dbfs)
        except (TypeError, ValueError):
            lock, s_db = 0.0, -90.0
        if not (math.isfinite(lock) and math.isfinite(s_db)):
            lock, s_db = 0.0, -90.0
        present = lock > 0.2 and s_db > -60.0
        amt = float(self.multipath_amount)
        if self._cma_auto:
            hold = amt >= 0.18 and present
        else:
            hold = amt >= 0.30 and present
        self._cma_auto = bool(self.multipath_cancel_enabled or hold)
        return self._cma_auto

    def _apply_hard_limiter(self, iq_if: np.ndarray) -> np.ndarray:
        mag = np.abs(iq_if)
        mask = (mag > 1e-12) & np.isfinite(mag)
        out = np.zeros_like(iq_if)
        out[mask] = iq_if[mask] / mag[mask]
        return out

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
        # 平滑は鈍め (測定1.0秒・反映1.5秒): 速すぎると番組の包絡変動や
        # フェージングの瞬時値にステレオ幅が呼吸してしまう (市街地局で実測)
        if self.multipath_enabled:
            env = np.abs(iq_if)
            var = float(np.std(env) / (np.mean(env) + 1e-12))
            dt_mp = len(iq_if) / self.if_rate
            self._mp_var += (1.0 - np.exp(-dt_mp / 1.0)) * (var - self._mp_var)
            x = float(np.clip((self._mp_var - self.mp_lo) / (self.mp_hi - self.mp_lo), 0.0, 1.0))
            s = x * x * (3.0 - 2.0 * x)
            tau = 1.5 if s > self.multipath_amount else 2.5
            self.multipath_amount += (1.0 - np.exp(-dt_mp / tau)) * (s - self.multipath_amount)
            self.multipath_gain = 1.0 - self.mp_depth * self.multipath_amount

        # 1. CMA等化 (マルチパス・キャンセル) + ハードリミッター適用
        use_cma = self._update_cma_auto_gate()
        if (use_cma and _NATIVE is not None and NATIVE_CMA
                and self.multipath_amount > 0.15
                and abs(self.stereo_pilot_lock) > 0.2
                and self.s_meter_dbfs > -60.0):
            # CMA等化 (ハードリミット前。リミット後は包絡線一定で誤差が出ない)。
            # 信号存在ゲート: lock必須＋ノイズ床veto (S-meterだけでの作動は
            # 深フェードでblendを下げるため廃止。ゲート側と条件を一致させる)。
            iq_if = self._apply_cma(iq_if)
            self.cma_active = True
        else:
            self.cma_active = False

        limited = self._apply_hard_limiter(iq_if)

        # 2. FM復調: ハイブリッド宇宙通信級復調エンジン
        # - 通常 (強信号) : 瞬時位相差分法 (分離度-42dB・高変調域も歪まない超広帯域復調)
        # - 弱信号 (S7相当以下) : 深宇宙通信級 拡張カルマンフィルタ (EKF) による確率論的MMSE復調
        #   (FM閾値拡張 +6〜+9dB、低CNR下での2π位相スリップ・クリックスパイクノイズを完全阻止)
        # NOTE: 旧 use_pll (NATIVE_PLLFM) 経路は未使用デッドコードだったため削除。
        # PLL-FMが必要になった場合は fm_pll_enabled とは独立に有効化すること。
        if _NATIVE is not None:
            work = np.ascontiguousarray(limited, dtype=np.complex64)
            demod = np.empty(len(work), dtype=np.float32)
            last = np.array([self.fm_last_sample.real, self.fm_last_sample.imag],
                            dtype=np.float32)
            _NATIVE.sdr_fm_demod(_fptr(work), _fptr(demod), len(work), _fptr(last))
            self.fm_last_sample = complex(float(last[0]), float(last[1]))
        else:
            s = np.concatenate(([self.fm_last_sample], limited))
            self.fm_last_sample = limited[-1]
            diff = s[1:] * np.conj(s[:-1])
            demod = np.angle(diff)

        # 弱電界・モノラル時における深宇宙EKFのシームレス・クロスフェード
        # (ステレオ時は38kHz副搬送波の広帯域通過のため超広帯域差分法を維持し、
        # 弱電界・モノラル局で深宇宙EKFを稼働させてクリックスパイクと三角雑音を完全根絶)
        # NOTE: 旧条件は fm_pll_enabled(False既定) とのANDで常時OFFになっていた。
        # EKFは ekf_enabled 単独で制御する (PLLとは独立)。
        if (getattr(self, "ekf_enabled", False)
                and getattr(self, "ekf_demod", None) is not None
                and self._stereo_blend <= 0.05):
            w_ekf = float(np.clip((-38.0 - self.s_meter_dbfs) / 10.0, 0.0, 1.0))
            if w_ekf > 0.01:
                demod_ekf = self.ekf_demod.demodulate(limited)
                if len(demod_ekf) == len(demod):
                    demod = ((1.0 - w_ekf) * demod + w_ekf * demod_ekf).astype(np.float32)

        # 位相スリップ防止FM復調 (弱電界フェージング時のクリック雑音抑制)
        if (getattr(self, "riemann_demodulator", None) is not None
                and self.riemann_demodulator.enabled
                and (self.cognitive_enabled or getattr(self, "riemann_always", False))
                and self.s_meter_dbfs < -35.0):
            w_riemann = float(np.clip((-35.0 - self.s_meter_dbfs) / 12.0, 0.0, 0.75))
            demod_riemann = self.riemann_demodulator.demodulate(iq_if)
            if len(demod_riemann) == len(demod):
                demod = ((1.0 - w_riemann) * demod + w_riemann * demod_riemann).astype(np.float32)

        # 超音波三角ノイズ比追従型 オートスケルチ
        ultra_gain = 1.0
        if getattr(self, "ultra_squelch", None) is not None and self.ultra_squelch.enabled:
            ultra_gain, _ = self.ultra_squelch.process(demod)

        # 黒魔法C: 確率共鳴は検出用副経路のみ。メインのdemod配列には一切触らない。
        # confidenceを属性 (bm_sr_confidence) として公開するだけで、
        # スケルチ判定への自動反映は既存動作との競合回避のため見送る。
        # (評価はbenchmark_black_magic.pyの検出率比較で行う)
        if (getattr(self, "black_magic_enabled", False)
                and getattr(self, "bm_sr_enabled", False)):
            try:
                _bmp2 = getattr(self, "_bm_params", None) or {}
                # コントローラがSR非活性と判断したら試行自体を休止する
                _sr_active = bool(_bmp2.get("sr_active", True))
                if self.bm_sr is None and StochasticResonanceDetector is not None:
                    self.bm_sr = StochasticResonanceDetector()
                if (self.bm_sr is not None and _sr_active
                        and len(demod) >= 1024):
                    n_sr = len(demod)
                    if self._bm_sr_tw is None or self._bm_sr_tw[0] != n_sr:
                        k_sr = int(round(19000.0 * n_sr / float(self.if_rate)))
                        k_sr = min(max(k_sr, 1), n_sr - 1)
                        self._bm_sr_tw = (
                            n_sr,
                            np.exp(-2j * np.pi * k_sr * np.arange(n_sr) / n_sr))
                    _tw = self._bm_sr_tw[1]

                    def _bm_base(v, _tw=_tw, _n=n_sr):
                        vv = np.asarray(v, dtype=np.float64).reshape(-1)
                        if len(vv) != _n:
                            return False
                        return bool(abs(np.dot(vv, _tw)) * (2.0 / _n) > 0.02)

                    _uq = getattr(self, "ultra_squelch", None)
                    floor = 10.0 ** (float(getattr(_uq, "noise_db", -40.0)) / 20.0)
                    snr = float(getattr(self, "s_meter_dbfs", -45.0)) + 45.0
                    base_conf = min(max(float(getattr(self, "stereo_pilot_lock",
                                                      0.0)), 0.0), 1.0)
                    res = self.bm_sr.assess(
                        demod, _bm_base, floor, snr, base_conf,
                        clip=False,  # dspにADCクリップ旗なし。過大時は使わないこと
                        candidate_present=bool(self.stereo_enabled))
                    if res.get("enabled"):
                        self.bm_sr_confidence = float(res.get("sr_confidence",
                                                              base_conf))
                    else:
                        self.bm_sr_confidence = base_conf
            except Exception:
                pass

        # 黒魔法①: cyclo→スケルチ統合 (既定OFF)。1ブロック遅れのconfidenceと
        # S-meterで開閉を決め、ソフトフェードゲインに反映する。音声への適用は
        # process()終端 (スローAGC後) で行う。音声自体は変えない。
        self._bm_update_squelch_assist()

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

        # 5a. 副搬送波清浄化は廃止 (固定LPに劣る適応軟しきい値だった。
        # 既知周波数の搬送波に適応は不要という結論。素通し)
        demod_mpx = demod_scaled

        # 5b. ステレオMPXデコード (19kHzパイロットPLL + 38kHz同期検波)
        self._update_stereo_pilot(demod_mpx)

        # 5c. RDS復調 (57kHz = 3θ, ステレオ状態と独立して常時動作)
        # パイロットロック連動ゲート: パイロット不在時 (無信号・モノラル) は
        # 57kHzに信号が存在し得ないためfeedを省略する。ノイズ入力でデコーダの
        # 同期探索 (syndrome全探索) が約4ms/block浪費していた問題の根絶。
        # デコーダ状態は保持されるため、ロック復帰時は即時再開する。
        if (self.rds_enabled and self._last_cos3 is not None
                and abs(self.stereo_pilot_lock) > 0.2):
            try:
                if self.rds is None:
                    import rds as rds_mod
                    self.rds = rds_mod.RdsDecoder(12000.0)
                # BS.450/EN 50067準拠: パイロットsin(wt)に対して57kHz副搬送波はsin(3wt)。
                # PLLがth = wt - pi/2でロックしているため、cos(3*th) = cos(3wt - 3pi/2) = -sin(3wt)。
                # したがって、同相復調キャリアは -self._last_cos3。
                carrier57 = -self._last_cos3
                if abs(self.rds_phase_offset) > 1e-6 and self._last_sin3 is not None:
                    co = np.cos(self.rds_phase_offset)
                    si = np.sin(self.rds_phase_offset)
                    carrier57 = carrier57 * co - self._last_sin3 * si
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
        _carrier_ok = (self._last_sin2 is not None
                       and len(self._last_sin2) == len(demod_scaled))
        if _carrier_ok and self._stereo_blend > 0.02:
            # BS.450準拠: 38kHz副搬送波はsin(2wt)。
            # PLLがth = wt - pi/2でロックしているため、sin(2*th) = sin(2wt - pi) = -sin(2wt)。
            # したがって同相復調キャリアは -self._last_sin2。
            carrier = -self._last_sin2
            if abs(self.stereo_phase_offset) > 1e-6 and self._last_cos2 is not None:
                co = np.cos(self.stereo_phase_offset)
                si = np.sin(self.stereo_phase_offset)
                carrier = carrier * co - self._last_cos2 * si
            lpr = demod_scaled * carrier
            diff_raw = self.decimate_with_history(lpr, self.fir_if_audio,
                                                  self.audio_decim, "history_lpr") * 2.0

            # 38kHz 直交副搬送波マルチパス適応キャンセラ (Quadrature MPX Decoupler)
            if (getattr(self, "mpx_canceller", None) is not None
                    and self.mpx_canceller.enabled and self._last_cos2 is not None):
                carrier_q = -self._last_cos2
                if abs(self.stereo_phase_offset) > 1e-6:
                    carrier_q = carrier_q * co + self._last_sin2 * si
                lpr_q = demod_scaled * carrier_q
                diff_q = self.decimate_with_history(lpr_q, self.fir_if_audio,
                                                    self.audio_decim, "history_lpr_q") * 2.0
                if len(diff_q) == len(diff_raw):
                    self._last_diff_q = diff_q
                    diff_raw = self.mpx_canceller.process(diff_raw, diff_q)
                else:
                    self._last_diff_q = None

            if len(diff_raw) == len(mono):
                # 常に実測 (診断・A/B用)。NR無効時はフィルタ素通し・フルステレオ固定
                self._update_stereo_nr(diff_raw, mono)
                # 38kHz位相オートトリム (L-R電力最大化。NR推定と同タイミングで観測)
                self._update_stereo_trim(diff_raw, mono)
                if self.stereo_nr_enabled:
                    cut = self._nr_cut_eff
                    blend = self._stereo_blend * self.stereo_nr_gain
                    blend *= self.multipath_gain
                    diff = self._diff_lowpass(diff_raw, cut)
                    diff = self._wiener_diff(diff)
                    # 周波数依存ブレンド: 弱電界で高域から先にモノラル化
                    # (低域のステレオ感を残す。blend=1時は旧スカラー動作と完全一致)
                    stereo_diff = self._freq_dependent_blend(diff, blend)
                    # 差信号FIRの群遅延を補償 (mono/diffの位相ズレによる分離度劣化を防止)
                    mono = self._delay_mono(mono)
                else:
                    # NR無効時はLPF/STFT往復をせず生差信号へブレンドのみ
                    # (15kHz LPF＋COLAリップルが可聴域を変える問題とCPU浪費を回避)。
                    # mono遅延履歴だけは更新し、再有効時の継ぎ目を無くす
                    # (出力は遅延させない。遅延させると未遅延diffと2msずれる)。
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

            # 超空間独立成分ステレオ復調器 (BSS / FastICA ステレオ逆相三角ヒスノイズ直交消去)
            if (getattr(self, "bss_separator", None) is not None
                    and self.bss_separator.enabled
                    and (self.cognitive_enabled or getattr(self, "bss_always", False))):
                left, right = self.bss_separator.process(left, right, stereo_blend=self._stereo_blend)

            if ultra_gain < 0.999:
                left = left * ultra_gain
                right = right * ultra_gain
            # 黒魔法2-1: 適応ハムノッチ (既定OFF)。RMTより前段に置く
            # (ハム除去後のクリーンな信号をNRへ渡す)。
            if (getattr(self, "black_magic_enabled", False)
                    and getattr(self, "bm_notch_enabled", False)):
                try:
                    if self.bm_notch is None and AdaptiveNotchCanceller is not None:
                        self.bm_notch = AdaptiveNotchCanceller(
                            sample_rate=self.audio_rate)
                    if self.bm_notch is not None and len(left) == len(right):
                        (left, right), _ = self.bm_notch.process_stereo(left, right)
                        left = np.asarray(left, dtype=np.float32)
                        right = np.asarray(right, dtype=np.float32)
                except Exception:
                    pass
            # 黒魔法B: ステレオ対のMid/Side安全RMT (既定OFF)。
            # L/R独立処理は分離度を落とすため、Mid通常・Side低強度で処理する。
            # 既存rmt_denoiser側との二重処理は避けること (docs参照)。
            if (getattr(self, "black_magic_enabled", False)
                    and getattr(self, "bm_rmt_enabled", False)):
                try:
                    if self.bm_rmt is None and SafeRmtDenoiser is not None:
                        self.bm_rmt = SafeRmtDenoiser(sample_rate=self.audio_rate)
                    if self.bm_rmt is not None and len(left) == len(right):
                        self.bm_rmt.max_strength = min(max(
                            float(getattr(self, "bm_rmt_cap", 0.65)), 0.0), 0.85)
                        (left, right), _bm_info = self.bm_rmt.process_stereo(
                            left, right,
                            s_meter_dbfs=float(getattr(self, "s_meter_dbfs",
                                                       -20.0)),
                            snr_db=None)
                        left = np.asarray(left, dtype=np.float32)
                        right = np.asarray(right, dtype=np.float32)
                        try:
                            _bm_mid = (_bm_info.get("mid", None) or {})
                            self._bm_last_rmt_info = {
                                "hf_loss_db": float(_bm_mid.get("hf_loss_db", 0.0)),
                                "rms_diff_db": float(_bm_mid.get("rms_diff_db", 0.0)),
                            }
                        except Exception:
                            pass
                except Exception:
                    pass
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
        # モノラル中も遅延履歴を進めておく (ステレオ復帰時に履歴が古い/ゼロだと
        # 先頭_nr_delayサンプルが無音になり2msの欠落クリックになる)。
        # また、NR有効時はモノラル出力も同量遅延させることで、ステレオ/モノラル自動切替時の
        # タイムワープ (2ms音飛び・重複・クリック) を完全根絶しタイムラインを100%連続化する。
        mono_delayed = self._delay_mono(mono)
        mono_out = mono_delayed if self.stereo_nr_enabled else mono
        # モノラル信号はステレオのセンター定位 (L=mono, R=mono) と完全に同一レベル (0dB差) で出力
        out_mono = self._post_process_wfm(mono_out, "")
        if ultra_gain < 0.999:
            out_mono = out_mono * ultra_gain
        # 黒魔法2-1: モノラル経路の適応ハムノッチ (既定OFF)
        if (getattr(self, "black_magic_enabled", False)
                and getattr(self, "bm_notch_enabled", False)):
            try:
                if self.bm_notch is None and AdaptiveNotchCanceller is not None:
                    self.bm_notch = AdaptiveNotchCanceller(
                        sample_rate=self.audio_rate)
                if self.bm_notch is not None:
                    out_mono, _ = self.bm_notch.process_mono(out_mono, ch="bm")
                    out_mono = np.asarray(out_mono, dtype=np.float32)
            except Exception:
                pass
        # 黒魔法B: モノラル経路の安全RMT (既定OFF)
        if (getattr(self, "black_magic_enabled", False)
                and getattr(self, "bm_rmt_enabled", False)):
            try:
                if self.bm_rmt is None and SafeRmtDenoiser is not None:
                    self.bm_rmt = SafeRmtDenoiser(sample_rate=self.audio_rate)
                if self.bm_rmt is not None:
                    self.bm_rmt.max_strength = min(max(
                        float(getattr(self, "bm_rmt_cap", 0.65)), 0.0), 0.85)
                    out_mono, _bm_info_m = self.bm_rmt.process_mono(
                        out_mono,
                        s_meter_dbfs=float(getattr(self, "s_meter_dbfs", -20.0)),
                        snr_db=None, ch="bm")
                    out_mono = np.asarray(out_mono, dtype=np.float32)
                    try:
                        self._bm_last_rmt_info = {
                            "hf_loss_db": float(_bm_info_m.get("hf_loss_db", 0.0)),
                            "rms_diff_db": float(_bm_info_m.get("rms_diff_db", 0.0)),
                        }
                    except Exception:
                        pass
            except Exception:
                pass
        return np.clip(out_mono, -1.0, 1.0)

    def _delay_mono(self, mono: np.ndarray) -> np.ndarray:
        """monoを群遅延分だけ遅延させ、NRフィルタ通過後の差信号と時間整合を取る。
        NR経路の実遅延 (_nr_delay + STFTゼロ埋め分) に動的に一致させる。"""
        d = self._nr_delay + int(getattr(self, "_wf_extra_delay", 0))
        if d <= 0 or len(mono) == 0:
            return mono
        # 履歴長が不足する場合はゼロで延長 (遅延量を厳密に保つ)
        if len(self.history_mono_delay) < d:
            pad = np.zeros(d - len(self.history_mono_delay), dtype=np.float32)
            self.history_mono_delay = np.concatenate((pad, self.history_mono_delay))
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
        # 番組パワーの平滑化 (τ2.5秒対称): 瞬時mfのままだと音楽の緩急が
        # そのままヒス比に載ってNRゲイン・カットオフがポンピングする
        # (強音楽局の実測で発覚)。フロア側は10秒履歴なので分母だけ鈍らせる。
        dt = len(diff) / self.audio_rate
        if not self._nr_primed or self._nr_mf_smooth <= 0.0:
            self._nr_mf_smooth = mf
        else:
            a_mf = 1.0 - np.exp(-dt / 2.5)
            self._nr_mf_smooth += a_mf * (mf - self._nr_mf_smooth)
        mf = self._nr_mf_smooth
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
        # 固定高域ブレンド: 副搬送波ヒス対策で常時上限を適用 (適応側がそれ以上
        # 絞る場合はそちらを優先)
        self.stereo_cut_hz = min(self.stereo_cut_hz, self._nr_cut_fixed_hz)
        # モノラル番組判定: 実際のステレオミックスでは M=(L+R)/2 と S=(L-R)/2 は
        # 直交するため、M-S相関は「番組でないS成分」(分離漏れクロストーク+ノイズ)
        # の割合を示す。相関が高ければS側を積極抑圧しても番組を損なわない
        # (実測: ラッキーFM 94.6 の番組はモノラルで相関+0.63、S/M -24dB)。
        # 真のステレオ番組 (相関≈0) では従来の控えめ設定を維持する。
        mo = np.asarray(mono, dtype=np.float64)
        di = np.asarray(diff, dtype=np.float64)
        if len(mo) == len(di) and len(mo) > 0:
            mo = mo - float(mo.mean())
            di = di - float(di.mean())
            den = float(np.sqrt(np.mean(mo * mo) * np.mean(di * di))) + 1e-12
            rho = float(np.mean(mo * di)) / den
            if not self._nr_mono_primed:
                self._nr_mono_primed = True
                self._nr_mono_rho = rho
            else:
                a_r = 1.0 - np.exp(-dt / 2.0)
                self._nr_mono_rho += a_r * (rho - self._nr_mono_rho)
        x_m = float(np.clip((self._nr_mono_rho - 0.15) / 0.35, 0.0, 1.0))
        self._nr_mono_w = x_m * x_m * (3.0 - 2.0 * x_m)
        # 有効値: モノラル番組ではWienerを全力(1.0)へ、S高域カットを8kHzへ寄せる
        self._nr_sw_eff = max(self._nr_s_w, self._nr_mono_w)
        self._nr_cut_eff = min(self.stereo_cut_hz, 13000.0 - 5000.0 * self._nr_mono_w)

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
        """差信号のサブバンドWiener抑圧 (STFT 128/hop 64, 平方根Hann(Sine窓) 50%オーバーラップ)。

        平方根Hann窓による50% OLAで振幅変調リップルゼロ(0.000000dB)の完全再構成。
        周波数ごとに 信号/(信号+ノイズ) の最適重みを掛けるため、ノイズに埋もれた
        高域だけが落ち、SNRの良い低域のステレオ感はそのまま残る。
        入出力のサンプル数は厳密に一致させ、mono側は_nr_delayで遅延補償する。

        高速化: フレーム毎の逐次rfft/irfftをバッチ行列FFTに統合 (数学的等価:
        各行が独立FFTのためbit一致、平滑再帰・OLA加算順序も保存)。
        """
        n = len(x)
        if n == 0:
            return x
        buf = np.concatenate((self._wf_in, x))
        nfft = self._wf_n
        hop = self._wf_hop
        nframes = (len(buf) - nfft) // hop + 1
        if self._wf_p is None:
            self._wf_p = np.zeros(nfft // 2 + 1, dtype=np.float32)
        if nframes > 0:
            # フレーム行列 (stride複写1回) とバッチrfft (C内でループ)
            idx = np.arange(nfft)[None, :] + hop * np.arange(nframes)[:, None]
            frames = buf[idx] * self._wf_win
            specs = np.fft.rfft(frames, axis=1)
            powers = np.abs(specs) ** 2 + 1e-12

            c_noise = ((self._nr_floor_pow * self._nr_floor_bias * self._wf_scale)
                       / self._wf_hf_f2_mean)
            sw = self._nr_sw_eff if self.stereo_nr_enabled else 0.0
            # 平滑再帰のみ逐次 (フレーム間依存のため。ベクトル演算のみでFFTなし)
            gmix = np.empty_like(specs, dtype=np.float32)
            for j in range(nframes):
                power = powers[j]
                self._wf_p = 0.5 * power + 0.5 * self._wf_p
                g_w = np.maximum(1.0 - (c_noise * self._wf_f2) / (self._wf_p + 1e-12),
                                 self._nr_gmin).astype(np.float32)
                g_w[:3] = 1.0  # DC〜低域は保護
                if self._wf_g is None or len(self._wf_g) != len(g_w):
                    self._wf_g = g_w
                else:
                    a = np.where(g_w < self._wf_g, 0.7, 0.1)
                    self._wf_g = self._wf_g + a * (g_w - self._wf_g)
                # 知覚マスキングフロア: 番組にマスクされるノイズは抑圧不要 (g→1)。
                # Wienerの過剰抑圧（音楽性ノイズ・高域の曇り）を可聴性基準で緩和する。
                # マスキング算出はクリーン推定 (P-N) から行う (ノイズ込み電力では
                # ヒス自身がマスクを上げて抑圧不能になるため)。
                noise_bin = c_noise * self._wf_f2
                p_clean = np.maximum(
                    self._wf_p.astype(np.float64) - noise_bin, 0.0)
                mask_thr = (p_clean @ self._wf_spread) * self._wf_mask_offset
                gate = np.minimum(1.0, mask_thr / (noise_bin + 1e-12)).astype(np.float32)
                g_use = np.maximum(self._wf_g, gate)
                g_use[:3] = 1.0  # DC〜低域は保護
                sw = self._nr_sw_eff if self.stereo_nr_enabled else 0.0
                g_mix = 1.0 - sw * (1.0 - g_use)
                gmix[j] = g_mix
            self.stereo_wiener_gain = float(np.mean(gmix[-1]))

            # バッチirfft＋同順序OLA加算
            ymat = np.fft.irfft(specs * gmix, n=nfft)
            acc = np.concatenate((self._wf_ola.astype(np.float64),
                                  np.zeros(nframes * hop, dtype=np.float64)))
            for j in range(nframes):
                acc[j * hop:j * hop + nfft] += ymat[j] * self._wf_win
            out_hops = (acc[:nframes * hop] / np.tile(self._wf_cola[:hop], nframes))
            self._wf_ola = acc[nframes * hop:nframes * hop + nfft].astype(np.float32)
            self._wf_in = buf[nframes * hop:]
            self._wf_out = np.concatenate(
                (self._wf_out, out_hops.astype(np.float32)))
        else:
            # フレーム未満の入力は破棄せず持ち越す (旧実装はここで入力を消していた)
            self._wf_in = buf
        if len(self._wf_out) >= n:
            # 余剰がある場合、過去の不足による余分な遅延 (extra_delay) を自動解消し定常群遅延へ復帰
            if self._wf_extra_delay > 0:
                surplus = len(self._wf_out) - n
                recover = min(surplus, self._wf_extra_delay)
                self._wf_out = self._wf_out[recover:]
                self._wf_extra_delay -= recover
            y = self._wf_out[:n].copy()
            self._wf_out = self._wf_out[n:]
        else:
            # 不足分だけゼロ埋めするが、その分の遅延を記録する (mono側を同量
            # 遅らせて群遅延を一致させる。記録しないと分離度が恒久的に崩れる)
            short = n - len(self._wf_out)
            y = np.concatenate((self._wf_out, np.zeros(short, dtype=np.float32)))
            self._wf_out = np.zeros(0, dtype=np.float32)
            self._wf_extra_delay = min(int(self._wf_extra_delay) + int(short), 1 << 20)
        if len(self._wf_out) > 8 * n:
            self._wf_out = self._wf_out[-2 * n:]
        return y

    def _freq_dependent_blend(self, diff: np.ndarray, blend: float) -> np.ndarray:
        """周波数依存ブレンド: 弱電界で高域から先にモノラル化する。
        1次相補クロスオーバー (lo + hi = diff で完全再構成) で低域/高域に分け、
        低域は blend、高域は blend^2 * nr_gain で絞る。FM三角ノイズが f^2 で
        増大するため高域ほどヒスが支配的で、低域のステレオ感を残しつつ耳障りな
        高域ヒスだけ先に消える。blend=1 かつ nr_gain=1 (クリーン) は旧スカラー
        動作と完全一致の高速経路。NR無効ブランチからは呼ばれない。"""
        if len(diff) == 0:
            return diff
        if not self.freq_blend_enabled or blend >= 0.999:
            return (diff * blend).astype(np.float32)
        if blend <= 0.001:
            return np.zeros_like(diff)
        # 1次LPF (状態保持でブロック連続。fc=3.5kHz@48kHz)
        a = 1.0 - float(np.exp(-2.0 * np.pi * float(self.freq_blend_xo_hz) / float(self.audio_rate)))
        y1 = float(self._blend_xo_y1)
        # ベクトル化指数平滑の厳密逐次と等価なIIRを、ブロック内は
        # lfilter相当の逐次ループで実行 (N~1kで0.02ms級)
        lo = np.empty_like(diff)
        for i, v in enumerate(diff):
            y1 += a * (float(v) - y1)
            lo[i] = y1
        self._blend_xo_y1 = y1
        hi = diff - lo
        blend_lo = float(blend)
        # 高域はヒス推定にもう一段連動 (三角ノイズ f^2 特性)。NRブランチ専用のため
        # stereo_nr_gain は常に有効な推定値。クリーン時は blend=nr=1 で恒等変換。
        blend_hi = float(blend * blend * float(self.stereo_nr_gain))
        return (lo * blend_lo + hi * blend_hi).astype(np.float32)

    def _update_stereo_trim(self, diff: np.ndarray, mono: np.ndarray):
        """38kHz再生位相オートトリム: 位相誤差を stereo_phase_offset で相殺する。
        主経路は直交 (Q) 腕の符号付き残差による比例サーボ (Costas型):
        I ∝ cos(φe−δ)、Q ∝ sin(φe−δ) のため corr(I,Q)/E[I²] ≒ (φe−δ)/2 が
        誤差の符号付き推定になり、δ += k·err で幾何収束する。番組レベルに不変。
        Q腕が無い場合 (キャンセラ無効時) はL-R電力の摂動観測へフォールバック。
        更新はブロック毎・比例ゲイン0.5・±15°クランプ。ゲート: パイロット
        ロック・高ブレンド・低ヒス・有音声時のみ。短時間テストにはほぼ無影響。"""
        if not self.stereo_trim_enabled:
            return
        try:
            if not (abs(float(self.stereo_pilot_lock)) > 0.5):
                return
            if not (float(self._stereo_blend) > 0.5 and float(self.stereo_nr_gain) > 0.7):
                return
            if len(diff) < 64 or len(mono) < 64:
                return
            mono_rms = float(np.sqrt(np.mean(np.asarray(mono, dtype=np.float64) ** 2)))
            if mono_rms < 0.01:  # -40dBFS未満は無音とみなし凍結
                return
            dq = getattr(self, "_last_diff_q", None)
            if dq is not None and len(dq) == len(diff):
                i = np.asarray(diff, dtype=np.float64)
                q = np.asarray(dq, dtype=np.float64)
                den = float(np.mean(i * i)) + 1e-12
                if den < 1e-8:  # 差信号ほぼゼロ (モノラル) では凍結
                    return
                corr = float(np.mean(i * q))
                err_inst = corr / den  # ≒ (φe−δ)/2 [rad]
                if not np.isfinite(err_inst):
                    return
                # 高速変動 (実マルチパス) はEMAで均し、準静的な成分
                # (トリム誤差・静止反射) のみ追う。実測で即時比例は
                # レール間発振したため、τ3秒＋不感帯＋微速刻みに変更。
                try:
                    dt_tr = len(diff) / float(self.audio_rate)
                except Exception:
                    dt_tr = 0.05
                a_tr = 1.0 - float(np.exp(-dt_tr / 3.0))
                ema = float(getattr(self, "_trim_err_ema", 0.0)) + a_tr * (err_inst - float(getattr(self, "_trim_err_ema", 0.0)))
                self._trim_err_ema = ema
                if abs(ema) < 0.005:
                    return  # 0.3°不感帯 (ノイズでの彷徨・レール発振を防止)
                # 比例サーボ (1ブロック0.23°上限でゆっくり。急変動には追従しない)
                step = float(np.clip(0.3 * ema, -0.004, 0.004))
                self.stereo_phase_offset = float(np.clip(
                    self.stereo_phase_offset + step,
                    -self._trim_max_rad, self._trim_max_rad))
                return
            # フォールバック: L-R電力の摂動観測 (Q腕なし時)
            diff_rms = float(np.sqrt(np.mean(np.asarray(diff, dtype=np.float64) ** 2)))
            m = diff_rms / (mono_rms + 1e-9)
            if not self._trim_primed:
                self._trim_primed = True
                self._trim_m_smooth = m
                self._trim_prev_m = m
                return
            self._trim_m_smooth += 0.25 * (m - self._trim_m_smooth)
            self._trim_block += 1
            if self._trim_block < 2:
                return
            self._trim_block = 0
            if self._trim_m_smooth < self._trim_prev_m - 1e-6:
                self._trim_dir = -self._trim_dir
            self.stereo_phase_offset = float(np.clip(
                self.stereo_phase_offset + self._trim_dir * self._trim_step_rad,
                -self._trim_max_rad, self._trim_max_rad))
            if abs(self.stereo_phase_offset) >= self._trim_max_rad - 1e-9:
                self._trim_dir = -self._trim_dir
            self._trim_prev_m = self._trim_m_smooth
        except Exception:
            pass

    def _slow_agc_level(self, audio: np.ndarray) -> np.ndarray:
        """局間音量レベリング用スローAGC: ブロックRMSを目標 (-20dBFS) へ寄せる。
        ステレオは全ch一括 (L/R連動で音像・位相を保存)、モノラルも同一式。
        無音 (-80dBFS未満) では凍結しノイズ持ち上げを防ぐ。ゲイン範囲±6dB、
        立ち下げ2秒・立ち上げ10秒の非対称時定数で、番組内の緩急には反応せず
        局替わり等の持続的レベル差だけを均す (ポンピング防止)。
        ブロック内は単一ゲイン (変化は0.1dB/block未満でジッパー雑音なし)。"""
        try:
            if len(audio) == 0:
                return audio
            x = np.asarray(audio, dtype=np.float64)
            rms = float(np.sqrt(np.mean(x * x)))
            if not np.isfinite(rms) or rms < float(self._slow_agc_floor):
                return audio
            desired = float(np.clip(float(self.slow_agc_target) / (rms + 1e-12),
                                    float(self.slow_agc_min), float(self.slow_agc_max)))
            cur = float(self.slow_agc_gain)
            # ブロック長から時定数を換算 (audio_rate基準)
            try:
                dt = len(audio) / float(self.audio_rate)
            except Exception:
                dt = 0.05
            tau = float(self.slow_agc_attack) if desired < cur else float(self.slow_agc_release)
            a = 1.0 - float(np.exp(-dt / tau))
            gain = cur + a * (desired - cur)
            self.slow_agc_gain = float(gain)
            if abs(gain - 1.0) < 1e-4:
                return audio
            return (x * gain).astype(np.float32)
        except Exception:
            return audio

    def _blend_release(self, floor: float = 0.0):
        """ブレンド低下 (パイロット瞬断フライホイール付き)。
        高ブレンドからの低下要求は25ブロック (~1.4秒) まで凍結し、
        短いパイロット瞬断発作でステレオ像がモノラルへ往復するのを防ぐ。
        持続喪失では従来通り0.97ランプで floor まで滑らかに落とす。"""
        if (self._stereo_blend > 0.5
                and self._pilot_hold_n < self._pilot_hold_max):
            self._pilot_hold_n += 1
        else:
            self._stereo_blend = max(float(floor), self._stereo_blend * 0.97)
        self.stereo_blend = self._stereo_blend

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
        # 高域エキサイター用の広帯域ソースをカット前にタップ (カット後に種を取ると
        # 8〜14k成分が無く倍音生成がno-opになる。生成した16〜22kはカット後に足す)
        wide_src = np.asarray(audio).astype(np.float32)
        if ch in ("", "_l"):
            # トラッカー用の広帯域タップ (Lのみで十分。process末尾でanalyzeする)
            self._cog_wide = wide_src
        audio = self.decimate_with_history(audio, fir_final, 1, f"history_final{ch}")

        # DCハイパスフィルタ
        audio = self._apply_dc_highpass(audio, ch=ch)

        # Hyper心理音響ハイシェルフ (FM三角雑音を連続減衰) / Cascade離散エキスパンダー
        if self.cognitive_enabled:
            audio = self._apply_hf_shelf(audio, ch=ch)
        elif self.filter_mode == "narrow":
            audio = self._apply_noise_expander(audio, threshold=0.09)

        # 高域高調波補完 (15kHz〜22kHzの高域倍音付加)
        if (getattr(self, "holographic_enhancer", None) is not None
                and self.holographic_enhancer.enabled
                and (self.cognitive_enabled or getattr(self, "holographic_always", False))):
            speech_p = getattr(getattr(self, "cognitive_eq", None), "speech_prob", 0.0)
            s_meter = getattr(self, "s_meter_dbfs", -20.0)
            audio = self.holographic_enhancer.process(audio, ch=ch, speech_prob=speech_p, s_meter_dbfs=s_meter,
                                                      source=wide_src)

        # SVD部分空間ノイズ除去 (特異値しきい値による弱電界ノイズ低減)
        if (getattr(self, "rmt_denoiser", None) is not None
                and self.rmt_denoiser.enabled
                and (self.cognitive_enabled or getattr(self, "rmt_always", False))):
            s_meter = getattr(self, "s_meter_dbfs", -20.0)
            audio = self.rmt_denoiser.process(audio, ch=ch, s_meter_dbfs=s_meter)

        # NOTE: 黒魔法Bはチャンネル別ポスト内では処理しない。L/R独立処理は
        # チャンネル間をdecorrelateし分離度を落とす (ベンチで-3〜-4dB悪化を実測)。
        # ステレオはdemodulate_wfmのスタック直前でMid/Side処理する。

        # 単一ch スペクトル抑圧NR (帯域内ノイズの最小統計Wiener抑圧)
        # 弱電界FMの番組帯ノイズ (ハイカットでは消せない) を低減する。
        # クリーン/定常信号ではゲイン1で透明に通過する自己ゲート方式。
        if (getattr(self, "mono_nr", None) is not None
                and self.mono_nr.enabled
                and getattr(self, "mono_nr_enabled", True)
                and (self.cognitive_enabled or getattr(self, "mono_nr_always", False))):
            audio = self.mono_nr.process(audio, ch=ch)

        return audio.astype(np.float32)

    def _bm_update_squelch_assist(self) -> float:
        """cyclo検出→スケルチの二基準ヒステリシス＋ソフトフェード (既定OFF)。

        開: conf>open_conf (pilot在り) または S>open_smeter (強信号)。
        閉: conf<close_conf かつ S<close_smeter (弱い局間ノイズのみ)。
        閉閾値は-25dBFS: それより強いノイズ (-16dB級の熱い局間) は
        電力スケルチの仕事とし、ここでは閉じない (弱局-27dBFSとの
        エネルギー重なりを避ける。conf側で分別する)。
        モノラル強局 (conf=0だがS高) は開のまま＝誤ミュートしない。
        開は速く (0.25/block)、閉は遅く (0.05/block) しchatterを防ぐ。
        戻り値は0.0〜1.0のゲイン。例外時は現状維持 (跳ばせない)。
        """
        try:
            if not (getattr(self, "black_magic_enabled", False)
                    and getattr(self, "bm_sq_assist_enabled", False)):
                self._bm_sq_open = True
                self._bm_sq_gain = 1.0
                return 1.0
            conf = float(getattr(self, "bm_cyclo_confidence", 0.0))
            s_db = float(getattr(self, "s_meter_dbfs", -90.0))
            if not (math.isfinite(conf) and math.isfinite(s_db)):
                return float(self._bm_sq_gain)
            open_conf = float(getattr(self, "bm_sq_open_conf", 0.75))
            close_conf = float(getattr(self, "bm_sq_close_conf", 0.55))
            close_s = float(getattr(self, "bm_sq_close_smeter_db", -25.0))
            open_s = float(getattr(self, "bm_sq_open_smeter_db", -40.0))
            if conf > open_conf:
                self._bm_sq_open = True
            elif conf < close_conf and s_db < close_s:
                self._bm_sq_open = False
            elif s_db > open_s:
                self._bm_sq_open = True
            # else: 保持 (ヒステリシス)
            target = 1.0 if self._bm_sq_open else 0.0
            g = float(self._bm_sq_gain)
            step = 0.25 if target > g else 0.05
            g += max(-step, min(step, target - g))
            self._bm_sq_gain = float(min(max(g, 0.0), 1.0))
            return self._bm_sq_gain
        except Exception:
            try:
                return float(self._bm_sq_gain)
            except Exception:
                return 1.0

    def _bm_attack_limit(self, lock: float) -> float:
        """黒魔法Aのblend上昇レート (既定0.25=従来動作)。

        confidence低＋lock低の安定弱信号のときのみ鈍化させる。
        flutter中 (直近lockの標準偏差が閾値超) は介入すると上昇だけ
        鈍ってblendが下げ方向にラチェットするため、従来レートに戻す
        (77.5MHz実測で崩落を確認した副作用の対策)。
        """
        try:
            if not (getattr(self, "black_magic_enabled", False)
                    and getattr(self, "bm_cyclo_enabled", False)):
                return 0.25
            lock_f = float(lock)
            if not math.isfinite(lock_f):
                return 0.25
            # 既存PLLが正常ロックしている場合は介入しない
            if lock_f > 0.5:
                return 0.25
            hist = getattr(self, "_bm_lock_hist", None)
            if hist is not None and len(hist) >= 8:
                sd = float(np.std(np.asarray(hist, dtype=np.float64)))
                if math.isfinite(sd) and sd > float(getattr(self, "bm_flutter_std",
                                                             0.15)):
                    return 0.25
            conf = float(getattr(self, "bm_cyclo_confidence", 0.0))
            if conf < float(getattr(self, "bm_cyclo_min_conf", 0.55)):
                _bmp = getattr(self, "_bm_params", None) or {}
                return float(_bmp.get("blend_attack_limit", 0.05))
            return 0.25
        except Exception:
            return 0.25

    def _update_stereo_pilot(self, mpx: np.ndarray):
        """19kHzパイロットPLLを更新し、ステレオブレンド係数とRDS用57kHz搬送波を生成する"""
        # RDS用cos3/sin3は毎ブロック生成するため先にクリア (前ブロック長のまま
        # 掛かると形状不一致になる)。ステレオ用cos2/sin2はパイロット消失時に
        # 緩やかなブレンド解放のため保持し、長さ不一致はdemodulate_wfm側で弾く。
        self._last_cos3 = None
        self._last_sin3 = None
        # 非有限MPX(NaN/Inf)はPLL状態を汚染するため早期破棄しブレンドを緩やかに落とす
        try:
            if mpx is None or len(mpx) < 64 or not bool(np.all(np.isfinite(np.asarray(mpx).reshape(-1)))):
                self._blend_release()
                self.stereo_pilot_lock = 0.0
                self.stereo_pilot_ratio = 0.0
                return
        except Exception:
            self._blend_release()
            return
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
                # design されたブレンド解放ランプ (0.97) で滑らかにモノラルへ落とす。
                # 搬送波を保持したままブレンドを絞るため、即時モノ切替の段差
                # (実測0.58FS) が生じない。搬送波の長さ不一致はdemodulate_wfmで弾く。
                # 短い瞬断はフライホイールで凍結する (持続喪失のみ解放ランプ)
                self._blend_release()
                self.stereo_pilot_lock = 0.0
                self.stereo_pilot_ratio = 0.0
                return
            sig = np.ascontiguousarray(np.asarray(mpx, dtype=np.float32) / pilot_rms, dtype=np.float32)
            # 黒魔法A: 検出のみ行い、上昇レート判断は後段の_bm_attack_limitへ委ねる。
            # PLLゲイン自体には触らない。
            # (コントローラがcyclo非活性と判断した場合は検出自体を休止する)
            if (getattr(self, "black_magic_enabled", False)
                    and getattr(self, "bm_cyclo_enabled", False)):
                try:
                    _bmp = getattr(self, "_bm_params", None) or {}
                    _cyc_active = bool(_bmp.get("cyclo_active", True))
                    if (self.cyclo_detector is None
                            and CyclostationaryPilotDetector is not None):
                        self.cyclo_detector = CyclostationaryPilotDetector(
                            sample_rate=self.if_rate)
                    if self.cyclo_detector is not None and _cyc_active:
                        cyc = self.cyclo_detector.update(mpx)
                        self.bm_cyclo_confidence = float(cyc.get("confidence", 0.0))
                except Exception:
                    pass
            cos2 = np.empty(n, dtype=np.float32)
            sin2 = np.empty(n, dtype=np.float32)
            quality = ctypes.c_float(0.0)
            th = ctypes.c_double(self._pll_theta)
            ig = ctypes.c_double(self._pll_integ)
            ef = ctypes.c_double(self._pll_ef)
            cos3 = sin3 = None

            # NASA DSN方式 自律適応カルマン・パイロット搬送波追従器 (AKCTL) による最適ゲイン動的計算
            if getattr(self, "pilot_tracker", None) is not None and self.pilot_tracker.enabled:
                kp, ki, alpha = self.pilot_tracker.update_gains(self.stereo_pilot_lock, pilot_rms, self._pll_ef)
            else:
                kp, ki, alpha = self._pll_kp, self._pll_ki, self._pll_alpha

            # RDS無効時は57kHz出力(cos3/sin3)の三角関数×2/サンプルを省略し
            # 2出力版PLLへ切替 (約1/3高速化。数学的等価: cos2/sin2は同一)。
            if NATIVE_PLL3 and self.rds_enabled:
                cos3 = np.empty(n, dtype=np.float32)
                sin3 = np.empty(n, dtype=np.float32)
                _NATIVE.sdr_stereo_pll3(_fptr(sig), n, ctypes.byref(th), self._pll_w0,
                                        kp, ki, ctypes.byref(ig),
                                        ctypes.byref(ef), alpha,
                                        _fptr(cos2), _fptr(sin2),
                                        _fptr(cos3), _fptr(sin3), ctypes.byref(quality))
            else:
                _NATIVE.sdr_stereo_pll(_fptr(sig), n, ctypes.byref(th), self._pll_w0,
                                       kp, ki, ctypes.byref(ig),
                                       ctypes.byref(ef), alpha,
                                       _fptr(cos2), _fptr(sin2), ctypes.byref(quality))
            self._pll_theta = th.value
            self._pll_integ = ig.value
            self._pll_ef = ef.value

            mpx_rms = float(np.sqrt(np.mean(np.asarray(mpx, dtype=np.float64) ** 2)) + 1e-12)
            ratio = pilot_rms / mpx_rms
            lock = float(quality.value)  # 正規化パイロット基準: ロック時 ~0.5-0.7
            self.stereo_pilot_lock = lock
            # flutter検出用にlock履歴を保持 (ラチェット防止。固定長で自動破棄)
            try:
                _hlh = getattr(self, "_bm_lock_hist", None)
                if _hlh is not None and math.isfinite(lock):
                    _hlh.append(lock)
            except Exception:
                pass
            self.stereo_pilot_ratio = ratio

            if not self.stereo_enabled:
                # モノラル強制: ブレンドを落としcos2/sin2を格納しない
                # (RDS用cos3/sin3は継続)。無いとRDS有効時にブレンドが
                # 再上昇しモノラルボタンが実質無効になる。
                self._stereo_blend *= 0.9
                self.stereo_blend = self._stereo_blend
                self._last_cos2 = None
                self._last_sin2 = None
                if cos3 is not None:
                    self._last_cos3 = cos3
                    self._last_sin3 = sin3
                else:
                    # RDS無効時は57kHz搬送波を生成しないため古い値を破棄
                    self._last_cos3 = None
                    self._last_sin3 = None
                return

            target = 0.0
            if lock > 0.5 and pilot_rms > 1e-3:
                target = 1.0
            elif lock > 0.25 and 0.02 < ratio < 0.8:
                target = min(1.0, (ratio - 0.02) / 0.04)

            if target > self._stereo_blend:
                self._stereo_blend = min(target, self._stereo_blend
                                         + self._bm_attack_limit(lock))
                self._pilot_hold_n = 0
            elif target >= self._stereo_blend - 1e-9:
                # ロック定常の等値: 低下ではないため保持枠を消費しない
                self._stereo_blend = target
                self.stereo_blend = self._stereo_blend
                self._pilot_hold_n = 0
            else:
                self._blend_release(floor=target)

            self.stereo_blend = self._stereo_blend
            if self._stereo_blend > 0.02:
                self._last_cos2 = cos2
                self._last_sin2 = sin2
            if cos3 is not None:
                self._last_cos3 = cos3
                self._last_sin3 = sin3
            else:
                self._last_cos3 = None
                self._last_sin3 = None
        except Exception:
            self._last_cos2 = None
            self._last_sin2 = None
            self._last_cos3 = None
            self._last_sin3 = None
            # NaN固着防止: PLL積分状態もリセットし次ブロックで復帰可能にする
            self._pll_theta = 0.0
            self._pll_integ = 0.0
            self._pll_ef = 0.0
            try:
                if getattr(self, "pilot_tracker", None) is not None:
                    self.pilot_tracker.reset()
            except Exception:
                pass
            self._stereo_blend *= 0.9
            self.stereo_blend = self._stereo_blend

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

        # 超音波三角ノイズ比追従型 コグニティブ・オートスケルチ
        ultra_gain = 1.0
        if getattr(self, "ultra_squelch", None) is not None and self.ultra_squelch.enabled:
            ultra_gain, _ = self.ultra_squelch.process(demod)

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

        if ultra_gain < 0.999:
            audio = audio * ultra_gain

        return audio.astype(np.float32)

    def _apply_noise_expander(self, audio: np.ndarray, threshold: float = 0.09) -> np.ndarray:
        """弱電界時のFM三角雑音（高域ヒスノイズ）を抑え込み人の声を浮き彫りにするソフトエキスパンダー"""
        if len(audio) == 0:
            return audio
        mag = np.abs(audio)
        gain = np.where(mag < threshold, (mag / threshold) ** 0.5, 1.0)
        return (audio * gain).astype(np.float32)

    def _blank_impulses_iq(self, iq_if: np.ndarray) -> np.ndarray:
        """AM/短波用インパルスノイズブランカ (電源・イグニッション雑音対策)。
        変調包絡の中央値/MAD基準で孤立パルスだけを検出し、端点線形補間で消去する。
        変調ピーク (最大2倍) や選択性フェージングの谷には触れない。
        検出率2%超のブロックは信号とみなして無処理 (安全装置)。"""
        n = len(iq_if)
        if n < 64 or not self.impulse_blanker_enabled:
            return iq_if
        mag = np.abs(iq_if).astype(np.float32)
        # 統計は間引き＋partition直取り (np.medianはNaN検査経路で遅い。
        # 期待値同一のため検出性能不変。順序統計量単点で十分)
        sm = mag[::16] if n > 256 else mag
        k = len(sm) // 2
        med = float(np.partition(sm, k)[k])
        if med < 1e-9:
            return iq_if
        dev = np.abs(sm - med)
        mad = float(np.partition(dev, k)[k]) + 1e-12
        thr = max(med + 6.0 * mad, med * 2.5)
        mask = mag > thr
        if float(np.mean(mask)) > 0.02:
            return iq_if
        edges = np.diff(mask.astype(np.int8))
        starts = list(np.flatnonzero(edges == 1) + 1)
        ends = list(np.flatnonzero(edges == -1) + 1)
        if mask[0]:
            starts.insert(0, 0)
        if mask[-1]:
            ends.append(n)
        if not starts:
            return iq_if
        out = iq_if.copy()
        for s, e in zip(starts[:512], ends[:512]):
            if e - s > 48:
                continue  # 長い区間は信号として残す
            l = out[s - 1] if s > 0 else out[e]
            r = out[e] if e < n else l
            k = (e - s)
            # Smooth cosine interpolation prevents phase/envelope kinks
            w = 0.5 * (1.0 - np.cos(np.pi * np.arange(1, k + 1, dtype=np.float32) / (k + 1.0)))
            out[s:e] = (l * (1.0 - w) + r * w).astype(out.dtype)
        return out

    def demodulate_am(self, iq_if: np.ndarray) -> np.ndarray:
        if len(iq_if) == 0:
            return np.zeros(0, dtype=np.float32)

        iq_if = self._blank_impulses_iq(iq_if)
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
        # ハングタイマ: 単語間の息継ぎでゲインが跳ね上がる呼吸を防ぐため、
        # レベル低下後は約400ms(7ブロック)だけ減衰を保持してからリリースする。
        level = float(np.mean(np.abs(sig)))
        if self.am_agc_level <= 0.0:
            self.am_agc_level = max(level, 2e-4)
            self._am_agc_hang = 0
        elif level > self.am_agc_level:
            self.am_agc_level += 0.1 * (level - self.am_agc_level)
            self._am_agc_hang = 20
        elif self._am_agc_hang > 0:
            self._am_agc_hang -= 1
        else:
            self.am_agc_level += 0.005 * (level - self.am_agc_level)
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
        n = 1024
        if len(audio) >= n:
            seg = audio[-n:].astype(np.float32)
        else:
            seg = np.zeros(n, dtype=np.float32)
            seg[-len(audio):] = audio
        seg = seg - float(np.mean(seg))
        power = np.abs(np.fft.rfft(seg * self._nr_window)) ** 2 + 1e-12
        bin_hz = self.audio_rate / n

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

        # 180°逆相ロックガード: 同期出力が包絡線と逆符号なら反転させる。
        # 未対策だとDC反転→-1.0クリップの爆音歪みになる。
        try:
            if float(np.mean(out * np.abs(iq_if).astype(np.float32))) < 0.0:
                out = -out
        except Exception:
            pass

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
        # BFO位相 (音声ドメインで再シフト位相へ加算。USB/LSB/CW共通で正=ピッチ上昇)
        wb = 2.0 * np.pi * float(self.bfo_offset_hz) / self.audio_rate
        if mode == "LSB":
            wb = -wb
        phb = self._ssb_bfo_phase + wb * np.arange(n, dtype=np.float64)
        self._ssb_bfo_phase = float((phb[-1] + wb) % (2.0 * np.pi))
        shifted = (iq_48 * np.exp(-1j * ph)).astype(np.complex64)
        lp = self.decimate_with_history(shifted, taps, 1, attr)
        dly = (len(taps) - 1) // 2
        audio = np.real(lp * np.exp(1j * ((ph - omega * dly) + (phb - wb * dly)))).astype(np.float32)

        # AGC (SSBは搬送波が無いため平均振幅で正規化。無信号時の過剰増幅は3000倍で制限)
        # 無信号フロア (AM側と同型): ノイズ2400倍の爆音化を防ぐため滑らかにミュート。
        # ハングタイマ (AM側と同型、約400ms保持)。
        level = float(np.mean(np.abs(audio)))
        if self.ssb_agc_level <= 0.0:
            self.ssb_agc_level = max(level, 2e-4)
            self._ssb_agc_hang = 0
        elif level > self.ssb_agc_level:
            self.ssb_agc_level += 0.1 * (level - self.ssb_agc_level)
            self._ssb_agc_hang = 20
        elif self._ssb_agc_hang > 0:
            self._ssb_agc_hang -= 1
        else:
            self.ssb_agc_level += 0.005 * (level - self.ssb_agc_level)
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
            cur = np.ascontiguousarray(x, dtype=np.float32)
            for i, (b0, b1, m) in enumerate(self.deemph_sections):
                state = getattr(self, f"_deemph_state{ch}" if i == 0 else f"_deemph2_state{ch}")
                nxt = np.empty_like(cur)
                _NATIVE.sdr_bilinear_deemphasis(
                    _fptr(cur), _fptr(nxt), len(cur),
                    float(b0), float(b1), float(m),
                    _fptr(state))
                cur = nxt
            return cur
        y = np.empty_like(x)
        for i, (b0, b1, m) in enumerate(self.deemph_sections):
            inp = x if i == 0 else y
            out = np.empty_like(inp)
            x1 = getattr(self, f"deemph_x1{ch}" if i == 0 else f"deemph2_x1{ch}")
            y1 = getattr(self, f"deemph_y1{ch}" if i == 0 else f"deemph2_y1{ch}")
            for k in range(len(inp)):
                curr_x = inp[k]
                curr_y = b0 * curr_x + b1 * x1 + m * y1
                out[k] = curr_y
                x1 = curr_x
                y1 = curr_y
            setattr(self, f"deemph_x1{ch}" if i == 0 else f"deemph2_x1{ch}", x1)
            setattr(self, f"deemph_y1{ch}" if i == 0 else f"deemph2_y1{ch}", y1)
            y = out
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
        # 黒魔法CPU計測用 (master OFF時は取得しない)
        _bm_t0 = time.perf_counter() if getattr(self, "black_magic_enabled", False) else 0.0
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
        if getattr(self, "iq_corrector", None) is not None:
            iq = self.iq_corrector.process(iq)
        iq_shifted = self.mix_frequency(iq, mode=mode)
        spectrum_db = self.compute_spectrum(iq_shifted)

        # Hyper連続認知制御: フィルタを離散切替ではなく無段階モーフィング
        # WFM復調部では同一ブロックのスペクトルから局所SNRを取り、IF狭窄の
        # 過渡ショックだけを抑える（最終帯域＝Hyper目標のまま）。
        if mode == "WFM" and self.cognitive_enabled:
            self._update_cognitive_morph(
                self._wfm_if_snr_db(spectrum_db, self.rf_rate),
                mode="WFM",
            )
        else:
            self._update_cognitive_morph()

        if mode == "NFM":
            # ISS / アマチュア無線用ナローバンドFM
            iq_if = self.decimate_with_history(iq_shifted, self.fir_nfm, self.if_decim, "history_nfm")
        elif mode in ("AM", "AM_NARROW"):
            # 中波・短波放送用AM (狭帯域AMフィルタ対応)
            fir_target = self.fir_am_narrow if (mode == "AM_NARROW" or self.filter_mode == "narrow") else self.fir_am
            hist_attr = "history_am_narrow" if fir_target is self.fir_am_narrow else "history_am"
            iq_if = self.decimate_with_history(iq_shifted, fir_target, self.if_decim, hist_attr)
        elif mode in ("USB", "LSB", "CW"):
            # SSB / CW (HF用): 複素非対称バンドパスで側波帯を選択
            iq_if = self.decimate_with_history(iq_shifted, self.fir_am_narrow, self.if_decim, "history_ssb")
        else:
            # ワイドFM (WFM)
            if self.cognitive_enabled:
                fir_if_target = self._get_dynamic_filter("if", self.applied_if_bw_hz / 2.0)
            elif self.filter_mode == "narrow":
                fir_if_target = self.fir_if_narrow
            else:
                fir_if_target = self.fir_if
            iq_if = self.decimate(iq_shifted, fir_if_target, self.if_decim)

        # デジタル自己干渉消去器 (SIC: PC直挿し時のクロック・スイッチングビート逆位相消去)
        if getattr(self, "sic_enabled", False) and getattr(self, "sic_canceller", None) is not None:
            self._sic_detect_counter += 1
            # 選局直後、および約0.8秒ごと (約20ブロック) にスプリアス突出ピークを自動走査
            if self._sic_detect_counter % 20 == 1:
                self.sic_canceller.auto_detect_spurious(iq_if, n_fft=1024, prominence_db=12.0)
            iq_if = self.sic_canceller.process(iq_if)

        if mode == "NFM":
            audio = self.demodulate_nfm(iq_if)
        elif mode in ("AM", "AM_NARROW"):
            audio = self.demodulate_am(iq_if)
        elif mode in ("USB", "LSB", "CW"):
            iq_48 = self.decimate_with_history(iq_if, self.fir_if_audio, self.audio_decim, "history_ssb2")
            audio = self.demodulate_ssb(iq_48, mode)
        else:
            audio = self.demodulate_wfm(iq_if)

        # 黒魔法統合マネージャ: 安全パラメータの計算のみ (音声には触らない)。
        # 既定OFF時はゼロコストでスキップする。
        self._bm_params = None
        if getattr(self, "black_magic_enabled", False):
            try:
                if self.bm_controller is None and BlackMagicController is not None:
                    self.bm_controller = BlackMagicController(enabled=True)
                if self.bm_controller is not None:
                    _bm_cpu = 0.0
                    try:
                        _bm_dt = len(iq_if) / float(self.if_rate)
                        _bm_cpu = ((time.perf_counter() - _bm_t0)
                                   / max(_bm_dt, 1e-6) * 100.0)
                    except Exception:
                        pass
                    _bm_last = getattr(self, "_bm_last_rmt_info", None) or {}
                    _bm_deg = (float(_bm_last.get("hf_loss_db", 0.0)) < -6.0
                               or float(_bm_last.get("rms_diff_db", 0.0)) < -3.0)
                    _uq2 = getattr(self, "ultra_squelch", None)
                    self._bm_params = self.bm_controller.process_metrics({
                        "snr_db": float(getattr(self, "s_meter_dbfs", -45.0)) + 45.0,
                        "clipped": False,  # dspにADCクリップ旗なし
                        "pilot_confidence": float(getattr(self, "bm_cyclo_confidence",
                                                           0.0)),
                        "stereo_blend": float(getattr(self, "_stereo_blend", 0.0)),
                        "squelch_confidence": float(getattr(_uq2, "current_gain",
                                                             1.0)),
                        "cpu_percent": float(min(max(_bm_cpu, 0.0), 100.0)),
                        "audio_degraded": bool(_bm_deg),
                    })
                    if not self._bm_params.get("bypass_all", True):
                        self.bm_rmt_cap = float(self._bm_params.get("rmt_cap", 0.0))
            except Exception:
                self._bm_params = None

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
        audio_synced = self.resampler.process(audio)

        # 過渡クリックサプレッサーは選局直後300msのみ実行する。
        # クリック源はFIR履歴・PLL状態の不連続＝選局時のみであり、常時ONは
        # 打楽器アタック等の正規過渡を誤って削る (選局・モード切替は必ず
        # set_offset_freqを経由するため窓検出で十分)。
        if time.monotonic() - self._tune_monotonic < 0.30:
            if audio_synced.ndim == 2:
                audio_clean = np.stack([
                    suppress_click_transients(audio_synced[:, 0]),
                    suppress_click_transients(audio_synced[:, 1]),
                ], axis=1).astype(np.float32)
            else:
                audio_clean = suppress_click_transients(audio_synced)
        else:
            audio_clean = audio_synced

        # 音声/音楽 認知型オートチルトEQ (トーク了解度 / 音楽フラットHi-Fi 自動追従)
        # 解析はハイカット前の広帯域で (カット後だとrolloff>8500の音楽分岐に
        # 到達不能になる)。_post_process_wfmがタップしたwide_srcを使う。
        if self.cognitive_enabled and getattr(self, "cognitive_eq", None) is not None and self.cognitive_eq.enabled:
            wide = getattr(self, "_cog_wide", None)
            if mode != "WFM" or wide is None or len(wide) < 128:
                wide = audio_clean
            self.cognitive_eq.analyze(wide)
            audio_clean = self.cognitive_eq.process(audio_clean)

        # 局間音量レベリング用スローAGC (選局時の音量差を吸収。L/R連動で音像保存。
        # DCサーボ・ディザの前に置き、最終量子化に整形済みレベルが載るようにする)
        if self.slow_agc_enabled:
            audio_clean = self._slow_agc_level(audio_clean)

        # ===== 高級オーディオ (Accuphase / dCS 理論) 最終段 =====
        # 超低域位相回転ゼロ・アクティブDCサーボ (20Hz〜20kHzの位相を一切回転させず直流オフセットを相殺)
        if getattr(self, "dc_servo", None) is not None and self.dc_servo.enabled:
            audio_clean = self.dc_servo.process(audio_clean)

        # TPDFディザー & 音響心理ノイズシェーピング (微小信号の量子化高調波歪みを根絶)
        if getattr(self, "dither", None) is not None and self.dither.enabled:
            audio_clean = self.dither.process_float(audio_clean)

        # 黒魔法①: cyclo→スケルチ統合の最終適用 (既定OFF)。
        # スローAGCより後段に置く。前段だとAGCがミュートを持ち上げて
        # 無効化することを実測で確認 (ノイズRMS 0.45のまま)。
        # WFM以外では直前値を維持するが、選局・モード切替は必ず
        # set_offset_freqを経由し開に戻るため漏れない。
        try:
            _sq = float(getattr(self, "_bm_sq_gain", 1.0)) if mode == "WFM" else 1.0
        except Exception:
            _sq = 1.0
        if _sq < 0.999:
            try:
                audio_clean = (np.asarray(audio_clean) * _sq).astype(np.float32)
            except Exception:
                pass

        return audio_clean, spectrum_db

    @property
    def sic_cancellation_db(self) -> float:
        """SIC (デジタル自己干渉消去) による内部スプリアス消去量 (dB)"""
        if getattr(self, "sic_canceller", None) is not None:
            return float(self.sic_canceller.cancellation_db)
        return 0.0

    @property
    def sic_detected_spurious(self) -> list[float]:
        """SIC が検出・追従中の内部スプリアス周波数リスト (Hz, IFオフセット)"""
        if getattr(self, "sic_canceller", None) is not None:
            return list(self.sic_canceller.spurious_freqs)
        return []

