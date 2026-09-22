"""
Digital Signal Processing (DSP) Module for SDR - Ultra-Clear Hi-Fi Edition.
歪み・クリッピング・19kHzパイロットトーン・ヒスノイズを抑えた高音質DSPパイプライン。

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


# 復調器のmixin分割 (純粋移動・動作同一)。各ファイルに等価ハッシュで検証。
from dsp_bm import DspBlackMagicMixin
from dsp_am import DspAmMixin
from dsp_nfm import DspNfmMixin
from dsp_wfm import DspWfmMixin


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


class SdrDspPipeline(DspBlackMagicMixin, DspAmMixin, DspNfmMixin,
                       DspWfmMixin):
    """低ノイズ SDR 信号処理パイプライン"""

    def __init__(self, sample_rate: int = 1152000, audio_rate: int = 48000):
        self.rf_rate = sample_rate
        self.audio_rate = audio_rate
        # 選局リセットとDSP処理の同時実行によるtorn history防止用
        self._state_lock = threading.Lock()

        self.if_decim = 4
        self.if_rate = self.rf_rate // self.if_decim  # 288 kHz
        self.audio_decim = self.if_rate // self.audio_rate  # 6
        self.total_decim = self.if_decim * self.audio_decim  # 24

        # 端数IQサンプル持ち越し用バッファ (時間軸断絶・クリック音の抑止)
        self.raw_leftover = np.empty(0, dtype=np.uint8)

        self.offset_freq = 0.0
        self.mixer_phase = 0.0
        self._tune_monotonic = time.monotonic()

        # グラム・シュミット直交化によるリアルタイム適応IQインバランス補正器 (鏡像ゴースト自動消去)
        self.iq_corrector = AdaptiveIqCorrector(sample_rate=self.rf_rate, time_constant_sec=3.0)

        # 全二重通信理論 (IBFD) デジタル自己干渉消去器 (SIC: PC直挿し時のクロック・スイッチングビート消去)
        # IF段 (288kHz) でスプリアスを自動検出し、NLMS直交基底追従で逆位相ノッチ消去。
        # 延長ケーブル使用時などスプリアスが存在しない場合は素通し (相関0.999+) となり副作用は小さい。
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

        # 19kHzパイロットトーンを抑えるIF段オーディオフィルタ (288kHzレート)
        # カットオフ 15kHz, 19kHzで -60dB以上の急峻減衰 (257タップ。
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

        # 5.5kHz: DXボイスフィルタ (微弱局のノイズフロアを抑え声の明瞭度を上げる, 65タップ)
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

        # 適応型分数リサンプラ (SDRとサウンドカードのクロックドリフトを微小補正し音飛び抑制)
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
        self.cognitive_alpha = 0.18  # 1フレームあたりの平滑追従率 (ポップノイズ抑制)
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
        # 低ジッターPLL設計 (fn=16Hz, ζ=0.85, ループフィルタ遮断 20Hz)
        # 従来の過大帯域(205Hz)による低音変調漏れ・位相揺らぎ・定位のあるノイズを抑制
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
        self.bm_sq_min_close_blocks = 20
        self.bm_seek_enabled = False
        self._bm_sq_open = True  # 起動時は開 (いきなりミュートしない)
        self._bm_sq_gain = 1.0
        self._bm_sq_hold = 0
        self.cyclo_detector = None
        self.bm_rmt = None
        self.bm_sr = None
        self.bm_notch = None
        self.bm_controller = None
        self.bm_cyclo_min_conf = 0.55
        self.bm_cyclo_confidence = 0.0
        self.bm_sr_confidence = 0.0
        self.bm_rmt_cap = 0.65
        # main.pyから渡される黒魔法パラメータ (configのblack_magic節)。
        # 遅延生成インスタンスのコンストラクタに反映する。既定は空=内蔵既定。
        self.bm_cfg = {}
        # flutter検出用lock履歴 (直近32ブロック) と判定閾値
        self._bm_lock_hist = deque(maxlen=32)
        self.bm_flutter_std = 0.15
        self._bm_params = None
        self._bm_last_rmt_info = None

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
        # ステレオ2ch同期 適応ドリフトリサンプラ (後方互換参照)
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
        # ほぼゼロ(0.000000dB)になる。
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
        # 周波数依存ブレンド用クロスオーバー状態 (1次相補: lo + hi = diff で再構成。
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
        # 拡張カルマンフィルタ (EKF) FM復調エンジン
        self.ekf_demod = DeepSpaceEkfDemodulator(sample_rate=self.if_rate)
        self.ekf_enabled = True

        # ===== オーディオ統合モジュール =====
        # 位相回転の少ないDCサーボ (20Hz〜300Hzの低音位相進み歪みの抑制)
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
        self._bm_sq_hold = 0
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
        self._bm_sq_hold = 0
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
        高級オーディオ処理の動的設定。
        :param apodizing: 最小位相アポダイジングフィルタ (インパルス応答のプリリンギング低減)
        :param dc_servo: 位相回転の少ないDCサーボ (20Hz〜300Hzの低域位相歪み抑制)
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
        """離散切替ではなくサンプル単位で滑らかにフィルタを変形 (クリック・ポップ抑制)

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
        """線形位相クロスオーバーにより高域ヒスノイズのみを連続可変減衰 (低域は素通し)"""
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
        """過去サンプルを保持したシームレスなFIR畳み込み & デシメーション (任意のフィルタ長に動的同期)"""
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

        # ポリフェーズ間引き (valid畳み込み[::factor]と等価、計算量1/factor)
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

        # 畳み込み (validモードで境界アーティファクトを排除)
        if np.iscomplexobj(x_ext):
            r = self._convolve_valid(x_ext.real, fir_taps)
            i = self._convolve_valid(x_ext.imag, fir_taps)
            filtered = r + 1j * i
        else:
            filtered = self._convolve_valid(x_ext, fir_taps)

        return filtered[::factor]

    def decimate(self, x: np.ndarray, fir_taps: np.ndarray, factor: int) -> np.ndarray:
        return self.decimate_with_history(x, fir_taps, factor, "history_if")

    # WFM-CMA等化器は dsp_wfm.py (DspWfmMixin) へ移動 (純粋移動・動作同一)。

    def _apply_hard_limiter(self, iq_if: np.ndarray) -> np.ndarray:
        mag = np.abs(iq_if)
        mask = (mag > 1e-12) & np.isfinite(mag)
        out = np.zeros_like(iq_if)
        out[mask] = iq_if[mask] / mag[mask]
        return out

    def _apply_dc_highpass(self, audio: np.ndarray, ch: str = "") -> np.ndarray:
        """30Hz以下の不要な直流・ボコボコ音をカット (Cコア: GIL解放で並行実行)"""
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

        # WFM復調は dsp_wfm

    # 黒魔法配線は dsp_bm.py (DspBlackMagicMixin) へ移動 (純粋移動・動作同一)。

        # WFMパイロットPLLは dsp_wfm

    # NFM/SSB復調は dsp_nfm.py (DspNfmMixin) へ移動 (純粋移動・動作同一)。

        # WFMエキスパンダーは dsp_wfm

    # AM復調は dsp_am.py (DspAmMixin) へ移動 (純粋移動・動作同一)。

    # SSB/CW復調は dsp_nfm.py (DspNfmMixin) へ移動 (純粋移動・動作同一)。

        # WFMデエンファシスは dsp_wfm

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

        # 48バイト (24 IQサンプル = IFデシメーション4 × オーディオデシメーション6) の整数倍に切り分ける
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
                    self.bm_controller.seek_enabled = bool(
                        getattr(self, "bm_seek_enabled", False))
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
                        "hf_loss_db": float(_bm_last.get("hf_loss_db", 0.0)),
                        "rms_diff_db": float(_bm_last.get("rms_diff_db", 0.0)),
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

        # 適応型分数リサンプラ (独立クロック間のドリフトを微小補正し連続再生)
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

        # ===== オーディオ最終段 =====
        # 位相回転の少ないDCサーボ (20Hz〜20kHzの位相変化を抑えつつ直流オフセットを除去)
        if getattr(self, "dc_servo", None) is not None and self.dc_servo.enabled:
            audio_clean = self.dc_servo.process(audio_clean)

        # TPDFディザー & 音響心理ノイズシェーピング (微小信号の量子化高調波歪みを抑制)
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

