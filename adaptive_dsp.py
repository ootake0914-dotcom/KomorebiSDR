"""
Adaptive DSP Tracking Modules for Antigravity SDR Radio.
特許フリーの古典数学（グラム・シュミット直交化・カーソン則）に基づく
自律型リアルタイム信号適応トラッキングモジュール。

他の作業員の既存DSPバグ修正と競合しない独立モジュールとして設計。

NOTE (リファクタ): 各クラスの正準は機能別モジュールへ移動済み。
本ファイルは後方互換のための再エクスポート・ファサードである。
新規コードは `adaptive_rf / adaptive_demod / adaptive_stereo / adaptive_audio / adaptive_multipath`
から直接 import すること。

- adaptive_rf: RFフロントエンド系 (IQ補正・IF帯域・スケルチ)
- adaptive_demod: FM復調系 (EKF・パイロット・ターボ・リーマン・ビタビ・シンプレクティック)
- adaptive_stereo: ステレオ・副搬送波系 (MPX・BSS・スパース抽出)
- adaptive_audio: オーディオ帯域系 (音声音楽判別・倍音外挿・RMT)
- adaptive_multipath: マルチパス等化系 (部分空間・最適輸送)
"""

from adaptive_rf import (
    AdaptiveIqCorrector,
    DynamicIfBandwidthTracker,
    UltrasonicSquelchTracker,
    CyclostationaryFeatureDetector,
    DigitalSelfInterferenceCanceller,
)
from adaptive_demod import (
    DeepSpaceEkfDemodulator,
    KalmanPilotTracker,
    TimeReversalTurboEqualizer,
    RiemannianTopologicalDemodulator,
    ViterbiPhaseDemodulator,
    SymplecticHamiltonianDemodulator,
    TopologicalClickSuppressor,
)
from adaptive_stereo import (
    QuadratureMpxCanceller,
    SuperSpatialBssStereoSeparator,
    SparseSubcarrierExtractor,
    QuaternionMpxDecoupler,
    BistableStochasticResonator,
)
from adaptive_audio import (
    CognitiveSpeechMusicTracker,
    HolographicAudioEnhancer,
    RmtHankelDenoiser,
    FractionalDeemphasis,
    WaveletNoiseShrinkage,
)
from adaptive_multipath import (
    SubspaceMultipathEqualizer,
    WassersteinMultipathEqualizer,
)

__all__ = [
    "AdaptiveIqCorrector",
    "DynamicIfBandwidthTracker",
    "UltrasonicSquelchTracker",
    "CognitiveSpeechMusicTracker",
    "QuadratureMpxCanceller",
    "DeepSpaceEkfDemodulator",
    "KalmanPilotTracker",
    "TimeReversalTurboEqualizer",
    "HolographicAudioEnhancer",
    "RiemannianTopologicalDemodulator",
    "ViterbiPhaseDemodulator",
    "SuperSpatialBssStereoSeparator",
    "SubspaceMultipathEqualizer",
    "RmtHankelDenoiser",
    "WassersteinMultipathEqualizer",
    "SymplecticHamiltonianDemodulator",
    "SparseSubcarrierExtractor",
    "FractionalDeemphasis",
    "TopologicalClickSuppressor",
    "QuaternionMpxDecoupler",
    "BistableStochasticResonator",
    "CyclostationaryFeatureDetector",
    "DigitalSelfInterferenceCanceller",
    "WaveletNoiseShrinkage",
]
