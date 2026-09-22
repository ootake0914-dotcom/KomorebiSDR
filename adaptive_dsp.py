"""
Adaptive DSP Tracking Modules for KomorebiSDR.
古典数学（グラム・シュミット直交化・カーソン則等）に基づく
リアルタイム信号適応トラッキングモジュール群。

NOTE: 各クラスの正準は機能別モジュールへ移動済み。
本ファイルは後方互換のための再エクスポート・ファサードである。
新規コードは `adaptive_rf / adaptive_demod / adaptive_stereo / adaptive_audio`
から直接 import すること。

- adaptive_rf: RFフロントエンド系 (IQ補正・スケルチ・自己干渉消去)
- adaptive_demod: FM復調系 (EKF・パイロット・位相スリップ抑制・クリック補修)
- adaptive_stereo: ステレオ・副搬送波系 (MPX・BSS・直交位相補正)
- adaptive_audio: オーディオ帯域系 (音声音楽判別・倍音補完・SVDノイズ除去・スペクトル抑圧)
"""

from adaptive_rf import (
    AdaptiveIqCorrector,
    UltrasonicSquelchTracker,
    DigitalSelfInterferenceCanceller,
)
from adaptive_demod import (
    DeepSpaceEkfDemodulator,
    KalmanPilotTracker,
    RiemannianTopologicalDemodulator,
    TopologicalClickSuppressor,
)
from adaptive_stereo import (
    QuadratureMpxCanceller,
    SuperSpatialBssStereoSeparator,
    QuaternionMpxDecoupler,
)
from adaptive_audio import (
    CognitiveSpeechMusicTracker,
    HolographicAudioEnhancer,
    RmtHankelDenoiser,
    MonoNoiseSuppressor,
)

__all__ = [
    "AdaptiveIqCorrector",
    "UltrasonicSquelchTracker",
    "CognitiveSpeechMusicTracker",
    "QuadratureMpxCanceller",
    "DeepSpaceEkfDemodulator",
    "KalmanPilotTracker",
    "HolographicAudioEnhancer",
    "RiemannianTopologicalDemodulator",
    "SuperSpatialBssStereoSeparator",
    "RmtHankelDenoiser",
    "MonoNoiseSuppressor",
    "TopologicalClickSuppressor",
    "QuaternionMpxDecoupler",
    "DigitalSelfInterferenceCanceller",
]
