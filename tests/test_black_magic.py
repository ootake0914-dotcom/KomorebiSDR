"""統合コントローラ＋設定＋全OFF同一性のテスト (合成信号のみ・実機不要)。"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from black_magic import BlackMagicController
from config import DEFAULT_CONFIG, _clean_black_magic


def test_disabled_bypass():
    c = BlackMagicController(enabled=False)
    out = c.process_metrics({"snr_db": 0.0})
    assert out["bypass_all"] is True
    assert out["rmt_cap"] == 0.0 and out["sr_active"] is False


def test_failsafe_nonfinite():
    c = BlackMagicController(enabled=True)
    out = c.process_metrics({"snr_db": float("nan")})
    assert out["bypass_all"] is True
    assert "failsafe" in out["reason"]
    out2 = c.process_metrics("not-a-dict")
    assert out2["bypass_all"] is True


def test_strong_signal_reduces():
    c = BlackMagicController(enabled=True)
    out = None
    # SNR平滑の収束まで回す (EMA初回は安全側0始まりのため)
    for _ in range(30):
        out = c.process_metrics({"snr_db": 40.0, "pilot_confidence": 0.9,
                                 "cpu_percent": 5.0})
    assert out["bypass_all"] is False
    assert out["sr_active"] is False
    assert out["rmt_cap"] <= 0.06


def test_weak_signal_assists():
    c = BlackMagicController(enabled=True)
    out = None
    for _ in range(5):
        out = c.process_metrics({"snr_db": 0.0, "pilot_confidence": 0.5,
                                 "cpu_percent": 5.0})
    assert out["sr_active"] is True
    assert out["cyclo_active"] is True
    assert out["rmt_cap"] > 0.3


def test_cpu_sheds_rmt_first():
    c = BlackMagicController(enabled=True)
    base = None
    for _ in range(5):
        base = c.process_metrics({"snr_db": 0.0, "pilot_confidence": 0.5,
                                  "cpu_percent": 5.0})
    hot = None
    # CPU平滑がcpu_max(90)を超えるまで十分回す (EMAのため)
    for _ in range(25):
        hot = c.process_metrics({"snr_db": 0.0, "pilot_confidence": 0.5,
                                 "cpu_percent": 95.0})
    assert hot["rmt_cap"] < base["rmt_cap"] * 0.5
    assert hot["reason"] == "cpu-max"


def test_degraded_latch_and_recovery():
    c = BlackMagicController(enabled=True)
    out = c.process_metrics({"snr_db": 20.0, "audio_degraded": True})
    assert out["rmt_cap"] == 0.0
    assert out["reason"] == "audio-degraded"
    # 劣化フラグが消えても弱信号中は回復しない
    for _ in range(10):
        out = c.process_metrics({"snr_db": 0.0, "audio_degraded": False})
    assert out["rmt_cap"] == 0.0
    # 非弱信号で8ブロック連続クリア→回復 (SNR平滑の遅れを考慮して多めに回す)
    for _ in range(20):
        out = c.process_metrics({"snr_db": 20.0, "audio_degraded": False})
    assert out["rmt_cap"] > 0.0
    # resetは即時解除
    c.process_metrics({"snr_db": 20.0, "audio_degraded": True})
    c.reset()
    out = c.process_metrics({"snr_db": 20.0, "audio_degraded": False})
    assert out["rmt_cap"] > 0.0


def test_params_smoothed():
    c = BlackMagicController(enabled=True, smooth_alpha=0.2)
    caps = [c.process_metrics({"snr_db": 0.0, "pilot_confidence": 0.5})["rmt_cap"]
            for _ in range(6)]
    # EMAで単調に目標へ近づく (急変しない)
    assert all(b >= a for a, b in zip(caps, caps[1:]))
    assert caps[-1] - caps[0] > 0.1
    assert caps[1] - caps[0] < caps[-1] - caps[0]


def test_config_defaults_and_clamp():
    bm = DEFAULT_CONFIG["black_magic"]
    assert bm["enabled"] is False
    assert bm["rmt_denoiser"]["enabled"] is False
    assert bm["stochastic_resonance"]["enabled"] is False
    assert bm["stochastic_resonance"]["detector_only"] is True
    bad = {"enabled": True,
           "cyclostationary": {"enabled": "yes", "min_confidence": 5.0},
           "rmt_denoiser": {"max_strength": 99.0, "trials": 1},
           "stochastic_resonance": {"detector_only": False, "trials": 100,
                                    "unknown_key": 1},
           "unknown_section": {}}
    out = _clean_black_magic(bad)
    assert out["enabled"] is True
    assert out["cyclostationary"]["enabled"] is True  # 型不一致は既定維持
    assert out["cyclostationary"]["min_confidence"] == 1.0  # クランプ
    assert out["rmt_denoiser"]["max_strength"] == 1.0
    assert out["stochastic_resonance"]["detector_only"] is True  # False拒否
    assert out["stochastic_resonance"]["trials"] == 16  # クランプ
    assert "unknown_key" not in out["stochastic_resonance"]
    assert "unknown_section" not in out
    assert _clean_black_magic(None)["enabled"] is False


def test_all_off_matches_baseline():
    # D統合: 全機能OFFで既存経路と同一出力 (dsp既定フラグがOFFであることの検証)
    from dsp import SdrDspPipeline
    from test_stereo import make_raw, BLOCK, NBLK
    raw = make_raw(True, snr_db=20.0)
    outs = []
    for _ in range(2):
        dsp = SdrDspPipeline(1152000, 48000)
        dsp.set_offset_freq(0.0)
        dsp.afc_enabled = False
        dsp.cognitive_enabled = False
        assert dsp.black_magic_enabled is False
        chunks = []
        for k in range(NBLK):
            audio, _ = dsp.process(raw[k * BLOCK:(k + 1) * BLOCK], mode="WFM")
            chunks.append(np.asarray(audio).reshape(-1))
        outs.append(np.concatenate(chunks))
    assert np.array_equal(outs[0], outs[1]), "既定OFFで出力が変動した"


def test_attack_limit_ratchet_guard():
    # ラチェット対策: flutter中は介入せず0.25、安定弱信号でのみ鈍化
    from dsp import SdrDspPipeline
    from collections import deque
    dsp = SdrDspPipeline(1152000, 48000)
    # 既定OFFでは常に従来レート
    assert dsp._bm_attack_limit(0.0) == 0.25
    dsp.black_magic_enabled = True
    dsp.bm_cyclo_enabled = True
    # 安定弱信号 (lock低・conf低・履歴平坦) →鈍化
    dsp._bm_lock_hist = deque([0.1] * 16, maxlen=32)
    dsp.bm_cyclo_confidence = 0.0
    assert dsp._bm_attack_limit(0.1) == 0.05
    # flutter (lock分散大) →非介入
    dsp._bm_lock_hist = deque(([0.0, 0.8] * 8), maxlen=32)
    assert dsp._bm_attack_limit(0.1) == 0.25
    # lock強→非介入
    dsp._bm_lock_hist = deque([0.1] * 16, maxlen=32)
    assert dsp._bm_attack_limit(0.8) == 0.25
    # conf高→非介入
    dsp.bm_cyclo_confidence = 0.9
    assert dsp._bm_attack_limit(0.1) == 0.25
    # コントローラ指定値は尊重
    dsp.bm_cyclo_confidence = 0.0
    dsp._bm_params = {"blend_attack_limit": 0.1}
    assert dsp._bm_attack_limit(0.1) == 0.1
    # 非有限lock→安全側0.25
    assert dsp._bm_attack_limit(float("nan")) == 0.25


def main() -> int:
    try:
        test_disabled_bypass()
        print("[*] 無効時バイパス OK")
        test_failsafe_nonfinite()
        print("[*] 非有限フェイルセーフ OK")
        test_strong_signal_reduces()
        print("[*] 強信号削減 OK")
        test_weak_signal_assists()
        print("[*] 弱信号補助 OK")
        test_cpu_sheds_rmt_first()
        print("[*] CPU逼迫時RMT削減 OK")
        test_degraded_latch_and_recovery()
        print("[*] 劣化ラッチ/回復 OK")
        test_params_smoothed()
        print("[*] パラメータ平滑 OK")
        test_config_defaults_and_clamp()
        print("[*] 設定既定/検証 OK")
        test_all_off_matches_baseline()
        print("[*] 全OFF同一性 OK")
        test_attack_limit_ratchet_guard()
        print("[*] ラチェット防止 OK")
    except AssertionError as e:
        print(f"FAILED: {e}")
        return 1
    print("\nALL BLACK MAGIC TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
