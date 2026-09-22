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
    assert out["adaptive_notch"]["enabled"] is False  # 既定維持
    assert out["adaptive_notch"]["max_harmonic"] == 5  # 未指定は既定
    assert out["adaptive_notch"]["base_hz"] == 0.0
    clamped = _clean_black_magic({"adaptive_notch": {"enabled": True, "max_harmonic": 99,
                                                     "base_hz": 500.0, "line_on_db": -5.0}})
    assert clamped["adaptive_notch"]["enabled"] is True
    assert clamped["adaptive_notch"]["max_harmonic"] == 9  # 上限クランプ
    assert clamped["adaptive_notch"]["base_hz"] == 100.0  # 上限クランプ
    assert clamped["adaptive_notch"]["line_on_db"] == 0.0  # 下限クランプ
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


def test_attack_limit_ratchet_guard():    # ラチェット対策: flutter中は介入せず0.25、安定弱信号でのみ鈍化
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


def test_profile_and_describe():
    from black_magic import profile_for
    wfm_weak = profile_for("WFM", 5.0)
    assert wfm_weak["cyclo"] is True and wfm_weak["rmt"] is True
    assert wfm_weak["reason"] == "weak-signal"
    wfm_strong = profile_for("WFM", 40.0)
    assert wfm_strong["rmt"] is False and wfm_strong["sr"] is False
    assert wfm_strong["cyclo"] is True  # 検出補助のみ残る
    assert profile_for("AM", 5.0)["cyclo"] is False  # 非FMでcycloなし
    assert profile_for("CW", 5.0)["rmt"] is False
    unk = profile_for("NOPE", 5.0)
    assert unk["reason"] == "unknown-mode" and not any(
        unk[k] for k in ("cyclo", "rmt", "sr", "notch"))
    c = BlackMagicController(enabled=True)
    d0 = c.describe()
    assert d0["enabled"] is True and d0["reason"] == "disabled"
    c.process_metrics({"snr_db": 0.0, "pilot_confidence": 0.5})
    d1 = c.describe()
    assert d1["cyclo"] == "active" and d1["sr"] == "active"
    assert d1["blocks"] == 1 and isinstance(d1["snr_db"], float)


def test_squelch_assist_hysteresis():
    # 二基準ヒステリシス: ノイズで閉じ、弱局・強モノラルで開く。既定OFFは素通し
    from dsp import SdrDspPipeline
    dsp = SdrDspPipeline(1152000, 48000)
    assert dsp._bm_update_squelch_assist() == 1.0  # 既定OFF
    dsp.black_magic_enabled = True
    dsp.bm_sq_assist_enabled = True
    # 局間ノイズ (conf低・S低) →閉じる
    dsp.bm_cyclo_confidence = 0.1
    dsp.s_meter_dbfs = -60.0
    for _ in range(30):
        g = dsp._bm_update_squelch_assist()
    assert dsp._bm_sq_open is False and g == 0.0
    # 弱ステレオ局 (conf高・S中) →開く。ただし最小閉保持 (20blk) のため
    # 開き直しに約1秒かかる (一過性信号でのバースト防止。スケルチの常識)
    dsp.bm_cyclo_confidence = 0.9
    dsp.s_meter_dbfs = -27.0
    for _ in range(25):
        g = dsp._bm_update_squelch_assist()
    assert dsp._bm_sq_open is True and g == 1.0
    # 強モノラル局 (conf=0だがS高) →開のまま (誤ミュート防止)
    dsp.bm_cyclo_confidence = 0.0
    dsp.s_meter_dbfs = -10.0
    for _ in range(10):
        g = dsp._bm_update_squelch_assist()
    assert dsp._bm_sq_open is True and g == 1.0
    # 中間帯の往復でチャタらない (保持)
    dsp.bm_cyclo_confidence = 0.65
    dsp.s_meter_dbfs = -45.0
    states = set()
    for _ in range(20):
        dsp._bm_update_squelch_assist()
        states.add(dsp._bm_sq_open)
    assert len(states) == 1, f"中間帯でchatter: {states}"
    # 非有限→現状維持 (跳ばない)
    dsp.bm_cyclo_confidence = float("nan")
    assert dsp._bm_update_squelch_assist() == 1.0


def test_squelch_assist_pipeline():
    # 実パイプライン: モノラル強局はミュートされない／ノイズは消える
    from dsp import SdrDspPipeline
    from test_stereo import make_raw, BLOCK, NBLK
    raw_mono = make_raw(False)
    outs = {}
    for assist in (False, True):
        dsp = SdrDspPipeline(1152000, 48000)
        dsp.set_offset_freq(0.0)
        dsp.afc_enabled = False
        dsp.cognitive_enabled = False
        dsp.black_magic_enabled = assist
        dsp.bm_sq_assist_enabled = assist
        chunks = []
        for k in range(6):
            audio, _ = dsp.process(raw_mono[k * BLOCK:(k + 1) * BLOCK], mode="WFM")
            chunks.append(np.asarray(audio).reshape(-1))
        outs[assist] = np.concatenate(chunks)
    rms_off = float(np.sqrt(np.mean(outs[False] ** 2)))
    rms_on = float(np.mean(np.abs(outs[True])))
    assert rms_on > 0.3 * rms_off, f"モノラル強局がミュートされた: {rms_on:.4f}/{rms_off:.4f}"
    # ノイズraw (冷たい局間相当・S約-40dBFS) は消える。
    # 閉は0.05/blockのスローなため25ブロック回す (約1.4秒で無音化。急な
    # ミュートでアタックを切らない設計)。
    rng = np.random.default_rng(0)
    raw_n = rng.integers(125, 131, BLOCK * 25).astype(np.uint8)
    dsp = SdrDspPipeline(1152000, 48000)
    dsp.set_offset_freq(0.0)
    dsp.afc_enabled = False
    dsp.cognitive_enabled = False
    dsp.black_magic_enabled = True
    dsp.bm_sq_assist_enabled = True
    chunks = []
    for k in range(25):
        audio, _ = dsp.process(raw_n[k * BLOCK:(k + 1) * BLOCK], mode="WFM")
        chunks.append(np.asarray(audio).reshape(-1))
    tail = np.concatenate(chunks)[-2752 * 5:]
    rms_n = float(np.sqrt(np.mean(tail ** 2)))
    assert rms_n < 0.05, f"ノイズが消えない: {rms_n:.4f}"


def test_squelch_assist_config():
    bm = DEFAULT_CONFIG["black_magic"]["squelch_assist"]
    assert bm["enabled"] is False
    assert bm["open_conf"] == 0.75 and bm["close_conf"] == 0.55
    assert bm["min_close_blocks"] == 20
    assert DEFAULT_CONFIG["black_magic"]["seeking"]["enabled"] is False
    out = _clean_black_magic({"squelch_assist": {"enabled": True, "open_conf": 5.0,
                                                 "close_smeter_db": 99.0,
                                                 "min_close_blocks": 999},
                              "seeking": {"enabled": True}})
    assert out["squelch_assist"]["enabled"] is True
    assert out["squelch_assist"]["open_conf"] == 1.0
    assert out["squelch_assist"]["close_smeter_db"] == 0.0
    assert out["squelch_assist"]["min_close_blocks"] == 200
    assert out["seeking"]["enabled"] is True


def test_squelch_min_close_hold():    # 深フェード呼吸: 最小閉保持で遷移が減り、ミュートが決定的になる
    from dsp import SdrDspPipeline
    for hold, max_tr in ((0, 99), (20, 6)):
        dsp = SdrDspPipeline(1152000, 48000)
        dsp.black_magic_enabled = True
        dsp.bm_sq_assist_enabled = True
        dsp.bm_sq_min_close_blocks = hold
        states = []
        gains = []
        for _ in range(6):
            for conf in [0.4] * 8 + [0.8] * 8:
                dsp.bm_cyclo_confidence = conf
                dsp.s_meter_dbfs = -27.0
                gains.append(dsp._bm_update_squelch_assist())
                states.append(dsp._bm_sq_open)
        tr = sum(abs(int(b) - int(a)) for a, b in zip(states, states[1:]))
        assert tr <= max_tr, f"hold={hold}で遷移{tr}"
    assert min(gains) < 0.5, "閉保持中にgainが落ちない"


def test_quality_and_seeker():
    from black_magic import quality_score, ExtremumSeeker
    # Q: 良条件ほど高く、chatter/damage/CPUで減点。非有限は安全側
    q_good = quality_score(conf=0.9, blend=1.0)
    q_bad = quality_score(conf=0.1, blend=0.0, chatter_event=True,
                          cpu_percent=90.0, hf_loss_db=-8.0)
    assert q_good > q_bad
    assert quality_score(conf=float("nan")) <= q_good
    assert quality_score(hf_loss_db=float("inf")) <= q_good
    # Seeker: 放物線Qで最適点 (offset +0.1) に収束する
    s = ExtremumSeeker(lo=-0.2, hi=0.2, step=0.05, eval_blocks=4,
                       freeze_rounds=3)
    for _ in range(200):
        off = s.update(-((s.offset - 0.1) ** 2) + 1.0, 0.0)
    assert abs(s.offset - 0.1) < 0.03, f"未収束: {s.offset}"
    assert s.frozen is True  # 改善停止で凍結
    # 範囲外に出ない
    s2 = ExtremumSeeker(lo=-0.2, hi=0.2, step=0.05, eval_blocks=2)
    for _ in range(60):
        s2.update(10.0, 0.0)  # 単調増加でも上限止まり
    assert -0.2 <= s2.offset <= 0.2
    # 環境変化で再開
    assert s.frozen is True
    s.update(0.0, 20.0)
    assert s.frozen is False
    # 非有限入力で落ちない
    s.update(float("nan"), 0.0)


def test_controller_seek_integration():
    # seek無効時は従来通り (既存挙動の保護)
    c = BlackMagicController(enabled=True)
    outs = [c.process_metrics({"snr_db": 0.0, "pilot_confidence": 0.5})
            for _ in range(5)]
    assert all("q_score" in o for o in outs)
    base_caps = [o["rmt_cap"] for o in outs]
    c2 = BlackMagicController(enabled=True, seek_enabled=True)
    seen = [c2.process_metrics({"snr_db": 0.0, "pilot_confidence": 0.5})
            for _ in range(40)]
    caps = [o["rmt_cap"] for o in seen]
    assert all(0.0 <= v <= 0.85 for v in caps)  # 範囲遵守
    assert seen[-1]["q_score"] is not None
    c2.reset()
    assert c2.seeker.offset == 0.0 and c2.seeker.frozen is False


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
        test_profile_and_describe()
        print("[*] プロファイル・状態表示 OK")
        test_squelch_assist_hysteresis()
        print("[*] スケルチ統合ヒステリシス OK")
        test_squelch_assist_pipeline()
        print("[*] スケルチ実経路 OK")
        test_squelch_assist_config()
        print("[*] スケルチ設定 OK")
        test_squelch_min_close_hold()
        print("[*] 最小閉保持 OK")
        test_quality_and_seeker()
        print("[*] Q関数・収束 OK")
        test_controller_seek_integration()
        print("[*] 収束統合 OK")
    except AssertionError as e:
        print(f"FAILED: {e}")
        return 1
    print("\nALL BLACK MAGIC TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
