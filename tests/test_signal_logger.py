"""Signal health logger tests (no hardware required)."""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_siglog_test.csv")


def _clean():
    for p in (TMP, TMP + ".1"):
        try:
            os.remove(p)
        except OSError:
            pass


def test_log_and_load():
    print("===== test_log_and_load =====")
    from signal_logger import SignalLogger, load_rows, COLUMNS
    _clean()
    lg = SignalLogger(path=TMP, max_bytes=10 ** 9)
    assert lg.log({"freq_hz": 80000000, "mode": "WFM", "blend": 1.0})
    assert lg.log({"freq_hz": 80000000, "mode": "WFM", "blend": 0.2})
    rows = load_rows(TMP)
    assert len(rows) == 2, f"want 2 rows, got {len(rows)}"
    assert rows[0]["freq_hz"] == 80000000
    assert set(COLUMNS) <= set(rows[0]) | {"ts"}
    with open(TMP, encoding="utf-8") as f:
        assert f.readline().strip().split(",")[0] == "ts"
    print("[OK] log and load")


def test_rotation():
    print("===== test_rotation =====")
    from signal_logger import SignalLogger
    _clean()
    lg = SignalLogger(path=TMP, max_bytes=300)
    for i in range(20):
        assert lg.log({"freq_hz": 80000000 + i, "mode": "WFM", "blend": 0.5})
    assert os.path.exists(TMP + ".1"), "rotation backup missing"
    assert os.path.getsize(TMP) < 2000
    print("[OK] rotation")


def test_summarize_finds_blend_hunting():
    print("===== test_summarize_finds_blend_hunting =====")
    import numpy as np
    from signal_logger import SignalLogger, summarize
    _clean()
    lg = SignalLogger(path=TMP, max_bytes=10 ** 9)
    rng = np.random.default_rng(0)
    # ブレンドだけが大きく往復する合成ログ (パイロット不安定の模擬)
    for i in range(60):
        lg.log({
            "freq_hz": 80000000, "mode": "WFM", "gain_db": 33.8,
            "pilot_lock": 0.3 + 0.4 * abs(np.sin(i * 0.3)),
            "blend": 0.5 + 0.5 * np.sin(i * 0.3),
            "nr_gain": 1.0, "cut_hz": 15000.0, "wiener_gain": 1.0,
            "multipath_gain": 1.0, "afc_hz": 10.0, "s_units": 9.0,
        })
    rep = summarize(TMP, freq_hz=80000000)
    assert rep["n"] == 60, rep["n"]
    assert rep["order"][0] == "blend", rep["order"]
    assert any("パイロット" in h for h in rep["hints"]), rep["hints"]
    # 別局フィルタ: 存在しない局はデータ不足
    rep2 = summarize(TMP, freq_hz=90000000)
    assert "データ不足" in rep2["hints"][0]
    print("[OK] summarize finds blend hunting")


def test_is_settled_blanking():
    print("===== test_is_settled_blanking =====")
    from signal_logger import is_settled
    # 起動15秒未満は記録しない
    assert not is_settled(100.0, 50.0, 90.0)
    # 選局8秒未満は記録しない
    assert not is_settled(100.0, 95.0, 0.0)
    # 両方過ぎていれば記録する
    assert is_settled(100.0, 50.0, 0.0)
    # 境界ちょうどは記録する (>= 扱い)
    assert is_settled(23.0, 15.0, 8.0)
    # 不正入力は記録しない
    assert not is_settled(100.0, None, 0.0)
    print("[OK] is_settled blanking")


def test_diagnose_environment():
    print("===== test_diagnose_environment =====")
    import numpy as np
    from signal_logger import SignalLogger, diagnose_environment, format_environment_report
    _clean()
    lg = SignalLogger(path=TMP, max_bytes=10 ** 9)
    # 台風・フェージング・激しいマルチパスの模擬ログ
    for i in range(50):
        lg.log({
            "freq_hz": 79500000, "mode": "WFM", "gain_db": 49.6,
            "pilot_lock": 0.45 + 0.3 * np.sin(i * 0.5),
            "blend": 0.2 if (i % 4 == 0) else 0.9,
            "nr_gain": 0.5,
            "cut_hz": 7000.0 + 3000.0 * (i % 2),  # 激しいチャタリング
            "wiener_gain": 0.25,  # 強力抑圧
            "multipath_gain": 0.38,  # 重篤マルチパス
            "afc_hz": 20.0,
            "s_units": 11.5 + 1.2 * np.sin(i * 0.2),  # フェージング
            "snr_db": 14.0,
            "audio_snr_db": 10.0,
        })
    diag = diagnose_environment(TMP, freq_hz=79500000, recent_n=50)
    assert diag["status"] == "ok"
    assert diag["fading"]["is_fading"] is True
    assert "深刻" in diag["multipath"]["severity"]
    assert diag["musical_noise"]["risk_score"] > 50
    assert any("MONO" in r for r in diag["recommendations"])
    txt = format_environment_report(diag)
    assert "精密診断レポート" in txt
    print("[OK] diagnose_environment")


def main() -> int:
    try:
        test_log_and_load()
        test_rotation()
        test_summarize_finds_blend_hunting()
        test_is_settled_blanking()
        test_diagnose_environment()
    except AssertionError as e:
        print(f"FAILED: {e}")
        return 1
    finally:
        _clean()
    print("\nALL SIGNAL LOGGER TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
