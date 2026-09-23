"""自動モード解決 (第2章・自動化の完成)。

周波数帯と矛盾するモードの自動補正 (config.resolve_mode) と、
main.SdrApp._apply_frequency_and_mode への配線を検証する。
- FREQ経路 (auto=True): 帯域と矛盾するモードは自動補正
- MODE経路 (auto=False): ユーザー明示を尊重
"""
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np  # noqa: E402

from config import resolve_mode, ham_band_mode  # noqa: E402

JP_FM = (76000000, 95000000)


def test_resolve_table():
    cases = [
        # (freq, requested, expected, fm_range)
        (80000000, "AM", "WFM", JP_FM),          # FM帯でAM → WFM
        (80000000, "USB", "WFM", JP_FM),         # FM帯でSSB → WFM
        (80000000, "WFM", "WFM", JP_FM),
        (80000000, "NFM", "NFM", JP_FM),         # NFMはユーザー意図として残す
        (594000, "WFM", "AM", JP_FM),            # 中波でWFM → AM
        (6055000, "NFM", "AM", JP_FM),           # 短波でNFM → AM
        (7300000, "AM", "AM", JP_FM),            # 7.3MHzはSW放送 (ハム帯外) → AM
        (7100000, "AM", "LSB", JP_FM),           # 40mでAM → LSB
        (7100000, "WFM", "LSB", JP_FM),          # 40mでWFM → LSB
        (14200000, "AM", "USB", JP_FM),          # 20mでAM → USB
        (7100000, "USB", "USB", JP_FM),          # 明示SSBは尊重
        (7100000, "CW", "CW", JP_FM),            # 明示CWは尊重
        (145000000, "WFM", "NFM", JP_FM),        # 2mでWFM → NFM
        (145000000, "AM", "NFM", JP_FM),         # 2mでAM → NFM
        (439560000, "WFM", "NFM", JP_FM),        # 70cmでWFM → NFM
        (439560000, "NFM", "NFM", JP_FM),
        (125000000, "WFM", "AM", JP_FM),         # エアバンドでWFM → AM
        (125000000, "NFM", "AM", JP_FM),
        (125000000, "AM", "AM", JP_FM),
        (155000000, "NFM", "NFM", JP_FM),        # 業務帯はそのまま
        (100100000, "AM", "WFM", (87500000, 108000000)),  # US帯域
        (100100000, "AM", "AM", JP_FM),          # JP帯域外は補正しない
        (95600000, "WFM", "WFM", JP_FM),         # 帯域外のWFMも尊重
    ]
    for freq, req, exp, rng in cases:
        got = resolve_mode(freq, req, rng[0], rng[1])
        assert got == exp, f"{freq} {req}: {got} != {exp}"
    # ham_band_modeの慣例と整合
    assert ham_band_mode(7100000) == "LSB"
    assert ham_band_mode(14200000) == "USB"
    print(f"resolve table OK ({len(cases)} cases)")


def test_resolve_robust():
    assert resolve_mode(None, "WFM") == "WFM"  # 不正入力は要求を返す
    assert resolve_mode(80000000, "") == "WFM"
    assert resolve_mode(80000000, None) == "WFM"
    print("resolve robust OK")


def _mock_app():
    import main as main_mod
    app = main_mod.SdrApp.__new__(main_mod.SdrApp)
    app.driver = mock.MagicMock()
    app.dsp = mock.MagicMock()
    app.gui = mock.MagicMock()
    app.profile = {"fm_start_hz": 76000000, "fm_end_hz": 95000000}
    app.use_controller = False
    app.freq = 594000
    app.mode = "AM"
    return app, main_mod


def test_wiring_freq_path():
    app, main_mod = _mock_app()
    with mock.patch.object(main_mod.sw_schedule, "lookup", return_value=None), \
         mock.patch.object(main_mod, "match_station_name",
                           return_value="Unknown FM Station"):
        # FREQ経路: AMのまま80MHzへ飛ぶ → WFMへ自動補正
        app._apply_frequency_and_mode(80000000, app.mode)
        assert app.mode == "WFM", app.mode
        assert app.gui.mode == "WFM"
        app.gui._sync_bfo_visibility.assert_called()
        # さらに 145MHz へ → NFM
        app._apply_frequency_and_mode(145000000, app.mode)
        assert app.mode == "NFM", app.mode
    print("wiring FREQ OK")


def test_wiring_mode_explicit():
    app, main_mod = _mock_app()
    with mock.patch.object(main_mod.sw_schedule, "lookup", return_value=None), \
         mock.patch.object(main_mod, "match_station_name",
                           return_value="Unknown FM Station"):
        # MODE経路 (auto=False): ユーザーがFM帯でAMを明示 → 尊重
        app._apply_frequency_and_mode(80000000, "AM", auto=False)
        assert app.mode == "AM", app.mode
    print("wiring MODE explicit OK")


if __name__ == "__main__":
    test_resolve_table()
    test_resolve_robust()
    test_wiring_freq_path()
    test_wiring_mode_explicit()
    print("ALL AUTO-MODE TESTS PASSED!")
