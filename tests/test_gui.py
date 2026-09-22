"""
GUI Layout and Interaction Test Suite.
UI被り防止、静的事前ベイク、桁単位ホイール同調、Sメーター、立体ボタンの回帰防止テスト。
"""

import os
import sys

# headlessテスト用 (SDLのダミービデオドライバ)
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
import numpy as np

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gui import SdrGui, Button


def test_gui_panel_boundaries():
    """パネル間の領域衝突・UI被りがないことを幾何学的に検証"""
    gui = SdrGui()

    # 主要パネル矩形
    hero = gui.hero_rect
    info = gui.info_rect
    spec = gui.spec_rect
    tele = gui.tele_rect
    tune = gui.tune_rect
    # gain_rectは廃止 (GAIN/AUDIOパネル削除)。衝突検証の対象外。
    preset = gui.preset_rect
    status = gui.status_rect

    # 1. ヒーローパネルと情報パネルが重なっていないこと
    assert not hero.colliderect(info), f"Hero and Info panels overlap: {hero} vs {info}"
    assert hero.right <= info.left, f"Hero right {hero.right} > Info left {info.left}"

    # 2. 上部ヘッダー (hero/info) と 下部パネル (spec/tele) が縦方向に衝突していないこと
    assert hero.bottom < spec.top, f"Hero bottom {hero.bottom} >= Spec top {spec.top}"
    assert info.bottom < spec.top, f"Info bottom {info.bottom} >= Spec top {spec.top}"
    assert info.bottom < tele.top, f"Info bottom {info.bottom} >= Tele top {tele.top}"

    # 3. スペクトラム系と右側コントロール系が横方向に衝突していないこと
    assert spec.right <= tele.left, f"Spec right {spec.right} > Tele left {tele.left}"

    # 4. プリセットとステータスバーが縦方向に衝突していないこと
    assert preset.bottom < status.top, f"Preset bottom {preset.bottom} >= Status top {status.top}"

    print("[OK] UI panel boundary & layout collision test passed")
    gui.close()


def test_gui_baking_and_rendering():
    """静的シーンの事前ベイク (baked_bg) とレンダリングパイプラインの検証"""
    gui = SdrGui()
    assert gui.baked_bg is not None, "baked_bg must be generated on init"
    assert gui.baked_bg.get_size() == (gui.width, gui.height), "baked_bg size mismatch"

    # ダミースペクトラムと音声PCM
    dummy_spec = np.linspace(-80.0, -20.0, 1024, dtype=np.float32)
    dummy_audio = np.sin(np.linspace(0, 2 * np.pi * 10, 1024)).astype(np.float32)

    # レンダリング実行 (例外が出ないこと)
    gui.s_units = 9.5
    gui.rds_text = "Artist - Song Title (Testing RDS RadioText Ticker)"
    gui.sw_info = "CRI 中国国際放送 (日本語) 09:00-11:00UTC"
    gui.render(dummy_spec, dummy_audio)

    # Sメーターのピークホールド検証
    assert gui.s_peak_units >= 9.5, f"Peak hold failed: {gui.s_peak_units}"

    print("[OK] Baked background and render pipeline test passed")
    gui.close()


def test_digit_wheel_tuning():
    """周波数の桁単位ホイール同調の検証"""
    gui = SdrGui()
    gui.center_freq = 80000000  # 80.0000 MHz

    # レンダリングして周波数桁のhitboxを構築させる
    dummy_spec = np.zeros(1024, dtype=np.float32)
    gui.render(dummy_spec)

    assert len(gui.freq_digit_hitboxes) > 0, "Frequency digit hitboxes must be populated"

    # 1MHz桁のhitboxを特定
    hit_1mhz = [step for _, step in gui.freq_digit_hitboxes if step == 1000000]
    assert len(hit_1mhz) == 1, "1MHz step digit hitbox not found"

    # 100kHz桁のhitboxを特定
    hit_100k = [step for _, step in gui.freq_digit_hitboxes if step == 100000]
    assert len(hit_100k) == 1, "100kHz step digit hitbox not found"

    # 桁ホイール同調のシミュレート (+1MHz)
    gui.hovered_freq_digit = 1000000
    init_f = gui.center_freq
    gui._adjust_freq(1 * gui.hovered_freq_digit)
    assert gui.center_freq == init_f + 1000000, f"Freq adjustment failed: {gui.center_freq}"

    # 桁ホイール同調のシミュレート (-100kHz)
    gui.hovered_freq_digit = 100000
    gui._adjust_freq(-1 * gui.hovered_freq_digit)
    assert gui.center_freq == init_f + 900000, f"Freq adjustment failed: {gui.center_freq}"

    print("[OK] Digit-based mousewheel tuning test passed")
    gui.close()


def test_button_tactile_feedback():
    """ボタンの立体押し込みフィードバック (pressed) とイベントの検証"""
    clicked = False

    def on_click():
        nonlocal clicked
        clicked = True

    btn = Button((10, 10, 80, 30), "TEST", on_click)
    assert not btn.pressed
    assert not btn.hover

    # マウス移動でホバー
    e_motion = pygame.event.Event(pygame.MOUSEMOTION, {"pos": (20, 20)})
    btn.handle_event(e_motion)
    assert btn.hover

    # マウス押下 (pressed=True, まだコールバックは呼ばれない)
    e_down = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"pos": (20, 20), "button": 1})
    btn.handle_event(e_down)
    assert btn.pressed
    assert not clicked

    # マウス離脱時のUP (領域外ならコールバック呼ばれない)
    e_up_outside = pygame.event.Event(pygame.MOUSEBUTTONUP, {"pos": (200, 200), "button": 1})
    btn.handle_event(e_up_outside)
    assert not btn.pressed
    assert not clicked

    # 正常クリックシーケンス (DOWN -> UP)
    btn.handle_event(e_down)
    assert btn.pressed
    e_up = pygame.event.Event(pygame.MOUSEBUTTONUP, {"pos": (20, 20), "button": 1})
    btn.handle_event(e_up)
    assert not btn.pressed
    assert clicked, "Button click callback was not triggered"

    print("[OK] Button tactile pressed state and event handling test passed")


def test_header_vertical_separation():
    """周波数巨大数字がhero_rect内に収まり、ボリュームスライダーと被らないことを検証"""
    gui = SdrGui()
    dummy_spec = np.zeros(1024, dtype=np.float32)
    gui.render(dummy_spec)

    # 周波数各桁の底辺がボリュームスライダーのY座標(hero_rect.y + 92)より上にあること
    vol_y = gui.hero_rect.y + 100
    for rect, _ in gui.freq_digit_hitboxes:
        assert rect.bottom <= vol_y, f"Digit rect bottom {rect.bottom} overlaps vol_y {vol_y}"

    # 周波数各桁がhero_rect内に完全に収まっていること
    for rect, _ in gui.freq_digit_hitboxes:
        assert gui.hero_rect.contains(rect), f"Digit rect {rect} outside hero_rect {gui.hero_rect}"

    print("[OK] Header vertical text separation verified (no overlap)")
    gui.close()


if __name__ == "__main__":
    print("===== test_gui.py =====")
    test_gui_panel_boundaries()
    test_gui_baking_and_rendering()
    test_digit_wheel_tuning()
    test_button_tactile_feedback()
    test_header_vertical_separation()
    print("ALL GUI TESTS PASSED!")
