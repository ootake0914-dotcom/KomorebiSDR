"""
Pygame-based SDR GUI Module - Frosted Glass Edition.
スペクトラムアナライザ、ウォーターフォール、波形表示、選局コントロールを備えた
ガラス調 (Frosted Glass) モダンGUI。
"""

import numpy as np
import pygame

from i18n import t


# ウォーターフォール用のカラーマップ生成 (紺 -> 青緑 -> シアン -> 金 -> 白)
_COLOR_STOPS = [
    (8, 12, 24),
    (10, 46, 78),
    (12, 110, 130),
    (64, 200, 186),
    (226, 196, 112),
    (255, 246, 224),
]


def create_colormap(num_colors=256) -> np.ndarray:
    lut = np.zeros((num_colors, 3), dtype=np.uint8)
    stops = np.array(_COLOR_STOPS, dtype=np.float32)
    xs = np.linspace(0.0, 1.0, len(stops))
    q = np.linspace(0.0, 1.0, num_colors)
    for c in range(3):
        lut[:, c] = np.interp(q, xs, stops[:, c]).astype(np.uint8)
    return lut


# パレット
C_BG_TOP = (238, 242, 249)
C_BG_BOTTOM = (219, 227, 240)
C_TEXT = (43, 52, 69)
C_MUTED = (118, 130, 150)
C_ACCENT = (0, 172, 152)
C_ACCENT_DARK = (0, 122, 110)
C_GOLD = (214, 164, 62)
C_PANEL_DARK = (17, 23, 36)
C_BTN = (246, 249, 253)
C_BTN_BORDER = (206, 216, 231)
C_BTN_ACTIVE = (0, 172, 152)
C_BTN_ACTIVE2 = (86, 118, 168)

# 日本語対応フォント候補
JP_FONTS = ["meiryo ui", "yu gothic ui", "meiryo", "yu gothic", "msgothic", "arial"]
LATIN_FONTS = ["segoeui", "arial"]


def show_message_screen(title: str, lines: list, width: int = 760, height: int = 400):
    """起動失敗時の案内画面 (キー入力またはウィンドウを閉じると終了)"""
    try:
        pygame.init()
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Antigravity SDR Radio")
        f_title = pygame.font.SysFont(JP_FONTS, 22, bold=True)
        f_body = pygame.font.SysFont(JP_FONTS, 15)
        clock = pygame.time.Clock()
        while True:
            for e in pygame.event.get():
                if e.type in (pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                    pygame.quit()
                    return
            screen.fill(C_BG_TOP)
            ts = f_title.render(title, True, (178, 62, 62))
            screen.blit(ts, (28, 22))
            y = 76
            for line in lines:
                s = f_body.render(line, True, C_TEXT)
                screen.blit(s, (28, y))
                y += 24
            hint = f_body.render(t("quit_hint"), True, C_MUTED)
            screen.blit(hint, (28, height - 36))
            pygame.display.flip()
            clock.tick(30)
    except Exception:
        pass


class Button:
    """クリック可能なUIボタン (ガラス調)"""

    def __init__(self, rect, text, callback, bg_color=None, active_color=None, dark_text=True):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.callback = callback
        self.bg_color = bg_color if bg_color is not None else C_BTN
        self.active_color = active_color if active_color is not None else C_BTN_ACTIVE
        self.dark_text = dark_text
        self.is_active = False
        self.hover = False
        self.radius = 8

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.callback()
                return True
        return False

    def draw(self, surface, font):
        if self.is_active:
            fill = self.active_color
            txt_color = (255, 255, 255)
            border = self.active_color
        else:
            fill = self.bg_color
            if self.hover:
                fill = tuple(min(255, c + 8) for c in fill)
            txt_color = C_TEXT if self.dark_text else (255, 255, 255)
            border = C_BTN_BORDER
        pygame.draw.rect(surface, fill, self.rect, border_radius=self.radius)
        pygame.draw.rect(surface, border, self.rect, width=1, border_radius=self.radius)
        txt_surf = font.render(self.text, True, txt_color)
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        surface.blit(txt_surf, txt_rect)


class SdrGui:
    """SDR デスクトップGUIクラス (Frosted Glass Edition)"""

    def __init__(self, width=1120, height=720):
        pygame.init()
        pygame.font.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Antigravity Full-Scratch SDR Radio")

        # フォント (日本語対応: Meiryo系を優先)
        self.font_title = pygame.font.SysFont(JP_FONTS, 14, bold=True)
        self.font_huge = pygame.font.SysFont(LATIN_FONTS, 40, bold=True)
        self.font_med = pygame.font.SysFont(JP_FONTS, 15, bold=True)
        self.font_small = pygame.font.SysFont(JP_FONTS, 12)
        self.font_tiny = pygame.font.SysFont(JP_FONTS, 11)

        self.clock = pygame.time.Clock()
        self.running = True

        self.colormap = create_colormap()

        # ---- レイアウト (1120x720) ----
        self.hero_rect = pygame.Rect(14, 10, 668, 88)
        self.info_rect = pygame.Rect(694, 10, 412, 88)
        self.spec_rect = pygame.Rect(14, 110, 800, 210)
        self.wf_rect = pygame.Rect(14, 332, 800, 150)
        self.wave_rect = pygame.Rect(14, 494, 800, 54)
        self.tele_rect = pygame.Rect(826, 110, 280, 88)
        self.tune_rect = pygame.Rect(826, 210, 280, 228)
        self.gain_rect = pygame.Rect(826, 450, 280, 104)
        self.preset_rect = pygame.Rect(14, 560, 1092, 74)
        self.status_rect = pygame.Rect(14, 644, 1092, 28)

        self.wf_surface = pygame.Surface((self.wf_rect.width, self.wf_rect.height))
        self.wf_surface.fill((0, 0, 0))

        # パラメータコールバック
        self.on_freq_change = None
        self.on_mode_change = None
        self.on_gain_change = None
        self.on_gain_lock_toggle = None  # lambda: 決め打ち/再探索トグル
        self.on_volume_change = None
        self.on_filter_change = None
        self.on_seek_change = None      # lambda direction: ...
        self.on_scan_request = None     # lambda: ...
        self.on_dx_toggle = None        # lambda enabled: ...
        self.on_afc_toggle = None       # lambda enabled: ...

        # 内部状態
        self.center_freq = 80000000  # 80.0 MHz
        self.sample_rate = 1152000
        self.mode = "WFM"
        self.volume = 0.5
        self.is_auto_gain = True
        self.is_hard_locked = False     # 収束決め打ち中フラグ
        self.gain_auto_label = "Hyper: ON"
        self.gain_val = 30.0
        self.gains_list = []
        self.current_rssi = -50.0
        self.is_dx_mode = False
        self.is_afc_enabled = True
        self.is_stereo = False
        self.region_label = ""
        self.scan_range = (76.0, 95.0)
        self.presets_fm = []
        self.presets_am = []
        self.detected_stations = []
        self.scan_status_text = "待機中 (Auto Seek / 全帯域スキャン可能)"
        self.telemetry_text = ""

        # 静的パネルの事前描画 (ガラス表現)
        self._prerender_background()
        self._prerender_panels()

        # UIボタンのリスト
        self.buttons = []
        self._init_controls()

    # ================================================================
    # 事前描画 (ガラスパネル / 背景)
    # ================================================================
    def _prerender_background(self):
        col = np.linspace(C_BG_TOP, C_BG_BOTTOM, self.height)
        arr = np.repeat(col[:, None, :], self.width, axis=1).astype(np.uint8)
        bg = pygame.Surface((self.width, self.height))
        pygame.surfarray.blit_array(bg, np.transpose(arr, (1, 0, 2)))
        # 柔らかいパステルブロブ (低解像度で描いて拡大 = ぼかし)
        sc = 10
        overlay = pygame.Surface((self.width // sc, self.height // sc), pygame.SRCALPHA)
        blobs = [
            (self.width * 0.18, self.height * 0.12, 210, (170, 205, 255, 70)),
            (self.width * 0.85, self.height * 0.22, 180, (255, 205, 225, 60)),
            (self.width * 0.55, self.height * 0.92, 240, (185, 240, 220, 60)),
        ]
        for cx, cy, r, color in blobs:
            pygame.draw.circle(overlay, color, (int(cx / sc), int(cy / sc)), int(r / sc))
        overlay = pygame.transform.smoothscale(overlay, (self.width, self.height))
        bg.blit(overlay, (0, 0))
        self.bg_surface = bg

    def _glass(self, w, h, radius=18, alpha=180, tint=(255, 255, 255), dark=False):
        """ガラスパネル表面を生成 (pad付き)"""
        pad = 14
        surf = pygame.Surface((w + pad * 2, h + pad * 2), pygame.SRCALPHA)
        for i in range(8, 0, -1):
            a = int(13 * (9 - i) / 8)
            pygame.draw.rect(surf, (58, 74, 104, a),
                             (pad - i, pad - i + 3, w + 2 * i, h + 2 * i),
                             border_radius=radius + i)
        fill = (C_PANEL_DARK[0], C_PANEL_DARK[1], C_PANEL_DARK[2], 233) if dark else (*tint, alpha)
        pygame.draw.rect(surf, fill, (pad, pad, w, h), border_radius=radius)
        border = (92, 112, 142, 170) if dark else (255, 255, 255, 235)
        pygame.draw.rect(surf, border, (pad, pad, w, h), width=1, border_radius=radius)
        return surf, pad

    def _prerender_panels(self):
        self.panels = {}
        for name, rect in (
            ("hero", self.hero_rect), ("info", self.info_rect),
            ("tele", self.tele_rect), ("tune", self.tune_rect),
            ("gain", self.gain_rect), ("preset", self.preset_rect),
        ):
            self.panels[name] = self._glass(rect.width, rect.height)
        self.panels["spec"] = self._glass(self.spec_rect.width, self.spec_rect.height,
                                          radius=16, tint=(17, 23, 36), dark=True)
        self.panels["wf"] = self._glass(self.wf_rect.width, self.wf_rect.height,
                                        radius=16, tint=(17, 23, 36), dark=True)
        self.panels["wave"] = self._glass(self.wave_rect.width, self.wave_rect.height,
                                          radius=16, tint=(17, 23, 36), dark=True)
        self.panels["status"] = self._glass(self.status_rect.width, self.status_rect.height,
                                            radius=14, alpha=160)

    def _draw_panel(self, name, rect):
        surf, pad = self.panels[name]
        self.screen.blit(surf, (rect.x - pad, rect.y - pad))

    # ================================================================
    # コントロール配置
    # ================================================================
    def set_region(self, label: str, start_mhz: float, end_mhz: float):
        """地域プロファイルをGUIへ反映 (スキャンボタンの帯域表示)"""
        self.region_label = label
        self.scan_range = (start_mhz, end_mhz)
        if hasattr(self, "btn_scan_band"):
            self.btn_scan_band.text = t("scan_button", start=f"{start_mhz:g}", end=f"{end_mhz:g}")

    def set_presets(self, presets_fm: list, presets_am: list):
        """スキャン結果などからプリセットボタンを再構築する"""
        self.presets_fm = list(presets_fm or [])[:10]
        self.presets_am = list(presets_am or [])[:7]
        self._init_controls()

    def _init_controls(self):
        btns = []

        # ---- TUNINGパネル ----
        tx, tw = self.tune_rect.x + 12, self.tune_rect.width - 24
        bw = (tw - 12) // 3
        step_btns = [("-1M", -1000000), ("-100k", -100000), ("-10k", -10000),
                     ("+10k", 10000), ("+100k", 100000), ("+1M", 1000000)]
        for i, (text, step) in enumerate(step_btns):
            col, row = i % 3, i // 3
            cb = (lambda s=step: self._adjust_freq(s))
            btns.append(Button((tx + col * (bw + 6), 236 + row * 30, bw, 26), text, cb))
        half = (tw - 6) // 2
        self.btn_seek_prev = Button((tx, 300, half, 28), "<< Auto Seek", lambda: self._seek(-1),
                                    bg_color=(226, 235, 248), active_color=C_BTN_ACTIVE2)
        self.btn_seek_next = Button((tx + half + 6, 300, half, 28), "Auto Seek >>", lambda: self._seek(1),
                                    bg_color=(226, 235, 248), active_color=C_BTN_ACTIVE2)
        self.btn_scan_band = Button((tx, 332, tw, 30), "全帯域スキャン (FM 76〜95MHz)", self._request_scan,
                                    bg_color=(214, 240, 229), active_color=C_ACCENT)
        btns.extend([self.btn_seek_prev, self.btn_seek_next, self.btn_scan_band])
        mw = (tw - 12) // 3
        self.btn_wfm = Button((tx, 382, mw, 28), "WFM", lambda: self._set_mode("WFM"), bg_color=(212, 236, 248))
        self.btn_am = Button((tx + mw + 6, 382, mw, 28), "AM", lambda: self._set_mode("AM"), bg_color=(212, 236, 248))
        self.btn_nfm = Button((tx + 2 * (mw + 6), 382, mw, 28), "NFM", lambda: self._set_mode("NFM"), bg_color=(212, 236, 248))
        btns.extend([self.btn_wfm, self.btn_am, self.btn_nfm])

        # ---- GAIN / AUDIOパネル ----
        gx, gy = self.gain_rect.x + 12, self.gain_rect.y
        self.btn_gain_auto = Button((gx, 470, 142, 25), "Hyper: ON", self._toggle_gain_auto,
                                    bg_color=(230, 240, 250))
        self.btn_gain_up = Button((gx + 148, 470, 50, 25), "G+", lambda: self._step_gain(1))
        self.btn_gain_down = Button((gx + 204, 470, 50, 25), "G-", lambda: self._step_gain(-1))
        btns.extend([self.btn_gain_auto, self.btn_gain_up, self.btn_gain_down])
        self.btn_vol_down = Button((gx, 497, 46, 25), "V-", lambda: self._adjust_vol(-0.1))
        self.btn_vol_up = Button((gx + 52, 497, 46, 25), "V+", lambda: self._adjust_vol(0.1))
        self.btn_filter = Button((gx + 104, 497, 150, 25), "Filter: Clean", self._toggle_filter)
        btns.extend([self.btn_vol_down, self.btn_vol_up, self.btn_filter])
        self.btn_afc = Button((gx, 524, 125, 25), "AFC: ON", self._toggle_afc, bg_color=(212, 232, 248))
        self.btn_dx_mode = Button((gx + 131, 524, 123, 25), "DX: OFF", self._toggle_dx, bg_color=(234, 224, 246))
        btns.extend([self.btn_afc, self.btn_dx_mode])

        # ---- プリセット (地域/スキャン結果に応じて動的に設定される) ----
        fw, fgap, fx0 = 100, 4, self.preset_rect.x + 36
        for i, p in enumerate(self.presets_fm[:10]):
            f, m = int(p["freq_hz"]), p.get("mode", "WFM")
            cb = (lambda freq=f, mode=m: self._tune(freq, mode))
            btns.append(Button((fx0 + i * (fw + fgap), 572, fw, 26), p["name"], cb))
        aw = 144
        for i, p in enumerate(self.presets_am[:7]):
            f, m = int(p["freq_hz"]), p.get("mode", "AM")
            cb = (lambda freq=f, mode=m: self._tune(freq, mode))
            btns.append(Button((fx0 + i * (aw + fgap), 600, aw, 26), p["name"], cb))

        # アトミックに差し替え (スキャン完了時の再構築と描画の競合防止)
        self.buttons = btns

    # ================================================================
    # 操作ハンドラ
    # ================================================================
    def _adjust_freq(self, delta):
        new_freq = max(100000, self.center_freq + delta)
        self.center_freq = new_freq
        if self.on_freq_change:
            self.on_freq_change(self.center_freq)

    def _set_mode(self, mode):
        self.mode = mode
        if self.on_mode_change:
            self.on_mode_change(self.mode)

    def _tune(self, freq, mode):
        self.center_freq = freq
        self.mode = mode
        if self.on_freq_change:
            self.on_freq_change(self.center_freq)
        if self.on_mode_change:
            self.on_mode_change(self.mode)

    def _toggle_gain_auto(self):
        if self.on_gain_lock_toggle:
            self.on_gain_lock_toggle()
        elif self.on_gain_change:
            self.is_auto_gain = not self.is_auto_gain
            self.btn_gain_auto.text = self.gain_auto_label if self.is_auto_gain else f"Gain: {self.gain_val:.1f}dB"
            self.on_gain_change(self.is_auto_gain, self.gain_val)

    def _step_gain(self, dir_step):
        if not self.gains_list:
            return
        min_idx = min(range(len(self.gains_list)), key=lambda i: abs(self.gains_list[i] - 12.5))
        idx = min(range(len(self.gains_list)), key=lambda i: abs(self.gains_list[i] - self.gain_val))
        idx = max(min_idx, min(len(self.gains_list) - 1, idx + dir_step))
        self.gain_val = self.gains_list[idx]
        self.is_hard_locked = True
        self.btn_gain_auto.text = f"手動 {self.gain_val:.1f}dB"
        self.btn_gain_auto.bg_color = (216, 240, 230)
        if self.on_gain_change:
            self.on_gain_change(False, self.gain_val)

    def _adjust_vol(self, delta):
        self.volume = max(0.0, min(1.0, self.volume + delta))
        if self.on_volume_change:
            self.on_volume_change(self.volume)

    def _toggle_filter(self):
        if not hasattr(self, "filter_mode"):
            self.filter_mode = "clean"
        self.filter_mode = "wide" if self.filter_mode == "clean" else "clean"
        self.btn_filter.text = f"Filter: {self.filter_mode.capitalize()}"
        if hasattr(self, "on_filter_change") and self.on_filter_change:
            self.on_filter_change(self.filter_mode)

    def _seek(self, direction):
        if self.on_seek_change:
            self.on_seek_change(direction)

    def _request_scan(self):
        if self.on_scan_request:
            self.scan_status_text = "全帯域スキャン実行中..."
            self.on_scan_request()

    def _toggle_dx(self):
        self.is_dx_mode = not self.is_dx_mode
        self.btn_dx_mode.text = "DX Boost: ON" if self.is_dx_mode else "DX: OFF"
        if self.on_dx_toggle:
            self.on_dx_toggle(self.is_dx_mode)

    def _toggle_afc(self):
        self.is_afc_enabled = not self.is_afc_enabled
        self.btn_afc.text = "AFC: ON" if self.is_afc_enabled else "AFC: OFF"
        if self.on_afc_toggle:
            self.on_afc_toggle(self.is_afc_enabled)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            # スペクトラム・ウォーターフォールクリックによる同調 (検出局マーカーへの自動吸着対応)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if self.spec_rect.collidepoint(mx, my) or self.wf_rect.collidepoint(mx, my):
                    ratio = (mx - self.spec_rect.x) / self.spec_rect.width
                    clicked_freq = (self.center_freq - self.sample_rate / 2) + ratio * self.sample_rate

                    snapped_station = None
                    min_dist = float("inf")
                    for st in self.detected_stations:
                        dist_hz = abs(st["freq_hz"] - clicked_freq)
                        if dist_hz <= 45000 and dist_hz < min_dist:
                            min_dist = dist_hz
                            snapped_station = st

                    if snapped_station:
                        self.center_freq = snapped_station["freq_hz"]
                        self.scan_status_text = (f"局吸着同調: {snapped_station['name']} "
                                                 f"({snapped_station['freq_mhz']:.2f}MHz, SNR:+{snapped_station['snr_db']}dB)")
                    else:
                        self.center_freq = int(round(clicked_freq / 10000) * 10000)

                    if self.on_freq_change:
                        self.on_freq_change(self.center_freq)

            # ボタンイベント処理
            for btn in self.buttons:
                btn.handle_event(event)

    # ================================================================
    # 描画
    # ================================================================
    def update_waterfall(self, spectrum_db: np.ndarray):
        """ウォーターフォールにライン追加 (スクロール描画)"""
        wf_w = self.wf_rect.width
        self.wf_surface.scroll(0, 2)

        norm = np.clip((spectrum_db + 70.0) / 55.0, 0.0, 1.0)
        indices = (norm * 255).astype(np.uint8)
        rgb_line = self.colormap[indices]

        x_src = np.linspace(0, len(indices) - 1, wf_w)
        r = np.interp(x_src, np.arange(len(indices)), rgb_line[:, 0]).astype(np.uint8)
        g = np.interp(x_src, np.arange(len(indices)), rgb_line[:, 1]).astype(np.uint8)
        b = np.interp(x_src, np.arange(len(indices)), rgb_line[:, 2]).astype(np.uint8)

        line_surf = pygame.Surface((wf_w, 2))
        arr = np.stack([r, g, b], axis=-1)
        arr2 = np.repeat(arr[:, np.newaxis, :], 2, axis=1)
        pygame.surfarray.blit_array(line_surf, arr2)
        self.wf_surface.blit(line_surf, (0, 0))

    def _draw_chip(self, text, x, y, fill, txt_color=C_TEXT):
        font = self.font_small
        w = font.size(text)[0] + 16
        rect = pygame.Rect(x, y, w, 20)
        pygame.draw.rect(self.screen, fill, rect, border_radius=10)
        surf = font.render(text, True, txt_color)
        self.screen.blit(surf, (x + 8, y + 3))
        return w

    def _draw_header(self):
        # 周波数ヒーローパネル
        self._draw_panel("hero", self.hero_rect)
        lbl = self.font_title.render(t("freq"), True, C_MUTED)
        self.screen.blit(lbl, (self.hero_rect.x + 18, self.hero_rect.y + 10))

        if self.center_freq >= 1000000:
            freq_str = f"{self.center_freq / 1e6:,.4f}"
            unit = "MHz"
        else:
            freq_str = f"{self.center_freq / 1e3:,.1f}"
            unit = "kHz"
        freq_surf = self.font_huge.render(freq_str, True, C_TEXT)
        self.screen.blit(freq_surf, (self.hero_rect.x + 16, self.hero_rect.y + 22))
        unit_surf = self.font_med.render(unit, True, C_ACCENT_DARK)
        self.screen.blit(unit_surf, (self.hero_rect.x + 20 + freq_surf.get_width(), self.hero_rect.y + 48))

        # 局名 (既知局データベース or 検出リストから表示)
        st_name = ""
        for st in self.detected_stations:
            if abs(st["freq_hz"] - self.center_freq) <= 50000:
                st_name = st["name"]
                break
        if not st_name:
            try:
                from auto_tuner import match_station_name
                cand = match_station_name(self.center_freq)
                if cand != "Unknown FM Station":
                    st_name = cand
            except Exception:
                st_name = ""
        if st_name:
            name_surf = self.font_small.render(st_name, True, C_MUTED)
            self.screen.blit(name_surf, (self.hero_rect.x + 20, self.hero_rect.y + 68))

        # 情報パネル (モード/音量/ゲイン + テレメトリ)
        self._draw_panel("info", self.info_rect)
        x0 = self.info_rect.x + 14
        y0 = self.info_rect.y + 10
        auto_name = self.gain_auto_label.split(":")[0]
        gain_txt = auto_name if self.is_auto_gain else f"{self.gain_val:.1f}dB"
        w1 = self._draw_chip(f"MODE {self.mode}", x0, y0, (210, 236, 246))
        w2 = self._draw_chip(f"VOL {int(self.volume * 100)}%", x0 + w1 + 6, y0, (226, 222, 246))
        w3 = self._draw_chip(f"GAIN {gain_txt}", x0 + w1 + w2 + 12, y0, (246, 232, 210))
        st_txt = t("stereo") if self.is_stereo else t("mono")
        self._draw_chip(st_txt, x0 + w1 + w2 + w3 + 18, y0,
                        (206, 240, 226) if self.is_stereo else (236, 238, 244))

        if self.region_label:
            rl = self.font_tiny.render(f"{t('region')}: {self.region_label}", True, C_MUTED)
            self.screen.blit(rl, (self.hero_rect.right - 12 - rl.get_width(), self.hero_rect.y + 68))

        # テレメトリ (「|」区切りをチップ化して2行に)
        parts = [p.strip() for p in self.telemetry_text.split("|") if p.strip()]
        cx, cy = x0, y0 + 26
        for i, p in enumerate(parts):
            w = self.font_small.size(p)[0] + 16
            if cx + w > self.info_rect.right - 7:
                cx = x0
                cy += 24
            self._draw_chip(p, cx, cy, (240, 243, 248))
            cx += w + 6

    def _draw_spectrum(self, spectrum_db):
        self._draw_panel("spec", self.spec_rect)
        r = self.spec_rect

        for i in range(1, 5):
            gy = r.y + int(r.height * i / 5)
            pygame.draw.line(self.screen, (33, 42, 60), (r.x + 4, gy), (r.right - 4, gy), 1)
        for i in range(1, 8):
            gx = r.x + int(r.width * i / 8)
            pygame.draw.line(self.screen, (28, 36, 52), (gx, r.y + 4), (gx, r.bottom - 4), 1)

        lbl = self.font_tiny.render(t("spectrum"), True, (110, 128, 150))
        self.screen.blit(lbl, (r.x + 12, r.y + 8))

        if len(spectrum_db) > 1:
            pw = min(420, max(64, r.width))
            db_min, db_max = -80.0, -10.0
            norm = np.clip((spectrum_db - db_min) / (db_max - db_min), 0.0, 1.0)
            norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
            xs = np.linspace(0, len(norm) - 1, pw)
            ys = np.interp(xs, np.arange(len(norm)), norm)
            px = np.linspace(r.x + 4, r.right - 4, pw)
            py = r.bottom - 10 - ys * (r.height - 34)
            pts = [(float(a), float(b)) for a, b in zip(px, py)]

            poly = [(r.x + 4, r.bottom - 4)] + pts + [(r.right - 4, r.bottom - 4)]
            pygame.draw.polygon(self.screen, (14, 78, 88), poly)
            pygame.draw.lines(self.screen, (0, 230, 190), False, pts, 2)

        # センター同調マーカー + 帯域ハイライト
        cx = r.centerx
        bw_hz = 200000 if self.mode == "WFM" else 16000 if self.mode == "NFM" else 12000
        bw_px = max(2, int((bw_hz / self.sample_rate) * r.width))
        shade = pygame.Surface((bw_px, r.height - 12), pygame.SRCALPHA)
        shade.fill((226, 190, 110, 26))
        self.screen.blit(shade, (cx - bw_px // 2, r.y + 6))
        pygame.draw.line(self.screen, C_GOLD, (cx, r.y + 4), (cx, r.bottom - 4), 2)

        # 検出局マーカー
        f_min = self.center_freq - self.sample_rate / 2
        f_max = self.center_freq + self.sample_rate / 2
        for st in self.detected_stations:
            sfreq = st["freq_hz"]
            if f_min <= sfreq <= f_max:
                ratio = (sfreq - f_min) / (f_max - f_min)
                m_x = r.x + int(ratio * r.width)
                color = (58, 200, 140) if st["quality"] == "STRONG" else (230, 178, 75) if st["quality"] == "MEDIUM" else (186, 130, 210)
                pygame.draw.polygon(self.screen, color,
                                    [(m_x, r.y + 16), (m_x - 5, r.y + 6), (m_x + 5, r.y + 6)])
                lbl = self.font_tiny.render(f"{st['freq_mhz']:.1f}", True, color)
                self.screen.blit(lbl, (m_x - 11, r.y + 18))

        f_start = (self.center_freq - self.sample_rate / 2) / 1e6
        f_end = (self.center_freq + self.sample_rate / 2) / 1e6
        l1 = self.font_tiny.render(f"{f_start:.3f} MHz", True, (150, 166, 190))
        l2 = self.font_tiny.render(f"{f_end:.3f} MHz", True, (150, 166, 190))
        self.screen.blit(l1, (r.x + 12, r.bottom - 18))
        self.screen.blit(l2, (r.right - 86, r.bottom - 18))

    def _draw_waterfall(self, spectrum_db):
        self._draw_panel("wf", self.wf_rect)
        self.update_waterfall(spectrum_db)
        self.screen.blit(self.wf_surface, self.wf_rect.topleft)
        pygame.draw.rect(self.screen, (60, 76, 100), self.wf_rect, width=1, border_radius=14)
        cx = self.wf_rect.centerx
        pygame.draw.line(self.screen, C_GOLD, (cx, self.wf_rect.y + 2), (cx, self.wf_rect.bottom - 2), 1)
        lbl = self.font_tiny.render(t("waterfall"), True, (110, 128, 150))
        self.screen.blit(lbl, (self.wf_rect.x + 12, self.wf_rect.y + 6))

    def _draw_waveform(self, audio_pcm):
        self._draw_panel("wave", self.wave_rect)
        r = self.wave_rect
        lbl = self.font_tiny.render(t("waveform"), True, (110, 128, 150))
        self.screen.blit(lbl, (r.x + 12, r.y + 5))
        if audio_pcm is None or len(audio_pcm) < 32:
            return
        n = min(len(audio_pcm), 8192)
        chunk = np.asarray(audio_pcm[-n:], dtype=np.float32)
        pw = r.width - 24
        seg = max(1, n // pw)
        m = (n // seg) * seg
        env = np.max(np.abs(chunk[-m:].reshape(-1, seg)), axis=1).astype(np.float32)
        peak = max(0.01, float(np.percentile(env, 99)))
        env = np.clip(env / peak * 0.92, 0.0, 1.0)
        xs = np.linspace(r.x + 12, r.right - 12, len(env))
        mid = r.centery + 7
        amp = r.height / 2 - 13
        top = [(float(x), float(mid - e * amp)) for x, e in zip(xs, env)]
        bot = [(float(x), float(mid + e * amp)) for x, e in zip(xs, env)]
        if len(top) >= 2:
            pygame.draw.polygon(self.screen, (14, 78, 88), top + bot[::-1])
            pygame.draw.lines(self.screen, (0, 220, 184), False, top, 1)
            pygame.draw.lines(self.screen, (0, 220, 184), False, bot, 1)

    def _draw_telemetry(self):
        self._draw_panel("tele", self.tele_rect)
        lbl = self.font_title.render(t("status"), True, C_MUTED)
        self.screen.blit(lbl, (self.tele_rect.x + 14, self.tele_rect.y + 8))
        parts = [p.strip() for p in self.telemetry_text.split("|") if p.strip()]
        for i, part in enumerate(parts[:6]):
            col, row = i // 3, i % 3
            x = self.tele_rect.x + 14 + col * 132
            y = self.tele_rect.y + 28 + row * 16
            surf = self.font_small.render(part, True, C_TEXT)
            self.screen.blit(surf, (x, y))

        # 音量 / ゲインのスリムインジケータ
        bx = self.tele_rect.x + 14
        by = self.tele_rect.bottom - 13
        pygame.draw.rect(self.screen, (222, 229, 240), (bx, by, 250, 5), border_radius=3)
        pygame.draw.rect(self.screen, C_ACCENT, (bx, by, int(250 * self.volume), 5), border_radius=3)
        gain_norm = min(1.0, max(0.0, (self.gain_val - 10.0) / 45.0)) if self.gains_list else 0.5
        pygame.draw.rect(self.screen, (222, 229, 240), (bx + 130, by, 120, 5), border_radius=3)
        pygame.draw.rect(self.screen, C_GOLD, (bx + 130, by, int(120 * gain_norm), 5), border_radius=3)

    def _draw_controls(self):
        # TUNINGパネル
        self._draw_panel("tune", self.tune_rect)
        lbl = self.font_title.render(t("tuning"), True, C_MUTED)
        self.screen.blit(lbl, (self.tune_rect.x + 14, self.tune_rect.y + 8))
        lbl2 = self.font_tiny.render(t("mode"), True, C_MUTED)
        self.screen.blit(lbl2, (self.tune_rect.x + 14, self.tune_rect.y + 156))

        # GAIN/AUDIOパネル
        self._draw_panel("gain", self.gain_rect)
        lbl = self.font_title.render(t("gain_audio"), True, C_MUTED)
        self.screen.blit(lbl, (self.gain_rect.x + 14, self.gain_rect.y + 6))

        # プリセットパネル
        self._draw_panel("preset", self.preset_rect)
        self.screen.blit(self.font_tiny.render("FM", True, C_MUTED), (self.preset_rect.x + 16, 578))
        self.screen.blit(self.font_tiny.render("AM", True, C_MUTED), (self.preset_rect.x + 16, 606))

        # ステータスバー
        self._draw_panel("status", self.status_rect)
        st_surf = self.font_small.render(f"{t('rx_status')}: {self.scan_status_text}", True, C_ACCENT_DARK)
        self.screen.blit(st_surf, (self.status_rect.x + 16, self.status_rect.y + 6))
        cnt = self.font_small.render(f"{t('detected')}: {len(self.detected_stations)}", True, C_MUTED)
        self.screen.blit(cnt, (self.status_rect.right - 110, self.status_rect.y + 6))

        # 全ボタン描画 (アクティブ状態更新)
        self.btn_wfm.is_active = (self.mode == "WFM")
        self.btn_am.is_active = (self.mode == "AM")
        self.btn_nfm.is_active = (self.mode == "NFM")
        for btn in self.buttons:
            btn.draw(self.screen, self.font_small)

    def render(self, spectrum_db, audio_pcm=None):
        """1フレームの描画処理"""
        self.screen.blit(self.bg_surface, (0, 0))

        self._draw_header()
        self._draw_spectrum(spectrum_db)
        self._draw_waterfall(spectrum_db)
        self._draw_waveform(audio_pcm)
        self._draw_telemetry()
        self._draw_controls()

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()
