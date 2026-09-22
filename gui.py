"""
Pygame-based SDR GUI Module - Frosted Glass Edition.
スペクトラムアナライザ、ウォーターフォール、波形表示、選局コントロールを備えた
ガラス調 (Frosted Glass) モダンGUI。
"""

import numpy as np
import pygame
import time
from collections import OrderedDict

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
        pygame.display.set_caption("KomorebiSDR")
        f_title = pygame.font.SysFont(JP_FONTS, 22, bold=True)
        f_body = pygame.font.SysFont(JP_FONTS, 15)
        clock = pygame.time.Clock()
        while True:
            for e in pygame.event.get():
                if e.type in (pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                    pygame.quit()
                    return
            screen.fill(C_BG_TOP)
            ts = cached_text(f_title, title, (178, 62, 62))
            screen.blit(ts, (28, 22))
            y = 76
            for line in lines:
                s = cached_text(f_body, line, C_TEXT)
                screen.blit(s, (28, y))
                y += 24
            hint = cached_text(f_body, t("quit_hint"), C_MUTED)
            screen.blit(hint, (28, height - 36))
            pygame.display.flip()
            clock.tick(30)
    except Exception:
        pass


# 文字サーフェスのキャッシュ (dirty update方式)
# 毎フレームの font.render (ラスタライズ) を排除。内容が変わった時だけ再生成する。
_TEXT_CACHE = OrderedDict()
_TEXT_CACHE_MAX = 512


def find_spectrum_peaks(spectrum_db, sample_rate: float, center_freq: float,
                        top_n: int = 14, min_snr_db: float = 6.0,
                        min_sep_hz: float = 25000.0) -> list:
    """表示スペクトラムからliveピークを抽出 (スキャン不要のワンクリック選局用)。
    ノイズフロア=中央値、局所最大＋3点放物線補間で周波数を精密化し、
    min_sep_hzで間引いた上位top_n件を返す。
    戻り値: [{"freq_hz": float, "snr_db": float, "db": float}] (snr降順)。
    """
    try:
        spec = np.asarray(spectrum_db, dtype=np.float64)
    except Exception:
        return []
    n = len(spec)
    if n < 8 or not np.isfinite(sample_rate) or sample_rate <= 0:
        return []
    spec = np.nan_to_num(spec, nan=-120.0, posinf=0.0, neginf=-120.0)
    floor = float(np.partition(spec, n // 2)[n // 2])
    if not np.isfinite(floor):
        return []
    inner = spec[1:-1]
    is_peak = ((inner > spec[:-2]) & (inner >= spec[2:])
               & (inner >= floor + min_snr_db))
    idx = np.flatnonzero(is_peak) + 1
    if len(idx) == 0:
        return []
    # 3点放物線補間 (ビン内精度)
    denom = (spec[idx - 1] - 2.0 * spec[idx] + spec[idx + 1])
    shift = np.zeros(len(idx))
    nz = denom != 0.0
    shift[nz] = 0.5 * (spec[idx[nz] - 1] - spec[idx[nz] + 1]) / denom[nz]
    shift = np.clip(shift, -0.5, 0.5)
    order = np.argsort(spec[idx])[::-1]
    peaks = []
    taken_hz = []
    for k in order:
        b = float(idx[k]) + float(shift[k])
        f_hz = center_freq - sample_rate / 2.0 + b / n * sample_rate
        if any(abs(f_hz - t) < min_sep_hz for t in taken_hz):
            continue
        taken_hz.append(f_hz)
        peaks.append({"freq_hz": float(f_hz),
                      "snr_db": float(spec[idx[k]] - floor),
                      "db": float(spec[idx[k]])})
        if len(peaks) >= top_n:
            break
    return peaks


def cached_text(font, text, color):
    # fontオブジェクト自体をキーに (id()は解放後の再利用で誤ヒットし得る)。
    # 実使用フォントはSdrGuiが保持するため参照保持コストは実質ゼロ。
    # 変動文字列のスラッシング対策にLRU方式 (上限超で全消去スパイクを排除)。
    key = (font, text, color)
    surf = _TEXT_CACHE.get(key)
    if surf is None:
        surf = font.render(text, True, color)
        _TEXT_CACHE[key] = surf
        if len(_TEXT_CACHE) > _TEXT_CACHE_MAX:
            _TEXT_CACHE.popitem(last=False)
    else:
        _TEXT_CACHE.move_to_end(key)
    return surf


class Button:
    """クリック可能なUIボタン (ガラス調・立体押し込みフィードバック付き)"""

    def __init__(self, rect, text, callback, bg_color=None, active_color=None, dark_text=True, radius=8):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.callback = callback
        self.bg_color = bg_color if bg_color is not None else C_BTN
        self.active_color = active_color if active_color is not None else C_BTN_ACTIVE
        self.dark_text = dark_text
        self.is_active = False
        self.hover = False
        self.pressed = False
        self.radius = radius
        self.visible = True  # SSB/CW時のみ表示するボタン用 (BFO±)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
            if not self.hover:
                self.pressed = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.pressed = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            was_pressed = self.pressed
            self.pressed = False
            if was_pressed and self.rect.collidepoint(event.pos):
                self.callback()
                return True
        return False

    def draw(self, surface, font):
        # 押下時は1px下に沈み込む立体物理フィードバック
        draw_rect = self.rect.move(0, 1) if self.pressed else self.rect

        if self.is_active:
            fill = self.active_color
            txt_color = (255, 255, 255)
            border = self.active_color
        else:
            fill = self.bg_color
            if self.pressed:
                fill = tuple(max(0, c - 15) for c in fill)
            elif self.hover:
                fill = tuple(min(255, c + 18) for c in fill)
            txt_color = C_TEXT if self.dark_text else (255, 255, 255)
            border = (160, 185, 215) if self.hover else C_BTN_BORDER

        border_w = 2 if self.is_active else 1
        pygame.draw.rect(surface, fill, draw_rect, border_radius=self.radius)
        pygame.draw.rect(surface, border, draw_rect, width=border_w, border_radius=self.radius)
        txt_surf = cached_text(font, self.text, txt_color)
        txt_rect = txt_surf.get_rect(center=draw_rect.center)
        surface.blit(txt_surf, txt_rect)


class SdrGui:
    """SDR デスクトップGUIクラス (Frosted Glass Edition)"""

    def __init__(self, width=1120, height=720):
        pygame.init()
        pygame.font.init()
        # ゼロ・極小サイズでのゼロ除算・Surface生成失敗を防止
        self.width = max(320, int(width))
        self.height = max(240, int(height))
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("KomorebiSDR")

        # フォント (日本語対応: Meiryo系を優先)
        self.font_title = pygame.font.SysFont(JP_FONTS, 14, bold=True)
        self.font_huge = pygame.font.SysFont(LATIN_FONTS, 40, bold=True)
        self.font_station = pygame.font.SysFont(JP_FONTS, 22, bold=True)
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
        self.tele_rect = pygame.Rect(826, 110, 280, 102)
        self.tune_rect = pygame.Rect(826, 222, 280, 326)
        self.gain_rect = pygame.Rect(826, 548, 0, 0)
        self.preset_rect = pygame.Rect(14, 560, 1092, 74)
        self.status_rect = pygame.Rect(14, 644, 1092, 28)

        self.wf_surface = pygame.Surface((self.wf_rect.width, self.wf_rect.height))
        self.wf_surface.fill((0, 0, 0))

        # パラメータコールバック
        self.on_freq_change = None
        self.on_mode_change = None
        # ゲイン・音量ボタンは廃止 (自動＋システム音量に一本化・木漏れ日整理)。
        # Filterはclean固定 (ボタン削除・木漏れ日整理)。
        self.on_seek_change = None      # lambda direction: ...
        self.on_scan_request = None     # lambda: ...
        self.on_sw_scan_request = None  # lambda: ...
        # PPM手動較正は廃止 (背景自動収集＋自動適用に一本化)。
        # ステレオは自動ブレンドに一本化 (ボタン削除・木漏れ日整理)。
        self.on_bfo_change = None       # lambda delta_hz: ...
        # AFCは常時ON固定・DXはC/N連動の自動絞り (ボタン削除・木漏れ日整理)。

        # 内部状態
        self.center_freq = 80000000  # 80.0 MHz
        self.sample_rate = 1152000
        self.mode = "WFM"
        self.volume = 0.5  # 表示用。実音量は起動時config＋システム音量。
        self.is_hard_locked = False     # 収束決め打ち中フラグ (表示用)
        self.filter_mode = "clean"      # 常時clean固定 (ボタン廃止)
        self.current_rssi = -50.0
        self.is_stereo = False
        self.stereo_status = "MONO"
        self.stereo_enabled = True  # 常時True固定 (自動ブレンドに一本化)
        # NRは常時ON固定 (ボタン削除・木漏れ日整理)。nr_enabled属性は廃止。
        # 描画用の事前確保バッファ (毎フレームの確保を排除)
        self._wave_xs = None
        self._spec_px = None
        self._wf_x_src = None
        self._wf_lo = None
        self._wf_w = None
        self._wf_src_n = -1
        self._wf_line_surf = None
        self._wf_arr = None
        self.region_label = ""
        self.station_name = ""   # メインから設定 (RDS PS / 番組表 / 既知局)
        self.scan_range = (76.0, 95.0)
        self.presets_fm = []
        self.presets_am = []
        self.detected_stations = []
        self.live_peaks = []          # 表示中スペクトラムのliveピーク (ワンクリック選局用)
        self._last_peak_time = 0.0    # liveピーク更新時刻 (5Hz間引き)
        self.hover_freq_hz = None     # スペクトラム上のマウス位置の周波数
        self.station_list_open = False  # 検出局プルダウンの開閉
        self.station_list_scroll = 0    # プルダウンのスクロール行位置
        self.scan_status_text = "待機中 (Auto Seek / 全帯域スキャン可能)"
        self.telemetry_text = ""

        # Sメーター / RDS / 短波番組表示属性
        self.rds_text = ""              # RDS RadioText (♪ 楽曲名/番組名)
        self.sw_info = ""               # 短波EiBi番組情報
        self.s_units = 0.0              # Sメーター値 (0.0〜9.0+, S9=+0dB)
        self.s_peak_units = 0.0         # ピークホールド値
        self.s_peak_time = 0.0          # ピーク保持タイムスタンプ
        self.hovered_freq_digit = None  # マウスホバー中の周波数桁 (1e6, 1e5, 1e4 等)
        self.freq_digit_hitboxes = []   # 周波数の各桁当たり判定 [(rect, step_hz), ...]
        self.baked_bg = None            # 全ガラスパネル合成済みの背景Surface

        # 静的パネルの事前描画 (ガラス表現 & シーンベイク)
        self._prerender_background()
        self._prerender_panels()
        self._bake_static_scene()

        # UIボタンのリスト
        self.buttons = []
        self._init_controls()

    # ================================================================
    # 事前描画 (ガラスパネル / 背景 / 静的ベイク)
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
            ("preset", self.preset_rect),
        ):
            self.panels[name] = self._glass(rect.width, rect.height)
        self.panels["gain"] = (pygame.Surface((1, 1), pygame.SRCALPHA), 0)
        self.panels["spec"] = self._glass(self.spec_rect.width, self.spec_rect.height,
                                          radius=16, tint=(17, 23, 36), dark=True)
        self.panels["wf"] = self._glass(self.wf_rect.width, self.wf_rect.height,
                                        radius=16, tint=(17, 23, 36), dark=True)
        self.panels["wave"] = self._glass(self.wave_rect.width, self.wave_rect.height,
                                          radius=16, tint=(17, 23, 36), dark=True)
        self.panels["status"] = self._glass(self.status_rect.width, self.status_rect.height,
                                            radius=14, alpha=160)

    def _bake_static_scene(self):
        """全Frosted Glassパネルと静的見出しを1枚のSurfaceに事前ベイク (毎フレームの半透明合成を全廃)"""
        baked = self.bg_surface.copy()
        for name, rect in (
            ("hero", self.hero_rect), ("info", self.info_rect),
            ("tele", self.tele_rect), ("tune", self.tune_rect),
            ("preset", self.preset_rect),
            ("spec", self.spec_rect), ("wf", self.wf_rect),
            ("wave", self.wave_rect), ("status", self.status_rect),
        ):
            surf, pad = self.panels[name]
            baked.blit(surf, (rect.x - pad, rect.y - pad))

        # 静的見出しラベルの事前ベイク
        lbl_tune = cached_text(self.font_title, t("tuning"), C_MUTED)
        baked.blit(lbl_tune, (self.tune_rect.x + 14, self.tune_rect.y + 8))
        lbl_mode = cached_text(self.font_tiny, t("mode"), C_MUTED)
        baked.blit(lbl_mode, (self.tune_rect.x + 14, self.tune_rect.y + 170))

        lbl_tele = cached_text(self.font_title, t("status"), C_MUTED)
        baked.blit(lbl_tele, (self.tele_rect.x + 14, self.tele_rect.y + 8))

        lbl_wf = cached_text(self.font_tiny, t("waterfall"), (110, 128, 150))
        baked.blit(lbl_wf, (self.wf_rect.x + 12, self.wf_rect.y + 6))
        lbl_wave = cached_text(self.font_tiny, t("waveform"), (110, 128, 150))
        baked.blit(lbl_wave, (self.wave_rect.x + 12, self.wave_rect.y + 5))

        self.baked_bg = baked

    def _draw_panel(self, name, rect):
        """互換用 (ベイク済みの場合は呼び出し省略可能)"""
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
            self.btn_scan_band.text = t("scan_button")

    def set_presets(self, presets_fm: list, presets_am: list):
        """スキャン結果などからプリセットボタンを再構築する。
        破損エントリ (freq_hz欠落・範囲外・未知mode) は除去してKeyErrorを防止。
        Noneでは既存ボタンを維持する (FM/SW両スキャンの片側維持用)。
        空リスト[]は明示的全消去として扱う。"""
        fm = self._clean_presets(presets_fm, "WFM")
        am = self._clean_presets(presets_am, "AM")
        if presets_fm is not None:
            self.presets_fm = fm[:10]
        if presets_am is not None:
            self.presets_am = am[:7]
        self._init_controls()
        # トグル系ボタンは全廃止したため再同期不要 (_sync_control_states削除)。

    @staticmethod
    def _clean_presets(items, default_mode: str) -> list:
        valid_modes = {"WFM", "AM", "NFM", "USB", "LSB", "CW"}
        out = []
        for p in (items or []):
            if not isinstance(p, dict):
                continue
            f = p.get("freq_hz")
            if isinstance(f, bool) or not isinstance(f, (int, float)):
                continue
            fi = int(f)
            if not (100000 <= fi <= 1750000000):
                continue
            m = p.get("mode", default_mode)
            if not isinstance(m, str) or m not in valid_modes:
                m = default_mode
            out.append({"freq_hz": fi, "name": str(p.get("name", "?")),
                        "mode": m})
        return out

    def _init_controls(self):
        btns = []

        # ---- TUNINGパネル (上からバランスよくゆったり配置) ----
        tx, tw = self.tune_rect.x + 14, self.tune_rect.width - 28
        half = (tw - 8) // 2
        y_seek = self.tune_rect.y + 36
        self.btn_seek_prev = Button((tx, y_seek, half, 34), t("seek_prev"), lambda: self._seek(-1),
                                    bg_color=(226, 235, 248), active_color=C_BTN_ACTIVE2)
        self.btn_seek_next = Button((tx + half + 8, y_seek, half, 34), t("seek_next"), lambda: self._seek(1),
                                    bg_color=(226, 235, 248), active_color=C_BTN_ACTIVE2)
        fm_w = int((tw - 8) * 0.52)
        sw_w = tw - 8 - fm_w
        y_scan = y_seek + 34 + 10
        self.btn_scan_band = Button((tx, y_scan, fm_w, 34), t("scan_button"), self._request_scan,
                                    bg_color=(214, 240, 229), active_color=C_ACCENT)
        self.btn_scan_sw = Button((tx + fm_w + 8, y_scan, sw_w, 34), t("scan_sw_button"),
                                   self._request_sw_scan, bg_color=(226, 236, 248), active_color=C_ACCENT)
        btns.extend([self.btn_seek_prev, self.btn_seek_next, self.btn_scan_band, self.btn_scan_sw])

        # 検出局プルダウン (生成画像スタイルの淡いエメラルドアクセント)
        y_list = y_scan + 34 + 10
        n_st = len(self.detected_stations)
        self.btn_station_list = Button((tx, y_list, tw, 30), t("station_list_btn", n=n_st),
                                       self._toggle_station_list,
                                       bg_color=(200, 238, 226), active_color=C_ACCENT)
        btns.append(self.btn_station_list)

        gap = 4
        mw = (tw - 5 * gap) // 6
        mode_defs = [("WFM", (212, 236, 248)), ("AM", (212, 236, 248)), ("NFM", (212, 236, 248)),
                     ("USB", (226, 236, 250)), ("LSB", (226, 236, 250)), ("CW", (226, 236, 250))]
        mode_buttons = {}
        y_mode = self.tune_rect.y + 192
        for i, (name, col) in enumerate(mode_defs):
            b = Button((tx + i * (mw + gap), y_mode, mw, 30), name,
                       (lambda m=name: self._set_mode(m)), bg_color=col)
            mode_buttons[name] = b
            btns.append(b)
        self.mode_buttons = mode_buttons
        self.btn_wfm = self.mode_buttons["WFM"]
        self.btn_am = self.mode_buttons["AM"]
        self.btn_nfm = self.mode_buttons["NFM"]

        # BFO微調整 (SSB・CW時に表示)
        gap2 = 8
        hw = (tw - gap2) // 2
        y_bfo = y_mode + 30 + 10
        self.btn_bfo_down = Button((tx, y_bfo, hw, 28), "BFO-",
                                   lambda: self._step_bfo(-50), bg_color=(240, 243, 248))
        self.btn_bfo_up = Button((tx + hw + gap2, y_bfo, hw, 28), "BFO+",
                                 lambda: self._step_bfo(50), bg_color=(240, 243, 248))
        btns.extend([self.btn_bfo_down, self.btn_bfo_up])

        # ---- チャンネルカード / プリセット (どこでも・だれでも・どんなアンテナでも) ----
        self.preset_buttons = []
        has_presets = bool(self.presets_fm or self.presets_am)
        if not has_presets:
            # 未スキャン時: 白紙にせず、ワンクリックでスキャンできる大きなウェルカムボタンを配置
            px = self.preset_rect.x + 20
            py = self.preset_rect.y + 16
            pw = (self.preset_rect.width - 56) // 2
            self.btn_welcome_fm = Button(
                (px, py, pw, 42),
                t("welcome_scan_fm"),
                self._request_scan,
                bg_color=(216, 242, 230), active_color=C_ACCENT
            )
            self.btn_welcome_sw = Button(
                (px + pw + 16, py, pw, 42),
                t("welcome_scan_sw"),
                self._request_sw_scan,
                bg_color=(228, 238, 252), active_color=C_ACCENT
            )
            btns.extend([self.btn_welcome_fm, self.btn_welcome_sw])
        else:
            # スキャン完了後: 検出局を美しいカードタイルとして均等配置 (押しやすいH=28px)
            if self.presets_fm:
                n_fm = min(10, len(self.presets_fm))
                gap_p = 6
                fw = (self.preset_rect.width - 60 - (n_fm - 1) * gap_p) // n_fm
                fx0 = self.preset_rect.x + 48
                for i, p in enumerate(self.presets_fm[:n_fm]):
                    f, m = int(p["freq_hz"]), p.get("mode", "WFM")
                    cb = (lambda freq=f, mode=m: self._tune(freq, mode))
                    b = Button((fx0 + i * (fw + gap_p), self.preset_rect.y + 7, fw, 28),
                               p["name"], cb, bg_color=(235, 242, 250), radius=6)
                    b.freq_hz = f
                    self.preset_buttons.append(b)
                    btns.append(b)
            if self.presets_am:
                n_am = min(8, len(self.presets_am))
                gap_a = 6
                aw = (self.preset_rect.width - 60 - (n_am - 1) * gap_a) // n_am
                ax0 = self.preset_rect.x + 48
                for i, p in enumerate(self.presets_am[:n_am]):
                    f, m = int(p["freq_hz"]), p.get("mode", "AM")
                    cb = (lambda freq=f, mode=m: self._tune(freq, mode))
                    b = Button((ax0 + i * (aw + gap_a), self.preset_rect.y + 39, aw, 28),
                               p["name"], cb, bg_color=(238, 240, 246), radius=6)
                    b.freq_hz = f
                    self.preset_buttons.append(b)
                    btns.append(b)

        # アトミックに差し替え (スキャン完了時の再構築と描画の競合防止)
        self.buttons = btns
        self._sync_bfo_visibility()

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
        self._sync_bfo_visibility()
        if self.on_mode_change:
            self.on_mode_change(self.mode)

    def _sync_bfo_visibility(self):
        """BFO±はSSB/CW選択時のみ表示 (木漏れ日整理)。"""
        show = self.mode in ("USB", "LSB", "CW")
        for name in ("btn_bfo_down", "btn_bfo_up"):
            btn = getattr(self, name, None)
            if btn is not None:
                btn.visible = show

    def _tune(self, freq, mode):
        self.center_freq = freq
        self.mode = mode
        self._sync_bfo_visibility()
        if self.on_freq_change:
            self.on_freq_change(self.center_freq)
        if self.on_mode_change:
            self.on_mode_change(self.mode)

    def _seek(self, direction):
        if self.on_seek_change:
            self.on_seek_change(direction)

    def _request_scan(self):
        if self.on_scan_request:
            self.scan_status_text = t("scanning", start=f"{self.scan_range[0]:g}",
                                      end=f"{self.scan_range[1]:g}")
            self.on_scan_request()

    def _request_sw_scan(self):
        if self.on_sw_scan_request:
            self.scan_status_text = t("sw_scanning")
            self.on_sw_scan_request()

    # ---- 検出局プルダウン ----
    _SL_ROW_H = 26
    _SL_HEADER_H = 30
    _SL_VISIBLE = 12
    _SL_WIDTH = 340

    def _toggle_station_list(self):
        if not self.detected_stations:
            self.scan_status_text = "検出局なし (スキャンしてください)"
            return
        self.station_list_open = not self.station_list_open
        self.station_list_scroll = 0

    def _station_list_layout(self):
        """プルダウンの配置を返す (panel_rect, row_rects, total)。純粋計算のみ。"""
        total = len(self.detected_stations)
        vis = min(total, self._SL_VISIBLE)
        w, rh, hh = self._SL_WIDTH, self._SL_ROW_H, self._SL_HEADER_H
        h = hh + vis * rh + 8
        panel = pygame.Rect(self.width // 2 - w // 2, self.height // 2 - h // 2, w, h)
        rows = [pygame.Rect(panel.x + 8, panel.y + hh + i * rh, w - 16, rh)
                for i in range(vis)]
        return panel, rows, total

    def _station_list_click(self, mx: int, my: int) -> bool:
        """プルダウン開閉中のクリック処理。消費したらTrue。"""
        if not self.station_list_open:
            return False
        # ワーカーによる detected_stations 差し替えとのレース防止: スナップショットで一貫参照
        sts = list(self.detected_stations) if self.detected_stations else []
        panel, rows, total = self._station_list_layout()
        # レイアウトが旧リスト長で計算される場合に備え、スナップショット長で補正
        total = len(sts)
        start = max(0, min(self.station_list_scroll, max(0, total - len(rows))))
        for i, rc in enumerate(rows):
            if rc.collidepoint(mx, my):
                idx = start + i
                if 0 <= idx < total:
                    try:
                        st = sts[idx]
                        fh = st.get("freq_hz")
                        if fh is None:
                            continue
                        self.center_freq = int(fh)
                        self.scan_status_text = (
                            f"局リスト選局: {st.get('name', '')} "
                            f"({st.get('freq_mhz', float(fh) / 1e6):.2f}MHz, "
                            f"SNR:+{st.get('snr_db', 0.0):.1f}dB)")
                        if self.on_freq_change:
                            self.on_freq_change(self.center_freq)
                    except Exception:
                        pass
                self.station_list_open = False
                return True
        # パネル外クリックで閉じる
        if not panel.collidepoint(mx, my):
            self.station_list_open = False
        return True

    def _draw_station_list(self):
        if not self.station_list_open or not self.detected_stations:
            return
        panel, rows, total = self._station_list_layout()
        dim = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        dim.fill((10, 16, 28, 90))
        self.screen.blit(dim, (0, 0))
        pygame.draw.rect(self.screen, (248, 250, 253), panel, border_radius=12)
        pygame.draw.rect(self.screen, (90, 110, 140), panel, width=2, border_radius=12)
        title = cached_text(self.font_med, f"検出局 ({total})  — クリックで選局",
                            (30, 60, 90))
        self.screen.blit(title, (panel.x + 14, panel.y + 6))
        start = max(0, min(self.station_list_scroll, max(0, total - len(rows))))
        cur = int(self.center_freq)
        sts_draw = list(self.detected_stations)
        for i, rc in enumerate(rows):
            idx = start + i
            if idx >= total or idx >= len(sts_draw):
                break
            try:
                st = sts_draw[idx]
                fh = st.get("freq_hz")
                if fh is None:
                    continue
                sel = abs(int(fh) - cur) < 50000
            except Exception:
                continue
            if sel:
                pygame.draw.rect(self.screen, (214, 236, 248), rc, border_radius=6)
            name = str(st.get("name", ""))[:18]
            try:
                freq = st.get("freq_mhz", float(st.get("freq_hz", 0)) / 1e6)
            except Exception:
                freq = 0.0
            snr = st.get("snr_db", 0.0)
            line = cached_text(self.font_small, f"{name}  {freq:.2f}MHz  +{snr:.1f}dB",
                               (30, 50, 80))
            self.screen.blit(line, (rc.x + 8, rc.y + 5))
        if total > len(rows):
            hint = cached_text(self.font_tiny, "ホイールでスクロール・Escで閉じる",
                               (110, 128, 150))
            self.screen.blit(hint, (panel.x + 14, panel.bottom - 20))

    def _step_bfo(self, delta):
        if self.on_bfo_change:
            self.on_bfo_change(delta)

    def handle_events(self):
        cursor_hand = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.station_list_open = False

            # マウス移動時の桁ホバー検出
            if event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                self.hovered_freq_digit = None
                if self.hero_rect.collidepoint(mx, my):
                    for rect, step in self.freq_digit_hitboxes:
                        if rect.collidepoint(mx, my):
                            self.hovered_freq_digit = step
                            break

            # マウスホイールによる周波数同調 (桁単位ホイール同調)
            # pygame2はMOUSEWHEELを出すため、MOUSEBUTTONDOWN(4/5)は旧SDLの
            # フォールバック限定 (両対応だと1ノッチで二重に動く)
            use_legacy_wheel = pygame.version.vernum[0] < 2
            wheel_delta = 0
            if event.type == pygame.MOUSEWHEEL:
                wheel_delta = event.y
            elif use_legacy_wheel and event.type == pygame.MOUSEBUTTONDOWN and event.button in (4, 5):
                wheel_delta = 1 if event.button == 4 else -1

            if wheel_delta != 0:
                mx, my = pygame.mouse.get_pos()
                if self.station_list_open:
                    panel, rows, total = self._station_list_layout()
                    if panel.collidepoint(mx, my) and total > len(rows):
                        self.station_list_scroll = max(
                            0, min(total - len(rows),
                                    self.station_list_scroll - wheel_delta))
                        wheel_delta = 0
            if wheel_delta != 0:
                mx, my = pygame.mouse.get_pos()
                if self.hero_rect.collidepoint(mx, my):
                    # ホバー中の桁、またはデフォルト100kHz刻みで同調
                    step = self.hovered_freq_digit if self.hovered_freq_digit is not None else 100000
                    self._adjust_freq(wheel_delta * step)
                elif self.spec_rect.collidepoint(mx, my) or self.wf_rect.collidepoint(mx, my):
                    # スペクトラム上のホイールは10kHzまたは1kHz刻み
                    step = 1000 if self.mode in ("USB", "LSB", "CW") else 10000
                    self._adjust_freq(wheel_delta * step)

            # スペクトラム・ウォーターフォールクリックによる同調 (検出局マーカーへの自動吸着対応)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if self.station_list_open:
                    # モーダル消費時は下層ボタン/同調へ素通りさせない
                    self._station_list_click(mx, my)
                    continue
                elif self.spec_rect.collidepoint(mx, my) or self.wf_rect.collidepoint(mx, my):
                    w = self.spec_rect.width
                    ratio = (mx - self.spec_rect.x) / w if w > 0 else 0.5
                    sr = self.sample_rate if self.sample_rate > 0 else 1152000
                    clicked_freq = (self.center_freq - sr / 2) + ratio * sr

                    snapped_station = None
                    min_dist = float("inf")
                    for st in list(self.detected_stations):
                        try:
                            fh = st.get("freq_hz")
                            if fh is None:
                                continue
                            dist_hz = abs(float(fh) - clicked_freq)
                            if dist_hz <= 45000 and dist_hz < min_dist:
                                min_dist = dist_hz
                                snapped_station = st
                        except Exception:
                            continue

                    if snapped_station:
                        try:
                            self.center_freq = int(snapped_station.get("freq_hz", self.center_freq))
                            self.scan_status_text = (f"局吸着同調: {snapped_station.get('name', '')} "
                                                     f"({snapped_station.get('freq_mhz', float(self.center_freq) / 1e6):.2f}MHz, SNR:+{snapped_station.get('snr_db', 0.0)}dB)")
                        except Exception:
                            pass
                    else:
                        # 第2段: liveピーク吸着 (スキャン不要。30kHz以内で正確周波数へ)
                        best_pk = None
                        best_pd = 30000.0
                        for pk in list(self.live_peaks):
                            try:
                                pf = pk.get("freq_hz")
                                if pf is None:
                                    continue
                                pd = abs(float(pf) - clicked_freq)
                                if pd < best_pd:
                                    best_pd = pd
                                    best_pk = pk
                            except Exception:
                                continue
                        if best_pk is not None:
                            try:
                                from auto_tuner import match_station_name
                                pk_name = match_station_name(int(best_pk.get("freq_hz", 0)))
                            except Exception:
                                pk_name = ""
                            try:
                                self.center_freq = int(best_pk.get("freq_hz", self.center_freq))
                                self.scan_status_text = (
                                    f"ピーク同調: {pk_name} "
                                    f"({float(best_pk.get('freq_hz', 0)) / 1e6:.2f}MHz, SNR:+{float(best_pk.get('snr_db', 0.0)):.1f}dB)")
                            except Exception:
                                pass
                        else:
                            self.center_freq = int(round(clicked_freq / 10000) * 10000)

                    if self.on_freq_change:
                        self.on_freq_change(self.center_freq)

            # ボタンイベント処理 (非表示ボタンは無視)
            for btn in self.buttons:
                if btn.visible:
                    btn.handle_event(event)

        # カーソル形状の更新 (ボタンホバー中または周波数桁ホバー中は指先ハンドカーソル)
        mx, my = pygame.mouse.get_pos()
        is_hover_btn = any(btn.visible and btn.rect.collidepoint(mx, my) for btn in self.buttons)
        is_hover_digit = (self.hovered_freq_digit is not None)
        # スペクトラム上のホバー周波数 (ワンクリック選局の照準表示)
        if self.spec_rect.collidepoint(mx, my) and self.spec_rect.width > 0:
            sr = self.sample_rate if self.sample_rate > 0 else 1152000
            self.hover_freq_hz = ((self.center_freq - sr / 2)
                                  + (mx - self.spec_rect.x) / self.spec_rect.width * sr)
        else:
            self.hover_freq_hz = None
        if is_hover_btn or is_hover_digit or self.hover_freq_hz is not None:
            try:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
            except Exception:
                pass
        else:
            try:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
            except Exception:
                pass

    # ================================================================
    # 描画
    # ================================================================
    def update_waterfall(self, spectrum_db: np.ndarray):
        """ウォーターフォールにライン追加 (スクロール描画)"""
        wf_w = self.wf_rect.width
        self.wf_surface.scroll(0, 2)

        n = len(spectrum_db)
        # 補間係数とサーフェスを事前確保 (内容が同じ間は再計算しない)
        if (self._wf_line_surf is None or self._wf_src_n != n
                or self._wf_line_surf.get_width() != wf_w):
            self._wf_x_src = np.linspace(0, n - 1, wf_w)
            self._wf_lo = np.clip(np.floor(self._wf_x_src).astype(np.int32), 0, n - 2)
            self._wf_w = self._wf_x_src - self._wf_lo
            self._wf_src_n = n
            self._wf_line_surf = pygame.Surface((wf_w, 2))
            self._wf_arr = np.empty((wf_w, 2, 3), dtype=np.uint8)

        norm = np.clip((spectrum_db + 70.0) / 55.0, 0.0, 1.0)
        indices = (norm * 255).astype(np.uint8)
        rgb_line = self.colormap[indices]

        lo, w = self._wf_lo, self._wf_w
        r = (rgb_line[lo, 0] * (1.0 - w) + rgb_line[lo + 1, 0] * w).astype(np.uint8)
        g = (rgb_line[lo, 1] * (1.0 - w) + rgb_line[lo + 1, 1] * w).astype(np.uint8)
        b = (rgb_line[lo, 2] * (1.0 - w) + rgb_line[lo + 1, 2] * w).astype(np.uint8)

        arr = self._wf_arr
        arr[:, 0, 0] = r
        arr[:, 0, 1] = g
        arr[:, 0, 2] = b
        arr[:, 1, :] = arr[:, 0, :]
        pygame.surfarray.blit_array(self._wf_line_surf, arr)
        self.wf_surface.blit(self._wf_line_surf, (0, 0))

    def _draw_chip(self, text, x, y, fill, txt_color=C_TEXT):
        font = self.font_small
        w = font.size(text)[0] + 16
        rect = pygame.Rect(x, y, w, 20)
        pygame.draw.rect(self.screen, fill, rect, border_radius=10)
        surf = cached_text(font, text, txt_color)
        self.screen.blit(surf, (x + 8, y + 3))
        return w

    def _draw_smeter(self, x, y, width, height):
        """本格的マルチセグメント Sメーター LEDバー (S1〜S9, +10〜+60dB)"""
        # 現在値とピークホールドの更新
        su = max(0.0, float(self.s_units))
        now = pygame.time.get_ticks() / 1000.0
        if su >= self.s_peak_units:
            self.s_peak_units = su
            self.s_peak_time = now
        elif now - self.s_peak_time > 0.8:
            # 0.8秒ホールド後に毎秒約6S-unitsの速度でスムーズに減衰
            self.s_peak_units = max(su, self.s_peak_units - 6.0 * 0.033)

        # 1. ヘッダーテキスト (S-METER見出し & 現在値バッジ)
        lbl = cached_text(self.font_tiny, "S-METER", C_MUTED)
        self.screen.blit(lbl, (x, y))

        if su >= 9.0:
            plus_db = (su - 9.0) * 6.0
            val_str = f"S9+{plus_db:.0f}dB"
            val_col = (235, 75, 75)   # 赤色 (強電界)
        else:
            val_str = f"S{su:.1f}"
            val_col = C_ACCENT_DARK  # シアン (通常)

        val_surf = cached_text(self.font_small, val_str, val_col)
        self.screen.blit(val_surf, (x + width - val_surf.get_width(), y - 1))

        # 2. スケール目盛り (S1, S3, S5, S7, S9, +20, +40, +60)
        scale_y = y + 14
        # 前半50%を S0〜S9、後半50%を +0〜+60dB に割り当てる無線機標準スケール
        scale_marks = [
            (1.0 / 9.0 * 0.5, "1", (120, 136, 155)),
            (3.0 / 9.0 * 0.5, "3", (120, 136, 155)),
            (5.0 / 9.0 * 0.5, "5", (120, 136, 155)),
            (7.0 / 9.0 * 0.5, "7", (120, 136, 155)),
            (0.5, "9", (120, 136, 155)),
            (0.5 + (20.0 / 60.0) * 0.5, "+20", (220, 110, 80)),
            (0.5 + (40.0 / 60.0) * 0.5, "+40", (220, 90, 80)),
            (1.0, "+60", (225, 70, 70)),
        ]
        for ratio, text, col in scale_marks:
            tx = int(x + ratio * (width - 4))
            surf = cached_text(self.font_tiny, text, col)
            self.screen.blit(surf, (tx - surf.get_width() // 2, scale_y))

        # 3. LEDセグメントバー (計24セグメント)
        bar_y = y + 27
        bar_h = 10
        num_segs = 24
        gap = 2
        seg_w = max(4, int((width - (num_segs - 1) * gap) / num_segs))

        # 点灯セグメント数の計算
        if su <= 9.0:
            active_segs = int(round((su / 9.0) * 12))
        else:
            over = min(60.0, (su - 9.0) * 6.0)
            active_segs = 12 + int(round((over / 60.0) * 12))
        active_segs = max(0, min(num_segs, active_segs))

        # ピークセグメント
        if self.s_peak_units <= 9.0:
            peak_seg = int(round((self.s_peak_units / 9.0) * 12))
        else:
            over_pk = min(60.0, (self.s_peak_units - 9.0) * 6.0)
            peak_seg = 12 + int(round((over_pk / 60.0) * 12))
        peak_seg = max(0, min(num_segs - 1, peak_seg))

        # 背景トラック (薄い溝)
        pygame.draw.rect(self.screen, (226, 232, 240), (x, bar_y - 1, width, bar_h + 2), border_radius=4)

        for i in range(num_segs):
            sx = x + i * (seg_w + gap)
            is_lit = (i < active_segs)
            is_peak = (i == peak_seg and peak_seg > 0)

            # セグメントの色定義 (0〜11: シアン, 12〜16: アンバー, 17〜23: レッド)
            if i < 12:
                lit_col = (0, 195, 160)
                dim_col = (205, 218, 228)
            elif i < 17:
                lit_col = (240, 180, 45)
                dim_col = (225, 216, 205)
            else:
                lit_col = (245, 65, 65)
                dim_col = (230, 210, 210)

            seg_rect = pygame.Rect(sx, bar_y, seg_w, bar_h)
            if is_peak and not is_lit:
                # ピークホールド表示 (明るいアウトライン)
                pygame.draw.rect(self.screen, dim_col, seg_rect, border_radius=2)
                pygame.draw.rect(self.screen, lit_col, seg_rect, width=1, border_radius=2)
            elif is_lit:
                pygame.draw.rect(self.screen, lit_col, seg_rect, border_radius=2)
            else:
                pygame.draw.rect(self.screen, dim_col, seg_rect, border_radius=2)

    def _draw_header(self):
        # 1. 周波数ヒーローパネル
        lbl = cached_text(self.font_title, t("freq"), C_MUTED)
        self.screen.blit(lbl, (self.hero_rect.x + 18, self.hero_rect.y + 10))

        # 地域プロファイルラベル (局名との衝突を避けるため右上段へ退避！)
        if self.region_label:
            rl = cached_text(self.font_tiny, f"{t('region')}: {self.region_label}", C_MUTED)
            self.screen.blit(rl, (self.hero_rect.right - 18 - rl.get_width(), self.hero_rect.y + 12))

        # 周波数文字列のフォーマットと桁単位の当たり判定構築
        # カンマ区切りは1GHz超で桁インデックスを狂わせ、同調ステップが10倍ずれるため使わない
        if self.center_freq >= 1000000:
            freq_str = f"{self.center_freq / 1e6:.4f}"
            unit = "MHz"
        else:
            freq_str = f"{self.center_freq / 1e3:.1f}"
            unit = "kHz"

        # 周波数各桁の当たり判定とアンダーライン描画
        self.freq_digit_hitboxes = []
        cur_x = self.hero_rect.x + 18
        base_y = self.hero_rect.y + 14

        # 桁の重み付け (MHz/kHz表示時)
        dot_idx = freq_str.find(".")
        for i, ch in enumerate(freq_str):
            ch_surf = cached_text(self.font_huge, ch, C_TEXT)
            ch_w = ch_surf.get_width()
            ch_rect = pygame.Rect(cur_x, base_y, ch_w, ch_surf.get_height())
            self.screen.blit(ch_surf, (cur_x, base_y))

            # 数字であればステップ周波数を算出
            if ch.isdigit():
                if self.center_freq >= 1000000:
                    power = dot_idx - 1 - i if i < dot_idx else dot_idx - i
                    step_hz = int(round(10 ** power * 1e6))
                else:
                    power = dot_idx - 1 - i if i < dot_idx else dot_idx - i
                    step_hz = int(round(10 ** power * 1e3))

                if step_hz >= 100:  # 100Hz以上を同調可能ステップとする
                    self.freq_digit_hitboxes.append((ch_rect, step_hz))
                    # 現在マウスがホバーしている桁であればアクセント下線を描画 (文字下端から綺麗に分離)
                    if self.hovered_freq_digit == step_hz:
                        pygame.draw.line(self.screen, C_ACCENT,
                                         (cur_x + 1, base_y + ch_surf.get_height() - 6),
                                         (cur_x + ch_w - 2, base_y + ch_surf.get_height() - 6), 3)

            cur_x += ch_w

        unit_surf = cached_text(self.font_med, unit, C_ACCENT_DARK)
        self.screen.blit(unit_surf, (cur_x + 8, self.hero_rect.y + 36))

        # アナログ調 周波数バンドスケール (FM/AMの受信位置を精密視覚化)
        scale_x = max(cur_x + 30, self.hero_rect.x + 330)
        scale_w = self.hero_rect.right - 20 - scale_x
        scale_y = self.hero_rect.y + 26
        scale_h = 24
        if scale_w >= 140:
            track_rect = pygame.Rect(scale_x, scale_y, scale_w, scale_h)
            pygame.draw.rect(self.screen, (240, 244, 250), track_rect, border_radius=6)
            pygame.draw.rect(self.screen, (214, 224, 236), track_rect, width=1, border_radius=6)

            # バンド範囲と目盛りの判定
            if 76000000 <= self.center_freq <= 108000000:
                b_min, b_max = 76.0, (95.0 if self.center_freq <= 95000000 else 108.0)
                ticks_major = [76.0, 80.0, 85.0, 90.0, 95.0] if b_max == 95.0 else [88.0, 92.0, 96.0, 100.0, 104.0, 108.0]
                ticks_minor = np.arange(b_min, b_max + 0.1, 1.0)
                unit_lbl = "FM BAND (76-95MHz)" if b_max == 95.0 else "FM BAND (88-108MHz)"
            elif self.center_freq < 30000000:
                b_min, b_max = 0.5, 15.0
                ticks_major = [1.0, 3.0, 6.0, 9.0, 12.0, 15.0]
                ticks_minor = np.arange(1.0, 15.1, 0.5)
                unit_lbl = "AM / SHORTWAVE"
            else:
                b_min, b_max = 118.0, 144.0
                ticks_major = [118.0, 124.0, 130.0, 136.0, 144.0]
                ticks_minor = np.arange(118.0, 144.1, 2.0)
                unit_lbl = "VHF AIR / AMATEUR"

            # 小目盛り
            for tk in ticks_minor:
                if b_min <= tk <= b_max:
                    tx_pos = scale_x + int((tk - b_min) / (b_max - b_min) * (scale_w - 12)) + 6
                    pygame.draw.line(self.screen, (200, 212, 226), (tx_pos, scale_y + 12), (tx_pos, scale_y + scale_h - 4), 1)

            # 大目盛り
            for tk in ticks_major:
                if b_min <= tk <= b_max:
                    tx_pos = scale_x + int((tk - b_min) / (b_max - b_min) * (scale_w - 12)) + 6
                    pygame.draw.line(self.screen, (160, 180, 205), (tx_pos, scale_y + 4), (tx_pos, scale_y + scale_h - 4), 1)

            # 現在周波数の赤いダイヤル指針
            cur_mhz = self.center_freq / 1e6
            ratio = float(np.clip((cur_mhz - b_min) / (b_max - b_min), 0.0, 1.0))
            needle_x = scale_x + int(ratio * (scale_w - 12)) + 6
            pygame.draw.rect(self.screen, (235, 55, 55), (needle_x - 1, scale_y - 2, 3, scale_h + 4), border_radius=1)

            # 指針上の現在周波数フロート数値 (例: 80.000)
            cur_f_str = f"{cur_mhz:.3f}"
            lbl_needle = cached_text(self.font_tiny, cur_f_str, (210, 45, 45))
            nl_x = min(scale_x + scale_w - lbl_needle.get_width(), max(scale_x, needle_x - lbl_needle.get_width() // 2))
            self.screen.blit(lbl_needle, (nl_x, scale_y - 14))

            # バンド種別ラベル
            lbl_scale = cached_text(self.font_tiny, unit_lbl, (140, 155, 175))
            self.screen.blit(lbl_scale, (scale_x + 6, scale_y + scale_h + 2))

        # 下段: 放送局名＆情報ティッカー (主役として大きく太字で堂々表示)
        ticker_text = ""
        ticker_color = C_TEXT
        if self.sw_info:
            ticker_text = f"[短波EiBi] {self.sw_info}"
            ticker_color = (30, 80, 145)
        elif self.rds_text:
            ticker_text = f"♪ {self.rds_text}"
            ticker_color = (0, 135, 115)
        elif self.station_name:
            ticker_text = self.station_name
            ticker_color = C_TEXT
        else:
            for st in self.detected_stations:
                if abs(st["freq_hz"] - self.center_freq) <= 50000:
                    ticker_text = st["name"]
                    break
            if not ticker_text:
                try:
                    from auto_tuner import match_station_name
                    cand = match_station_name(self.center_freq)
                    if cand != "Unknown FM Station":
                        ticker_text = cand
                except Exception:
                    pass

        if ticker_text:
            ticker_surf = cached_text(self.font_station, ticker_text, ticker_color)
            self.screen.blit(ticker_surf, (self.hero_rect.x + 20, self.hero_rect.y + 57))

        # 2. 情報パネル (ステータスバッジ + 本格SメーターLEDバー)
        x0 = self.info_rect.x + 14
        y0 = self.info_rect.y + 8
        w1 = self._draw_chip(f"MODE {self.mode}", x0, y0, (210, 236, 248))
        status = getattr(self, "stereo_status", None) or ("STEREO" if self.is_stereo else "MONO")
        if status == "STEREO":
            st_txt, st_col = t("stereo"), (206, 240, 226)
        elif status == "BLEND":
            st_txt, st_col = t("blend"), (250, 234, 206)
        else:
            st_txt, st_col = t("mono"), (236, 238, 244)
        self._draw_chip(st_txt, x0 + w1 + 18, y0, st_col)

        # 本格的SメーターLEDバーの描画 (テレメトリチップの溢れを廃止し、美しいLEDメーターに！)
        sm_w = self.info_rect.width - 28
        self._draw_smeter(x0, y0 + 30, sm_w, 42)

    def _draw_spectrum(self, spectrum_db):
        self._draw_panel("spec", self.spec_rect)
        r = self.spec_rect

        for i in range(1, 5):
            gy = r.y + int(r.height * i / 5)
            pygame.draw.line(self.screen, (33, 42, 60), (r.x + 4, gy), (r.right - 4, gy), 1)
        for i in range(1, 8):
            gx = r.x + int(r.width * i / 8)
            pygame.draw.line(self.screen, (28, 36, 52), (gx, r.y + 4), (gx, r.bottom - 4), 1)

        lbl = cached_text(self.font_tiny, t("spectrum"), (110, 128, 150))
        self.screen.blit(lbl, (r.x + 12, r.y + 8))

        if len(spectrum_db) > 1:
            pw = min(420, max(64, r.width))
            db_min, db_max = -80.0, -10.0
            norm = np.clip((spectrum_db - db_min) / (db_max - db_min), 0.0, 1.0)
            norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
            xs = np.linspace(0, len(norm) - 1, pw)
            ys = np.interp(xs, np.arange(len(norm)), norm)
            if self._spec_px is None or len(self._spec_px) != pw:
                self._spec_px = np.linspace(r.x + 4, r.right - 4, pw)
            px = self._spec_px
            py = r.bottom - 10 - ys * (r.height - 34)
            pts = np.stack((px, py), axis=1).tolist()

            poly = [(r.x + 4, r.bottom - 4)] + pts + [(r.right - 4, r.bottom - 4)]
            pygame.draw.polygon(self.screen, (14, 78, 88), poly)
            pygame.draw.lines(self.screen, (0, 230, 190), False, pts, 2)

        # センター同調マーカー + 帯域ハイライト
        cx = r.centerx
        bw_hz = 200000 if self.mode == "WFM" else 16000 if self.mode == "NFM" else 12000
        sr = self.sample_rate if self.sample_rate > 0 else 1152000
        bw_px = max(2, int((bw_hz / sr) * r.width))
        shade = pygame.Surface((bw_px, r.height - 12), pygame.SRCALPHA)
        shade.fill((226, 190, 110, 26))
        self.screen.blit(shade, (cx - bw_px // 2, r.y + 6))
        pygame.draw.line(self.screen, C_GOLD, (cx, r.y + 4), (cx, r.bottom - 4), 2)

        # 検出局マーカー
        f_min = self.center_freq - self.sample_rate / 2
        f_max = self.center_freq + self.sample_rate / 2
        if (f_max - f_min) > 0:
            for st in list(self.detected_stations):
                try:
                    sfreq = st.get("freq_hz")
                    if sfreq is None:
                        continue
                    if f_min <= sfreq <= f_max:
                        ratio = (sfreq - f_min) / (f_max - f_min)
                        m_x = r.x + int(ratio * r.width)
                        st_tag = st.get("name") or f"{st.get('freq_mhz', float(sfreq) / 1e6):.1f}"
                        lbl = cached_text(self.font_tiny, st_tag, (230, 246, 255))
                        lw, lh = lbl.get_width(), lbl.get_height()
                        bx = max(r.x + 4, min(r.right - 4 - lw - 8, m_x - lw // 2 - 4))
                        by = r.y + 12
                        badge_rect = pygame.Rect(bx, by, lw + 8, lh + 4)
                        pygame.draw.rect(self.screen, (14, 24, 38), badge_rect, border_radius=4)
                        pygame.draw.rect(self.screen, (0, 190, 160), badge_rect, width=1, border_radius=4)
                        pygame.draw.polygon(self.screen, (0, 190, 160),
                                            [(m_x, by + lh + 7), (m_x - 4, by + lh + 4), (m_x + 4, by + lh + 4)])
                        self.screen.blit(lbl, (bx + 4, by + 2))
                except Exception:
                    continue

        # liveピーク (スキャン不要のワンクリック選局マーカー。5Hz更新)
        try:
            now_p = time.monotonic()
            if now_p - self._last_peak_time >= 0.2 and len(spectrum_db) > 8:
                self.live_peaks = find_spectrum_peaks(
                    spectrum_db, self.sample_rate, self.center_freq)
                self._last_peak_time = now_p
        except Exception:
            pass
        for pk in self.live_peaks[:8]:
            pf = pk["freq_hz"]
            if f_min <= pf <= f_max and (f_max - f_min) > 0:
                p_x = r.x + int((pf - f_min) / (f_max - f_min) * r.width)
                pygame.draw.line(self.screen, (0, 210, 170), (p_x, r.y + 30), (p_x, r.y + 40), 2)

        # ホバー周波数表示
        if self.hover_freq_hz is not None and f_min <= self.hover_freq_hz <= f_max:
            hov = cached_text(self.font_tiny, f"{self.hover_freq_hz / 1e6:.4f} MHz",
                              (255, 215, 130))
            self.screen.blit(hov, (r.right - 128, r.y + 8))

        f_start = (self.center_freq - self.sample_rate / 2) / 1e6
        f_end = (self.center_freq + self.sample_rate / 2) / 1e6
        l1 = cached_text(self.font_tiny, f"{f_start:.3f} MHz", (150, 166, 190))
        l2 = cached_text(self.font_tiny, f"{f_end:.3f} MHz", (150, 166, 190))
        self.screen.blit(l1, (r.x + 12, r.bottom - 18))
        self.screen.blit(l2, (r.right - 80, r.bottom - 18))

    def _draw_waterfall(self, spectrum_db):
        self._draw_panel("wf", self.wf_rect)
        self.update_waterfall(spectrum_db)
        r = self.wf_rect
        if self.wf_surface:
            self.screen.blit(self.wf_surface, (r.x + 4, r.y + 4))
        cx = r.centerx
        pygame.draw.line(self.screen, C_GOLD, (cx, r.y + 4), (cx, r.bottom - 4), 1)

    def _draw_waveform(self, audio_pcm):
        self._draw_panel("wave", self.wave_rect)
        r = self.wave_rect
        lbl = cached_text(self.font_tiny, t("waveform"), (110, 128, 150))
        self.screen.blit(lbl, (r.x + 12, r.y + 5))
        # オーディオモニターステータスバッジ
        mon_tag = cached_text(self.font_tiny, "HI-FI AUDIO", (60, 160, 140))
        self.screen.blit(mon_tag, (r.right - mon_tag.get_width() - 14, r.y + 5))

        if audio_pcm is None or len(audio_pcm) < 32:
            return
        n = min(len(audio_pcm), 8192)
        chunk = np.asarray(audio_pcm[-n:], dtype=np.float32)
        if chunk.ndim == 2:
            # ステレオはL/Rを混ぜず片ch (L) を表示 (reshapeで交互に混ざるのを防ぐ)
            chunk = chunk[:, 0]
        pw = r.width - 24
        seg = max(1, n // pw)
        m = (n // seg) * seg
        env = np.max(np.abs(chunk[-m:].reshape(-1, seg)), axis=1).astype(np.float32)
        k = int(0.99 * (len(env) - 1)) if len(env) > 1 else 0
        peak = max(0.01, float(np.partition(env, k)[k])) if len(env) else 0.01
        env = np.clip(env / peak * 0.92, 0.0, 1.0)
        # 座標配列は長さが変わった時だけ再生成 (O(n)リスト内包をCレベルのtolistへ)
        if self._wave_xs is None or len(self._wave_xs) != len(env):
            self._wave_xs = np.linspace(r.x + 12, r.right - 12, len(env))
        xs = self._wave_xs
        mid = r.centery + 7
        amp = r.height / 2 - 13
        top = np.stack((xs, mid - env * amp), axis=1).tolist()
        bot = np.stack((xs, mid + env * amp), axis=1).tolist()
        if len(top) >= 2:
            pygame.draw.polygon(self.screen, (14, 78, 88), top + bot[::-1])
            pygame.draw.lines(self.screen, (0, 220, 184), False, top, 1)
            pygame.draw.lines(self.screen, (0, 220, 184), False, bot, 1)

    def _draw_telemetry(self):
        # 初心者にもわかる電波クオリティ判定バッジ (どんなアンテナでも状況把握)
        if self.s_units >= 9.0:
            qual_text = "● 受信クリア"
            qual_col = (16, 160, 100)
        elif self.s_units >= 5.0:
            qual_text = "● 受信良好"
            qual_col = (20, 140, 170)
        elif self.s_units >= 2.0:
            qual_text = "▲ 受信中 (並)"
            qual_col = (200, 130, 20)
        elif self.s_units > 0.5:
            qual_text = "△ 微弱電波"
            qual_col = (180, 100, 80)
        else:
            qual_text = "○ 探索中"
            qual_col = C_MUTED
        qual_surf = cached_text(self.font_tiny, qual_text, qual_col)
        self.screen.blit(qual_surf, (self.tele_rect.right - qual_surf.get_width() - 14, self.tele_rect.y + 8))

        parts = [p.strip() for p in self.telemetry_text.split("|") if p.strip()]

        # 2列×3行の精密ダッシュボード・ミニバッジ形式で整然と表示
        bw, bh = 122, 20
        gap_x = 8
        for i, part in enumerate(parts[:6]):
            col, row = i // 3, i % 3
            bx = self.tele_rect.x + 14 + col * (bw + gap_x)
            by = self.tele_rect.y + 30 + row * 22
            pygame.draw.rect(self.screen, (234, 240, 248), (bx, by, bw, bh), border_radius=4)
            pygame.draw.rect(self.screen, (214, 224, 238), (bx, by, bw, bh), width=1, border_radius=4)
            surf = cached_text(self.font_tiny, part, C_TEXT)
            self.screen.blit(surf, (bx + 6, by + 3))

    def _draw_controls(self):
        # プリセットが存在する場合のみ、左端にバンドタグを描画
        if self.presets_fm:
            self.screen.blit(cached_text(self.font_tiny, "FM", C_MUTED),
                             (self.preset_rect.x + 16, self.preset_rect.y + 14))
        if self.presets_am:
            self.screen.blit(cached_text(self.font_tiny, "AM", C_MUTED),
                             (self.preset_rect.x + 16, self.preset_rect.y + 45))

        # ステータスバー (右端の検出局数と被らないよう幅をガード)
        max_st_w = self.status_rect.width - 150
        st_surf = cached_text(self.font_small, f"{t('rx_status')}: {self.scan_status_text}", C_ACCENT_DARK)
        if st_surf.get_width() > max_st_w:
            self.screen.blit(st_surf, (self.status_rect.x + 16, self.status_rect.y + 6),
                             (0, 0, max_st_w, st_surf.get_height()))
        else:
            self.screen.blit(st_surf, (self.status_rect.x + 16, self.status_rect.y + 6))

        cnt = cached_text(self.font_small, f"{t('detected')}: {len(self.detected_stations)}", C_MUTED)
        self.screen.blit(cnt, (self.status_rect.right - 110, self.status_rect.y + 6))

        # 全ボタン描画 (アクティブ状態更新)
        for name, b in self.mode_buttons.items():
            b.is_active = (self.mode == name)
        for b in getattr(self, "preset_buttons", []):
            b.is_active = (hasattr(b, "freq_hz") and abs(self.center_freq - b.freq_hz) < 50000)
        for btn in self.buttons:
            if btn.visible:
                btn.draw(self.screen, self.font_small)

    def render(self, spectrum_db, audio_pcm=None):
        """1フレームの描画処理 (静的シーン事前ベイクで高速化)"""
        if self.baked_bg is not None:
            self.screen.blit(self.baked_bg, (0, 0))
        else:
            self.screen.blit(self.bg_surface, (0, 0))

        self._draw_header()
        self._draw_spectrum(spectrum_db)
        self._draw_waterfall(spectrum_db)
        self._draw_waveform(audio_pcm)
        self._draw_telemetry()
        if hasattr(self, "btn_station_list"):
            self.btn_station_list.text = t("station_list_btn", n=len(self.detected_stations))
        self._draw_controls()
        self._draw_station_list()

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()
