"""RDS (Radio Data System) decoder.

57kHz副搬送波 (BPSK, 1187.5bps) を復調し、PI / PS / PTY / RadioText / 時計を取り出す。
搬送波はパイロットPLLの3θ (57kHz) とコヒーレント。差動復号のため搬送波位相誤差に強い。

処理:
  1. バイフェーズ相関 (累積和の線形補間 = ベクトル演算) でソフトビット抽出
  2. 差動復号 → ビット列
  3. オフセットワード同期 + CRC-10 チェック
  4. グループ組立 (0A/0B: PS, 2A/2B: RadioText, 4A: 時計)
"""

import numpy as np
from collections import Counter

RDS_SYMBOL_RATE = 1187.5
RDS_FS = 12000.0

# オフセットワード (10bit CRCの期待値)
OFFSET_WORDS = {
    "A": 0x0FC, "B": 0x198, "C": 0x168,
    "C'": 0x350, "D": 0x1B4, "E": 0x000,
}

# RDS G0/E1文字セット (0x80以上; 未定義は置換)
_E1 = {
    0x80: "á", 0x81: "à", 0x82: "é", 0x83: "è", 0x84: "í", 0x85: "ì",
    0x86: "ó", 0x87: "ò", 0x88: "ú", 0x89: "ù", 0x8A: "Ñ", 0x8B: "Ç",
    0x8C: "Ş", 0x8D: "ß", 0x8E: "¡", 0x8F: "Ĳ", 0x90: "â", 0x91: "ä",
    0x92: "ê", 0x93: "ë", 0x94: "î", 0x95: "ï", 0x96: "ô", 0x97: "ö",
    0x98: "û", 0x99: "ü", 0x9A: "ñ", 0x9B: "ç", 0x9C: "ş", 0x9D: "ğ",
    0x9E: "ı", 0x9F: "ĳ", 0xA0: "ª", 0xA1: "α", 0xA2: "©", 0xA3: "‰",
    0xA4: "Ğ", 0xA5: "ě", 0xA6: "ň", 0xA7: "ő", 0xA8: "π", 0xA9: "€",
    0xAA: "£", 0xAB: "$", 0xAC: "←", 0xAD: "↑", 0xAE: "→", 0xAF: "↓",
    0xB0: "º", 0xB1: "¹", 0xB2: "²", 0xB3: "³", 0xB4: "±", 0xB5: "İ",
    0xB6: "ń", 0xB7: "ű", 0xB8: "µ", 0xB9: "¿", 0xBA: "÷", 0xBB: "°",
    0xBC: "¼", 0xBD: "½", 0xBE: "¾", 0xBF: "§", 0xC0: "Á", 0xC1: "À",
    0xC2: "É", 0xC3: "È", 0xC4: "Í", 0xC5: "Ì", 0xC6: "Ó", 0xC7: "Ò",
    0xC8: "Ú", 0xC9: "Ù", 0xCA: "Ř", 0xCB: "Č", 0xCC: "Š", 0xCD: "Ž",
    0xCE: "Ð", 0xCF: "Ŀ", 0xD0: "Â", 0xD1: "Ä", 0xD2: "Ê", 0xD3: "Ë",
    0xD4: "Î", 0xD5: "Ï", 0xD6: "Ô", 0xD7: "Ö", 0xD8: "Û", 0xD9: "Ü",
    0xDA: "ř", 0xDB: "č", 0xDC: "š", 0xDD: "ž", 0xDE: "đ", 0xDF: "ÿ",
}


def char_of(byte: int) -> str:
    if 0x20 <= byte < 0x7F:
        return chr(byte)
    return _E1.get(byte, "?")


def syndrome(bits) -> int:
    """RDS CRC-10 (g(x)=x^10+x^8+x^7+x^5+x^4+x^3+1) のシンドローム"""
    reg = 0
    for b in bits:
        reg = (reg << 1) | int(b)
        if reg & 0x400:
            reg ^= 0x5B9
    return reg & 0x3FF


def block_offset(bits26) -> str | None:
    syn = syndrome(bits26)
    for name, val in OFFSET_WORDS.items():
        if syn == val:
            return name
    return None


def bits16(data: int):
    return [(data >> (15 - i)) & 1 for i in range(16)]


def make_block(data16: int, offset: str) -> list:
    """16bitデータ + オフセットワードから26bitブロックを生成 (テスト/送信側用)"""
    data_bits = bits16(data16)
    syn_data = syndrome(data_bits + [0] * 10)
    check = syn_data ^ OFFSET_WORDS[offset]
    return data_bits + [(check >> (9 - i)) & 1 for i in range(10)]


class RdsDecoder:
    """12kHzのBPSKベースバンド(実数)を受け取りRDS情報を復号する"""

    def __init__(self, fs: float = RDS_FS):
        self.fs = float(fs)
        self.sps = self.fs / RDS_SYMBOL_RATE
        self._buf = np.zeros(0, dtype=np.float32)
        self._phase = None
        self._prev_soft = 0.0
        self._pending = []
        self._sync = False
        self._search_from = 0
        self.pi = 0
        self._pi_hist = Counter()
        self.pty = None
        self.tp = None
        self.ps_name = ""
        self.radio_text = ""
        self.clock = None
        self.groups = 0
        self.blocks = 0
        self._ps = [None] * 8
        self._rt = [None] * 64      # 生バイト値 (0x0D=CR 終端を検出可能にする)
        self._rt_ab = None          # RadioText A/Bフラグ (切替でバッファリセット)

    def reset(self):
        """選局・モード切替時にデコーダ状態をリセットする。"""
        self._buf = np.zeros(0, dtype=np.float32)
        self._phase = None
        self._prev_soft = 0.0
        self._pending = []
        self._sync = False
        self._search_from = 0
        self.pi = 0
        self._pi_hist.clear()
        self.pty = None
        self.tp = None
        self.ps_name = ""
        self.radio_text = ""
        self.clock = None
        self.groups = 0
        self.blocks = 0
        self._ps = [None] * 8
        self._rt = [None] * 64
        self._rt_ab = None

    # ---------------- 信号処理 ----------------
    def _soft_bits(self, buf: np.ndarray, phase: float) -> np.ndarray:
        """バイフェーズ相関 (前半-後半) を累積和の補間でベクトル計算"""
        n = len(buf)
        count = int(np.floor((n - phase - self.sps) / self.sps)) + 1
        if count <= 0:
            return np.zeros(0, dtype=np.float64)
        starts = phase + self.sps * np.arange(count, dtype=np.float64)
        idx = np.arange(n + 1, dtype=np.float64)
        cum = np.concatenate(([0.0], np.cumsum(buf, dtype=np.float64)))
        a = np.interp(starts, idx, cum)
        b = np.interp(starts + self.sps * 0.5, idx, cum)
        c = np.interp(starts + self.sps, idx, cum)
        return (b - a) - (c - b)

    def _acquire_timing(self) -> bool:
        """シンボルタイミングを相関電力最大の位相で獲得"""
        if len(self._buf) < int(self.sps * 500):
            return False
        best_phase, best_score = 0.0, -1.0
        for ph in np.linspace(0.0, self.sps, 24, endpoint=False):
            soft = self._soft_bits(self._buf, float(ph))
            if len(soft) < 100:
                continue
            score = float(np.mean(soft ** 2))
            if score > best_score:
                best_phase, best_score = float(ph), score
        self._phase = best_phase
        return True

    def feed(self, x: np.ndarray):
        """12kHzの実数ベースバンドを入力 (ブロック単位)"""
        if len(x) == 0:
            return
        self._buf = np.concatenate((self._buf, np.asarray(x, dtype=np.float32)))
        # タイミング未獲得時のバッファ無制限成長を防止 (最大2000シンボル分で切り捨て)
        max_buf = int(self.sps * 2000)
        if len(self._buf) > max_buf:
            self._buf = self._buf[-max_buf:]
        if self._phase is None and not self._acquire_timing():
            return
        soft = self._soft_bits(self._buf, self._phase)
        if len(soft) == 0:
            return
        # 差動復号
        prev = self._prev_soft
        for s in soft:
            self._pending.append(1 if (s * prev) < 0.0 else 0)
            prev = float(s)
        self._prev_soft = prev
        # 消費サンプルの破棄 (次シンボル開始まで)
        next_start = self._phase + self.sps * len(soft)
        consumed = int(next_start)
        self._buf = self._buf[consumed:]
        self._phase = next_start - consumed
        if len(self._pending) > 600:
            del self._pending[:-208]
        self._decode_groups()

    # ---------------- グループ復号 ----------------
    def _group_valid(self, bits: list) -> bool:
        if len(bits) < 104:
            return False
        o1 = block_offset(bits[0:26])
        o2 = block_offset(bits[26:52])
        o3 = block_offset(bits[52:78])
        o4 = block_offset(bits[78:104])
        if o2 != "B" or o4 != "D":
            return False
        # RDS規格上 Block 1 のオフセットは常に "A" (PI符号)。
        # "C"/"C'" を許容するとノイズからの誤検出率が3倍になるため厳格化。
        return o1 == "A" and o3 in ("C", "C'")

    @staticmethod
    def _data16(bits: list) -> int:
        v = 0
        for b in bits[:16]:
            v = (v << 1) | b
        return v

    def _decode_groups(self):
        if not self._sync:
            if len(self._pending) < 130:
                return
            # 最初に見つかった有効グループ位置へ同期 (以降のグループを全て活かす)
            # 前回探索済みの位置は再探索しない (未同期時のCRC計算を約1/6に削減)
            start = max(0, self._search_from)
            off = -1
            for i in range(start, len(self._pending) - 104 + 1):
                if self._group_valid(self._pending[i:i + 104]):
                    off = i
                    break
            if off < 0:
                if len(self._pending) > 160:
                    drop = len(self._pending) - 160
                    del self._pending[:drop]
                    self._search_from = max(0, len(self._pending) - 104 + 1)
                else:
                    self._search_from = max(0, len(self._pending) - 104 + 1)
                return
            del self._pending[:off]
            self._sync = True

        while len(self._pending) >= 104:
            group = self._pending[:104]
            del self._pending[:104]
            # 同期後もCRCを再検証する (ノイズ・ビットずれによる誤PS/RT防止)。
            # 1ブロックでも不正なら同期を外して再探索へ (ずれ続けると誤情報を出すため)。
            if not self._group_valid(group):
                self._sync = False
                self._search_from = 0
                self._pending = group + self._pending  # 捨てずに再探索へ
                break
            self._process_group(group)

    def _process_group(self, bits: list):
        self.groups += 1
        self.blocks += 4
        b1 = self._data16(bits[0:26])
        b2 = self._data16(bits[26:52])
        b3 = self._data16(bits[52:78])
        b4 = self._data16(bits[78:104])

        self._pi_hist[b1] += 1
        self.pi = self._pi_hist.most_common(1)[0][0]

        gtype = (b2 >> 12) & 0xF
        version_b = (b2 >> 11) & 1
        self.tp = (b2 >> 10) & 1
        self.pty = (b2 >> 5) & 0x1F

        if gtype == 0:  # PS name: ブロック4の2文字 (ブロック3は0A=AF / 0B=PI)
            addr = b2 & 0x3
            chars = [char_of((b4 >> 8) & 0xFF), char_of(b4 & 0xFF)]
            for i, ch in enumerate(chars):
                idx = addr * 2 + i
                if 0 <= idx < 8:
                    self._ps[idx] = ch
            if all(c is not None for c in self._ps):
                self.ps_name = "".join(self._ps).strip()
        elif gtype in (2,) and not version_b:  # RadioText (2A: 64 chars)
            ab = (b2 >> 4) & 1
            if self._rt_ab is not None and ab != self._rt_ab:
                # A/Bフラグ切替: 新しい文が始まるためバッファと表示をリセット
                # (古い曲名が新文の表示条件成立まで残るのを防ぐ)
                self._rt = [None] * 64
                self.radio_text = ""
            self._rt_ab = ab
            addr = b2 & 0xF
            raw3 = [(b3 >> 8) & 0xFF, b3 & 0xFF,
                    (b4 >> 8) & 0xFF, b4 & 0xFF]
            for i, rb in enumerate(raw3):
                idx = addr * 4 + i
                if 0 <= idx < 64:
                    self._rt[idx] = rb
            # 表示: 32文字全部埋まった場合、またはCR (0x0D) 終端の短い文を表示
            txt_raw = "".join(chr(c) if c == 0x0D else (char_of(c) if c is not None else " ")
                              for c in self._rt)
            if "\r" in txt_raw:
                txt = txt_raw.split("\r")[0].rstrip()
                if txt:
                    self.radio_text = txt
            elif all(c is not None for c in self._rt[:32]):
                self.radio_text = txt_raw.rstrip()
        elif gtype == 4 and not version_b:  # 時計 (EN 50067 4A: MJD 17bit + 5bit時 + 6bit分)
            # MJD = (Bの下位2bit << 15) | (b3 >> 1)  (b3のbit0は時のMSB)
            mjd = ((b2 & 3) << 15) | (b3 >> 1)
            hour = ((b3 & 1) << 4) | ((b4 >> 12) & 0xF)
            minute = (b4 >> 6) & 0x3F
            if mjd > 15079 and hour < 24 and minute < 60:
                # MJD -> 年月日 (Fliegel-Van Flandern / Richards アルゴリズム)
                j = mjd + 2400001
                a = j + 32044
                b = (4 * a + 3) // 146097
                c = a - (146097 * b) // 4
                d = (4 * c + 3) // 1461
                e = c - (1461 * d) // 4
                m = (5 * e + 2) // 153
                day = e - (153 * m + 2) // 5 + 1
                month = m + 3 - 12 * (m // 10)
                year = 100 * b + d - 4800 + m // 10
                self.clock = (year, month, day, hour, minute)

    @property
    def ps_valid(self) -> bool:
        return bool(self.ps_name)
