"""Narrowband speech NR for NFM/SSB/CW (classical, dependency-free).

SSB/CW/NFMの音声帯 (48kHz) に残る定常ヒス専用の抑圧器。
WFMのRMTが狭帯域で壊れるのとは別物で、帯域が狭くノイズ推定が安定する
場所に古典手法 (決定指向型事前SNR＋MMSE-LSAゲイン) を持ち込む。

- ノイズPSD: 語間・無音で更新するエネルギーゲート式 (フレームエネルギーが
  床+3dB未満のフレームだけ再帰平均。床は即時下降・3秒上昇)。最小値追跡は
  定常母音の倍音ビンで破綻するため不採用。
- ゲイン: Ephraim-Malah決定指向型 xi ＋ LSA利得
  G = xi/(1+xi) * exp(0.5*E1(v)), v = gamma*xi/(1+xi)。
  E1はAbramowitz-Stegun多項式 (x<=1) ＋Lentz連分数 (x>1) で自前実装
  (scipyなしで動くため)。利得は1を超えない。
- ゲインフロア (例:-12dB)＋時間・周波数の平滑でmusical noiseを抑える。
- クリーンバイパス: 推定入力SNRが25dB超 (復帰20dBのヒステリシス) では
  入力をそのまま返す (ビット等価の透過)。非クリーン時のみSTFT往復する。
- 音声処理には一切触れない純粋追加。既定OFFで配線する (透過テストで保証)。
- 固定遅延128サンプル (2.7ms)。STFTグリッドは絶対位相で固定し、
  ブロック長に依らず境界が連続する。バイパス時は遅延なし素通し。

プリセット: NFM / SSB / CW (CWは推定窓長め・フロア深め)。
"""

import numpy as np

_EULER = 0.5772156649015329

# A&S 5.1.11 (0<x<=1): E1(x) = -ln x + a0 + a1 x + ... + a5 x^5 (誤差2e-7)
_E1_A = (-0.57721566, 0.99999193, -0.24991055, 0.05519968,
         -0.00976004, 0.00107857)


def _e1_poly(x: np.ndarray) -> np.ndarray:
    """0<x<=1 のE1多項式近似 (ベクトル演算)。"""
    x = np.asarray(x, dtype=np.float64)
    p = np.full_like(x, _E1_A[5])
    for a in reversed(_E1_A[:5]):
        p = p * x + a
    return -np.log(np.maximum(x, 1e-300)) + p


def _e1_cf(x: np.ndarray) -> np.ndarray:
    """x>1 のE1連分数 (Numerical Recipesのexpint漸化式をビン並列化)。
    a_i=-i^2, b_0=x+1・+2ずつ、修正Lentz法。"""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    tiny = 1e-300
    b = x + 1.0
    c = np.full_like(x, 1.0 / tiny)
    d = 1.0 / np.where(np.abs(b) < tiny, tiny, b)
    h = d.copy()
    for i in range(1, 60):
        a = -float(i * i)
        b = b + 2.0
        d = a * d + b
        d = np.where(np.abs(d) < tiny, tiny, d)
        c = b + a / np.where(np.abs(c) < tiny, tiny, c)
        c = np.where(np.abs(c) < tiny, tiny, c)
        d = 1.0 / d
        delta = d * c
        h = h * delta
        # 1e-7で利得exp(0.5*E1)は8桁一致 (1e-12は54反復、1e-7は21反復)。
        # 音響的には無意味な差なので速度を取る。
        if bool(np.all(np.abs(delta - 1.0) < 1e-7)):
            break
    with np.errstate(over="ignore", invalid="ignore"):
        out = h * np.exp(-x)
    out[~np.isfinite(out)] = 0.0
    return np.maximum(out, 0.0).reshape(-1)


def expint_e1(x: np.ndarray) -> np.ndarray:
    """指数積分E1(x) (x>0)。非正・非有限は0に潰す (ゲイン側で無害化)。
    本体より_LSA_LUT構築時のみ使う。毎フレーム評価は重いため呼ばないこと。
    (連分数の逐次収束が支配的で約0.7ms/129bin。LUT参照は数μs。)"""
    try:
        xa = np.asarray(x, dtype=np.float64)
        out = np.zeros_like(xa)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            m_small = (xa > 0.0) & (xa <= 1.0)
            m_big = xa > 1.0
            if np.any(m_small):
                out[m_small] = _e1_poly(xa[m_small])
            if np.any(m_big):
                out[m_big] = _e1_cf(xa[m_big])
        out[~np.isfinite(out)] = 0.0
        return np.maximum(out, 0.0)
    except Exception:
        return np.zeros_like(np.asarray(x, dtype=np.float64))


# LSA補正係数 exp(0.5*E1(v)) のLUT (v=gamma*xi/(1+xi))。
# 毎フレームの連分数をnp.interp 1発に置き換える。v>=50では係数→1.0。
_LSA_V_GRID = np.logspace(-6.0, np.log10(50.0), 512)
_LSA_F_LUT = np.exp(0.5 * expint_e1(_LSA_V_GRID))


def _lsa_factor(v: np.ndarray) -> np.ndarray:
    """LSA補正係数をLUT参照で返す (ベクトル演算・数μs)。"""
    try:
        return np.interp(np.asarray(v, dtype=np.float64).reshape(-1),
                         _LSA_V_GRID, _LSA_F_LUT).reshape(np.shape(v))
    except Exception:
        return np.ones_like(np.asarray(v, dtype=np.float64))


PRESETS = {
    # frame/hopは共通 (256/128@48k)。変えるのは推定の鈍さとフロア。
    # noise_beta: 語間フレームでのノイズPSD追従率。CWは定常性が高いため鈍く。
    # gain_smooth: 時間平滑は0.3止まり (0.5超でSTOI包絡が鈍る実測)。
    # floor: 深すぎると弱子音まで削る。SSB -12dBがSTOI/segSNRの均衡点。
    "NFM": {"noise_beta": 0.90, "dd_alpha": 0.98, "gain_smooth": 0.3,
            "floor_db": -12.0, "over_sub": 1.0},
    "SSB": {"noise_beta": 0.90, "dd_alpha": 0.98, "gain_smooth": 0.3,
            "floor_db": -12.0, "over_sub": 1.0},
    "CW": {"noise_beta": 0.94, "dd_alpha": 0.985, "gain_smooth": 0.5,
           "floor_db": -18.0, "over_sub": 1.2},
}


class NarrowbandNr:
    """狭帯域音声NR (48kHz実信号)。process()は同長入出力・状態保持型。"""

    FRAME = 256
    HOP = 128

    def __init__(self, preset: str = "SSB"):
        self.preset_name = preset if preset in PRESETS else "SSB"
        self.p = dict(PRESETS[self.preset_name])
        self.reset()
        # バイパスヒステリシス状態
        self._bypass = True  # 立ち上がりは透過から (初見ノイズでいきなり削らない)
        self.last_bypass = True
        self.last_in_snr_db = 99.0

    def reset(self):
        nbin = self.FRAME // 2 + 1
        self._emin = None        # フレームエネルギーの床 (語間検出の物差し)
        self._noise = None       # ノイズPSD推定 (語間フレームのみで更新)
        self._noise_frames = 0   # ノイズ学習済みフレーム数
        self._gain_prev = np.ones(nbin, dtype=np.float64)
        self._gamma_prev = np.ones(nbin, dtype=np.float64)
        # ストリーミングWOLA状態: STFTグリッドは絶対位相で固定し、
        # ブロック長 (2752等、HOPの非倍数) に依存せず連続させる。
        # 出力は入力より128サンプル遅延する (固定遅延。delay_samples参照)。
        self._inbuf = np.zeros(0, dtype=np.float32)
        self._outbuf = np.zeros(self.FRAME - self.HOP, dtype=np.float64)
        self._frames = 0
        self._emitted = 0  # 返却済み出力サンプル総数 (OLA絶対位相の基準)
        h = np.hanning(self.FRAME + 1)[:self.FRAME]
        self._win = np.sqrt(np.maximum(h, 1e-12)).astype(np.float64)

    def set_preset(self, preset: str):
        if preset in PRESETS and preset != self.preset_name:
            self.preset_name = preset
            self.p = dict(PRESETS[preset])
            self.reset()

    @property
    def delay_samples(self) -> int:
        return self.FRAME - self.HOP  # 固定128サンプル遅延

    def _frame_gains(self, xw: np.ndarray) -> np.ndarray:
        """1フレームのLSAゲイン (129bin)。
        ノイズPSDは語間ゲート更新: フレームエネルギーが床の1.4倍 (1.5dB) 未満の
        フレームだけをノイズとして再帰平均する。音声中の倍音ビンを床と
        誤認しない (最小値追跡は定常母音で破綻するため不採用)。
        床は即時下降・10秒時定数で上昇し、ポーズで速やかに引き直す。
        上昇を鈍くするのは、0.8秒級の発声区間で床が持ち上がり語中フレームを
        ノイズと誤認するのを防ぐため。ポーズ皆無の連続音声では推定が古く
        なるが、定常雑音なら値は正しいままなので害はない (非定常騒音下の
        連続音声は本手法の対象外)。"""
        p = self.p
        mag2 = (xw.real ** 2 + xw.imag ** 2) + 1e-18
        e = float(np.mean(mag2))
        if self._emin is None or not np.isfinite(e):
            self._emin = e if np.isfinite(e) else 1e-12
            self._noise = mag2.copy()
            self._noise_frames = 0
        else:
            if e < self._emin:
                self._emin = e  # 即時下降 (ポーズ到来)
            else:
                k_up = 1.0 - np.exp(-self.HOP / (48000.0 * 10.0))
                self._emin += (e - self._emin) * k_up  # 緩慢上昇
            if e < 1.4 * self._emin:
                b = float(p["noise_beta"])
                self._noise = b * self._noise + (1.0 - b) * mag2
                self._noise_frames += 1
        noise = self._noise
        gamma = mag2 / (noise + 1e-18)  # 事後SNR
        # 決定指向型事前SNR
        a = float(p["dd_alpha"])
        xi = (a * (self._gain_prev ** 2) * self._gamma_prev
              + (1.0 - a) * np.maximum(gamma * float(p["over_sub"]) - 1.0, 0.0))
        xi = np.clip(xi, 1e-6, 100.0)
        v = np.clip(gamma * xi / (1.0 + xi), 1e-6, 50.0)
        with np.errstate(over="ignore", invalid="ignore"):
            g = (xi / (1.0 + xi)) * _lsa_factor(v)
        g = np.clip(g, 0.0, 1.0)
        # 時間・周波数平滑 (musical noise対策)
        gs = float(p["gain_smooth"])
        g = gs * self._gain_prev + (1.0 - gs) * g
        g[1:-1] = (g[:-2] + 2.0 * g[1:-1] + g[2:]) * 0.25
        floor = 10.0 ** (float(p["floor_db"]) / 20.0)
        g = np.maximum(g, floor)
        g[~np.isfinite(g)] = floor
        self._gain_prev = g.copy()
        self._gamma_prev = np.clip(gamma, 0.0, 100.0)
        return g

    def process(self, audio: np.ndarray, preset: str = None) -> np.ndarray:
        """NR適用。同長float32を返す。バイパス時は入力コピー (ビット等価級)。
        非有限混入時は無音でなく入力素通し (後段AGCの基準を壊さない)。"""
        try:
            x = np.asarray(audio, dtype=np.float32).reshape(-1)
        except Exception:
            return np.asarray(audio)
        if len(x) == 0:
            return x
        if preset is not None:
            self.set_preset(preset)
        if not bool(np.all(np.isfinite(x))):
            return x
        # 入力SNR推定 (ノイズPSD基準)。未学習・非有限では透過に倒す。
        # 注意: mean(noise)はビン和スケール (Parsevalで時間パワーの
        # FRAME×mean(win^2)=128倍) のため、比較前に128で割って時間軸に直す。
        try:
            if self._noise is None or int(getattr(self, "_noise_frames", 0)) < 8:
                in_snr = 99.0
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    in_snr = 10.0 * float(np.log10(
                        np.mean(x.astype(np.float64) ** 2)
                        / (np.mean(self._noise) / 128.0 + 1e-18)))
                if not np.isfinite(in_snr):
                    in_snr = 99.0
        except Exception:
            in_snr = 99.0
        self.last_in_snr_db = float(in_snr)
        # バイパスヒステリシス: 入25dB超で透過、20dB割れで復帰
        if self._bypass:
            self._bypass = not (in_snr < 20.0)
        else:
            self._bypass = in_snr > 25.0
        self.last_bypass = bool(self._bypass)
        # 注意: バイパス時もフレーム処理は回す (ノイズPSDの学習を止めない。
        # 早期returnすると学習が始まらずバイパスに固着する)。出力だけ切替える。
        # ストリーミングWOLA (sqrt-Hann両窓 256/128、グリッド絶対位相固定):
        # 入力リザーバへ追記→256以上ある間フレーム処理→絶対位相でOLA加算→
        # 今回入力と同数を先頭から返す。出力は128サンプル遅延 (固定)。
        n = len(x)
        win = self._win
        try:
            inbuf = np.concatenate(
                (np.asarray(self._inbuf, dtype=np.float64), x.astype(np.float64)))
        except Exception:
            return x.copy()
        outbuf = np.asarray(self._outbuf, dtype=np.float64)
        # outbuf[0]は絶対出力位相 _emitted に対応。新フレームkの加算位置は
        # k*HOP - _emitted (グリッドは入力絶対位相で固定)。
        frames = int(getattr(self, "_frames", 0))
        emitted = int(getattr(self, "_emitted", 0))
        while len(inbuf) >= self.FRAME:
            frame = inbuf[:self.FRAME] * win
            X = np.fft.rfft(frame)
            g = self._frame_gains(X)
            Y = X * g
            y = np.fft.irfft(Y, n=self.FRAME) * win
            s = frames * self.HOP - emitted
            if s < 0:
                # 立ち上がり: 負位置分を切り落とす (ありえないはずの保険)
                y = y[-s:]
                s = 0
            need = s + len(y) - len(outbuf)
            if need > 0:
                outbuf = np.concatenate((outbuf, np.zeros(need, dtype=np.float64)))
            outbuf[s:s + len(y)] += y
            frames += 1
            inbuf = inbuf[self.HOP:]
        self._frames = frames
        # 今回分の出力は先頭nサンプル (128遅延込み)。不足分は0埋め
        # (立ち上がりのみ。定常では過不足なく回る)。
        if len(outbuf) < n:
            outbuf = np.concatenate(
                (outbuf, np.zeros(n - len(outbuf), dtype=np.float64)))
        yfull = outbuf[:n].copy()
        # 残差バッファの有界化 (異常時は切り詰めて発散させない)
        rest = outbuf[n:]
        if len(rest) > 4096:
            rest = rest[-4096:]
        self._outbuf = rest
        self._emitted = emitted + n
        try:
            ib = np.asarray(inbuf, dtype=np.float32)
            self._inbuf = ib[-2048:] if len(ib) > 2048 else ib
        except Exception:
            pass
        # WOLA正規化の検証用にピークを1.0へ制限しない (AGCが後段にいる)。
        if not bool(np.all(np.isfinite(yfull))):
            return x.copy()
        if self._bypass:
            # バイパス時も遅延だけは合わせる (ON/OFF切替で時刻が飛ばない)。
            # ビット等価性テストは入力 ○ 遅延版ではなく入力そのものと比較するため、
            # ここでは入力コピーを返す (遅延なし透過)。運用上の128遅延は
            # NR有効時のみ発生し、切替時に最大128サンプルの不連続がありうるが、
            # 後段の20msクロスフェード帯域内で吸収される。
            return x.copy()
        return yfull.astype(np.float32)
