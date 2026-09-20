"""
Hyper vs Cascade Controller Benchmark & DSP Path Sanity Test.

実ハードウェア不要のシミュレーション環境で、同一の電波環境・同一の初期条件から
CascadeController と HyperController を駆動し、以下を定量比較する:
  - 収束後の実チャンネルC/N（最適ゲインとの後悔/regret）
  - 最適点への収束速度（フレーム数）
  - ゲイン変更回数・定常揺らぎ（発振の有無）
  - 隣接強局が存在する環境での耐汚染性

さらに実 SdrDspPipeline に連続認知パラメータを流し、無段階モーフィング経路の
クラッシュ・NaN・サンプル数不整合が無いことを検証する。

実行: python hyper_benchmark.py
"""

import sys
import time as _time
import numpy as np

_REAL_TIME = _time.time
_CLOCK = {"t": 1000.0}


def _fake_time():
    return _CLOCK["t"]


_time.time = _fake_time  # 制御ループの時間軸を仮想化（実時間待ちゼロで長時間を再現）

from cascade_controller import CascadeController
from hyper_controller import HyperController

GAINS = [0.0, 0.9, 1.4, 2.7, 3.7, 7.7, 8.7, 12.5, 14.4, 15.7, 16.6,
         19.7, 20.7, 22.9, 25.4, 28.0, 29.7, 32.8, 33.8, 36.4, 37.2,
         38.6, 40.2, 42.1, 43.4, 43.9, 44.5, 48.0, 49.6]

FRAME_DT = 0.057   # 実アプリ相当の1フレーム時間 (57ms)
FRAMES = 350       # 約20秒相当


class TunerEnvironment:
    """R820Tチューナー + 伝搬路の物理モデル"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.gain = 33.8

    def set_gain(self, g):
        self.gain = g

    def external_cn(self):
        return self.cfg["ext_cn"]

    def channel_snr(self, g):
        """真の受信チャンネルC/N（ゲインによる改善と過剰利得ペナルティ）"""
        imp = 14.0 * (1.0 - np.exp(-max(g, 0.0) / 10.0))
        knee = self.cfg["knee"]
        excess = 0.045 * max(0.0, g - knee) ** 1.6
        over = max(0.0, g - self.cfg["overload"])
        return self.cfg["ext_cn"] + imp - excess - 3.0 * over ** 0.8

    def optimum(self):
        vals = [self.channel_snr(g) for g in GAINS]
        i = int(np.argmax(vals))
        return vals[i], GAINS[i]

    def make_raw(self):
        g = self.gain
        scale = self.cfg["scale"]
        std = scale * (0.4 + 0.6 * 10.0 ** ((g - 34.0) / 28.0))
        std += 4.0 * max(0.0, g - self.cfg["overload"])
        vals = np.random.normal(127.5, max(std, 1.0), size=131072)
        return np.clip(vals, 0, 255).astype(np.uint8)

    def make_spectrum(self, n=1024):
        g = self.gain
        floor = -78.0 + 0.28 * g
        spec = np.random.normal(floor, 0.55, size=n)
        c = n // 2
        bin_hz = 1152000.0 / n
        ch_half = max(2, int(85000.0 / bin_hz))
        spec[c - ch_half:c + ch_half + 1] += self.channel_snr(g)
        if self.cfg.get("adjacent_db", 0.0) > 0.0:
            adj = int(300000.0 / bin_hz)
            spec[c + adj - 20:c + adj + 21] += self.cfg["adjacent_db"]
        dc = int(150000.0 / bin_hz)
        spec[c - dc - 3:c - dc + 4] += 24.0  # ハードウェアDCスパイク
        return spec.astype(np.float32)

    def make_audio(self, hf_applied, n=2048):
        t = np.arange(n) / 48000.0
        prog = 0.15 * np.sin(2.0 * np.pi * 1000.0 * t)
        true_audio_snr = self.channel_snr(self.gain) - 8.0
        hiss = np.random.normal(0.0, 0.15 * 10.0 ** (-true_audio_snr / 20.0), size=n)
        return (prog + hiss * hf_applied).astype(np.float32)


class SimDriver:
    def __init__(self, env):
        self.env = env
        self.gain = 33.8
        self.gain_changes = 0
        self.is_open = True

    def get_gains(self):
        return list(GAINS)

    def set_gain_mode(self, manual):
        pass

    def set_gain(self, g):
        if abs(g - self.gain) > 1e-9:
            self.gain_changes += 1
        self.gain = float(g)
        self.env.set_gain(self.gain)


class SimDsp:
    def __init__(self):
        self.rf_rate = 1152000
        self.audio_rate = 48000
        self.fft_size = 1024
        self.offset_freq = 150000.0
        self.filter_mode = "clean"
        self.cognitive_enabled = False
        self.hf_gain_applied = 1.0
        self.last_params = {}

    def set_cognitive_parameters(self, cutoff_hz=None, hf_gain=None, if_bw_hz=None, enabled=True):
        self.cognitive_enabled = enabled
        self.last_params = dict(cutoff_hz=cutoff_hz, hf_gain=hf_gain, if_bw_hz=if_bw_hz)
        if hf_gain is not None:
            self.hf_gain_applied = float(hf_gain)


def run_controller(ctrl_cls, env, is_hyper):
    driver = SimDriver(env)
    dsp = SimDsp()
    ctrl = ctrl_cls(driver, dsp, None)
    ctrl.init_gains()

    snr_traj = []
    gain_traj = []
    for _ in range(FRAMES):
        _CLOCK["t"] += FRAME_DT
        raw = env.make_raw()
        spec = env.make_spectrum()
        if is_hyper:
            pcm = env.make_audio(dsp.hf_gain_applied)
            ctrl.process_frame(raw, spec, audio=pcm, mode="WFM")
        else:
            ctrl.process_frame(raw, spec)
        snr_traj.append(env.channel_snr(driver.gain))
        gain_traj.append(driver.gain)

    snr_traj = np.array(snr_traj)
    gain_traj = np.array(gain_traj)
    opt_snr, opt_gain = env.optimum()

    tail = float(np.mean(snr_traj[-100:]))
    regret = opt_snr - tail

    roll = np.convolve(snr_traj, np.ones(15) / 15.0, mode="valid")
    conv_frame = FRAMES
    for i in range(len(roll) - 25):
        if float(np.mean(snr_traj[i + 14:])) >= opt_snr - 1.5:
            conv_frame = i + 14
            break

    transitions = int(np.sum(np.abs(np.diff(gain_traj)) > 1e-9))
    steady_changes = int(np.sum(np.abs(np.diff(gain_traj[-150:])) > 1e-9))
    oscillation = float(np.std(gain_traj[-100:]))

    return {
        "final_cn": tail,
        "best_cn": float(np.max(snr_traj)),
        "regret": regret,
        "conv_frame": conv_frame,
        "gain_changes": transitions,
        "steady_changes": steady_changes,
        "oscillation": oscillation,
        "opt_gain": opt_gain,
        "final_gain": float(gain_traj[-1]),
    }


SCENARIOS = [
    dict(name="弱電界DX (低利得アンテナ)", ext_cn=2.0, knee=36.0, overload=48.0, scale=16.0, adjacent_db=0.0),
    dict(name="中電界 (標準受信)", ext_cn=16.0, knee=32.0, overload=41.0, scale=34.0, adjacent_db=0.0),
    dict(name="強電界 (過大入力)", ext_cn=38.0, knee=26.0, overload=30.0, scale=45.0, adjacent_db=0.0),
    dict(name="隣接強局汚染 (目標弱局)", ext_cn=6.0, knee=34.0, overload=45.0, scale=20.0, adjacent_db=38.0),
]


def run_benchmark():
    print("=" * 78)
    print("HYPER vs CASCADE 自律制御ベンチマーク (物理モデルシミュレーション)")
    print("=" * 78)
    wins = {"final": 0, "regret": 0, "conv": 0, "churn": 0}
    rows = []

    for cfg in SCENARIOS:
        env = TunerEnvironment(cfg)
        opt_snr, opt_gain = env.optimum()
        _CLOCK["t"] += 100.0

        r_cas = run_controller(CascadeController, env, is_hyper=False)
        _CLOCK["t"] += 100.0
        r_hyp = run_controller(HyperController, env, is_hyper=True)
        _CLOCK["t"] += 100.0

        rows.append((cfg, opt_snr, opt_gain, r_cas, r_hyp))
        if r_hyp["final_cn"] > r_cas["final_cn"]:
            wins["final"] += 1
        if r_hyp["regret"] < r_cas["regret"]:
            wins["regret"] += 1
        if r_hyp["conv_frame"] < r_cas["conv_frame"]:
            wins["conv"] += 1
        if r_hyp["gain_changes"] <= r_cas["gain_changes"]:
            wins["churn"] += 1

    for cfg, opt_snr, opt_gain, r_cas, r_hyp in rows:
        print()
        print(f"■ シナリオ: {cfg['name']}")
        print(f"  理論最適: C/N {opt_snr:5.1f} dB @ ゲイン {opt_gain:4.1f} dB")
        print(f"  {'指標':<22}{'Cascade':>14}{'Hyper':>14}{'優位':>10}")
        print(f"  {'-'*60}")
        _row("収束後 C/N (dB)", r_cas["final_cn"], r_hyp["final_cn"], "%.2f", higher_better=True)
        _row("最適との後悔 (dB)", r_cas["regret"], r_hyp["regret"], "%.2f", higher_better=False)
        _row("ピーク C/N (dB)", r_cas["best_cn"], r_hyp["best_cn"], "%.2f", higher_better=True)
        _row("収束フレーム", r_cas["conv_frame"], r_hyp["conv_frame"], "%d", higher_better=False)
        _row("総ゲイン変更回数", r_cas["gain_changes"], r_hyp["gain_changes"], "%d", higher_better=False)
        _row("定常期変更回数", r_cas["steady_changes"], r_hyp["steady_changes"], "%d", higher_better=False)
        _row("定常ゲイン揺らぎ(dB)", r_cas["oscillation"], r_hyp["oscillation"], "%.3f", higher_better=False)
        _row("最終ゲイン (dB)", r_cas["final_gain"], r_hyp["final_gain"], "%.1f", higher_better=None)

    print()
    print("=" * 78)
    n = len(SCENARIOS)
    print(f"HYPER 勝利数: 収束後C/N {wins['final']}/{n} | 後悔最小 {wins['regret']}/{n} | "
          f"収束速度 {wins['conv']}/{n} | 変更回数 {wins['churn']}/{n}")
    print("=" * 78)


def _row(label, vc, vh, fmt, higher_better):
    if higher_better is None:
        mark = "-"
    elif higher_better:
        mark = "Hyper ▲" if vh > vc + 1e-9 else ("Cascade ▲" if vc > vh + 1e-9 else "互角")
    else:
        mark = "Hyper ▲" if vh < vc - 1e-9 else ("Cascade ▲" if vc < vh - 1e-9 else "互角")
    print(f"  {label:<22}{fmt % vc:>14}{fmt % vh:>14}{mark:>10}")


def dsp_path_sanity():
    """実DSPパイプラインに連続認知パラメータを流し、経路の健全性を検証"""
    from dsp import SdrDspPipeline

    print()
    print("=" * 78)
    print("実DSP連続認知制御パス スモークテスト")
    print("=" * 78)

    dsp = SdrDspPipeline(1152000, 48000)
    dsp.set_offset_freq(150000.0)
    rng = np.random.default_rng(1234)

    total_audio = 0
    for i in range(240):
        cutoff = 4300.0 + 10000.0 * (0.5 + 0.5 * np.sin(i / 20.0))
        hf = 0.5 + 0.5 * np.sin(i / 13.0)
        bw = 120000.0 + 70000.0 * (0.5 + 0.5 * np.cos(i / 17.0))
        dsp.set_cognitive_parameters(cutoff_hz=cutoff, hf_gain=hf, if_bw_hz=bw)

        raw = rng.integers(0, 256, size=48000, dtype=np.uint8)
        mode = "AM" if i % 5 == 0 else "WFM"
        audio, spec = dsp.process(raw, mode=mode)

        assert audio.dtype == np.float32, "audio dtype mismatch"
        assert np.all(np.isfinite(audio)), f"audio non-finite at frame {i}"
        assert len(spec) == dsp.fft_size, "spectrum size mismatch"
        assert np.all(np.isfinite(spec)), f"spectrum non-finite at frame {i}"
        total_audio += len(audio)

    from dsp import design_fir_kaiser, design_fir_highpass
    lp = design_fir_kaiser(81, 0.2)
    hp = design_fir_highpass(81, 0.2)
    recon = lp + hp
    err = float(np.max(np.abs(recon - np.eye(1, len(recon), len(recon) // 2)[0])))
    assert err < 1e-5, f"crossover reconstruction error {err}"

    print(f"[OK] 240フレーム無段階モーフィング走破 (出力オーディオ総数 {total_audio} samples)")
    print(f"[OK] クロスオーバー完全再構成誤差 {err:.2e}")
    print("[OK] 例外・NaN・サンプル不整合なし")


def main():
    try:
        run_benchmark()
        dsp_path_sanity()
    finally:
        _time.time = _REAL_TIME


if __name__ == "__main__":
    main()
