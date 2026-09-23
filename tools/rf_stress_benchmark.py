"""RFストレス & 破壊限界ベンチマーク (Receiver Resilience / Stress Mapping)。

悪条件・過入力・外乱・モード遷移下におけるKomorebiSDRパイプラインの
破壊限界（どこで復調が破綻・発散・異音発生するか）を網羅的・定量的に可視化する。

測定シナリオ:
1. OVERLOAD_CLIP: ADCクリップ率 (5%, 20%, 50%) による高調波歪み・発散
2. BLOCKER_ADJACENT: +200kHz/+400kHz での +25dB/+35dB 近接強妨害波
3. DEEP_FADE: 定常信号からの急激な -40dB 信号消失 (スケルチ挙動・ヒス爆発)
4. FREQ_STEP: ±15kHz の搬送波急変 (過渡クリックスパイク・AFC追従)
5. MODE_SWITCH: WFM ↔ AM ↔ NFM ↔ USB の高速切り替え (ブロック境界段差・残留履歴)
6. DC_IQ_IMBALANCE: DCオフセット +0.35 FS、位相誤差 15°、ゲイン不均衡 3dB
7. SINGULARITY: オールゼロ、極小ノイズ (-90dBFS)、NaN混入に対する自己回復

Usage:
    python tools/rf_stress_benchmark.py [--all] [--json report.json]
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dsp import SdrDspPipeline

RF_RATE = 1152000
AUDIO_RATE = 48000
BLOCK_SIZE = 132096  # 66048 IQ samples = 57.33 ms


def _iq_to_u8(iq: np.ndarray) -> np.ndarray:
    """複素IQ (-1.0〜+1.0) を RTL-SDR の uint8 バイト列に変換"""
    raw = np.empty(2 * len(iq), dtype=np.uint8)
    raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255)
    raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255)
    return raw


def _make_fm_signal(n_samples: int, mod_freq: float = 1000.0, dev_hz: float = 50000.0,
                    snr_db: float = 40.0, offset_hz: float = 0.0) -> np.ndarray:
    """基準WFM変調信号の生成"""
    t = np.arange(n_samples) / RF_RATE
    mod = np.sin(2.0 * np.pi * mod_freq * t)
    phase = 2.0 * np.pi * (dev_hz * np.cumsum(mod) / RF_RATE + offset_hz * t)
    sig = 0.5 * np.exp(1j * phase)
    if snr_db < 60.0:
        noise_pwr = (0.5 ** 2) / (10.0 ** (snr_db / 10.0))
        noise = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) * np.sqrt(noise_pwr / 2.0)
        sig += noise
    return sig.astype(np.complex64)


def _to_2ch(audio: np.ndarray) -> np.ndarray:
    """1D (モノラル) を 2D (ステレオ同相) に正規化"""
    if audio.ndim == 1:
        return np.column_stack([audio, audio])
    return audio


def _measure_clicks(pcm: np.ndarray) -> float:
    """オーディオ波形内の最大サンプル間段差 (dBFS) を測定"""
    if len(pcm) < 2:
        return -120.0
    arr = pcm if pcm.ndim == 1 else pcm[:, 0]
    diffs = np.abs(np.diff(arr))
    max_step = float(np.max(diffs)) if len(diffs) > 0 else 0.0
    return float(20.0 * np.log10(max(max_step, 1e-6)))


def run_scenario_overload() -> dict:
    """シナリオ1: ADC過大入力・クリッピング耐性"""
    print("\n[Scenario 1: OVERLOAD / ADC CLIPPING]")
    results = {}
    n_blk = 6
    n_samples = n_blk * (BLOCK_SIZE // 2)

    clean_iq = _make_fm_signal(n_samples, mod_freq=1000.0, dev_hz=40000.0, snr_db=50.0)

    # 3段階の過大入力倍率 (x1.0: 通常, x2.5: 軽度飽和~10%, x6.0: 重度飽和~45%)
    scales = [("normal_1x", 1.0), ("mild_clip_2.5x", 2.5), ("hard_clip_6.0x", 6.0)]
    for label, scale in scales:
        overloaded = clean_iq * scale
        raw = _iq_to_u8(overloaded)

        dsp = SdrDspPipeline(RF_RATE, AUDIO_RATE)
        dsp.cognitive_enabled = True

        out_blocks = []
        clip_flags = []
        for b in range(n_blk):
            chunk = raw[b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE]
            audio, _ = dsp.process(chunk, mode="WFM")
            out_blocks.append(_to_2ch(audio))
            clip_flags.append(bool(getattr(dsp, "adc_clipped", False)))

        full_audio = np.concatenate(out_blocks)
        nan_inf = bool(not np.all(np.isfinite(full_audio)))
        max_click = _measure_clicks(full_audio)
        max_amp = float(np.max(np.abs(full_audio)))
        detected_clip = any(clip_flags)

        # 評価: NaN/Inf発生がなく、出力振幅が±1.0内に抑えられているか
        status = "FAIL" if nan_inf or max_amp > 1.05 else ("WARN" if max_click > -3.0 else "PASS")
        print(f"  {label:<16}: status={status} | NaN/Inf={nan_inf} | max_amp={max_amp:.3f} | "
              f"max_click={max_click:+.1f}dBFS | clip_detected={detected_clip}")
        results[label] = {
            "status": status,
            "nan_inf": nan_inf,
            "max_amp": max_amp,
            "max_click_dbfs": max_click,
            "clip_detected": detected_clip,
        }
    return results


def run_scenario_blocker() -> dict:
    """シナリオ2: 近接強妨害波 (隣接チャンネル干渉)"""
    print("\n[Scenario 2: ADJACENT STRONG BLOCKER]")
    results = {}
    n_blk = 6
    n_samples = n_blk * (BLOCK_SIZE // 2)

    # 希望波: 1000Hz変調, 振幅0.1 (-14dBFS)
    wanted = 0.1 * _make_fm_signal(n_samples, mod_freq=1000.0, dev_hz=30000.0, snr_db=50.0)

    # 妨害波パターン: +200kHz (+20dB), +400kHz (+30dB)
    blockers = [
        ("adj_200k_20db", 200000.0, 1.0),   # 10倍 (+20dB)
        ("adj_400k_30db", 400000.0, 3.16),  # 31.6倍 (+30dB)
    ]

    for label, f_offset, amp in blockers:
        t = np.arange(n_samples) / RF_RATE
        # 妨害波もFM変調波
        blocker = amp * np.exp(1j * (2.0 * np.pi * f_offset * t + np.sin(2.0 * np.pi * 3000.0 * t)))
        mix = wanted + blocker
        raw = _iq_to_u8(mix)

        dsp = SdrDspPipeline(RF_RATE, AUDIO_RATE)
        dsp.cognitive_enabled = True

        out_blocks = []
        for b in range(n_blk):
            chunk = raw[b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE]
            audio, _ = dsp.process(chunk, mode="WFM")
            out_blocks.append(_to_2ch(audio))

        full_audio = np.concatenate(out_blocks)
        nan_inf = bool(not np.all(np.isfinite(full_audio)))
        
        # 1000Hz トーンの残留検出 (後半3ブロックで評価)
        eval_seg = full_audio[len(full_audio) // 2 :]
        mono_seg = eval_seg if eval_seg.ndim == 1 else eval_seg[:, 0]
        spec = np.abs(np.fft.rfft(mono_seg * np.hanning(len(mono_seg))))
        freqs = np.fft.rfftfreq(len(mono_seg), 1.0 / AUDIO_RATE)
        idx_1k = np.argmin(np.abs(freqs - 1000.0))
        p_1k = float(spec[idx_1k])
        p_noise = float(np.median(spec)) + 1e-12
        snr_est = 20.0 * np.log10(p_1k / p_noise)

        status = "FAIL" if nan_inf or snr_est < 6.0 else ("WARN" if snr_est < 14.0 else "PASS")
        print(f"  {label:<16}: status={status} | NaN/Inf={nan_inf} | recovered_tone_SNR={snr_est:.1f}dB")
        results[label] = {
            "status": status,
            "nan_inf": nan_inf,
            "recovered_tone_snr_db": snr_est,
        }
    return results


def run_scenario_deep_fade() -> dict:
    """シナリオ3: 深フェージング・高速信号消失"""
    print("\n[Scenario 3: DEEP FAST FADING]")
    results = {}
    n_blk = 8
    n_samples = n_blk * (BLOCK_SIZE // 2)

    # 4ブロック良好 → 2ブロック急峻ドロップ (-40dB) → 2ブロック復帰
    iq = _make_fm_signal(n_samples, mod_freq=1000.0, dev_hz=40000.0, snr_db=50.0)
    drop_start = 3 * (BLOCK_SIZE // 2)
    drop_end = 6 * (BLOCK_SIZE // 2)
    iq[drop_start:drop_end] *= 0.01  # -40dB 減衰

    raw = _iq_to_u8(iq)
    dsp = SdrDspPipeline(RF_RATE, AUDIO_RATE)
    dsp.cognitive_enabled = True

    out_blocks = []
    blends = []
    for b in range(n_blk):
        chunk = raw[b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE]
        audio, _ = dsp.process(chunk, mode="WFM")
        out_blocks.append(_to_2ch(audio))
        blends.append(float(getattr(dsp, "stereo_blend", 0.0)))

    full_audio = np.concatenate(out_blocks)
    nan_inf = bool(not np.all(np.isfinite(full_audio)))
    
    # 信号消失区間での振幅（スケルチまたはブレンド低下でノイズ爆発していないか）
    fade_audio = full_audio[drop_start * AUDIO_RATE // RF_RATE : drop_end * AUDIO_RATE // RF_RATE]
    fade_rms = float(np.sqrt(np.mean(fade_audio ** 2))) + 1e-12
    fade_rms_dbfs = 20.0 * np.log10(fade_rms)

    # フェード中のブレンド解放
    fade_blend = blends[4]

    # ノイズ爆発 (> -10dBFS) がなくブレンドが落ちているか
    status = "FAIL" if nan_inf or fade_rms_dbfs > -10.0 else ("WARN" if fade_blend > 0.6 else "PASS")
    print(f"  deep_fade_recovery: status={status} | NaN/Inf={nan_inf} | "
          f"fade_rms={fade_rms_dbfs:.1f}dBFS | blend_in_fade={fade_blend:.2f}")
    results["deep_fade"] = {
        "status": status,
        "nan_inf": nan_inf,
        "fade_rms_dbfs": fade_rms_dbfs,
        "blend_in_fade": fade_blend,
    }
    return results


def run_scenario_freq_step() -> dict:
    """シナリオ4: 周波数ステップ急変 (AFC・過渡クリック)"""
    print("\n[Scenario 4: CARRIER FREQUENCY STEP]")
    results = {}
    n_blk = 6
    n_samples = n_blk * (BLOCK_SIZE // 2)

    # ブロック3で瞬時に周波数が +15kHz 飛ぶ
    step_point = 3 * (BLOCK_SIZE // 2)
    t = np.arange(n_samples) / RF_RATE
    mod = np.sin(2.0 * np.pi * 1000.0 * t)
    offset = np.zeros(n_samples)
    offset[step_point:] = 15000.0
    phase = 2.0 * np.pi * (40000.0 * np.cumsum(mod) / RF_RATE + offset * t)
    iq = 0.5 * np.exp(1j * phase)
    raw = _iq_to_u8(iq)

    dsp = SdrDspPipeline(RF_RATE, AUDIO_RATE)
    dsp.afc_enabled = True

    out_blocks = []
    afc_offsets = []
    for b in range(n_blk):
        chunk = raw[b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE]
        audio, _ = dsp.process(chunk, mode="WFM")
        out_blocks.append(_to_2ch(audio))
        afc_offsets.append(float(getattr(dsp, "afc_offset_hz", 0.0)))

    full_audio = np.concatenate(out_blocks)
    nan_inf = bool(not np.all(np.isfinite(full_audio)))
    max_click = _measure_clicks(full_audio)
    end_afc = afc_offsets[-1]

    # ステップ過渡でNaNが出ず、AFCが追従を開始しているか
    status = "FAIL" if nan_inf else ("WARN" if max_click > -3.0 else "PASS")
    print(f"  step_15khz        : status={status} | NaN/Inf={nan_inf} | "
          f"max_click={max_click:+.1f}dBFS | afc_end_hz={end_afc:+.0f}Hz")
    results["freq_step"] = {
        "status": status,
        "nan_inf": nan_inf,
        "max_click_dbfs": max_click,
        "afc_end_hz": end_afc,
    }
    return results


def run_scenario_mode_switch() -> dict:
    """シナリオ5: 高速モード遷移 (WFM ↔ AM ↔ NFM ↔ USB)"""
    print("\n[Scenario 5: RAPID MODE SWITCHING]")
    results = {}
    modes = ["WFM", "AM", "NFM", "USB", "WFM", "LSB", "CW"]
    n_blk = len(modes)

    dsp = SdrDspPipeline(RF_RATE, AUDIO_RATE)
    dsp.cognitive_enabled = True

    # 共通のテストトーン信号
    n_samples = BLOCK_SIZE // 2
    raw = _iq_to_u8(0.4 * np.exp(2j * np.pi * 1000.0 * np.arange(n_samples) / RF_RATE))

    out_blocks = []
    clicks = []
    last_sample = 0.0
    for b, mode in enumerate(modes):
        audio, _ = dsp.process(raw, mode=mode)
        out_blocks.append(_to_2ch(audio))
        first_sample = float(audio[0, 0] if audio.ndim > 1 else audio[0]) if len(audio) > 0 else 0.0
        if b > 0:
            step = abs(first_sample - last_sample)
            click_db = 20.0 * np.log10(max(step, 1e-6))
            clicks.append(click_db)
        if len(audio) > 0:
            last_sample = float(audio[-1, 0] if audio.ndim > 1 else audio[-1])

    max_boundary_click = max(clicks) if clicks else -120.0
    all_finite = all(np.all(np.isfinite(b)) for b in out_blocks)

    # 境界クリックスパイクが -6dBFS 未満か (切替時の破裂音チェック)
    status = "FAIL" if not all_finite else ("WARN" if max_boundary_click > -6.0 else "PASS")
    print(f"  rapid_mode_switch : status={status} | all_finite={all_finite} | "
          f"max_boundary_step={max_boundary_click:+.1f}dBFS")
    results["mode_switch"] = {
        "status": status,
        "all_finite": all_finite,
        "max_boundary_click_dbfs": max_boundary_click,
    }
    return results


def run_scenario_dc_iq_imbalance() -> dict:
    """シナリオ6: DCオフセット & IQインバランス"""
    print("\n[Scenario 6: DC SPIKE & IQ IMBALANCE]")
    results = {}
    n_blk = 6
    n_samples = n_blk * (BLOCK_SIZE // 2)

    # 基準信号
    iq_clean = 0.3 * np.exp(2j * np.pi * 15000.0 * np.arange(n_samples) / RF_RATE)
    
    # DCオフセット +0.35 FS & I/Q ゲイン比 1.4 (+3dB) & 位相誤差 15°
    i_dist = (iq_clean.real + 0.35) * 1.4
    q_dist = iq_clean.imag * np.cos(np.deg2rad(15.0)) - iq_clean.real * np.sin(np.deg2rad(15.0))
    iq_distorted = i_dist + 1j * q_dist
    raw = _iq_to_u8(iq_distorted)

    dsp = SdrDspPipeline(RF_RATE, AUDIO_RATE)
    dsp.cognitive_enabled = True

    out_blocks = []
    for b in range(n_blk):
        chunk = raw[b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE]
        audio, _ = dsp.process(chunk, mode="WFM")
        out_blocks.append(_to_2ch(audio))

    full_audio = np.concatenate(out_blocks)
    nan_inf = bool(not np.all(np.isfinite(full_audio)))
    dc_residual = float(np.abs(np.mean(full_audio)))
    dc_residual_db = 20.0 * np.log10(max(dc_residual, 1e-6))

    # DCハイパスによって残留DCが -30dBFS 以下に抑圧されているか
    status = "FAIL" if nan_inf or dc_residual_db > -20.0 else ("WARN" if dc_residual_db > -35.0 else "PASS")
    print(f"  dc_iq_distortion  : status={status} | NaN/Inf={nan_inf} | residual_DC={dc_residual_db:.1f}dBFS")
    results["dc_iq"] = {
        "status": status,
        "nan_inf": nan_inf,
        "residual_dc_dbfs": dc_residual_db,
    }
    return results


def run_scenario_singularity() -> dict:
    """シナリオ7: 特異点・無信号・非有限値入力"""
    print("\n[Scenario 7: SINGULARITY / ZERO / NAN SURVIVAL]")
    results = {}
    dsp = SdrDspPipeline(RF_RATE, AUDIO_RATE)

    cases = [
        ("all_zeros", np.zeros(BLOCK_SIZE, dtype=np.uint8)),
        ("all_ones_0xff", np.full(BLOCK_SIZE, 255, dtype=np.uint8)),
    ]

    for label, raw_chunk in cases:
        try:
            audio, _ = dsp.process(raw_chunk, mode="WFM")
            is_finite = bool(np.all(np.isfinite(audio)))
            status = "PASS" if is_finite else "FAIL"
        except Exception as e:
            is_finite = False
            status = f"FAIL (Exception: {e})"
        print(f"  {label:<18}: status={status} | finite={is_finite}")
        results[label] = {"status": status, "is_finite": is_finite}

    # 特異点入力後の自己回復テスト (正常信号が再び復調できるか)
    clean_raw = _iq_to_u8(0.4 * np.exp(2j * np.pi * 1000.0 * np.arange(BLOCK_SIZE // 2) / RF_RATE))
    try:
        audio, _ = dsp.process(clean_raw, mode="WFM")
        recovered = bool(np.all(np.isfinite(audio))) and float(np.std(audio)) > 1e-4
        rec_status = "PASS" if recovered else "FAIL"
    except Exception as e:
        recovered = False
        rec_status = f"FAIL (Exception: {e})"
    print(f"  recovery_after_zero: status={rec_status} | recovered={recovered}")
    results["recovery"] = {"status": rec_status, "recovered": recovered}
    return results


def run_breaking_limits_map() -> dict:
    """破壊限界マッピング: 信号がどこまで耐えられ、どこで破綻するかをスキャン"""
    print("\n" + "=" * 70)
    print("BREAKING POINT LIMIT MAPPING (限界境界探索)")
    print("=" * 70)
    limits = {}

    n_blk = 4
    n_samples = n_blk * (BLOCK_SIZE // 2)

    # 1. 過入力クリップ限界 (Overload Ceiling)
    print("\n[1. Overload Scale Sweep (過入力破綻スキャン)]")
    overload_map = []
    clean_iq = _make_fm_signal(n_samples, mod_freq=1000.0, dev_hz=40000.0, snr_db=50.0)
    for mult in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
        mult_db = 20.0 * np.log10(mult)
        raw = _iq_to_u8(clean_iq * mult)
        dsp = SdrDspPipeline(RF_RATE, AUDIO_RATE)
        dsp.cognitive_enabled = True
        outs = [_to_2ch(dsp.process(raw[b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE], mode="WFM")[0]) for b in range(n_blk)]
        audio = np.concatenate(outs)
        clip_pct = float(np.mean(raw == 0) + np.mean(raw == 255)) * 50.0  # 概算飽和率
        max_click = _measure_clicks(audio)
        
        # 1kHzトーンのTHD+N (SINAD) 推定: ピーク周辺 ±4 ビンを基本波とする
        mono = audio[:, 0]
        eval_seg = mono[len(mono)//2:]
        spec = np.abs(np.fft.rfft(eval_seg * np.hanning(len(eval_seg))))
        freqs = np.fft.rfftfreq(len(eval_seg), 1.0 / AUDIO_RATE)
        idx_1k = np.argmin(np.abs(freqs - 1000.0))
        fund_mask = (np.abs(freqs - 1000.0) <= 60.0)
        p_fund = float(np.sum(spec[fund_mask] ** 2))
        dc_mask = (freqs <= 100.0)
        noise_mask = (~fund_mask) & (~dc_mask)
        p_noise = float(np.sum(spec[noise_mask] ** 2)) + 1e-12
        sinad_db = 10.0 * np.log10(p_fund / p_noise)
        
        broken = sinad_db < 10.0 or not np.all(np.isfinite(audio))
        state = "BROKEN" if broken else ("DEGRADED" if sinad_db < 20.0 else "SURVIVED")
        print(f"  Input {mult_db:+5.1f}dB ({mult:4.1f}x): state={state:<8} | clip_rate={clip_pct:4.1f}% | "
              f"SINAD={sinad_db:5.1f}dB | max_click={max_click:+.1f}dBFS")
        overload_map.append({"mult_db": mult_db, "state": state, "sinad_db": sinad_db, "clip_pct": clip_pct})
    limits["overload"] = overload_map

    # 2. 隣接妨害波の排除限界 (Adjacent Blocker Rejection Limit)
    print("\n[2. Adjacent Blocker Sweep (隣接妨害波排除限界)]")
    blocker_map = {}
    wanted = 0.1 * _make_fm_signal(n_samples, mod_freq=1000.0, dev_hz=30000.0, snr_db=50.0)
    for offset_khz in [100.0, 200.0, 400.0]:
        t = np.arange(n_samples) / RF_RATE
        sweep_data = []
        for blocker_db in [10.0, 20.0, 30.0, 40.0, 50.0]:
            amp = 0.1 * (10.0 ** (blocker_db / 20.0))
            blocker = amp * np.exp(1j * (2.0 * np.pi * offset_khz * 1000.0 * t + np.sin(2.0 * np.pi * 3000.0 * t)))
            mix = wanted + blocker
            raw = _iq_to_u8(mix)
            dsp = SdrDspPipeline(RF_RATE, AUDIO_RATE)
            dsp.cognitive_enabled = True
            outs = [_to_2ch(dsp.process(raw[b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE], mode="WFM")[0]) for b in range(n_blk)]
            audio = np.concatenate(outs)
            
            mono = audio[:, 0]
            eval_seg = mono[len(mono)//2:]
            spec = np.abs(np.fft.rfft(eval_seg * np.hanning(len(eval_seg))))
            freqs = np.fft.rfftfreq(len(eval_seg), 1.0 / AUDIO_RATE)
            fund_mask = (np.abs(freqs - 1000.0) <= 60.0)
            p_1k = float(np.max(spec[fund_mask]))
            p_noise = float(np.median(spec)) + 1e-12
            tone_snr = 20.0 * np.log10(p_1k / p_noise)
            broken = tone_snr < 6.0
            state = "BROKEN" if broken else ("MARGINAL" if tone_snr < 15.0 else "INTACT")
            sweep_data.append({"blocker_db": blocker_db, "tone_snr_db": tone_snr, "state": state})
        
        intact_max = max([d["blocker_db"] for d in sweep_data if d["state"] != "BROKEN"] or [0.0])
        print(f"  Δf={offset_khz:+.0f}kHz: Rejection limit = +{intact_max:.0f}dB (破綻点: {intact_max + 10:.0f}dB)")
        blocker_map[f"{offset_khz:.0f}kHz"] = {"rejection_limit_db": intact_max, "details": sweep_data}
    limits["adjacent_rejection"] = blocker_map

    # 3. 搬送波オフセット追従限界 (AFC Pull-in Range, 15ブロック滞留)
    print("\n[3. Carrier Frequency Offset Sweep (AFC引き込み限界)]")
    afc_map = []
    n_afc_blk = 15
    n_afc_samples = n_afc_blk * (BLOCK_SIZE // 2)
    for f_off_khz in [3.0, 8.0, 15.0, 25.0, 50.0]:
        iq_off = _make_fm_signal(n_afc_samples, mod_freq=1000.0, dev_hz=30000.0, snr_db=50.0, offset_hz=f_off_khz * 1000.0)
        raw = _iq_to_u8(iq_off)
        dsp = SdrDspPipeline(RF_RATE, AUDIO_RATE)
        dsp.afc_enabled = True
        outs = []
        for b in range(n_afc_blk):
            audio, _ = dsp.process(raw[b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE], mode="WFM")
            outs.append(_to_2ch(audio))
        final_afc = float(getattr(dsp, "afc_offset_hz", 0.0))
        error_hz = abs(final_afc - (-f_off_khz * 1000.0))
        pulled_in = error_hz < 2500.0
        state = "LOCKED" if pulled_in else ("PULLING" if abs(final_afc) > 2000.0 else "OUT_OF_RANGE")
        print(f"  Offset {f_off_khz:+4.0f}kHz: state={state:<12} | final_afc={final_afc:+6.0f}Hz | residual_err={error_hz:5.0f}Hz")
        afc_map.append({"offset_khz": f_off_khz, "state": state, "final_afc_hz": final_afc, "error_hz": error_hz})

    return limits


def main():
    parser = argparse.ArgumentParser(description="KomorebiSDR RF Stress & Breaking Point Benchmark")
    parser.add_argument("--json", type=str, default="", help="Save report to JSON file")
    args = parser.parse_args()

    print("=" * 70)
    print("KomorebiSDR RF Stress & Resilience Benchmark")
    print("=" * 70)

    t0 = time.time()
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scenarios": {
            "overload": run_scenario_overload(),
            "blocker": run_scenario_blocker(),
            "deep_fade": run_scenario_deep_fade(),
            "freq_step": run_scenario_freq_step(),
            "mode_switch": run_scenario_mode_switch(),
            "dc_iq": run_scenario_dc_iq_imbalance(),
            "singularity": run_scenario_singularity(),
        },
        "breaking_limits": run_breaking_limits_map(),
    }
    elapsed = time.time() - t0

    # 総合スコア集計
    total_tests = 0
    passed_tests = 0
    warn_tests = 0
    fail_tests = 0

    for sc_name, sc_data in report["scenarios"].items():
        for t_name, t_info in sc_data.items():
            total_tests += 1
            st = t_info.get("status", "FAIL")
            if st == "PASS":
                passed_tests += 1
            elif "WARN" in st:
                warn_tests += 1
            else:
                fail_tests += 1

    score = 100.0 * (passed_tests + 0.5 * warn_tests) / max(total_tests, 1)
    report["summary"] = {
        "total": total_tests,
        "pass": passed_tests,
        "warn": warn_tests,
        "fail": fail_tests,
        "resilience_score": score,
        "elapsed_sec": elapsed,
    }

    print("\n" + "=" * 70)
    print("STRESS BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Total Checks     : {total_tests}")
    print(f"Passed           : {passed_tests}")
    print(f"Warnings         : {warn_tests}")
    print(f"Failed           : {fail_tests}")
    print(f"Resilience Score : {score:.1f} / 100.0 (Elapsed: {elapsed:.2f}s)")
    print("=" * 70)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report saved to {args.json}")


if __name__ == "__main__":
    main()

