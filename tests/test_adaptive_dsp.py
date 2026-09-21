"""
Unit tests for adaptive_dsp.py
適応IQインバランス補正およびダイナミックIF帯域トラッカーの単体数理シミュレーション検証。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from adaptive_dsp import (
    AdaptiveIqCorrector,
    CognitiveSpeechMusicTracker,
    UltrasonicSquelchTracker,
    QuadratureMpxCanceller,
    DeepSpaceEkfDemodulator,
    KalmanPilotTracker,
    HolographicAudioEnhancer,
    RiemannianTopologicalDemodulator,
    SuperSpatialBssStereoSeparator,
    RmtHankelDenoiser,
)


def test_iq_imbalance_correction():
    print("===== test_iq_imbalance_correction =====")
    fs = 1152000.0
    duration = 0.5  # 0.5秒分
    t = np.arange(int(fs * duration)) / fs

    # 1. 理想的な信号 (100kHzのトーン信号)
    f_sig = 100000.0
    ideal_i = np.cos(2 * np.pi * f_sig * t)
    ideal_q = np.sin(2 * np.pi * f_sig * t)

    # 2. 人工的な不均衡の付与:
    # 振幅比 1.15 (+1.21dB), 直交位相ズレ 4.0度 (0.0698 rad)
    gain_err = 1.15
    phase_err_rad = np.radians(4.0)

    distorted_i = ideal_i
    distorted_q = gain_err * np.sin(2 * np.pi * f_sig * t + phase_err_rad)
    distorted_iq = (distorted_i + 1j * distorted_q).astype(np.complex64)

    # 補正前のイメージ抑圧比 (IRR) を計測
    # 期待される鏡像は -100kHz
    def measure_irr(iq, f_tone):
        spec = np.abs(np.fft.fft(iq))
        freqs = np.fft.fftfreq(len(iq), 1.0 / fs)
        idx_sig = np.argmin(np.abs(freqs - f_tone))
        idx_img = np.argmin(np.abs(freqs - (-f_tone)))
        sig_p = spec[idx_sig] ** 2
        img_p = spec[idx_img] ** 2
        irr_db = 10.0 * np.log10(sig_p / (img_p + 1e-12))
        return irr_db

    initial_irr = measure_irr(distorted_iq[:8192], f_sig)
    print(f"[*] 補正前のイメージ抑圧比 (IRR): {initial_irr:.2f} dB")
    assert initial_irr < 32.0, "補正前は鏡像が強く出ているはず"

    # 3. 適応補正器にブロック単位 (66048サンプル ≈ 57ms) で流し込む
    corrector = AdaptiveIqCorrector(sample_rate=fs, time_constant_sec=0.15)
    block_size = 32768
    corrected_blocks = []

    for start in range(0, len(distorted_iq), block_size):
        chunk = distorted_iq[start:start + block_size]
        corrected_chunk = corrector.process(chunk)
        corrected_blocks.append(corrected_chunk)

    corrected_iq = np.concatenate(corrected_blocks)

    # 収束後 (後半ブロック) での IRR を計測
    final_irr = measure_irr(corrected_iq[-8192:], f_sig)
    print(f"[*] 収束後のイメージ抑圧比 (IRR): {final_irr:.2f} dB (推定位相誤差: {corrector.estimated_phase_error_deg:.2f}°, 推定ゲイン比: {corrector.estimated_gain_imbalance_db:.2f} dB)")

    # 鏡像抑圧比が 30dB 未満から 50dB 以上へ大幅向上していることを検証
    assert final_irr >= 50.0, f"収束後のIRRが不十分です: {final_irr:.2f} dB"
    assert abs(corrector.estimated_phase_error_deg - 4.0) < 1.0, "位相誤差の推定精度が不十分です"
    print("[OK] IQインバランス自動補正テスト成功 (IRR改善: +{:.1f} dB)".format(final_irr - initial_irr))


def test_ultrasonic_squelch():
    print("\n===== test_ultrasonic_squelch =====")
    sr = 288000.0
    squelch = UltrasonicSquelchTracker(sample_rate=sr, noise_threshold_db=-38.0)

    # 1. 局間ノイズ（未変調・ホワイト三角雑音）: 超音波高域パワーが大きい
    noise_demod = np.random.normal(0.0, 0.4, 4000).astype(np.float32)
    gain_mute = 1.0
    is_open_mute = True
    for _ in range(40):
        gain_mute, is_open_mute = squelch.process(noise_demod)

    print(f"[*] 局間ノイズ時のスケルチ状態: is_open={is_open_mute}, gain={gain_mute:.2f} (ノイズ推定: {squelch.noise_db:.1f} dB)")
    assert not is_open_mute, "局間ノイズでスケルチが閉じませんでした"
    assert gain_mute < 0.05, f"ソフトフェードミュートゲインが十分に下がりませんでした: {gain_mute}"

    # 2. 本物のFM変調波キャリア受信時 (クワイエティング効果で超音波ノイズ急減)
    t = np.arange(4000) / sr
    clean_demod = (0.25 * np.sin(2 * np.pi * 1000.0 * t) + np.random.normal(0.0, 0.0005, 4000)).astype(np.float32)
    gain_open = 0.0
    is_open = False
    for _ in range(40):
        gain_open, is_open = squelch.process(clean_demod)

    print(f"[*] キャリア受信時のスケルチ状態: is_open={is_open}, gain={gain_open:.2f} (ノイズ推定: {squelch.noise_db:.1f} dB)")
    assert is_open, "本物のキャリアを受信してもスケルチが開きませんでした"
    assert gain_open > 0.95, f"ソフトフェードゲインが十分に開きませんでした: {gain_open}"
    print("[OK] 超音波ノイズ比追従型コグニティブ・オートスケルチテスト成功")


def test_cognitive_speech_music_tracker():
    print("\n===== test_cognitive_speech_music_tracker =====")
    sr = 48000.0
    eq = CognitiveSpeechMusicTracker(sample_rate=sr)

    # 1. トーク音声区間 (300Hz〜2500Hz中心、低音サブベースなし、ロールオフ低)
    t = np.arange(1024) / sr
    speech_audio = (
        0.4 * np.sin(2 * np.pi * 800.0 * t) +
        0.3 * np.sin(2 * np.pi * 1800.0 * t) +
        0.1 * np.sin(2 * np.pi * 2800.0 * t)
    ).astype(np.float32)

    prob_speech = 0.0
    for _ in range(30):
        prob_speech = eq.analyze(speech_audio)

    print(f"[*] トーク区間の推定音声確率: {prob_speech:.2f} (期待値 > 0.6)")
    assert prob_speech > 0.6, f"トーク区間を音声と認識できませんでした: {prob_speech}"

    # トーク時の了解度EQ処理
    speech_out = eq.process(speech_audio)
    assert speech_out.shape == speech_audio.shape
    assert not np.allclose(speech_out, speech_audio), "トーク用EQが適用されていません"

    # 2. 音楽区間 (重低音50Hz〜60Hzサブベース + 12kHz以上のハイハット)
    music_audio = (
        0.6 * np.sin(2 * np.pi * 55.0 * t) +
        0.3 * np.sin(2 * np.pi * 1000.0 * t) +
        0.3 * np.sin(2 * np.pi * 12000.0 * t)
    ).astype(np.float32)

    prob_music = 1.0
    for _ in range(60):
        prob_music = eq.analyze(music_audio)

    print(f"[*] 音楽区間の推定音声確率: {prob_music:.2f} (期待値 < 0.2)")
    assert prob_music < 0.2, f"音楽区間を音楽と認識できませんでした: {prob_music}"

    # 音楽時は完全フラット（原音維持）
    music_out = eq.process(music_audio)
    assert np.allclose(music_out, music_audio, atol=1e-5), "音楽区間で完全フラットが維持されていません"
    print("[OK] 音声/音楽 認知型オートチルトEQテスト成功")


def test_quadrature_mpx_canceller():
    print("\n===== test_quadrature_mpx_canceller =====")
    sr = 48000.0
    canceller = QuadratureMpxCanceller(sample_rate=sr, taps=5, mu=0.05)

    # 1. クリーン信号テスト (直交軸ゼロ): 原音ビットパーフェクト維持
    t = np.arange(2048) / sr
    clean_i = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    zero_q = np.zeros_like(clean_i)
    out_clean = canceller.process(clean_i, zero_q)
    assert np.allclose(out_clean, clean_i, atol=1e-5), "クリーン時に原音が維持されていません"
    print("[OK] クリーン信号ビットパーフェクト通過確認")

    # 2. マルチパス干渉シミュレーション
    # 直交軸 Q に非線形歪み成分が存在し、同相軸 I へ 0.35 倍で漏れ込んでいる状態
    dist_q = (0.4 * np.sin(2 * np.pi * 3500.0 * t) + 0.2 * np.cos(2 * np.pi * 7000.0 * t)).astype(np.float32)
    leak_target = 0.35 * dist_q
    distorted_i = (clean_i + leak_target).astype(np.float32)

    # 複数ブロック適応反復
    out = distorted_i
    for _ in range(60):
        out = canceller.process(distorted_i, dist_q)

    # 残留歪み測定
    res_err = float(np.mean((out - clean_i) ** 2))
    orig_err = float(np.mean((distorted_i - clean_i) ** 2))
    suppression_db = 10.0 * np.log10(orig_err / (res_err + 1e-12))
    assert suppression_db > 12.0, f"マルチパス直交歪みが十分に抑圧されていません: {suppression_db:.1f} dB"
    print("[OK] 38kHz直交副搬送波マルチパス適応キャンセラテスト成功")


def test_deep_space_ekf_demodulator():
    print("\n===== test_deep_space_ekf_demodulator =====")
    fs = 288000.0
    duration = 0.05
    t = np.arange(int(fs * duration)) / fs

    # 1. 1kHz 変調波 (偏移 ±50kHz) の生成
    mod_signal = np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)
    dev_hz = 50000.0
    phase = np.cumsum(2.0 * np.pi * dev_hz * mod_signal / fs)
    clean_iq = np.exp(1j * phase).astype(np.complex64)

    ekf = DeepSpaceEkfDemodulator(sample_rate=fs)
    demod_clean = ekf.demodulate(clean_iq)
    
    # 瞬時周波数の追従精度 (相関 > 0.98)
    ref_norm = (2.0 * np.pi * dev_hz / fs) * mod_signal
    # 位相遅延を考慮した相互相関
    corr = np.corrcoef(demod_clean[200:], ref_norm[200:])[0, 1]
    print(f"[*] クリーンFM波のEKF復調相関度: {corr:.4f} (期待値 > 0.95)")
    assert corr > 0.95, f"EKF復調の追従精度が不十分です: {corr:.4f}"

    # 2. 低CNR極限環境 (強いガウス雑音 + インパルスノイズ) でのクリックスパイク抑圧テスト
    np.random.seed(42)
    noise = (np.random.normal(0, 0.45, len(clean_iq)) + 1j * np.random.normal(0, 0.45, len(clean_iq))).astype(np.complex64)
    noisy_iq = clean_iq + noise

    # 通常の位相差分法 (ナイーブ復調)
    diff_demod = np.angle(noisy_iq[1:] * np.conj(noisy_iq[:-1]))
    diff_spikes = int(np.sum(np.abs(np.diff(diff_demod)) > 0.8))

    # 深宇宙EKF復調 (Huber M推定によるクリックスパイク消去)
    ekf.reset()
    ekf_demod = ekf.demodulate(noisy_iq)
    ekf_spikes = int(np.sum(np.abs(np.diff(ekf_demod)) > 0.8))

    print(f"[*] 低CNR環境におけるクリックスパイク数: ナイーブ差分法={diff_spikes}個 -> 深宇宙EKF={ekf_spikes}個")
    assert ekf_spikes < diff_spikes // 3, f"EKFによるクリックスパイク抑圧が不十分です: EKF {ekf_spikes} vs 差分 {diff_spikes}"
    print("[OK] 深宇宙通信級 拡張カルマンフィルタ (EKF) FM復調テスト成功")


def test_kalman_pilot_tracker():
    print("\n===== test_kalman_pilot_tracker =====")
    fs = 288000.0
    tracker = KalmanPilotTracker(sample_rate=fs, fn_min=3.0, fn_max=24.0, zeta=0.85)

    # 1. 弱電界・高ジッター環境シミュレーション (低品質・残差大)
    for _ in range(15):
        kp, ki, alpha = tracker.update_gains(lock_quality=0.10, pilot_rms=0.005, ef_state=0.35)

    print(f"[*] 弱電界適応後: fn={tracker.current_fn:.2f} Hz (期待値 < 10 Hz), CNR={tracker.current_cnr_db:.1f} dB, Ki={ki:.4e}")
    assert tracker.current_fn < 10.0, f"弱電界で帯域幅が十分に絞られていません: {tracker.current_fn}"
    assert ki < 5e-8, f"積分ゲインが十分に抑制されていません: {ki}"

    # 2. 強電界・クリーン環境シミュレーション (高品質・残差極小)
    for _ in range(25):
        kp, ki, alpha = tracker.update_gains(lock_quality=0.85, pilot_rms=0.15, ef_state=0.005)

    print(f"[*] 強電界適応後: fn={tracker.current_fn:.2f} Hz (期待値 > 20 Hz), CNR={tracker.current_cnr_db:.1f} dB, Ki={ki:.4e}")
    assert tracker.current_fn > 20.0, f"強電界で帯域幅が十分に拡大されていません: {tracker.current_fn}"
    assert ki > 1.8e-7, f"積分ゲインが十分に俊敏になっていません: {ki}"

    # 3. パラメータの連続性 (急峻なステップ変化によるクリック音の不在)
    # 1フレームでいきなり弱電界になった場合の1回更新での変化率
    old_fn = tracker.current_fn
    tracker.update_gains(lock_quality=0.05, pilot_rms=0.001, ef_state=0.5)
    delta_fn = abs(tracker.current_fn - old_fn)
    assert delta_fn < 5.0, f"パラメータが1フレームで急激に跳躍しました (クリック音リスク): delta={delta_fn}"
    print("[OK] NASA DSN方式 自律適応カルマン・パイロット搬送波追従テスト成功")


def test_holographic_audio_enhancer():
    print("\n===== test_holographic_audio_enhancer =====")
    fs = 48000.0
    enhancer = HolographicAudioEnhancer(sample_rate=fs, air_gain=0.08)
    n = 2048
    t = np.arange(n) / fs

    # 10kHzの倍音を持つ音楽信号 (15kHz以上は帯域制限で完全ゼロ)
    sig = (0.4 * np.sin(2 * np.pi * 1000.0 * t) + 0.2 * np.sin(2 * np.pi * 10000.0 * t)).astype(np.float32)

    # 1. 音楽再生時 (ハイレゾ外挿発動)
    out_music = enhancer.process(sig, ch="_l", speech_prob=0.0, s_meter_dbfs=-20.0)

    # 2. トーク時 (コグニティブ保護バイパス)
    enhancer._histories["_l"] = np.zeros(len(enhancer.fir_hp15k) - 1, dtype=np.float32)
    out_talk = enhancer.process(sig, ch="_l", speech_prob=0.9, s_meter_dbfs=-20.0)

    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    mask_air = freqs >= 15000.0

    spec_in = np.abs(np.fft.rfft(sig))
    spec_music = np.abs(np.fft.rfft(out_music))
    spec_talk = np.abs(np.fft.rfft(out_talk))

    air_in = float(np.max(spec_in[mask_air]))
    air_music = float(np.max(spec_music[mask_air]))
    air_talk = float(np.max(spec_talk[mask_air]))

    print(f"[*] 15kHz以上エアバンド最大振幅: 入力={air_in:.4f} -> 音楽外挿={air_music:.4f} -> トーク保護={air_talk:.4f}")
    assert air_music > air_in * 10.0, f"15kHz以上のエアバンド倍音が十分に外挿されていません: {air_music}"
    assert np.allclose(out_talk, sig, atol=1e-5), "トーク時に完全バイパスされていません"

    # 3. 弱電界ノイズ保護 (弱電界で外挿抑制)
    enhancer._histories["_l"] = np.zeros(len(enhancer.fir_hp15k) - 1, dtype=np.float32)
    out_weak = enhancer.process(sig, ch="_l", speech_prob=0.0, s_meter_dbfs=-50.0)
    assert np.allclose(out_weak, sig, atol=1e-5), "弱電界ノイズ環境で保護バイパスされていません"
    print("[OK] ホログラフィック・ハイレゾ倍音外挿テスト成功")


def test_riemannian_topological_demodulator():
    print("\n===== test_riemannian_topological_demodulator =====")
    fs = 288000.0
    duration = 0.05
    n = int(fs * duration)
    t = np.arange(n) / fs

    dev_hz = 50000.0
    mod_audio = np.sin(2.0 * np.pi * 1000.0 * t).astype(np.float32)
    phase = np.cumsum(2.0 * np.pi * dev_hz * mod_audio / fs)
    clean_iq = np.exp(1j * phase).astype(np.complex64)

    demodulator = RiemannianTopologicalDemodulator(sample_rate=fs, dev_limit_hz=75000.0)

    # 1. クリーン信号でのビット整合性検証
    demod_clean = demodulator.demodulate(clean_iq)
    ref_norm = (2.0 * np.pi * dev_hz / fs) * mod_audio
    corr_clean = float(np.corrcoef(demod_clean[100:], ref_norm[100:])[0, 1])
    print(f"[*] クリーンFM波のリーマン測地線復調相関度: {corr_clean:.6f} (期待値 > 0.999)")
    assert corr_clean > 0.999, f"クリーン波形の追従精度が不十分です: {corr_clean}"

    # 2. 分岐切断 (±π) 横断クリックの除去検証
    # 真の偏移 (±50kHz → ±1.09rad) はπに届かないため、±πまたぎは偽スリップ。
    # サンプルを切断上に押し付けて15発の偽スリップを発生させる。
    rng = np.random.default_rng(0)
    iq2 = clean_iq + 0.25 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    ks = rng.integers(500, n - 500, size=30)
    for k in ks:
        iq2[k] = abs(iq2[k]) * np.exp(1j * (np.pi - 0.02))
    naive2 = np.angle(np.concatenate(([iq2[0]], iq2))[1:]
                      * np.conj(np.concatenate(([iq2[0]], iq2))[:-1]))
    naive_clicks = int(np.sum(np.abs(naive2) > 2.5))
    demodulator.reset()
    riemann_demod = demodulator.demodulate(iq2)
    riemann_clicks = int(np.sum(np.abs(riemann_demod) > 2.5))

    corr_naive = float(np.corrcoef(naive2[200:], ref_norm[200:])[0, 1])
    corr_riemann = float(np.corrcoef(riemann_demod[200:], ref_norm[200:])[0, 1])

    print(f"[*] 分岐切断横断スパイク数: ナイーブ={naive_clicks}個 -> アンラップ={riemann_clicks}個")
    print(f"[*] ノイズ下波形相関度: ナイーブ={corr_naive:.4f} -> アンラップ={corr_riemann:.4f}")

    assert riemann_clicks < naive_clicks, "分岐切断クリックが除去されていません"
    assert corr_riemann >= corr_naive - 0.01, "アンラップで波形が劣化しています"
    print("[OK] リーマン多様体トポロジカル測地線復調テスト成功")


def test_super_spatial_bss_stereo_separator():
    print("===== test_super_spatial_bss_stereo_separator =====")
    fs = 48000.0
    duration = 0.1
    n = int(fs * duration)
    t = np.arange(n) / fs

    # 1. 理想ステレオ信号 (L: 1kHz + 4kHz, R: 1kHz - 4kHz)
    l_true = (0.5 * np.sin(2 * np.pi * 1000.0 * t) + 0.3 * np.sin(2 * np.pi * 4000.0 * t)).astype(np.float32)
    r_true = (0.5 * np.sin(2 * np.pi * 1000.0 * t) - 0.2 * np.sin(2 * np.pi * 4000.0 * t)).astype(np.float32)

    bss = SuperSpatialBssStereoSeparator(sample_rate=fs, crossover_hz=7500.0)

    # クリーン信号のビット通過性 (遅延整合のため delay=24 サンプル≒0.5ms遅延を補正して比較)
    l_clean, r_clean = bss.process(l_true, r_true, stereo_blend=1.0)
    dly = int(getattr(bss, "delay", 0))
    if dly > 0 and len(l_clean) > dly:
        corr_l_clean = float(np.corrcoef(l_clean[dly:], l_true[:-dly] if dly else l_true)[0, 1])
        corr_r_clean = float(np.corrcoef(r_clean[dly:], r_true[:-dly] if dly else r_true)[0, 1])
    else:
        corr_l_clean = float(np.corrcoef(l_clean, l_true)[0, 1])
        corr_r_clean = float(np.corrcoef(r_clean, r_true)[0, 1])
    print(f"[*] クリーンステレオ波形の忠実度相関: L={corr_l_clean:.6f}, R={corr_r_clean:.6f} (期待値 > 0.999)")
    assert corr_l_clean > 0.999 and corr_r_clean > 0.999, "クリーン信号が変形しています"

    # 2. 38kHz副搬送波FM逆相三角ノイズの重畳
    np.random.seed(42)
    white = np.random.normal(0, 0.03, n).astype(np.float32)
    tri_noise = (np.diff(white, prepend=white[0]) * 3.5).astype(np.float32)

    l_noisy = l_true + tri_noise
    r_noisy = r_true - tri_noise

    bss.reset()
    l_out, r_out = bss.process(l_noisy, r_noisy, stereo_blend=1.0)

    # 遅延補正して評価 (出力は delay だけ遅れるため)
    dly2 = int(getattr(bss, "delay", 0))
    if dly2 > 0 and len(l_out) > dly2:
        l_true_a = l_true[:-dly2]
        l_noisy_a = l_noisy[dly2:]
        l_out_a = l_out[dly2:]
    else:
        l_true_a, l_noisy_a, l_out_a = l_true, l_noisy, l_out
    noise_before = float(np.mean((l_noisy_a - l_true_a) ** 2))
    noise_after = float(np.mean((l_out_a - l_true_a) ** 2))
    snr_gain = 10.0 * np.log10(noise_before / (noise_after + 1e-12))
    corr_noisy = float(np.corrcoef(l_noisy_a, l_true_a)[0, 1])
    corr_out = float(np.corrcoef(l_out_a, l_true_a)[0, 1])

    print(f"[*] FM三角逆相ヒスノイズ低減比: +{snr_gain:.2f} dB (期待値 > +3.0 dB)")
    print(f"[*] ノイズ下波形相関度: 前={corr_noisy:.4f} -> BSS後={corr_out:.4f} (改善: +{(corr_out - corr_noisy):.4f})")

    assert snr_gain > 3.0, f"ノイズ低減効果が不十分です: {snr_gain:.2f} dB"
    assert corr_out > corr_noisy, "BSSによる波形相関度改善が見られません"

    print("[OK] 超空間独立成分ステレオ復調テスト成功")


def test_rmt_hankel_denoiser():
    print("\n===== test_rmt_hankel_denoiser =====")
    fs = 48000.0
    duration = 0.05
    n = int(fs * duration)
    t = np.arange(n) / fs

    # 1. クリーンな音声・音楽信号 (1kHz + 2.5kHz + 4kHz)
    clean = (0.4 * np.sin(2 * np.pi * 1000.0 * t) +
             0.2 * np.sin(2 * np.pi * 2500.0 * t) +
             0.15 * np.cos(2 * np.pi * 4000.0 * t)).astype(np.float32)

    rmt = RmtHankelDenoiser(sample_rate=fs, embed_dim=24)

    # クリーン通過テスト (S-Meter > -28dBFS 強電界で完全バイパス)
    out_clean = rmt.process(clean, ch="", s_meter_dbfs=-15.0)
    max_clean_diff = float(np.max(np.abs(out_clean - clean)))
    print(f"[*] 強電界クリーン波形の通過差分: {max_clean_diff:.2e} (完全ビット一致)")
    assert max_clean_diff == 0.0, "クリーン信号が変形しています"

    # 2. 弱電界ノイズ (FM三角ノイズ + ガウス雑音)
    np.random.seed(42)
    white = np.random.normal(0, 0.08, n).astype(np.float32)
    noisy = clean + white

    rmt.reset()
    # 弱電界 (S-Meter -38dBFS)
    out_denoised = rmt.process(noisy, ch="", s_meter_dbfs=-38.0)

    # 群遅延 (half_taps) を考慮した定常区間での評価
    half = rmt.half_taps
    eval_slice = slice(half * 2, -half)
    ref = np.concatenate((np.zeros(half, dtype=np.float32), clean))[:n]

    err_before = float(np.mean((noisy[eval_slice] - clean[eval_slice]) ** 2))
    err_after = float(np.mean((out_denoised[eval_slice] - ref[eval_slice]) ** 2))
    snr_gain = 10.0 * np.log10(err_before / (err_after + 1e-12))
    corr_before = float(np.corrcoef(noisy[eval_slice], clean[eval_slice])[0, 1])
    corr_after = float(np.corrcoef(out_denoised[eval_slice], ref[eval_slice])[0, 1])

    print(f"[*] RMTマルチェンコ・パスツール特異値切除 SNR改善度: +{snr_gain:.2f} dB (期待値 > +4.5 dB)")
    print(f"[*] ノイズ下波形相関度: 前={corr_before:.4f} -> RMT後={corr_after:.4f} (改善: +{(corr_after - corr_before):.4f})")

    assert snr_gain > 4.5, f"RMTによるノイズ低減効果が不十分です: {snr_gain:.2f} dB"
    assert corr_after > corr_before, "RMTによる相関度改善が見られません"

    # 3. 分割処理 (ストリーミング) と一括処理の完全一致性検証 (境界誤差ゼロ)
    rmt_split = RmtHankelDenoiser(sample_rate=fs, embed_dim=24)
    rmt_split._cached_kernel = rmt._cached_kernel.copy()
    rmt_split._cached_sigma2 = rmt._cached_sigma2
    rmt_split._frame_count = 1
    c1 = noisy[:n // 2]
    c2 = noisy[n // 2:]
    out_split = np.concatenate([
        rmt_split.process(c1, ch="", s_meter_dbfs=-38.0),
        rmt_split.process(c2, ch="", s_meter_dbfs=-38.0)
    ])
    rmt_bulk = RmtHankelDenoiser(sample_rate=fs, embed_dim=24)
    rmt_bulk._cached_kernel = rmt._cached_kernel.copy()
    rmt_bulk._cached_sigma2 = rmt._cached_sigma2
    rmt_bulk._frame_count = 1
    out_bulk = rmt_bulk.process(noisy, ch="", s_meter_dbfs=-38.0)

    diff_split = float(np.max(np.abs(out_split - out_bulk)))
    print(f"[*] 分割処理 vs 一括処理の最大差分: {diff_split:.2e} (完全一致)")
    assert diff_split < 1e-6, f"境界不連続が生じています: {diff_split}"

    print("[OK] ランダム行列特異値切除ノイズクリーナーテスト成功")


if __name__ == "__main__":
    test_iq_imbalance_correction()
    test_ultrasonic_squelch()
    test_cognitive_speech_music_tracker()
    test_quadrature_mpx_canceller()
    test_deep_space_ekf_demodulator()
    test_kalman_pilot_tracker()
    test_holographic_audio_enhancer()
    test_riemannian_topological_demodulator()
    test_super_spatial_bss_stereo_separator()
    test_rmt_hankel_denoiser()
    print("\nALL ADAPTIVE DSP TESTS PASSED!")


