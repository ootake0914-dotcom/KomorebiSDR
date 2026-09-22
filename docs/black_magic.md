# 黒魔法三点セット (弱電界検出補助)

RTL-SDRの弱電界受信時における (1) 局検出・ロック性能、(2) ステレオ復調の安定性、
(3) 復調後音声のノイズ感を改善するための3機能。いずれも独立モジュール＋
feature flagで、既定では既存経路のみが動作する (全OFFでビット同一をテストで検証)。

> 注意: 確率共鳴は真のSNRを改善しない。検出器の判定補助専用である。
> RMTの改善値 (+2.2dB) は合成ヒス条件の実測であり、実機効果は未検証。

## モジュール

- `cyclostationary_detector.py` — A: 19kHzパイロットのGoertzel検出＋位相連続性評価。
  PLLを置き換えず、blend上昇レート制限の助言のみ。
- `rmt_denoiser.py` — B: 既存`RmtHankelDenoiser` (adaptive_audio.py) の安全層。
  コア数学は不変。強度ブレンド・Mid/Side・CPU予算・監視つき。
- `stochastic_resonance.py` — C: 検出専用の確率共鳴アシスタント。音声入出力なし。
- `black_magic.py` — 統合マネージャ (`BlackMagicController`)。パラメータ計算のみ。
- `benchmark_black_magic.py` — 合成ベンチマーク (7条件×OFF/ON)。

## ON/OFF方法

設定ファイル (`config.py`の`black_magic`節。既定は全て安全側):

```json
black_magic: {
  enabled: false,                 // master (falseで全機能停止・既存経路のみ)
  cyclostationary: { enabled: true, ... },   // master ON時のみ有効
  rmt_denoiser: { enabled: false, ... },     // 既定無効
  stochastic_resonance: { enabled: false, ... }  // 既定無効・検出専用
}
```

`main.py`への設定→dsp属性の自動反映は**未配線 (見送り)**。有効化は暫定的に
dsp属性で直接行う (いずれも既定False):

```python
dsp.black_magic_enabled = True
dsp.bm_cyclo_enabled = True   # A (PLL助言のみ)
dsp.bm_rmt_enabled = True     # B (既存rmt_denoiser側と同時有効化は禁止＝二重処理)
dsp.bm_sr_enabled = True      # C (副経路のみ。スケルチ自動反映なし)
```

## 処理経路の変更点 (dsp.py のみ・全てflag guarded)

- `_update_stereo_pilot`: confidence低＋ネイティブlock低のときのみblend上昇を
  0.25→0.05/ブロックに制限。lock>0.5では従来通り (強信号の確定を遅らせない)。
- `demodulate_wfm` (ステレオ結合点・モノラル終端): BをMid/Side (Mid通常・
  Side×0.4) で処理。L/R独立処理は分離度を-3〜-4dB落とすため採用しない。
- `demodulate_wfm` (スケルチ直後): Cを副経路で評価し`bm_sr_confidence`として公開。
  **スケルチ判定への自動反映は既存動作との競合回避のため見送り。**
- `process`: コントローラで安全パラメータのみ計算 (音声不変)。CPU使用率は
  ブロック処理時間から実測。選局時 (`reset`) は全状態クリア。
- `adaptive_audio.py`: RMTコアに診断公開値 (`last_retained_rank`等) を追加した
  のみ。処理内容は不変。

## ベンチマーク結果 (`python benchmark_black_magic.py`)

12ブロック×OFF/ON。pilot検出率0.67は立上り (dwell 3＋EMA) 込み、定常は1.0。

| 条件 | pilot検出 | chatter | 音声SNRΔ | RMS/高域 | 分離度 | CPU |
|---|---|---|---|---|---|---|
| 1 ステレオ+白 SNR10 | 0.67/conf0.66 | 1/1 | ±0.0 | ±0.00/-0.64 | 17.5/17.5 | +17% |
| 1b 弱パイロット RF-SNR3 | 0.67/conf0.66 | 1/1 | ±0.0 | ±0.00/±0.00 | 11.6/11.6 | +17% |
| 2 +インパルス | 0.67 | 1/1 | ±0.0 | — | 20.9/20.9 | +18% |
| 3 搬送波+3kHz | 0.67 | 1/1 | ±0.0 | — | 20.2/20.2 | +12% |
| 4 マルチパス | 0.67 | 1/1 | ±0.0 | — | 6.7/6.7 | +18% |
| 5 ノイズのみ | 0.00/conf0.16 | 0/0 | -0.9※ | -0.13/+0.15 | — | +14% |
| 6 隣接+20dB | 0.25/conf0.31 | 1/1 | ±0.0 | ±0.00 | 0.9/0.9 | +11% |
| 7 音声+ヒス (RMT直接) | — | — | **+2.2** | -3.21/RMS | rank2 | 0.3ms/BLK |

※5のSNRΔはノイズ同士の比較で無意味。RMS-0.13dBで増幅なしを確認。
1〜4・6のON==OFFは「強信号・トーン信号で過剰処理しない」ことの証拠
(RMTは強信号バイパス＋コアsigma2ゲートで休止)。7でRMT単体の効果を確認。

## 既知の問題

1. RMTコアのノイズ分散推定がトーン性信号で負値になりバイパスする
   (音楽保護方向の誤作動だが、ヒス性噪音では動作する)。コア再設計は見送り。
2. SRの`clip`は常にFalse (dspにADCクリップ旗がない)。過大入力時は使わないこと。
3. SRの`noise_floor`/`snr`は暫定プロキシ (超音波ノイズ・Sメータ換算)。
4. コントローラのCPU値はdsp内実測のみ。外部負荷は見ない。
5. pilot検出率の立上り3〜4ブロックはdwell＋EMAの仕様 (最小継続時間のため)。
6. [対策済] attack制限の非対称ラチェット: flutter下でblendが下げ方向に
   バイアスされた (77.5MHz実測)。`_bm_attack_limit`でlock分散による
   flutter検出 (`bm_flutter_std=0.15`) を入れ、flutter中は非介入に戻す。

## 実機で追加確認が必要な項目

1. 弱局 (例: 77.5MHz) で`bm_cyclo_enabled=True`時のステレオ確定までの秒数と
   音の途切れ (blend軌跡を記録し、従来と比べ遅くなっていないか)。
2. 強局 (例: 94.6MHz) でON/OFFの聴感差がないこと (透明性の確認)。
3. 局間ノイズで`pilot_present`が立たないこと (誤検出の実地確認)。
4. RMT有効時の番組 (音楽・トーク) での人工ノイズ・高域劣化の有無。
   `bm_rmt_cap`を0.3→0.65と段階的に上げる。
5. 長時間受信でのCPU使用率 (`_bm_params`の`processing_ms`と実測の比較)。
6. `config.json`の`black_magic`節を壊した状態でも起動すること (検証済み: 既定復帰)。
