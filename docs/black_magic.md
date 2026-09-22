# 黒魔法三点＋一点 (弱電界検出補助)

## ユーザー向け: 何が効いてるか・いつ効くか・いつ止まるか

```
弱電界 (SNR<12dB)          通常 (12-30dB)          強信号 (SNR>30dB)
[検出系: 動作中]           [検出系: 待機]           [検出系: 助言のみ]
 pilot監視・SR試行         pilot監視のみ           すべて休止
[処理系: 弱く動作]         [処理系: 最小]           [処理系: 停止]
 RMT≦0.65・ノッチ          RMT≦0.2                 素通し (透明)
```

- **効く相手**: パイロットのふらつき (ステレオ確定の迷い)・定常ヒス・
  電源ハム。フェージングで信号自体が消える瞬間には何もできない。
- **止まる条件**: 強信号・CPU逼迫・音質劣化検出・NaN/例外。
  止まるときは既存経路へ無音で戻る (クリックなしが設計)。
- **モード別**: WFM=全部入り、NFM/AM=ノッチ＋RMTのみ、
  SSB=RMTのみ、CW=全停止 (`profile_for`の表どおり)。
- 状態は`BlackMagicController.describe()`で取得できる
  (GUIのチューニング表示は将来対応。DSP側APIは完成)。

> 注意: 確率共鳴は真のSNRを改善しない。検出器の判定補助専用である。
> RMTの改善値 (+2.2dB) は合成ヒス条件の実測であり、実機効果は未検証。

RTL-SDRの弱電界受信時における (1) 局検出・ロック性能、(2) ステレオ復調の安定性、
(3) 復調後音声のノイズ感、(4) 電源ハムを改善するための機能群。
いずれも独立モジュール＋feature flagで、既定では既存経路のみが動作する
(全OFFでビット同一をテストで検証)。

## モジュール

- `cyclostationary_detector.py` — A: 19kHzパイロットのGoertzel検出＋位相連続性評価。
  PLLを置き換えず、blend上昇レート制限の助言のみ。
- `rmt_denoiser.py` — B: 既存`RmtHankelDenoiser` (adaptive_audio.py) の安全層。
  コア数学は不変。強度ブレンド・Mid/Side・CPU予算・監視つき。
- `stochastic_resonance.py` — C: 検出専用の確率共鳴アシスタント。音声入出力なし。
- `black_magic.py` — 統合マネージャ (`BlackMagicController`)。
  パラメータ計算＋`profile_for` (モード別自動プロファイル)＋
  `describe` (状態表示API) のみ。
- `adaptive_notch.py` — Phase 2-1: 適応ハムノッチ (四点目)。
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

## Phase 1パラメータ最適化 (合成掃引＋録音検証の結果)

- 1-1 RMT強度マップ: 掃引中に**遅延不整合バグ**を発見・修正
  (コア出力の15サンプルlookaheadとブレンド相手の位相ずれで番組-2.4dB)。
  ラッパー側で入力遅延整合＋評価も整合化。修正後は合成で+10〜+30dB
  (定常トーンのため過大、実番組は+2dB級)。マップは(20,0.35)→(20,0.40)
  の微調整のみでほぼ現状肯定。高域損失<=1dBのパレートで選定。
- 1-2 SR: sigma 0.05〜2.0×trials 2〜8を掃引したが有意な山なし
  (subthresholdで+2/20程度、誤差範囲)。既定値維持。
  fa-worse-rejectは良性条件で発動0・悪条件で発動を確認 (効きすぎなし)。
- 1-3 cyclo: on/off/dwellを掃引。平滑narrow化は効果なし＋立上り2ブロック
  遅延のため不採用 (コードに理由を記録して revert)。既定6/3/dwell3を
  録音で検証: 強局4ブロック確定・フラッター追従・ノイズ不検出 (conf 0.07)。
  既定値維持。

## Phase 2-1 適応ハムノッチ (四点目)

- `adaptive_notch.py`: 50/60Hz＋高調波 (〜300Hz) の適応キャンセル。
  静的ノッチではなく最小二乗フィット波形を差し引く (穴あけなし)。
- 誤認防止: 基底確定＋2本以上の線確定でのみ動作 (音楽単一トーンは素通し)、
  除去量は信号RMSの30%上限、NaN即時バイパス、選局リセット。
- 基底は50/60Hz自動選択 (EMA＋4tick投票。単発スパイク・連続位相wobble対策済み)。
- ステレオはMid推定→L/R同量差引 (同相仮定)。
- 実測 (番組4トーン＋ハム4線＋ヒス): 50Hz -21dB、100Hz -15dB、
  番組変動±0.00dB、5.3ms/ブロック。200Hz以上 (-32dB以下) は残留しうる。
- 配線: RMT前段 (ハム除去後の信号をNRへ)。既定OFF。
  `black_magic.adaptive_notch.enabled`＋`dsp.bm_notch_enabled`。
  ハーネスは`--bm notch`。

## CMAゲート締め直し＋2-2評価 ( synth測定に基づく)

- 症状: 77.5MHz実測で`multipath_amount=1.0`飽和→CMAが常時作動。
  ゲートのS-meter ORバイパスがlock≈0の深フェードでも作動させていた。
- 測定: 静的/変動/弱エコーではCMAが+0〜+12dB改善。深フェード
  (lock≈0.15) ではblend 1.00→0.73に悪化。長遅延強エコー
  (d=40/g=1.2) では0.03でlock 0.18まで悪化。
- 対策: ゲートをlock>0.2必須＋S-meterは-60dBFS vetoのみに締め直し
  (使用側の条件も一致)。μは0.03→0.02
  (長エコーlock 0.18→0.39、短エコー・flutter同等以上)。
- 2-2 (第二等化器) は**見送り**: 既存CMAと正面競合するため。
  残る長エコー問題はCMA自体の捕捉問題であり、実録音データが揃ってから
  CMA側で対処する (別途CMA改善として扱う)。

## Phase 0実証基盤 (録音IQハーネス＋ゴールデンデータ)

- `python tools/ab_benchmark.py <iq...> [--bm cyclo|rmt|sr|all] [--out json]`:
  同一録音をOFF/ONで2回通す。対応形式は`.npy`(uint8 raw)、`.cs16`、
  `.wav`。初回2ブロック除外でp50/p95/p99・57.3ms超過率を出す。
- `testdata/`: `strong_946_3s.npy` (強局3秒)、`noise_762_3s.npy`
  (局間3秒)、`flutter_775_seq.npy`＋`meas_775_snap_*.npy`
  (77.5MHzフラッター)、`baseline_cyclo.json` (基準値)。
- ベースライン (cyclo): 強局 present 0.92/conf 0.83・ON==OFF完全一致、
  局間 present 0.00/conf 0.14、フラッター present 0.33/conf 0.46。
  遅延は強局p99 34ms・局間p99 57.5ms (超過率0.02、実機ジッタの範囲)。
  合格基準「p99<57.3ms厳守」は局間で僅かに超過するため要調整。
- ネガティブテスト (0-3) は単体テストでカバー済みのため再実装なし。
- BER/MERはWFM音声に既知系列がないため対象外。将来的にRDS CRC率を検討。

## 実機で追加確認が必要な項目
1. 弱局 (例: 77.5MHz) で`bm_cyclo_enabled=True`時のステレオ確定までの秒数と
   音の途切れ (blend軌跡を記録し、従来と比べ遅くなっていないか)。
2. 強局 (例: 94.6MHz) でON/OFFの聴感差がないこと (透明性の確認)。
3. 局間ノイズで`pilot_present`が立たないこと (誤検出の実地確認)。
4. RMT有効時の番組 (音楽・トーク) での人工ノイズ・高域劣化の有無。
   `bm_rmt_cap`を0.3→0.65と段階的に上げる。
5. 長時間受信でのCPU使用率 (`_bm_params`の`processing_ms`と実測の比較)。
6. `config.json`の`black_magic`節を壊した状態でも起動すること (検証済み: 既定復帰)。
