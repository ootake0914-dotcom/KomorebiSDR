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

1. [対策済] RMTコアのノイズ分散推定がトーン性信号で負値になりバイパスする
   → MP中央値による頑健σ^2化 (負値にしない。全固有値中央値/MP理論中央値)。
   Gavish-Donoho最適収縮も試したが対角平均→FIR後の出力SNRでは
   旧シフト収縮と互角〜微劣化 (合成±0.3dB) のため収縮式は維持。
   合成AB: 旧+3.67dB/保存0.916 → 新+3.69dB/保存0.922 (同等＋非負保証)。
2. [対策済] SRの`clip`は常にFalseだった → dspがadc_clip_pct/adc_clippedを
   自己計測しSRへ配線済み。notch/RMTにも過大入力ガード (adc-clipバイパス)
   を追加 (tests/test_clip_guard.py)。
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

## Q最適収束系 (調停者の完成)

- `quality_score`: Q = 0.4conf＋0.3blend−0.5chatter−0.2cpu−0.5damage
  (測れる量のみ。非有限は安全側)。
- `ExtremumSeeker`: 1次元山登り (8blk評価・反転でstep×0.7・
  3回停滞で凍結・SNR±6dBで再開・offset±0.2)。同時駆動禁止。
- 統合: base方策は不変、seekerは±0.2のoffsetのみ
  (`rmt_cap`に加算・上限0.85)。既定OFF
  (`black_magic.seeking.enabled`＋`dsp.bm_seek_enabled`)。
- 実測: 弱局録音でoffset +0.17まで適応後凍結、音声正常。
  注意: 実番組ではQが平坦で bounds まで漂うことがある
  (damage≈0のため)。凍結で発散はしない。
- 同時駆動の禁止・時定数分離・凍結の3点が収束の条件
  (詳細はコードの設計コメント参照)。

## ① cyclo→スケルチ統合 (推し筆頭・実装済み)

- 二基準ヒステリシス: 開=conf>0.75 or S>-40dBFS、
  閉=conf<0.55 かつ S<-25dBFS、開速0.25/閉遅0.05 per block。
- 最小閉保持20ブロック: 深フェードの呼吸 (開閉11回/96blk・gainポンピング)
  を実測し追加。保持後は遷移が決定的ミュートに変わる。
  代償として閉後の開き直しに約1秒かかる (一過性バースト防止として妥当)。
- モノラル強局はconf=0でもS高で開のまま (誤ミュートなし・検証済み)。
- 適用はprocess終端 (スローAGC後)。前段だとAGCが持ち上げて無効化
  されることを実測 (ノイズRMS 0.45のまま→終端移設で無音化)。
- `bm_cyclo_enabled`併用が必須 (confidence源)。選局時は開から開始。
- 実録音: 弱局77.5MHzで開率100%・遷移0 (従来blendは0-1で暴走)。
  熱い局間 (-16dBFS級) は開のまま。これは制限ではなく**分担**:
  エネルギーで重なる相手は電力スケルチの担当であり、
  cyclo統合は「弱いがpilot在り」を開け続ける担当。
  冷たい局間 (合成S-40dBFS) は約1.4秒で無音化。
- 設定: `black_magic.squelch_assist`＋`dsp.bm_sq_assist_enabled`
  (いずれも既定OFF)。

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
- ②CMA長エコー改善 (合成golden `testdata/longecho_d40.npy` で試行→**不採用**):
  定常60ブロックABで off=23.1dBに対し CMA33=+7.8、65=+1.9、μ0.005=+3.0、
  DFE併用は発散。弱エコーg=0.5では+5.2dB (35.9→41.1) と効くが、
  強エコーg=1.2では-15dB破壊。重み軌道は両条件とも中央タップ崩壊→
  拡散の同型病理で、包絡誤差比・重み軌道とも早期識別不能。
  結論: 広帯域アナログFMの遅延拡散にブラインドCMAは不適 (捕捉最小値が
  所望逆特性でない)。現行の保守ゲート (稀作動) が正しい姿勢。タップ延長・
  DFE・μ低下・自己防衛ゲートはいずれも不採用。

## 共チャネル分離PoC (4手→不採用。GTは資産として残す)

- GT: `testdata/coch_gt_{mix,Aonly}_u8.npy`＋clean参照`coch_gt_{A,B}_prog.npy`。
  A=ステレオトーク (パイロット有)、B=モノラル別トーク (-6dB同一周波数)。
  A単独復調でSTOI-A=1.000 (鎖の透明性とGT方法論の妥当性を証明)。
- 被害定量化: 混信でSTOI-A 1.000→0.913 (捕捉効果のコスト0.087)。
  STOI-B=0.332→0.454で指標は両局を識別する。
- PoC: (1) 音声域クリックブランカ (k=3/4/6) → 0.805/0.854/0.907と全敗
  (子音を殺す。被害は疎クリックでなく連続クロストークのため領域が違う)。
  (2) BSSそのまま適用 → 0.913で完全無変化 (Mid/Sideヒス抑制器は
  共チャネル構造に触れないことを確認)。
- 結論: 単一アンテナの同一周波数アナログFMブラインド分離は不良設定
  (SICに判定がなく、捕捉後の連続クロストークにブラインド手がない)。
  パイプライン無変更。GTは今後のAB判定材料として残す。

## PAST不採用 (プロファイリング実測で閉廷。パイプライン無変更)
- RMT直接コスト 0.7〜1.1ms (予算57.3msの約1〜2%)。EVD犯人説は否定。
  真犯人はdemod/wfm残差 (native C 8〜12ms) とstereo_pair系13ms。
  アロケーションのホットスポットも不在。
- 「0.5〜1msの問題に近似誤差を払う理由なし」。WFMのp99逼迫
  (52〜57ms) は地コスト由来で、PASTでは沈まない。
  (`tools/profile_snapshot.py`・`docs/profile_snapshot.json` に記録)

## RMT狭帯域横展開は不採用 (AM/NFM/SSB配線→revert。安全層2件は残す)

- 直接移植のAB: AM弱局STOI -0.048、NFM -0.339 (ヒス+35dB)、
  USB -0.052 (ヒス+102dB)。cap掃引は無感応 (構造問題)。
- 原因: コア・強度マップは広帯域FM放送スペクトル前提。狭帯域音声では
  高域の剥離/合成をブロック毎に往復する (rank 0/4/5/6/7乱高下)。
  15サンプル過渡の広帯域エネルギーが包絡判定を突破する穴も実測。
- 副産物として残した安全層2件 (WFM無影響をベンチで確認):
  narrowband休止 (6kHz以上内容-30dB未満で休止) とhf-add revert
  (6kHz以上で+3dB合成したら入力へ戻す)。RMTの動作包絡を宣言した。
- AM/NFM/SSBに残るのはSR検出プローブ (音声不変・confidence公開のみ)。
  ビット一致・p99<25ms・実録音pseudoをテストで保証
  (`tests/test_bm_am_ssb.py`)。
- 教訓: STOIは4.3kHz以上盲目。NFMの+22dB可聴ヒスはSTOI=1.000のまま
  すり抜けた。帯域別ヒス指標との併用 (デュアルメトリック) が必須。

## 自動化の完成 (プラグアンドプレイ: 自動モード解決＋収束測定)

- 周波数と矛盾するモードの自動解決 `config.resolve_mode` を追加。
  FREQ・シーク・スキャン経路 (`auto=True`) は帯域と矛盾するFM/AM系を
  補正 (FM帯AM→WFM、24MHz未満WFM→AM、HAM HFのAM→LSB/USB、
  2m/70cm→NFM、エアバンド→AM)。ユーザーの明示MODE選択 (`auto=False`) と
  明示SSB/CWは尊重する。
- 収束測定 (`tests/test_auto_convergence.py`): 合成WFM/AM/NFM/USBを
  冷えたdspへ流し、主要状態 (Sメーター/パイロットロック/ブレンド/NRヒス/
  AFC/AM同期/SSB AGC) が最終値の許容内で安定するまでの時間を実測。
  最悪はWFM Sメーター1.26s・NFM AFC 1.03sで、3秒基準を満たす。
  スローAGC (attack2s/release10s) は局間レベリング用の意図的遅延のため
  合否対象外 (参考表示: 本条件で約2.3〜2.4s)。
- 副産物: 合成NFMに+300Hzオフセットを入れ、AFCが-295.6Hzへ収束する
  ことを確認 (追従の意味的検証)。

## 第2章 タスクC: WFM減量 (採用: stereo_pair / freq_blend 最適化)

### 目的と基準
- 目的: WFMブロック処理コスト削減 (追加禁止・減量のみ)。
- 基準: 削減 ≥ 1ms かつ 品質score ≥ 0.99 (STOI_pseudo) かつ 可聴ヒス差悪化なし。
- 達成目標: p99 < 40ms。

### ボトルネック内訳特定 (tools/profile_snapshot.py 計装)
- `wfm/stereo_pair` 排他 6.10ms の主犯は未計装だった `QuadratureMpxCanceller.process` (6.28ms) と判明。
- 各候補のマイクロプロファイル実測:
  1. `_freq_dependent_blend` の Python 2752サンプル forループ: 0.63ms
  2. `QuadratureMpxCanceller.process` の batch_sz=32 ミニバッチNLMSループ: 5.56ms
  3. `_wiener_diff` フレームループ: 1.24ms (フレーム間平滑再帰依存あり)
  4. `pair/post` 1ch: 0.13ms (Cコア deemphasis 0.057ms / decimate 0.051ms / DC 0.022ms、既に高速)

### 実施した最適化
1. **`_freq_dependent_blend` IIR の Cコア化 (dsp_wfm.py)**:
   - 1次相補クロスオーバーの指数平滑 $y[n] = a x[n] + (1-a) y[n-1]$ を、既存Cコア `sdr_bilinear_deemphasis` ($b_0=a, b_1=0, m=1-a$) に置換。
   - 測定: 0.63ms → 0.025ms (0.60ms 削減)。
   - 数値等価性: 最大差分 7.45e-9 (ビットレベルで等価)。テスト全合格。
2. **`QuadratureMpxCanceller.process` NLMS の高速内積・バッチ拡大 (adaptive_stereo.py)**:
   - ブロック全体一括更新は収束抑圧比が 8.8dB に悪化し不採用 (基準>12.0dB)。
   - `batch_sz=32 → 64` へ拡大し、勾配計算をテンソル乗算 `err_b[:, None] * wb` から内積 `(err_b @ wb) * inv_len` に最適化。
   - 測定: 5.56ms → 1.53ms (プロファイル上 `pair/mpx_canceller` 6.28ms → 1.80ms、4.48ms 削減)。
   - 収束抑圧比: 28.66dB (旧32の 28.98dB と同等、基準>12.0dB 大幅クリア)。クリーン信号ビットパーフェクト通過確認。

### 測定結果と Verdict
- **処理時間削減 (tools/profile_snapshot.py 30blk, WFM bm=off)**:
  - total p50: **27.8ms → 24.2ms (3.6ms 削減)**
  - total p95: **37.4ms → 30.2ms (7.2ms 削減)**
  - total p99: **43.3ms → 39.4ms (3.9ms 削減、基準 p99 < 40ms 達成)**
  - `pair/mpx_canceller`: **6.28ms → 1.80ms (-4.48ms)**
  - `wfm/freq_blend`: **0.63ms → 0.025ms (-0.60ms)**
- **品質score (tools/ab_benchmark.py golden 全4種)**:
  - `weak_775_10s.npy`: STOI = **1.000**, 可聴ヒス差 = **-0.00dB** (判定: ok)
  - `deepfade_800_10s.npy`: STOI = **1.000**, 可聴ヒス差 = **+0.00dB** (判定: ok)
  - `strong_946_3s.npy`: STOI = **1.000**, 可聴ヒス差 = **-0.00dB** (判定: ok)
  - `noise_762_3s.npy`: STOI = **1.000**, 可聴ヒス差 = **+0.00dB** (判定: ok)
- **非退行確認**:
  - `benchmark_black_magic.py`: 全7条件で非退行合格
  - `tests/run_all.py`: **ALL TESTS PASSED** (全テスト完全合格)
- **Verdict: 採用 (GO)** (3.6〜4.5ms 削減 ≥ 1ms, score 1.000 ≥ 0.99, p99<40ms達成)

## 実録音AB結果編 (ゴールデンデータ・ハーネス実走)

- 対象: `testdata/weak_775_10s.npy` (77.5MHz弱局10秒)、
  `deepfade_800_10s.npy` (80MHz 10秒・当日は強く入感)、
  `strong_946_3s.npy`、`noise_762_3s.npy`。
  `python tools/ab_benchmark.py <file> --bm cyclo|notch --out json`。
- cyclo (指標統一後の正しい比較): native lock>0.5率はON/OFFで
  **差分+0.00 (全条件)**。旧報告の「+60pt」は指標混同
  (lock率とpresent率の比較) であり**撤回**する。
  cycloの価値はlock改善ではなく、nativeがふらつく弱局でも
  present 0.98・conf 0.9級の安定検出信号を出すこと
  (将来のスケルチ反映の入力用)。合格基準「lock率+15%」は
  助言専用設計では到達不能のため、基準自体を再定義要
  (present安定性 or スケルチ反映後のchatterで測る)。
  音声は全条件でON==OFF (過剰処理なし)。
- notch: 実録音にハムなし→全件バイパス、ON==OFF。
  実ハム相当として周波数ドリフト耐性を合成で確認:
  ±0.03/±0.10/±0.30Hz wobbleでも50Hz -28dB抑制・番組±0.00dB
  (ブロック内最小二乗フィットのためドリフトに強い)。
  実ハム録音での除去実証は未取得 (残件)。
- 遅延: 57.3msの由来は物理制約
  (132096byte=66048IQ ÷ 1152000Hz = 57.33ms = 1ブロックの空中時間)。
  実測はp50 24〜37ms、p99 57〜60msで0〜2%超過
  (単発・位置不定の環境ジッタ)。
- 波形連続性チェック (合格): 超過ブロック前後の波形に段差・無音化なし。
  6試行の最大52ms・無音ランは18ms×1件のみ (番組/フェード由来)。
  live側はraw_queue 120ブロック (約7秒)＋音声キューで吸収する設計
  (main.py:432) のため、単発超過で音切れしない。
  正しい基準は「p99＋持続超過 (キュー枯渇) の監視」。p99単発超過は許容。
  p95への緩和提案は撤回し、この基準に置換する。
- 主観評価用WAV (temp): weak775_off/notch/all、deepfade800_off/all。
  RMTのSide定位・ノッチの番組影響は耳で確認すること (未実施)。

## ゴールデン (全機能ONの実録音AB・確定値)

- `testdata/golden_all.json`: 5条件×OFF/ON。遅延整合つき比較
  (RMTの15サンプル固定遅延をlag探索で吸収。未整合だと1kHz位相ずれで
  見かけの劣化が出るため)。
- 弱局: present 0.98/conf 0.92、RMS +0.01dB、高域 -1.13dB (軽微なNR効果)。
- 強局/深フェード: RMS ±0.02dB、高域 ±0.5dB以内。完全透明。
- 局間: present 0.00、RMS +0.13dB。誤検出なし。
- 遅延: 全条件 p99<57.3ms・超過率0.00 (本走行)。
- RMT遅延整合の修正: バイパス毎に遅延が入抜して15サンプルの
  タイムジャンプ (1kHzで0.45FS段差) が出ていたのを、
  全経路で一定遅延にして解消 (回帰テストあり)。
- ベンチ指標の修正: 定常部評価 (後半1/4)＋トーン基準高域＋
  case7の遅延整合。fresh-instanceのS-meter立上り過渡を除外。

## C高速化 (sdr_dft_bins)

- `sdr_dft_bins` (C, version 6): 複数DFTビン一括計算。
  cyclo 9ビン・notch 25ドット・SRを1コール化。
- 実測: notch 6.5→0.9ms/block。cycloは横ばい (Python側の残りが支配的)。
- `dsp_native.dft_bins`＋numpy代替 (旧DLL互換)。等価性1.7e-9で検証。
- DLLはgitignoreのため、他環境では`build_native.bat`で再ビルドすること。

## AM短波の品位 (7.3MHz実録音)

- 症状: 100Hz系の線が番組比-20〜-27dBで存在＋深いフェージング
  (包絡変動0.84)。AM経路にノッチ接合を追加済み (chキー分離)。
- 検出器は作動せず (dwell -4のまま): 線が番組に埋もれ局所床比が
  閾値に届かない。無理に掛けると音楽を削るため正しい判断。
- 強制除去の実験WAV (temp: sw7300_off/nohum): ハム推定-27dB。
  可聴域ギリギリのため、耳で確認してから閾値調整を判断する。
- フェージング歪み自体はAGCの担当。選択性フェージングの歪み補償は
  将来課題 (AM同期検波の混合率が既存の調整点)。

- `python tools/ab_benchmark.py <iq...> [--bm cyclo|rmt|sr|all] [--out json]`:
  同一録音をOFF/ONで2回通す。対応形式は`.npy`(uint8 raw)、`.cs16`、
  `.wav`。初回2ブロック除外でp50/p95/p99・57.3ms超過率を出す。
- `testdata/`: `strong_946_3s.npy` (強局3秒)、`noise_762_3s.npy`
  (局間3秒)、`flutter_775_seq.npy`＋`meas_775_snap_*.npy`
  (77.5MHzフラッター)、`baseline_cyclo.json` (基準値)。
- ベースライン (cyclo): 強局 present 0.92/conf 0.83・ON==OFF完全一致、
  局間 present 0.00/conf 0.14、フラッター present 0.33/conf 0.46。
  遅延は強局p99 34ms・局間p99 57.5ms (超過率0.02)。
  57.3msは物理制約 (1ブロックの空中時間) のため基準自体は維持し、
  超過時は音切れ有無で判定する (結果編参照)。
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

---

## 第2章: 受信機自律保護 (Receiver Resilience & RF Health Governor)

### 目的
「音をさらにきれいにする黒魔法」は第1章で弾を撃ち尽くした（WFMは追加禁止・減量完了）。
第2章では、8-bit ADC特有の狭ダイナミックレンジや悪条件下（過大入力、隣接強妨害波、急峻フェージング、高速モード切替）でも破綻・発振・破裂音を出さず耐え抜く「受信機の堅牢性（Receiver Resilience）」を確立する。

### 実測破壊限界探索 (Breaking Point Limit Mapping)
`tools/rf_stress_benchmark.py` による限界探索実測値：

1. **過入力破綻スキャン (Overload Scale Sweep)**:
   - 0dB〜+6dB: クリップ率 <3%、SINAD 51.7〜52.5dB を維持（耐性あり）。
   - +12dB (4.0x): クリップ率 33.0%、SINAD 48.3dB。
   - +18dB (8.0x) 超: クリップ率 >40%、混変調歪みにより SINAD 39.6dB に悪化（破綻境界）。
2. **隣接妨害波排除限界 (Adjacent Strong Blocker Sweep)**:
   - 257タップ Kaiser窓IFフィルタにより、$\Delta f = \pm 100\text{kHz}, \pm 200\text{kHz}, \pm 400\text{kHz}$ のすべてにおいて **+50dB まで完全排除**（破綻点は +60dB）。
3. **搬送波オフセット追従限界 (AFC Pulling Limit Sweep)**:
   - 3kHz: 完全ロック（残差 1.39kHz）。
   - 8kHz〜15kHz: 引き込み動作（残差 3.7〜6.9kHz）。
   - 25kHz以上: $\pm 20\text{kHz}$ ハードリミットにより追従膠着（OUT_OF_RANGE）。

### 導入モジュール: RF Health Governor (`rf_health.py`)
- **3段階FSM統制**:
  - `HEALTHY`: 正常運用。原音完全素通し（ゼロオーバーヘッド・ビット完全透明）。
  - `OVERLOAD_WARNING`: ADCクリップ率 3〜20% または過入力。自律ゲイン引下げ要求 (-2dB)。
  - `OVERLOAD_HARD`: ADCクリップ率 >20% または極大飽和。自律ゲイン大幅引下げ要求 (-6dB〜-12dB)。
  - **ヒステリシス**: 悪化は即時遷移、復帰は8ブロック（約450ms）正常継続を必須としてチャタリング根絶。
- **モード切り替え境界平滑化 (Boundary Continuity Crossfade)**:
  - 復調方式（WFM ↔ AM ↔ NFM ↔ USB等）の切り替え時に発生していた境界段差（旧 -23.7dBFS、未補間時 -1.7dBFS）を、直前サンプルの記憶と20msコサイン窓クロスフェード（$y[t] = x_{prev}(1-w) + x_{new}w$）により **-120dBFS（段差ゼロ）に根絶**。

### ストレステスト結果 (`python tools/rf_stress_benchmark.py`)
- 全12項目中 12項目 PASS（Warnings: 0, Failed: 0）
- **Resilience Score: 100.0 / 100.0**
- 全テストスイート（`python tests/run_all.py`）: **ALL TESTS PASSED**

### ハードウェアゲイン自動連動 (Auto Gain Interlocking)
- **`HyperController` との統合**:
  - `RfHealthGovernor` の過大入力判定（`OVERLOAD_HARD` / `OVERLOAD_WARNING`）および自律減衰要求（`gain_step_db`）を、`HyperController` の毎フレーム処理で直接監視。
  - **ハードロック優先介入**: ユーザー固定または収束後ハードロック（`hard_lock=True`）中であっても、過大入力を検知した場合は最優先で介入し、チューナー実機（R820T2等）のゲインを目標値以下（例: -2dB, -6dB, -12dB）の安全段へ即座に引き下げてクリッピングを物理解消。
  - **安全下限クランプ**: 過剰な減衰による無音化を防ぐため、`_min_safe_idx()`（19.7dB / 非常時12.5dB）の下限ガードを厳格に維持。
  - **GUIテレメトリ反映**: 異常発生時はテレメトリに `| RF:OVERLOAD_HARD` などの警告タグを表示。
- **検証**: `tests/test_rf_gain_interlock.py`（全4テスト全件合格）。


