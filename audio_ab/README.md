# audio_ab — 耳ABプロジェクト

生放送は二度と同じ条件で聴けない。耳ABはすべて「同一録音・同一合成素材の
往復比較」で行う。測るのは機械、聴くのは人間。耳をやらない場合でも、
参照不要の機械指標 (MOS/CER) で順位付けまで自動化できる。

## 回し方

```bash
python audio_ab/run_ab.py --list              # 項目一覧
python audio_ab/run_ab.py                     # 全項目のAB生成 (数分)
python audio_ab/run_ab.py --items E1,E4       # 指定だけ
python audio_ab/score_mos.py --p835           # 全ペアをMOS採点 (Scoreq+DNSMOS)
python audio_ab/cer_ab.py --list              # シナリオ一覧
python audio_ab/cer_ab.py --scenario ssb --snrs 0,5   # 文単位CER+CI (既定6文×2シード)
python audio_ab/cer_ab.py --scenario ssb-clicks --mos # クリック耐性+MOS
python audio_ab/sweep.py --list               # スイープ軸一覧
python audio_ab/sweep.py --scenario ssb --axis nr.floor_db=-18,-12 --snrs 5
python audio_ab/regress.py                    # goldenドリフト検査 (約6s)
python audio_ab/regress.py --update [--full]  # 基準更新 (--fullでCER追加)
python audio_ab/subjective.py --xcorr         # 指標同士の相関 (主観記入後は一致率)
python audio_ab/summary.py --csv              # 結果一覧
python audio_ab/record_raw.py 7.100 LSB 30 --tag night   # 生IQ録音
```

出力 (`audio_ab/out/`、models/tts/outはgitignore済み):

- `{項目}_A.wav` / `{項目}_B.wav` — ブラインドペア (どっちがONかはランダム)
- `scores.json` — 機械スコア (STOI・RMS・ヒス・MOS・CER・P835)
- `key.json` — 正解 (聴取後に開封すること)
- `summary.csv` — 集計表
- `listen_log.csv` — 試聴記録 (雛形をoutへコピーして使う)

## 機械指標 (耳の代わり)

- **Scoreq NR-MOS** (`audio_ab/score_noref.py`): 参照なし総合MOS。
  ONNXを直接叩く (この環境のtorchaudioはtorchcodec/FFmpeg共有DLL不足で
  wavを読めないため)。絶対値でなく順位付けに使う。
- **DNSMOS P.835** (`score_mos.py --p835`): SIG/BAK/OVRL/P808。
  モデルは `audio_ab/models/*.onnx` (HuggingFace mirrorから取得、gitignore)。
  16k化してから渡す (48kを渡すとlibrosa新APIと衝突)。
- **CER** (`cer_ab.py`): VOICEVOX原稿 (既知) → 雑音チェーン → faster-whisper
  書き起こし → 文字誤り率。了解度の代用。カタカナ・固有名詞は床を上げるので
  原稿は平易な日本語にする。

## 測定基盤v2 (文単位試行 × 統計ゲート)

旧版は16.8秒連結原稿で1条件1サンプル (CER量子化±0.014、シード分散大) だった。
v2は文ごとに試行を増やし、平均±CIと対の符号検定で判定する:

- `corpus.py` — VOICEVOX 10文×4話者 (ずんだもん/めたん/つむぎ/武宏)。
  1文=1試行として使う。tts/にキャッシュ (gitignore)。
- `simulate.py` — 名前付き標準チャネル: `ssb` / `ssb-clicks` /
  `ssb-onesided` / **`ssb-fade` (2波フェージング)** / **`ssb-step`
  (選局切替)** / `am` / `am-onesided` / `nfm` / `nfm-clicks` / `wfm`
  (ステレオMPX・75µsプリエンファシス) / **`wfm-fade` (全帯域ディップ)** /
  **`wfm-multipath` (遅延エコー+ドップラー)** / **`wfm-adjacent` (隣接強
  トーン)**。SNR・クリック・妨害・フェードが引数で再現可能。
- `stats.py` — bootstrap CI・符号検定・実用ゲート (効果量0.01未満やCIが0を
  跨ぐものは「差なし」と報告し、1文字差を有意と誤認しない)。さらに
  `holm` (多重比較補正) と `n_for_effect` (効果量→必要試行数の逆算)。
- `cer_ab.py` — 文×シード×条件を同じ雑音実現で対にして回し、全試行を
  `out/cer_stats_{scenario}.json` に保存する。
- `metrics.py` — 参照あり指標 (SI-SDR / segSNR / STOI)。クリーン原稿が
  ある強みを活かし、相互相関で遅延整列してから評価する (AGC遅延・ゲイン差に
  ロバスト)。`cer_ab` は全試行で自動計算し、off/onのCI付きΔを表示する
  (`--no-ref`で無効)。CER(伝わるか)とSI-SDR(原音への忠実さ)は乖離しうる:
  実測例 SSB@0dBではNR-onでCER +0.51 (悪化) でもSI-SDR +2.4dB /
  segSNR +2.8dB (改善)。両方を残す理由。
- `realdata.py` — 実録レジストリ (FM3本。gitignore実素材は無ければ
  skip)。`regress --full` が実録のヒス抑圧 (Lucky -13.9dB / NHK -1.7dB) と
  幅推定器下限 (nr_gain_min 0.98 / 0.57) をgoldenに含める。

## スイープと回帰

- `sweep.py` — `--axis name=v1,v2` でDSPパラメータを宣言的にスイープ。
  全バリアントを同一雑音試行で対比較し、baseとのΔをCI・符号検定・
  実用ゲートで報告、全試行を `out/sweep_{scenario}.json` に保存する。
  軸 (`--list`): `nr.on` / `nr.over_sub` / `nr.floor_db` / `nr.gain_smooth` /
  `nr.noise_beta` / `nr.dd_alpha` / `ssb.blank_k` / `am.on` / `wfm.on` /
  `wf.mask_scale` / `wf.gmin` / `agc.hyst`。`--base nr.on=0` で基準条件を
  上書きできる (バリアントにも継承される)。出力に `p_holm` (Holm補正後) と
  `n(.05)` (その効果量を検出するのに必要な試行数) を含み、CIが0を外れても
  Holmで落ちたものは「trend only」に降格する。
- `regress.py` — 標準シナリオの決定的DSP指標6つ (ブランカ偽検出・クリック
  残差・WFM sideヒス/mid透過・AM SINR利得・NR指纹) を `audio_ab/golden.json`
  と比較し、ドリフトでexit 1。約6秒。`--full` でSSB CER 2指標も検査。
  run_allの二値テストでは拾えない「静かな悪化」(抑圧量が1dB変わる等) を
  検出する。goldenは `--update` で手動更新し、無断で書き換えない。

## 高速化と主観突き合わせ

- `fast.py` — ASR結果をデコード後音声のSHA1でディスクキャッシュ
  (`out/asr_cache/`)。DSPコード・パラメータを変えれば音声が変わるため
  自動で失効する。同一設定の再実行は実測27s→1s。スレッド並列はCT2が
  1推論で4コアを使い切るため効果なし (24.9→25.6s)、beam=5→1も26→29sで
  順位が変わるため探索にも不採用。キャッシュが本命。
  `cer_ab`/`sweep`/`regress --full` が自動で使う (`--no-cache` で無効、
  `--beam` で変更可。終了時にヒット率を表示)。20000件超は古い順に淘汰
  (放置しても肥大化しない)。
- 結果JSON (`cer_stats_*.json` / `sweep_*.json` / golden) には provenance
  (git HEAD・dirty・python/numpy/onnxruntime/torch版) を刻む。数値比較は
  同一来歴の間でのみ意味を持つため。
- `subjective.py` — `listen_log.csv` (prefer=A/B/same) と scores.json を
  突き合わせ、指標ごとの符号一致率・Spearmanを出す。未記入でも `--xcorr`
  で指標同士の相関行列 (どの指標が冗長か) を出せる。現状listen_logは
  未記入のため、AB試聴→記入後に「どの指標が耳を予測するか」が決まる。
  (参考: 現有16項目では scoreq–SIG 0.70、hiss–STOI 1.00 (n=4) など)

## 試聴手順 (耳をやる場合)

1. `key.json` を見ない。A/Bを順不同・複数回聴く (ヘッドホン推奨、音量固定)
2. `prefer` に A / B / same、確信度 1-3、コメントを書く
3. 全部終わったら `key.json` を開封して集計する

AB WAVはR128ラウドネス整合済み (「大きい方が選ばれる」バイアスを除去)。
`run_ab.py --no-match` で無効化できる。

## 項目表

| ID | 対象 | 素材 | 問うこと |
|---|---|---|---|
| E1 | NR-SSB白 | 合成 | ヒス減＋声の自然さ |
| E2 | NR-SSB桃 | 合成 | 同上 (帯域重なり条件) |
| E3 | NR-NFM | 合成 | 同上 |
| E4 | AM側波帯 | 合成片側妨害 | 妨害減＋番組保全 |
| E5 | AM側波帯 | 実録sw_7300 | 実録での透過/効果 |
| E6 | 適応ノッチ | 合成番組+ハム | ハム減＋番組保全 |
| E7 | RMT | 合成音楽 | 人工物・高域劣化の有無 |
| E8 | RMT | 合成トーク | 同上 |
| E9 | apodizing | 実録strong_946 | 最小位相化の可聴差 |
| E10 | RMT | 実録weak_775 | 弱局のSide定位・ステレオ像 |
| R1 | NR透過性 | 実録昼7.1MHz (デッドエア) | 実RFノイズでのバイパス/抑圧 |
| R4 | RMT定位 | 実録昼94.6 (音楽/トーク) | 実音楽のステレオ像・人工物 |
| R5 | apodizing | 実録昼94.6 (同上) | 実音楽での可聴差 |
| C1 | NR-CER | TTS音声+白色雑音 0/5/10dB | 了解度 (CER) の改善 |

## 機械スコア実測 (2026-10時点・非可聴判定)

Scoreq NR-MOS の off→on 比較 (差の向きのみ信じる):

| 項目 | MOS off → on | CER (off→on) | 所見 |
|---|---|---|---|
| C1 0dB (v2, 12対) | — | 0.225 → 0.296 (Δ+0.071, CI[-0.03,+0.21]) | 改善は確認できず |
| C1 5dB (v2, 12対) | — | 0.119 → 0.212 (Δ+0.093, CI[+0.04,+0.15], 符号検定n.s.) | 悪化方向 |
| C1 10dB | 1.421 → 1.245 (off) | 0.333 → 0.308 (on) | 参考値 (旧1シード) |
| E1 NR-SSB白 | 1.311 → 1.344 (on) | — | 母音プロキシではMOSも微改善 |
| E2 NR-SSB桃 | 1.425 → 1.434 (on) | — | 同等〜微改善 |
| E3 NR-NFM | 1.318 → 1.342 (on) | — | BAK (背景雑音) 改善が明確 |
| E4 AM側波帯 | 1.016 → 1.458 (on) | — | 片側妨害の除去が効く |
| E5 AM側波帯実録 | 1.140 → 1.109 (off) | — | デッドエアで差なし (期待通り) |
| E6 ノッチ | 1.161 → 1.369 (on) | — | MOSはon、P835はSIG減でoff寄り (指標不一致) |
| E7 RMT音楽 | 1.285 → 1.283 (off) | — | 透明 (差は誤差) |
| E8 RMTトーク | 1.206 → 1.210 (on) | — | 透明 |
| E9 apodizing | 1.067 → 1.046 (off) | — | 透明〜微減 |
| E10 RMT弱局 | 1.039 → 1.026 (off) | — | ほぼ透明 |
| E11 ブランカSSB | 1.245 → 1.486 (on, k8) | 0.265 → 0.209 (paired, 3シード) | パルス12x@15/s。BAK 1.91→2.72 |
| E12 WFMマスクoffset | — | — | 0.1→0.05。合成でside誤差≤-35dBのままヒス+5〜6dB |
| E13 AM側波帯自動ON | — | — | 既定ON化。クリーン/両側妨害はビット等価、片側は+3.6dB自動 |
| E14 LUFS-AGC修正 | — | — | 凍結パスがゲイン未適用のバグ修正。局間差12→0.98LU |
| R1 NRデッドエア | 1.241 → 1.324 (on) | — | ノイズのみ抑圧 (正常) |
| R4 実音楽RMT | 1.139 → 1.135 (off) | — | 透明 |
| R5 実音楽apod | 1.139 → 1.135 (off) | — | 透明 |

**読み方の注意**:
- C1はv2 (10文×4話者・文単位12対) で測り直した結果、NR-onのCER改善は
  確認できず、5dBでは悪化方向 (Δ+0.093, CI[+0.04,+0.15]。符号検定n.s.)。
  旧版の「0dBで相対-4%改善」は1話者・連結1サンプルで過小出力だった。
  一方P.835はBAK改善/SIG低下、Scoreqは微減 (雑音は下がるが声も傷む)。
  よって狭帯域NRは既定OFF・プリセット変更なしが結論 (v2でも変わらず)。
- 狭帯域NRパラメータスイープ (旧ハーネス: 4シード・連結原稿・話者1・0dB):
  over_sub 1.0→1.5/2.0/3.0、floor -12→-18/-6を比較したが、現行baseのみ
  改善方向だった (他は中立〜+0.045悪化)。v2の結果を踏まえると「攻めない」
  ことが本質で、baseはその中で最良。P.835はbaseで BAK +0.94 / SIG -0.87 /
  OVRL ±0.01。よってPRESETS変更はせず・既定OFFのまま。
- ブランカSSB (E11): thr_k 6→8へ変更 (dsp_nfm SSB経路)。クリーン音声での
  誤検出が1280→435サンプルに減り、パルス回復は3シード全てでk8>k6
  (k9以上は回復不足)。NFM/AM経路は今回未測定のためk=6のまま。
- WFMマスクoffset (E12): `_wf_mask_offset` 0.1→0.05。推定器 (幅制御) を固定した
  合成ステレオ (音声L/R・75µs、SNR14/20/26dB) で、side番組誤差-35dB以下・
  幅変化≤0.34dBを保ったままsideヒス10-14kが+5〜6dB深くなる。実録でも
  hiss8-12k: Lucky -8.4→-13.9dB / NHK -0.35→-1.7dB、番組帯・幅は不変。
  0.02相当まで攻めると誤差-21dBと可聴域のため不採用。midは-50dBで透明。
  `_nr_gmin` 0.03/0.05/0.10は差なし (mask gateが支配)。
  追記 (2026-10): radiko_lab (別フォルダ) のオラクル実験で「ギャップの無い
  連続音声では床推定が番組HFを拾い、強電界でもnr_gain 0.15まで全モノラル化
  (SI-SDRは素の受信より5〜11dB悪)」を実測。対策として復調前IF SNR
  (認知OFFでも常時更新) で床をクロスチェックし、IF>32dB側で床を最大30dB
  割り引く `_nr_snr_gate_*` を追加。合成最悪ケースは nr_gain 0.15→0.65・
  SI-SDR +2.7→+5.3dB (オラクル上限+15.4)、実録Luckyは -13.9→-13.4dB
  (ほぼ無傷)、NHK -1.7→0dB。IF推定は未校正でしきい値は受信機ローカル
  (goldenの real_* で固定)。
- AM側波帯合成 (E13): 片側妨害スイープで自動判定しきい値2dB/1dBを検証。
  SINR利得ゼロの条件は不平衡gap≤1.35dB、+3.6dB (-10dB妨害) はgap≥2.4dB、
  0dB妨害は+9.9dBでgap≈10.5dB。両側妨害・クリーンはgap≈-0.3dBで発動せず
  (ビット等価テスト追加)。利得境界としきい値が一致するため数値変更なしで
  既定ONへ (コールドスタート汚染時は中立だが、これは既知の残件)。
- LUFS-AGC (E14): 既存故障test_lufs_agcを実測で決着 (テストが正しかった)。
  原因は2つ: (1) `_slow_agc_level` の凍結パス (不感帯/ホールド/番組ゲート) が
  未スケール音声を返し、収束後はゲインが外れてレベリングが丸ごと無効化、
  (2) 不感帯1.5dBが設計目標±1LUに対し広く、両局が各1.5dB手前で停止。
  全凍結パスで現ゲインを適用し、不感帯を0.5dBへ。局間差12.0→0.98LU、
  ゲインstd 0.080→0.015 (ポンピング面も改善)。これで全テスト緑。
- E6は指標不一致 (Scoreq on / P835 SIG off)。P835はSIG低下 (信号への傷) を
  検出しており、合成ハム条件のノッチが番組に触れている可能性を示す。
  実ハム録音が取れたら再確認する。
- 絶対値は低い (狭帯域無線音声は学習分布の外)。**差の向きだけ**を信じる。

## 実録が必要な残件 (ハード or 放送待ち)

- 幅推定器の密番組検証 (済): 実録4本 (新規Lucky 120s含む) で
  `stereo_nr_gain` は定常≈1.0、全モノラル化なし。合成で崩壊する条件
  (連続強HFでhiss_db>-18dB) は実放送では未観測。選局直後0.5秒だけ
  0.4まで下がる初期過渡 (設計済み)。
- R1(R2): SSB/NFMの実音声IQ (夜/グレーライン) — CER/MOSの実証用
- R3: 実ハム混入AM (電源ハムのある環境) — E6の実証用
- R4: 強力FMステレオ音楽 — 取得済み (day_94.600)。聴取や実C/Nでの再録は任意
- R5: LuckyFM 94.6/88.1のRDS付き録音 (AFホップ素材)

## 開発メモ: SSB合成の正しい作り方

`dsp.demodulate_ssb` は周波数変換をしない (側波帯選択のみ)。複素IF上の
周波数がそのまま音声周波数になる。したがってSSB合成は:

- 正: 実音声の**解析信号** (負周波数ゼロ) をベースバンド直入れ
  → `audio_ab/ssb_synth.py` の `analytic()` / `noisy_ssb_iq()`
- 誤: 実信号に `exp(j2π1500t)` を掛ける → DSB (両側波重畳) になり
  了解度が壊れる (当初ハーネスがこれを踏み、CER 1.0で発覚)

解析信号は `np.fft.irfft` で作らないこと (実数化で負周波数が復活し、
上方変換になる)。`np.fft.ifft` + 負周波数ゼロで作る。

## 環境メモ

- VOICEVOX engine: `%LOCALAPPDATA%\Programs\VOICEVOX\vv-engine\run.exe`
  を起動 (API `127.0.0.1:50021`)。`cer_ab.py` は起動済み前提。
- faster-whisperは48k直入れが壊れるビルドがあるため、必ず自前16k化
  (`score_noref.fft_resample`) を経由する。
- Scoreq/DNSMOSの初回はモデルDLが必要 (`pandas`, `onnxruntime`,
  `librosa`, `huggingface_hub`)。RFC: torchはCPU版で足りる。
