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
python audio_ab/cer_ab.py                     # TTS→雑音→NRのCER/MOS
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
  原稿は平易な日本語にする。クリーン床を毎回校正して比べる。

## 試聴手順 (耳をやる場合)

1. `key.json` を見ない。A/Bを順不同・複数回聴く (ヘッドホン推奨、音量固定)
2. `prefer` に A / B / same、確信度 1-3、コメントを書く
3. 全部終わったら `key.json` を開封して集計する

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
| C1 0dB | 1.307 → 1.221 (on, 4シード) | 0.378 → 0.362 (paired Δ-0.016±0.011) | 了解度は微改善、MOSは減 |
| C1 5dB | — | 0.317 → 0.362 (paired Δ+0.045±0.087) | 有意差なし (分散大) |
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
| R1 NRデッドエア | 1.241 → 1.324 (on) | — | ノイズのみ抑圧 (正常) |
| R4 実音楽RMT | 1.139 → 1.135 (off) | — | 透明 |
| R5 実音楽apod | 1.139 → 1.135 (off) | — | 透明 |

**読み方の注意**:
- CERとMOSは役割が違う。C1は「NRで言葉が通じやすくなる (0dBで相対-4%。
  5dB以上は有意差なし) が総合MOSは微減」— 了解度と自然さの古典的トレードオフ。
- 狭帯域NRパラメータスイープ (4シード・0dB): over_sub 1.0→1.5/2.0/3.0、
  floor -12→-18/-6を比較したが、現行baseのみCERが改善 (他は中立〜+0.045悪化)。
  P.835はbaseで BAK +0.94 / SIG -0.87 / OVRL ±0.01 → 「雑音は下がるが音声も
  少し傷める、総合は横ばい」。  よってPRESETS変更はせず・既定OFFのまま。
  (原稿16.8秒ではCER量子化±0.014のため、0.01級の差は追わない)
- ブランカSSB (E11): thr_k 6→8へ変更 (dsp_nfm SSB経路)。クリーン音声での
  誤検出が1280→435サンプルに減り、パルス回復は3シード全てでk8>k6
  (k9以上は回復不足)。NFM/AM経路は今回未測定のためk=6のまま。
- WFMマスクoffset (E12): `_wf_mask_offset` 0.1→0.05。推定器 (幅制御) を固定した
  合成ステレオ (音声L/R・75µs、SNR14/20/26dB) で、side番組誤差-35dB以下・
  幅変化≤0.34dBを保ったままsideヒス10-14kが+5〜6dB深くなる。実録でも
  hiss8-12k: Lucky -8.4→-13.9dB / NHK -0.35→-1.7dB、番組帯・幅は不変。
  0.02相当まで攻めると誤差-21dBと可聴域のため不採用。midは-50dBで透明。
  `_nr_gmin` 0.03/0.05/0.10は差なし (mask gateが支配)。
- AM側波帯合成 (E13): 片側妨害スイープで自動判定しきい値2dB/1dBを検証。
  SINR利得ゼロの条件は不平衡gap≤1.35dB、+3.6dB (-10dB妨害) はgap≥2.4dB、
  0dB妨害は+9.9dBでgap≈10.5dB。両側妨害・クリーンはgap≈-0.3dBで発動せず
  (ビット等価テスト追加)。利得境界としきい値が一致するため数値変更なしで
  既定ONへ (コールドスタート汚染時は中立だが、これは既知の残件)。
- E6は指標不一致 (Scoreq on / P835 SIG off)。P835はSIG低下 (信号への傷) を
  検出しており、合成ハム条件のノッチが番組に触れている可能性を示す。
  実ハム録音が取れたら再確認する。
- 絶対値は低い (狭帯域無線音声は学習分布の外)。**差の向きだけ**を信じる。

## 実録が必要な残件 (ハード or 放送待ち)

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
