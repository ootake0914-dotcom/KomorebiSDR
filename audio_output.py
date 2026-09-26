"""
Audio Output Module using sounddevice - Hi-Fi Edition.
低遅延・高品位なオーディオ再生エンジン。
- スレッドセーフなジッター吸収キュー
- ソフトニー・ダイナミクスコントロール（歪み防止 & 音量安定化）
- ボリュームフェード & アンダーラン保護
"""

import threading
import time
import queue
import sys
import numpy as np
import sounddevice as sd


class AudioOutput:
    """sounddevice を用いた高品位オーディオ再生マネージャ"""

    def __init__(self, sample_rate: int = 48000, blocksize: int = 1024):
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.volume = 0.7  # デフォルト音量 70%
        self.is_muted = False

        # 音声サンプル用キューと残存バッファ
        # (深めにして、万一の処理遅延でも古い音声の切り捨て=早回しを起こさない)
        # 内部表現は常にステレオ (N, 2)。モノラル入力はL/Rへ複製する。
        self.audio_queue = queue.Queue(maxsize=200)
        self.remainder = np.empty((0, 2), dtype=np.float32)
        self.stream = None
        self.is_running = False
        self.is_prerolled = False
        self.current_device = None
        self._last_default_name = None
        self._reopened_for_name = None
        self._want_running = False
        self._device_watch_thread = None
        self._device_watch_stop = threading.Event()
        # 8→5へ短縮 (約459→287ms。初音・選局復帰を速めつつ
        # GIL詰まり耐性は維持。単体機並みの即応へ寄せる)
        self.preroll_threshold = 5  # ジッターバッファ: 約287ms (5チャンク) 蓄積して安定再生
        self.last_out_samples = np.zeros(2, dtype=np.float32)

        # リアルタイムコールバック内の動的確保を排除 (uac2: pre-allocated pool / single-copy)
        self._empty = np.empty((0, 2), dtype=np.float32)
        self._scratch = np.zeros((max(int(blocksize), 8192), 2), dtype=np.float32)
        self._absbuf = np.empty_like(self._scratch)
        self._mask = np.empty(self._scratch.shape, dtype=bool)
        # フェード曲線 (最大32) とソフトリミッタ作業域も事前確保
        self._fade = (0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, 32, dtype=np.float32)))).astype(np.float32)
        # 復帰用フェードイン (0→1)。無音から任意振幅で再開する段差を消す
        self._fade_in = (0.5 * (1.0 - np.cos(np.pi * np.arange(1, 33) / 33.0))
                         ).astype(np.float32)[:, None]
        self._needs_fade_in = False
        self._played_once = False
        self._vol_current = float(self.volume)
        self._vol_ramp = np.empty(len(self._scratch), dtype=np.float32)
        self._preroll_default = self.preroll_threshold
        self._signbuf = np.empty_like(self._scratch)
        self._compbuf = np.empty_like(self._scratch)
        # ステレオ連動ソフトリミッター作業域 (事前確保、アロケーションゼロ)
        self._peakbuf = np.empty(len(self._scratch), dtype=np.float32)
        self._mask1d = np.empty(len(self._scratch), dtype=bool)
        self._comp1d = np.empty(len(self._scratch), dtype=np.float32)
        self._gainbuf = np.empty(len(self._scratch), dtype=np.float32)

        # ルックアヘッドリミッタ (ワーカー側put_audioで実行。1.5ms先読みで
        # 過変調クリップを歪みなく抑止。コールバック側の瞬時リミッタは安全網として残す)
        self._lim_delay_n = 72  # 1.5ms @48kHz
        self._lim_delay = np.zeros((self._lim_delay_n, 2), dtype=np.float32)
        self._lim_env = 0.0
        self._lim_rel = float(np.exp(-1.0 / (0.040 * float(sample_rate))))
        self._lim_thr = 0.98

        # 長時間安定稼働モニタリング統計
        self.underrun_count = 0
        self.overflow_count = 0
        self.callback_errors = 0
        self.status_underflows = 0
        self.total_callbacks = 0
        self.total_frames_played = 0
        self.callback_us_max = 0.0
        self._fpu_initialized = False

    def _callback_core(self, outdata, frames, time_info, status):
        """ジッターバッファによる安定オーディオ再生コールバック。

        リアルタイムパスでは新規メモリ確保を行わず、事前確保した _scratch へ
        キュー内容を直接コピーする (uac2 の pre-allocated pool / single-copy 方式)。
        フェード曲線・リミッタ作業域も事前確保済み。Queue操作の内部ロックは
        sounddevice/PortAudio環境では避けられないため、例外防壁(外側)と
        状態スナップショットで影響を最小化する。
        """
        if not self._fpu_initialized:
            self._fpu_initialized = True
            try:
                from dsp import enable_fast_fpu
                enable_fast_fpu()
            except Exception:
                pass
        t0 = time.perf_counter()
        frames = int(frames)
        if frames > len(self._scratch):
            # 想定外の大きなブロック要求時のみ拡張 (通常は発生しない)
            self._scratch = np.zeros((frames, 2), dtype=np.float32)
            self._absbuf = np.empty_like(self._scratch)
            self._mask = np.empty(self._scratch.shape, dtype=bool)
            self._signbuf = np.empty_like(self._scratch)
            self._compbuf = np.empty_like(self._scratch)
            self._vol_ramp = np.empty(len(self._scratch), dtype=np.float32)
            self._peakbuf = np.empty(len(self._scratch), dtype=np.float32)
            self._mask1d = np.empty(len(self._scratch), dtype=bool)
            self._comp1d = np.empty(len(self._scratch), dtype=np.float32)
            self._gainbuf = np.empty(len(self._scratch), dtype=np.float32)

        # プレロール判定: バッファがクッション(約400ms)まで蓄積されるまで待機
        if not self.is_prerolled:
            if self.audio_queue.qsize() >= self.preroll_threshold:
                self.is_prerolled = True
                # 無音から復帰する場合のみフェードイン (初回再生は不要)
                self._needs_fade_in = self._played_once
            else:
                outdata.fill(0.0)
                self.last_out_samples[:] = 0.0
                return
        elif self.preroll_threshold < self._preroll_default and self.audio_queue.qsize() >= self._preroll_default:
            # アンダーランで縮めた閾値を、バッファが回復したら元へ戻す
            self.preroll_threshold = self._preroll_default

        scratch = self._scratch
        n = 0

        # 1. 前回のコールバックで余ったサンプルを先頭から消費
        if len(self.remainder) > 0:
            take = min(len(self.remainder), frames)
            scratch[:take] = self.remainder[:take]
            n = take
            self.remainder = self.remainder[take:] if take < len(self.remainder) else self._empty

        # 2. 不足分をキューから時系列順にコピー (余りはremainderへ、コピーなしのビュー)
        while n < frames:
            try:
                chunk = self.audio_queue.get_nowait()
            except queue.Empty:
                break
            take = min(len(chunk), frames - n)
            scratch[n:n + take] = chunk[:take]
            n += take
            if take < len(chunk):
                self.remainder = chunk[take:]

        self.total_callbacks += 1
        self.total_frames_played += frames

        # アンダーフロー補完 (直前サンプルから0Vへコサイン減衰、バッファへ直接書込)
        if n < frames:
            self.underrun_count += 1
            missing = frames - n
            fade_len = min(32, missing)
            if n > 0:
                last = scratch[n - 1]
            else:
                last = self.last_out_samples
            if fade_len > 0 and np.any(np.abs(last) > 1e-4):
                fade = self._fade[:fade_len]
                scratch[n:n + fade_len] = last[None, :] * fade[:, None]
                scratch[n + fade_len:frames] = 0.0
            else:
                scratch[n:frames] = 0.0
            # 次コールバックの復帰時にフェードインを掛ける (無音→任意振幅の段差防止)
            self._needs_fade_in = True

            # キューが空なら再プレロール(3チャンク)して小刻みなバタつきを防止
            if self.audio_queue.empty() and len(self.remainder) == 0:
                self.is_prerolled = False
                self.preroll_threshold = 3

        data = scratch[:frames]

        # フェード源は音量適用前の生サンプルを保存 (次回アンダーラン時、音量を
        # 二重に掛けて -6dB 段差になるのを防ぐ)。n==0では前回保存値を使う。
        # 事前確保域へcopytoし、コールバック内確保を避ける。
        if n > 0:
            np.copyto(self.last_out_samples, scratch[n - 1])
            raw_last = self.last_out_samples
        else:
            raw_last = self.last_out_samples

        # 無音からの復帰時は短いフェードイン (先頭クリック防止)
        if self._needs_fade_in:
            fl = min(len(self._fade_in), frames)
            data[:fl] *= self._fade_in[:fl]
            self._needs_fade_in = False

        # GUIスレッドと共有する状態は先頭でスナップショット (途中の書き換えを遮断)
        vol = self.volume
        muted = self.is_muted

        # ミュート処理
        if muted:
            outdata[:, 0] = 0.0
            outdata[:, 1] = 0.0
            self.last_out_samples[:] = raw_last
            us = (time.perf_counter() - t0) * 1e6
            if us > self.callback_us_max:
                self.callback_us_max = us
            return

        # 音量スケーリング (ブロック内で前回値から目標値へ線形ランプ。
        # スライダー操作・ミュート解除の段差クリックを消す。確保なし)
        vol = self.volume
        if abs(vol - self._vol_current) > 1e-4:
            # 線形ランプをin-place生成 (numpy版によりlinspaceのout=が使えないため)
            rb = self._vol_ramp[:frames]
            step = (vol - self._vol_current) / max(1, frames)
            rb.fill(step)
            np.cumsum(rb, out=rb)
            rb += (self._vol_current - step)
            data *= rb[:, None]
            self._vol_current = vol
        else:
            np.multiply(data, vol, out=data)
            self._vol_current = vol

        # ステレオ連動（Linked Stereo）ソフトリミッター (0.98超のみ圧縮)
        # 左右チャンネル独立圧縮による音像定位の揺れ・偏りを抑え、左右の音量比を保存。
        # 全て事前確保域で計算しアロケーションゼロ。
        # ワーカー側ルックアヘッドリミッタ (thr=0.98) と統一し、それ以下の
        # 瞬時tanh整形による高調波歪み (THD) を出さない。安全網として残す。
        threshold = 0.98
        abs_l = self._absbuf[:frames, 0]
        abs_r = self._absbuf[:frames, 1]
        np.absolute(data[:, 0], out=abs_l)
        np.absolute(data[:, 1], out=abs_r)
        peak = self._peakbuf[:frames]
        np.maximum(abs_l, abs_r, out=peak)
        mask = self._mask1d[:frames]
        np.greater(peak, threshold, out=mask)
        if mask.any():
            inv = 1.0 - threshold
            comp = self._comp1d[:frames]
            np.subtract(peak, threshold, out=comp)
            np.divide(comp, inv, out=comp)
            np.tanh(comp, out=comp)
            comp *= inv
            comp += threshold
            gain = self._gainbuf[:frames]
            np.divide(comp, np.maximum(peak, 1e-12), out=gain)
            # 左右両チャンネルに同一の減衰ゲインを適用 (L/R比率・音像を保存)
            # fancy-index (data[mask]*=) は一時配列を確保するため、where書戻しで確保回避
            np.copyto(self._compbuf[:frames, 0], data[:, 0])
            np.copyto(self._compbuf[:frames, 1], data[:, 1])
            self._compbuf[:frames, 0] *= gain
            self._compbuf[:frames, 1] *= gain
            np.copyto(data[:, 0], self._compbuf[:frames, 0], where=mask)
            np.copyto(data[:, 1], self._compbuf[:frames, 1], where=mask)

        # ステレオ出力 (outdata は C-contiguous な (frames,2))
        outdata[:, 0] = data[:, 0]
        outdata[:, 1] = data[:, 1]
        # フェード源は音量・リミッタ適用前の生サンプルを保存 (二重適用防止)
        self.last_out_samples[:] = raw_last
        self._played_once = True

        us = (time.perf_counter() - t0) * 1e6
        if us > self.callback_us_max:
            self.callback_us_max = us

    def _audio_callback(self, outdata, frames, time_info, status):
        """PortAudio実コールバック。例外1発でのストリーム死を防ぐ最終防壁。
        statusのoutput_underflowも検出してカウントする。"""
        try:
            try:
                if status is not None and bool(getattr(status, "output_underflow", False)):
                    self.status_underflows += 1
            except Exception:
                pass
            self._callback_core(outdata, frames, time_info, status)
        except Exception as e:
            self.callback_errors += 1
            if self.callback_errors % 50 == 1:
                print(f"[WARN] audio callback error x{self.callback_errors}: {e}",
                      file=sys.stderr)
            try:
                outdata.fill(0.0)
            except Exception:
                pass

    def _open_stream(self, device_idx=None):
        """指定デバイス(既定出力)でストリームを開いて再生開始する。
        デバイスは明示指定で固定し、後続の監視で変更を検出する。"""
        kwargs = dict(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            channels=2,
            dtype="float32",
            callback=self._audio_callback,
        )
        if device_idx is not None:
            kwargs["device"] = int(device_idx)
        s = sd.OutputStream(**kwargs)
        try:
            s.start()
        except Exception:
            # start失敗時に開いたストリームを必ず閉じる (リーク防止)
            try:
                s.close(ignore_errors=True)
            except Exception:
                try:
                    s.close()
                except Exception:
                    pass
            raise
        if not self._want_running:
            # 再オープン中にstop()された場合は開いたストリームを閉じて終了
            # (停止後にストリームが残りキューを消費し続けるのを防ぐ)
            try:
                s.stop()
                s.close()
            except Exception:
                pass
            return False
        self.stream = s
        try:
            # 実際に開いたデバイス番号を保持 (device指定なしの場合は解決値)
            self.current_device = int(s.device)
        except Exception:
            self.current_device = device_idx
        self.is_running = True
        self.is_prerolled = False
        return True

    @staticmethod
    def _default_output_index(devices=None) -> int | None:
        """Windows既定の再生デバイスインデックスを返す。
        - 古いsounddeviceは is_default_output_device を持つ
        - 現行sounddeviceは持たないため、Core Audio (MMDevice) で既定出力の
          フレンドリ名を取得し、PortAudioデバイス名と突合する
        検出できない場合は None (現状維持)。"""
        try:
            if devices is None:
                devices = sd.query_devices()
            for d in devices:
                if (d.get("is_default_output_device")
                        and d.get("max_output_channels", 0) > 0):
                    return int(d["index"])
        except Exception:
            pass
        # Core Audio 経由 (Windows動的検出)
        try:
            from win_audio import _win_default_output_name
            name = _win_default_output_name()
            if name:
                return AudioOutput._match_output_index(name, None)
        except Exception:
            pass
        return None

    @staticmethod
    def _match_output_index(name: str, current_index=None) -> int | None:
        """フレンドリ名に一致するPortAudio出力デバイスを返す。
        現在デバイスと同じホストAPIを優先し、無ければホストAPI昇順で最初の一致。"""
        try:
            devs = sd.query_devices()
            cur_host = None
            if current_index is not None:
                for d in devs:
                    if int(d["index"]) == int(current_index):
                        cur_host = d["hostapi"]
                        break
            cands = [d for d in devs
                     if d.get("max_output_channels", 0) > 0
                     and name.strip().lower() in d["name"].lower()]
            if not cands:
                return None
            if cur_host is not None:
                for d in cands:
                    if d["hostapi"] == cur_host:
                        return int(d["index"])
            cands.sort(key=lambda d: d["hostapi"])
            return int(cands[0]["index"])
        except Exception:
            return None

    def _reopen_for_device(self, device_idx):
        """デバイス変更 (イヤホン挿抜等) でストリームを開き直す。
        古いキューの滞留音(最大11秒)を破棄してライブから再開し、再プレロールで滑らかに再接続する。
        開き直し失敗に備え、指定→PortAudio既定の順に試す。意図外デバイスへの無言フォールバックを
        避けるため全デバイス掃引は行わず、フォールバック先はログ表示する。
        _want_running が立っていれば監視ループが毎秒リトライする。"""
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.is_prerolled = False
        # stale音の破棄 (再オープン前の最大200ch≒11秒の古音を再生しない)
        try:
            while True:
                self.audio_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.remainder = np.empty((0, 2), dtype=np.float32)
        except Exception:
            pass

        candidates = []
        if device_idx is not None:
            candidates.append(int(device_idx))
        candidates.append(None)  # PortAudio既定
        # 重複除去
        seen = set()
        uniq = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                uniq.append(c)

        for c in uniq:
            try:
                if self._open_stream(c):
                    try:
                        print(f"[*] audio reopen -> device {c}")
                    except Exception:
                        pass
                    return True
                # _want_running=False (stop済み) の場合は開かず終了
                return False
            except Exception as e:
                try:
                    print(f"[WARN] audio open failed device={c}: {e}", file=sys.stderr)
                except Exception:
                    pass
                self.stream = None
                continue
        # 最後の手段: PortAudioを再初期化して再試行 (デバイス抜き差しで
        # PortAudioのデバイス一覧が陳腐化した場合の回復)
        try:
            sd._terminate()
            sd._initialize()
            if self._open_stream(None):
                return True
        except Exception:
            pass
        self.stream = None
        self.is_running = False
        return False

    def _start_device_watch(self):
        """Windows既定再生デバイスを監視し、変更時にストリームを再オープンする。
        - 既定デバイス名の変化 (Headphones⇔Speakers)
        - 抜き差し後のPortAudioデバイス再列挙による番号ずれ
        - ストリーム死 (再オープン失敗)
        のいずれかを0.5秒間隔で検出し、複数候補へフォールバックしながら復帰する。"""
        # 世代ごとにEventを新規発行し、旧スレッドの停止要求が新スレッドに波及/取消されないようにする
        # (旧共有Event使い回しでは join失敗時に旧・新2スレッドが並行reopenした)
        if self._device_watch_thread is not None and self._device_watch_thread.is_alive():
            try:
                self._device_watch_stop.set()
                self._device_watch_thread.join(timeout=1.0)
            except Exception:
                pass
            if self._device_watch_thread.is_alive():
                # 旧スレッド残留: 新スレッドを立てず旧スレッドに任せる (二重reopen防止)
                return
        self._device_watch_stop = threading.Event()
        try:
            from win_audio import _win_default_output_name
            self._last_default_name = _win_default_output_name()
        except Exception:
            self._last_default_name = None

        def loop(stop_ev=None):
            if stop_ev is None:
                stop_ev = self._device_watch_stop
            last = self._last_default_name
            while not stop_ev.is_set():
                try:
                    if self._want_running:
                        try:
                            from win_audio import _win_default_output_name
                            name = _win_default_output_name()
                        except Exception:
                            name = None
                        need = False
                        matched = None
                        if name is not None and name != last:
                            last = name
                            self._last_default_name = name
                            need = True
                        # PortAudio内部エラー等でコールバックが止まった場合の死検出
                        if (not need and self.is_running and self.stream is not None
                                and not getattr(self.stream, "active", True)):
                            need = True
                        if not self.is_running:
                            need = True
                        elif name and name != self._reopened_for_name:
                            # 抜き差しでWindowsがデバイスを再列挙するとPortAudioの
                            # 番号がずれる (例: Headphones 4→3)。名前が同じでも現在の
                            # ストリーム実デバイスが既定名と不一致なら一度だけ再オープン。
                            # (毎周期リトライするとフォールバック後に0.5秒毎の音切れループに
                            #  なるため、同一名では1回に制限する)
                            matched = self._match_output_index(name, self.current_device)
                            if (matched is not None
                                    and self.current_device is not None
                                    and matched != self.current_device):
                                need = True
                        if need:
                            idx = matched if matched is not None else (
                                self._match_output_index(name, self.current_device)
                                if name else None)
                            self._reopen_for_device(idx)
                            self._reopened_for_name = name
                except Exception:
                    pass
                # 0.5秒間隔 (Core Audioクエリ約10ms。抜き差し検出の遅延を最小化)
                stop_ev.wait(0.5)

        t = threading.Thread(target=loop, daemon=True, name="audio-device-watch",
                             args=(self._device_watch_stop,))
        self._device_watch_thread = t
        t.start()

    def start(self):
        if self.is_running:
            return
        self._want_running = True
        self.is_prerolled = False
        self._played_once = False
        self.preroll_threshold = self._preroll_default
        self.remainder = np.empty((0, 2), dtype=np.float32)
        self.last_out_samples = np.zeros(2, dtype=np.float32)
        self._lim_delay = np.zeros((self._lim_delay_n, 2), dtype=np.float32)
        self._lim_env = 0.0
        dev = self._default_output_index()
        try:
            if not self._open_stream(dev):
                self.is_running = False
        except Exception:
            # オーディオデバイスが無い環境でも受信自体は継続する (無音)
            self.stream = None
            self.is_running = False
        self._start_device_watch()

    def stop(self):
        self._want_running = False
        self._device_watch_stop.set()
        if self._device_watch_thread is not None:
            self._device_watch_thread.join(timeout=1.5)
            self._device_watch_thread = None
        self.is_running = False
        self.is_prerolled = False
        self.remainder = np.empty((0, 2), dtype=np.float32)
        self.last_out_samples = np.zeros(2, dtype=np.float32)
        # is_runningがFalseでもstreamが残っていれば必ず閉じる (リーク防止)
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def _lookahead_limit(self, arr: np.ndarray) -> np.ndarray:
        """1.5ms先読みブリックウォールリミッタ (put_audio=ワーカースレッド側)。
        未来ピークからゲインを決めるためアタック歪みが出ない。リリース40ms。
        チャンク跨ぎは遅延線＋エンベロープ持越しで連続性を保つ。
        高速化: ピークが閾値以下 (通常時) は無処理で即復帰 (~20us)。
        ホット時のみ窓max＋リリース平滑を実行する。"""
        n = len(arr)
        if n == 0:
            return arr
        d = self._lim_delay_n
        ext = np.concatenate((self._lim_delay, arr), axis=0)
        self._lim_delay = ext[-d:].copy()
        absmax = float(np.max(np.abs(ext)))
        if absmax <= self._lim_thr:
            # コールドパス: 制限不要。エンベロープは減衰のみ継続。
            self._lim_env = max(absmax, float(self._lim_env) * (self._lim_rel ** n))
            return ext[:n]
        # ホットパス: モノラルピーク化して窓max (sliding_windowはviewで複写なし)
        pk = np.maximum(np.abs(ext[:, 0]), np.abs(ext[:, 1]))
        win = np.lib.stride_tricks.sliding_window_view(pk, d + 1)
        peak = np.max(win[:n], axis=1)
        env = np.empty(n, dtype=np.float32)
        e = float(self._lim_env)
        rel = self._lim_rel
        for i in range(n):
            p = peak[i]
            if p > e:
                e = p
            else:
                e *= rel
            env[i] = e
        self._lim_env = e
        g = np.minimum(1.0, self._lim_thr / np.maximum(env, 1e-6)).astype(np.float32)
        return (ext[:n] * g[:, None]).astype(np.float32)

    def put_audio(self, samples: np.ndarray):
        if len(samples) == 0:
            return
        if self.stream is None:
            # オーディオ出力なし: キューへ溜め込まず破棄
            return
        arr = np.asarray(samples, dtype=np.float32)
        # 非有限サニタイズ (DSP異常時のNaN/Infがキュー・last_out_samples・
        # リミッタ包絡へ拡散し恒久ノイズ化するのを入口で断つ。±1.0へ丸め)
        if not bool(np.all(np.isfinite(arr))):
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
        if arr.ndim == 1:
            # モノラル -> ステレオ複製
            arr = np.stack((arr, arr), axis=1)
        elif arr.ndim != 2 or arr.shape[1] != 2:
            flat = arr.ravel()
            arr = np.stack((flat, flat), axis=1)
        arr = self._lookahead_limit(arr)
        if self.audio_queue.full():
            self.overflow_count += 1
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.audio_queue.put_nowait(arr)
        except queue.Full:
            self.overflow_count += 1

    # リサンプラ水位換算の nominal 1チャンク (=DSP 1ブロック≈2752サンプル)。
    # remainderは1チャンク未満の端数のため、0/1の二値ではなく分数で加算する
    # (1サンプル残存で+1チャンク誤差→800ppm級の誤PI補正・ハンチングの防止)。
    _CHUNK_FRAMES = 2752

    def get_queue_size(self) -> float:
        """キュー内の現在の残存チャンク数 (目標: 3〜5チャンク。端数は分数)"""
        rem = len(self.remainder)
        frac = min(1.0, rem / float(self._CHUNK_FRAMES)) if rem > 0 else 0.0
        return self.audio_queue.qsize() + frac

    def get_buffer_fill_ratio(self) -> float:
        """バッファキュー充填率 (0.0〜1.0)"""
        return float(self.get_queue_size() / max(1, self.audio_queue.maxsize))

    def get_stats(self) -> dict:
        """長時間稼働の安定性統計を取得"""
        return {
            "fill_ratio": self.get_buffer_fill_ratio(),
            "underrun_count": self.underrun_count,
            "overflow_count": self.overflow_count,
            "total_callbacks": self.total_callbacks,
            "total_frames_played": self.total_frames_played,
        }

    def set_volume(self, vol: float):
        self.volume = max(0.0, min(1.0, vol))

    def set_mute(self, muted: bool):
        self.is_muted = muted
