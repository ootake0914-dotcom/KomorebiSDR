/*
 * sdr_core.c - Antigravity SDR ネイティブ高速DSPコア
 *
 * PythonループがGILを保持して音声処理を妨害していたホットスポットをCで実装。
 * ctypes経由で呼び出すため、計算中はGILが解放され、
 * GUI描画・オーディオコールバック・受信処理が並行して動作できる。
 *
 * ビルド: build_native.bat を実行 (cl /O2 /LD)
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <xmmintrin.h>
#include <pmmintrin.h>
#endif
#if defined(_M_X64) || defined(__SSE2__)
#include <emmintrin.h>
#define SDR_HAVE_SSE2 1
#endif

#ifdef _WIN32
#define SDR_EXPORT __declspec(dllexport)
#else
#define SDR_EXPORT
#endif

#define SDR_PI 3.14159265358979323846f
#define SDR_TWO_PI 6.28318530717958647692

SDR_EXPORT int sdr_version(void)
{
    return 5;
}

/* denormal(非正規化数)対策: FTZ/DAZを有効化。
 * IIRフィルタやPLLの減衰テールで非正規化数が発生すると、x86では数百サイクルの
 * ペナルティで周期的なジッタ (じりじりノイズ) の原因になる。
 */
SDR_EXPORT void sdr_fast_fpu(void)
{
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    unsigned int csr = _mm_getcsr();
    csr |= 0x8040u; /* bit15=FTZ, bit6=DAZ */
    _mm_setcsr(csr);
#endif
}

/* ---- 高速sin LUT (1024エントリ + 線形補間, 誤差 ~5e-6) ----
 * PLLのサンプル毎のlibm sin/cos呼び出しを置換 (1.2M呼び出し/秒を削減)
 */
#define SDR_LUT_BITS 10
#define SDR_LUT_SIZE (1 << SDR_LUT_BITS)
#define SDR_TWO_PI_F 6.283185307179586
static float g_sin_lut[SDR_LUT_SIZE + 1];
static volatile int g_sin_lut_ready = 0;
#ifdef _MSC_VER
static volatile long g_sin_lut_lock = 0;
#endif

static void sdr_lut_init(void)
{
    for (int i = 0; i < SDR_LUT_SIZE; ++i) {
        g_sin_lut[i] = (float)sin(SDR_TWO_PI_F * (double)i / (double)SDR_LUT_SIZE);
    }
    g_sin_lut[SDR_LUT_SIZE] = g_sin_lut[0];
    g_sin_lut_ready = 1;
}

/* 1回だけ初期化 (2スレッド同時初回呼の二重書込レースを排除) */
static void sdr_lut_init_once(void)
{
#ifdef _MSC_VER
    if (g_sin_lut_ready) {
        return;
    }
    if (_InterlockedCompareExchange(&g_sin_lut_lock, 1, 0) == 0) {
        sdr_lut_init();
    } else {
        while (!g_sin_lut_ready) {
            _mm_pause();
        }
    }
#else
    if (!g_sin_lut_ready) {
        sdr_lut_init();
    }
#endif
}

static inline float sdr_sin(double x)
{
    if (!g_sin_lut_ready) {
        sdr_lut_init_once();
    }
    /* NaN / Inf 入力に対するフェイルセーフ (メモリ保護違反・未定義動作を根絶) */
    if (x != x || x > 1e15 || x < -1e15) {
        return 0.0f;
    }
    x = fmod(x, SDR_TWO_PI_F);
    if (x < 0.0) {
        x += SDR_TWO_PI_F;
    }
    double f = x * ((double)SDR_LUT_SIZE / SDR_TWO_PI_F);
    int i = (int)f;
    if (i < 0) {
        i = 0;
    } else if (i >= SDR_LUT_SIZE) {
        i = SDR_LUT_SIZE - 1;
    }
    float fr = (float)(f - (double)i);
    if (fr < 0.0f) {
        fr = 0.0f;
    } else if (fr > 1.0f) {
        fr = 1.0f;
    }
    float a = g_sin_lut[i];
    return a + fr * (g_sin_lut[i + 1] - a);
}

static inline float sdr_cos(double x)
{
    return sdr_sin(x + 1.5707963267948966);
}

/* ---- SIMD実数FIR (valid畳み込み) ----
 * np.convolve (MSVC/NumPyのスカラー相関) をSSE2 4並列 + 4アキュムレータで置換。
 * y[i] = sum_k x[i+k]*h[k]  (n_out = len(x)-taps+1)
 */
SDR_EXPORT void sdr_fir_real(const float * __restrict x, const float * __restrict h,
                             float * __restrict y, int n_out, int taps)
{
    for (int i = 0; i < n_out; ++i) {
        const float *xp = x + i;
        float acc = 0.0f;
#ifdef SDR_HAVE_SSE2
        if (taps >= 8) {
            __m128 a0 = _mm_setzero_ps();
            __m128 a1 = _mm_setzero_ps();
            int k = 0;
            for (; k + 8 <= taps; k += 8) {
                a0 = _mm_add_ps(a0, _mm_mul_ps(_mm_loadu_ps(xp + k), _mm_loadu_ps(h + k)));
                a1 = _mm_add_ps(a1, _mm_mul_ps(_mm_loadu_ps(xp + k + 4), _mm_loadu_ps(h + k + 4)));
            }
            __m128 s = _mm_add_ps(a0, a1);
            s = _mm_add_ps(s, _mm_movehl_ps(s, s));
            s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
            acc = _mm_cvtss_f32(s);
            for (; k < taps; ++k) {
                acc += xp[k] * h[k];
            }
        } else
#endif
        {
            for (int k = 0; k < taps; ++k) {
                acc += xp[k] * h[k];
            }
        }
        y[i] = acc;
    }
}

/* ポリフェーズ間引きFIR (valid畳み込み [::decim] の 1/decim 計算量版)
 * y[j] = sum_k x[j*decim + k]*h[k], j=0..n_out-1
 * 呼出側は len(x) >= (n_out-1)*decim + taps を保証すること。
 * 畳み込み→間引きの定義そのものなので完全等価 (SSE加算順序差のみ)。
 */
SDR_EXPORT void sdr_polyphase_decim(const float * __restrict x, const float * __restrict h,
                                    float * __restrict y, int n_out, int taps, int decim)
{
    for (int j = 0; j < n_out; ++j) {
        const float *xp = x + (size_t)j * (size_t)decim;
        float acc = 0.0f;
#ifdef SDR_HAVE_SSE2
        if (taps >= 8) {
            __m128 a0 = _mm_setzero_ps();
            __m128 a1 = _mm_setzero_ps();
            int k = 0;
            for (; k + 8 <= taps; k += 8) {
                a0 = _mm_add_ps(a0, _mm_mul_ps(_mm_loadu_ps(xp + k), _mm_loadu_ps(h + k)));
                a1 = _mm_add_ps(a1, _mm_mul_ps(_mm_loadu_ps(xp + k + 4), _mm_loadu_ps(h + k + 4)));
            }
            __m128 s = _mm_add_ps(a0, a1);
            s = _mm_add_ps(s, _mm_movehl_ps(s, s));
            s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
            acc = _mm_cvtss_f32(s);
            for (; k < taps; ++k) {
                acc += xp[k] * h[k];
            }
        } else
#endif
        {
            for (int k = 0; k < taps; ++k) {
                acc += xp[k] * h[k];
            }
        }
        y[j] = acc;
    }
}

/* 19kHzパイロットPLL + RDS用57kHz(3θ)搬送波出力 (sdr_stereo_pllの拡張版)
 * cos3/sin3 = cos/sin(3*theta) を追加出力する。RDS復調は57kHzを3θから
 * 生成することでパイロットと完全にコヒーレントになる。
 */
SDR_EXPORT void sdr_stereo_pll3(const float *sig, int n, double *theta, double w0,
                                double kp, double ki, double *integ, double *ef_state,
                                double alpha, float *cos2, float *sin2,
                                float *cos3, float *sin3, float *quality)
{
    double th = *theta;
    double ig = *integ;
    double ef = *ef_state;
    double qsum = 0.0;
    for (int i = 0; i < n; ++i) {
        double s = (double)sdr_sin(th);
        double c = (double)sdr_cos(th);
        /* 搬送波は更新前位相から生成する。更新後から作ると1サンプル進み
         * (19kHz@288kHzで23.76°、38kHzで47.5°) の系統誤差になる。*/
        double c2 = (double)sdr_cos(2.0 * th);
        double s2 = (double)sdr_sin(2.0 * th);
        cos2[i] = (float)c2;
        sin2[i] = (float)s2;
        cos3[i] = (float)(c * c2 - s * s2);
        sin3[i] = (float)(s * c2 + c * s2);
        double e = -(double)sig[i] * s;
        ef += alpha * (e - ef);
        ig += ki * ef;
        th += w0 + kp * ef + ig;
        if (th > 3.141592653589793) {
            th -= 6.283185307179586;
        } else if (th < -3.141592653589793) {
            th += 6.283185307179586;
        }
        qsum += (double)sig[i] * c;
    }
    *theta = th;
    *integ = ig;
    *ef_state = ef;
    *quality = (float)(qsum / (n > 0 ? n : 1));
}

/* AM同期検波 (キャリア再生PLL + 同期検波)
 * - iq: 複素IFインタリーブ配列 (re,im,...) 288kHz想定、キャリアは0Hz付近
 * - out: 同相成分 (同期検波音声)
 * - phase/integ/ef_state: PLL状態
 * - lock: ロック指標 = |平均(同相)| / 平均(|IQ|)  (キャリア捕捉時 ~0.7-1.0)
 *
 * 包絡線検波と違い直交成分を捨てるため、選択性フェージング時の
 * ひずみ (AM成分) や隣接混信の影響を大幅に低減できる。
 */
SDR_EXPORT void sdr_am_sync(const float *iq, float *out, int n,
                            double *phase, double *integ, double *ef_state,
                            double kp, double ki, double alpha, float *lock)
{
    double th = *phase;
    double ig = *integ;
    double ef = *ef_state;
    double sum_i = 0.0;
    double sum_m = 0.0;
    for (int i = 0; i < n; ++i) {
        double re = (double)iq[2 * i];
        double im = (double)iq[2 * i + 1];
        double c = (double)sdr_cos(th);
        double s = (double)sdr_sin(th);
        double i_out = re * c + im * s;   /* 同相 = 同期検波出力 */
        double q = im * c - re * s;       /* 直交 = 位相誤差 */
        ef += alpha * (q - ef);
        ig += ki * ef;
        th += kp * ef + ig;
        if (th > 3.141592653589793) {
            th -= 6.283185307179586;
        } else if (th < -3.141592653589793) {
            th += 6.283185307179586;
        }
        out[i] = (float)i_out;
        sum_i += i_out;
        sum_m += sqrt(re * re + im * im);
    }
    *phase = th;
    *integ = ig;
    *ef_state = ef;
    double mi = fabs(sum_i) / (n > 0 ? n : 1);
    double mm = sum_m / (n > 0 ? n : 1) + 1e-12;
    *lock = (float)(mi / mm);
}

/* 双一次変換ディエンファシス (50us) - 1次IIR
 * y[n] = b0*x[n] + b1*x[n-1] + minus_a1*y[n-1]
 * state = {x1, y1}
 */
SDR_EXPORT void sdr_bilinear_deemphasis(const float *x, float *y, int n,
                                        float b0, float b1, float minus_a1,
                                        float *state)
{
    float x1 = state[0];
    float y1 = state[1];
    for (int i = 0; i < n; ++i) {
        float cx = x[i];
        float cy = b0 * cx + b1 * x1 + minus_a1 * y1;
        y[i] = cy;
        x1 = cx;
        y1 = cy;
    }
    state[0] = x1;
    state[1] = y1;
}

/* 1次ハイパス (DCカット / 300Hz音声用)
 * y[n] = x[n] - x[n-1] + r*y[n-1]
 */
SDR_EXPORT void sdr_one_pole_highpass(const float *x, float *y, int n,
                                      float r, float *state)
{
    float x1 = state[0];
    float y1 = state[1];
    for (int i = 0; i < n; ++i) {
        float cx = x[i];
        float cy = cx - x1 + r * y1;
        y[i] = cy;
        x1 = cx;
        y1 = cy;
    }
    state[0] = x1;
    state[1] = y1;
}

/* FM復調 (瞬時位相差分法) - 複素IQインタリーブ配列 (re,im,...)
 * demod[i] = atan2( imag(s[i]*conj(s[i-1])), real(...) )
 * last = {re, im} (前回最終サンプル)
 */
SDR_EXPORT void sdr_fm_demod(const float *iq, float *demod, int n, float *last)
{
    float li = last[0];
    float lq = last[1];
    for (int i = 0; i < n; ++i) {
        float ci = iq[2 * i];
        float cq = iq[2 * i + 1];
        /* s[i] * conj(s[i-1]) */
        float re = ci * li + cq * lq;
        float im = cq * li - ci * lq;
        demod[i] = atan2f(im, re);
        li = ci;
        lq = cq;
    }
    last[0] = li;
    last[1] = lq;
}

/* CMAブラインド等化器 (マルチパス・キャンセル用)。
 * 定包絡線(FM)信号の周波数選択性フェージングをパイロット不要で等化する。
 * - x: 入力複素IF (floatインタリーブ re,im,...)。history前置済み。
 * - y: 出力 (n_out)。y[j] は x[j..j+taps-1] からのフィルタ出力。
 * - w: タップ重み (complex64インタリーブ、呼出側で保持・継続適応)。
 * - mu: ステップ幅係数 (電力正規化NLMS型: 実効μ = mu/(電力+eps))。
 * 各サンプルで CMA誤差 e = y*(1-|y|^2) により w をLMS更新する。
 * 位相回転の不定性はFM復調 (差分/PLL) が吸収するため無害。
 */
SDR_EXPORT void sdr_cma_equalize(const float *x, float *y, int n_out,
                                 float *w, int taps, float mu)
{
    float leak = 0.99995f;
    int center = taps / 2;

    for (int j = 0; j < n_out; ++j) {
        const float *xp = x + 2 * j;
        float yr = 0.0f, yi = 0.0f, pwr = 0.0f;
        for (int k = 0; k < taps; ++k) {
            float xr = xp[2 * k];
            float xi = xp[2 * k + 1];
            float wr = w[2 * k];
            float wi = w[2 * k + 1];
            yr += xr * wr - xi * wi;
            yi += xr * wi + xi * wr;
            pwr += xr * xr + xi * xi;
        }

        /* 出力の有限性チェック (NaN/Infなら中央タップデルタへ即座に緊急リセット) */
        float m2 = yr * yr + yi * yi;
        if (m2 != m2 || m2 > 64.0f) {
            for (int k = 0; k < taps; ++k) {
                w[2 * k] = 0.0f;
                w[2 * k + 1] = 0.0f;
            }
            w[2 * center] = 1.0f;
            yr = xp[2 * center];
            yi = xp[2 * center + 1];
            m2 = yr * yr + yi * yi;
        }

        /* 誤差クリッピング: 大入力時の3次多項式爆発・発散を数学的に完全抑圧 */
        float g = 1.0f - m2;
        if (g < -2.0f) {
            g = -2.0f;
        } else if (g > 2.0f) {
            g = 2.0f;
        }

        float er = g * yr;
        float ei = g * yi;
        float step = mu / (pwr + 1e-4f);
        if (!(step > 0.0f)) {
            /* pwrがNaN/非正: 重み更新を停止 (NaN伝播防止)。出力のみ返す */
            y[2 * j]     = yr;
            y[2 * j + 1] = yi;
            continue;
        }
        if (step > 0.05f) {
            step = 0.05f;
        }

        /* Normalized Leaky-CMA更新 */
        for (int k = 0; k < taps; ++k) {
            float xr = xp[2 * k];
            float xi = xp[2 * k + 1];
            float nwr = leak * w[2 * k]     + step * (er * xr + ei * xi);
            float nwi = leak * w[2 * k + 1] + step * (ei * xr - er * xi);
            /* NaNは書き込まない (比較が常にfalseになるのを利用) */
            if (nwr != nwr || nwi != nwi) {
                continue;
            }
            /* 個別タップクリッピング (±Infもここで飽和) */
            if (nwr > 3.0f) nwr = 3.0f; else if (nwr < -3.0f) nwr = -3.0f;
            if (nwi > 3.0f) nwi = 3.0f; else if (nwi < -3.0f) nwi = -3.0f;
            w[2 * k]     = nwr;
            w[2 * k + 1] = nwi;
        }

        y[2 * j]     = yr;
        y[2 * j + 1] = yi;
    }
}

/* PLL周波数復調 (しきい値拡張型)。
 * 2次ループ+VCOで搬送波位相を追従し、瞬時角周波数 [rad/sample] を出力する。
 * angle差分法と同単位のためそのまま置換可能。入力はハードリミット済み (|x|=1)。
 * state = {th, fr} (double×2)。kp/kiはプローブ実測の勝ち値 (fn=25kHz, ζ=1.0)。
 * 狭帯域ループのためノイズ下でもクリックせず、CNR 6dBで約+18dBの改善を実測。
 */
SDR_EXPORT void sdr_pll_fm_demod(const float *iq, float *demod, int n,
                                 double *state, double kp, double ki)
{
    double th = state[0];
    double fr = state[1];
    for (int i = 0; i < n; ++i) {
        double re = (double)iq[2 * i];
        double im = (double)iq[2 * i + 1];
        double c = (double)sdr_cos(th);
        double s = (double)sdr_sin(th);
        double e = im * c - re * s;
        fr += ki * e;
        th += fr + kp * e;
        if (th > 3.141592653589793) {
            th -= 6.283185307179586;
        } else if (th < -3.141592653589793) {
            th += 6.283185307179586;
        }
        demod[i] = (float)(fr + kp * e);
    }
    state[0] = th;
    state[1] = fr;
}

/* 複素周波数ミキサー (in-place)
 * iq = iq * exp(j*phase), phase += phase_step
 */
SDR_EXPORT void sdr_mix_freq(float *iq, int n, double phase_step, double *phase_state)
{
    double ph = *phase_state;
    for (int i = 0; i < n; ++i) {
        float ci = iq[2 * i];
        float cq = iq[2 * i + 1];
        double c = cos(ph);
        double s = sin(ph);
        iq[2 * i] = (float)(ci * c - cq * s);
        iq[2 * i + 1] = (float)(ci * s + cq * c);
        ph += phase_step;
        if (ph > SDR_TWO_PI) {
            ph -= SDR_TWO_PI;
        } else if (ph < -SDR_TWO_PI) {
            ph += SDR_TWO_PI;
        }
    }
    *phase_state = ph;
}

/* FMステレオ MPX用 19kHzパイロットPLL
 * - pilot: 19kHz近傍に帯域制限されたパイロット信号 (288kHzレート想定)
 * - theta: PLL位相状態 (19kHz基準), w0 = 2*pi*19000/fs
 * - cos2: 出力 38kHz副搬送波 cos(2*theta)
 * - quality: 平均 pilot*cos(theta) (ロック時はパイロット振幅の約1/2)
 *
 * (L-R)復調: mpx * cos2 -> 15kHz LPF -> 2倍 で差信号が得られる。
 */
SDR_EXPORT void sdr_stereo_pll(const float *sig, int n, double *theta, double w0,
                               double kp, double ki, double *integ, double *ef_state,
                               double alpha, float *cos2, float *sin2, float *quality)
{
    double th = *theta;
    double ig = *integ;
    double ef = *ef_state;
    double qsum = 0.0;
    for (int i = 0; i < n; ++i) {
        double s = (double)sdr_sin(th);
        double c = (double)sdr_cos(th);
        /* 搬送波は更新前位相から生成 (1サンプル進み防止。pll3と同一理由) */
        cos2[i] = sdr_cos(2.0 * th);
        sin2[i] = sdr_sin(2.0 * th);
        /* 位相検波 (符号反転で負帰還) */
        double e = -(double)sig[i] * s;
        /* ループフィルタ: 音声成分(可聴帯域)を除去しパイロットのみで追従 */
        ef += alpha * (e - ef);
        ig += ki * ef;
        th += w0 + kp * ef + ig;
        if (th > 3.141592653589793) {
            th -= 6.283185307179586;
        } else if (th < -3.141592653589793) {
            th += 6.283185307179586;
        }
        qsum += (double)sig[i] * c;
    }
    *theta = th;
    *integ = ig;
    *ef_state = ef;
    *quality = (float)(qsum / (n > 0 ? n : 1));
}

/* クリック(孤立インパルス)除去 - コサインS字補間 (in-place)
 * Python版 suppress_click_transients と同一ロジック。
 * 戻り値: 補間修復したサンプル数
 */
SDR_EXPORT int sdr_suppress_clicks(float *x, int n, float threshold)
{
    if (n < 16) {
        return 0;
    }

    uint8_t *mask = (uint8_t *)calloc((size_t)n, 1);
    if (!mask) {
        return 0;
    }

    int any = 0;
    for (int b = 0; b < n - 1; ++b) {
        float diff = fabsf(x[b + 1] - x[b]);
        if (diff <= threshold) {
            continue;
        }
        int li = b - 1;
        if (li < 0) {
            li = 0;
        }
        int ri = b + 2;
        if (ri > n - 1) {
            ri = n - 1;
        }
        float local_span = fabsf(x[ri] - x[li]);
        /* 孤立条件を厳格化: 前後接続が戻っている突起のみ修復する。
         * 旧 `|| diff > 0.65` は打楽器アタック等の正規過渡を誤って削るため撤去。*/
        if (diff > local_span * 1.5f) {
            int s0 = b - 1;
            if (s0 < 0) {
                s0 = 0;
            }
            int e0 = b + 3;
            if (e0 > n) {
                e0 = n;
            }
            for (int k = s0; k < e0; ++k) {
                mask[k] = 1;
            }
            any = 1;
        }
    }

    if (!any) {
        free(mask);
        return 0;
    }

    int repaired = 0;
    for (int i = 0; i < n;) {
        if (!mask[i]) {
            ++i;
            continue;
        }
        int start = i;
        while (i < n && mask[i]) {
            ++i;
        }
        int end = i - 1;
        if (start > 0 && end < n - 1 && (end - start + 1) <= 4) {
            float v0 = x[start - 1];
            float v1 = x[end + 1];
            int L = (end + 1) - (start - 1);
            for (int k = start; k <= end; ++k) {
                float t = (float)(k - (start - 1)) / (float)L;
                float w = 0.5f * (1.0f - cosf(SDR_PI * t));
                x[k] = v0 + (v1 - v0) * w;
            }
            repaired += (end - start + 1);
        }
    }

    free(mask);
    return repaired;
}
