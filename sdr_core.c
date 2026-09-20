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

#ifdef _WIN32
#define SDR_EXPORT __declspec(dllexport)
#else
#define SDR_EXPORT
#endif

#define SDR_PI 3.14159265358979323846f
#define SDR_TWO_PI 6.28318530717958647692

SDR_EXPORT int sdr_version(void)
{
    return 2;
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
        double s = sin(th);
        double c = cos(th);
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
        cos2[i] = (float)cos(2.0 * th);
        sin2[i] = (float)sin(2.0 * th);
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
        if (diff > local_span * 1.5f || diff > 0.65f) {
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
