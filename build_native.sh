#!/bin/sh
# KomorebiSDR native core build script (Linux / macOS).
# Usage: sh build_native.sh
# Output: ./sdr_core.so (Linux) or ./sdr_core.dylib is NOT built here on
# macOS -- see comment below. dsp_native.py loads .dll/.so/.dylib in order.
set -eu
cd "$(dirname "$0")"
CC="${CC:-cc}"
OUT="sdr_core.so"
case "$(uname -s)" in
    Darwin)
        # macOS: clang with dynamiclib (unverified -- reports welcome)
        exec $CC -O2 -fPIC -dynamiclib -o sdr_core.dylib sdr_core.c -lm
        ;;
    MINGW*|MSYS*|CYGWIN*)
        echo "[ERROR] On Windows use build_native.bat (MSVC) instead." >&2
        exit 1
        ;;
esac
$CC -O2 -fPIC -shared -o "$OUT" sdr_core.c -lm
echo "[OK] $OUT built."
