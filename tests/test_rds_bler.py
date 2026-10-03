"""RDS BLER instrumentation tests (synthetic, no hardware).

4 (RDS BLER) の回帰テスト:
1) クリーン復号では失敗0・bler 0.0 (既存動作の透明性)
2) 破損ビットでは失敗が数えられ bler が上がり、同期喪失が記録される
3) reset() で計装値がクリアされる
4) DSPパイプラインが bler 系属性を公開する (既定0)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from rds import RdsDecoder, make_block
from dsp import SdrDspPipeline


def _clean_group_bits(pi=0x1234):
    b2 = (0 << 12) | (0 << 11) | (1 << 10) | ((10 & 0x1F) << 5) | 0
    return (make_block(pi, "A") + make_block(b2, "B")
            + make_block(0x0000, "C") + make_block(0x4142, "D"))


def test_clean_bler_zero() -> bool:
    dec = RdsDecoder(12000.0)
    dec._sync = True
    for _ in range(6):
        dec._pending = list(_clean_group_bits())
        dec._decode_groups()
    ok = (dec.groups == 6 and dec.groups_checked == 6
          and dec.groups_failed == 0 and dec.sync_losses == 0
          and dec.bler == 0.0)
    print(f"[{'OK' if ok else 'FAIL'}] clean: groups={dec.groups} "
          f"checked={dec.groups_checked} failed={dec.groups_failed} "
          f"losses={dec.sync_losses} bler={dec.bler:.3f}")
    return ok


def test_corrupt_counts_failure() -> bool:
    dec = RdsDecoder(12000.0)
    dec._sync = True
    dec._pending = list(_clean_group_bits())
    dec._decode_groups()
    assert dec.groups == 1 and dec.bler == 0.0
    # 1ビット反転でCRC不正にする (同期喪失するはず)
    bad = list(_clean_group_bits())
    bad[10] ^= 1
    dec._pending = bad + list(_clean_group_bits())
    dec._decode_groups()
    ok = (dec.groups_failed >= 1 and dec.sync_losses >= 1
          and dec.bler > 0.0 and dec.groups_checked >= 2)
    print(f"[{'OK' if ok else 'FAIL'}] corrupt: checked={dec.groups_checked} "
          f"failed={dec.groups_failed} losses={dec.sync_losses} "
          f"bler={dec.bler:.3f}")
    # 全損が続くとblerは1.0へ漸近する (EMAの向き確認)
    dec2 = RdsDecoder(12000.0)
    dec2._sync = True
    for _ in range(30):
        bad = list(_clean_group_bits())
        bad[5] ^= 1
        dec2._pending = bad
        # 失敗後は_syncが外れるので立て直して次群を評価させる
        dec2._sync = True
        dec2._search_from = 0
        dec2._decode_groups()
    ok2 = dec2.bler > 0.9
    print(f"[{'OK' if ok2 else 'FAIL'}] bler saturates: bler={dec2.bler:.3f}")
    return ok and ok2


def test_reset_clears() -> bool:
    dec = RdsDecoder(12000.0)
    dec._sync = True
    bad = list(_clean_group_bits())
    bad[5] ^= 1
    dec._pending = bad
    dec._decode_groups()
    assert dec.groups_checked >= 1
    dec.reset()
    ok = (dec.groups_checked == 0 and dec.groups_failed == 0
          and dec.sync_losses == 0 and dec.sync_misses == 0
          and dec.bler == 0.0 and dec.groups == 0)
    print(f"[{'OK' if ok else 'FAIL'}] reset clears instrumentation")
    return ok


def test_pipeline_exposes() -> bool:
    dsp = SdrDspPipeline(1152000, 48000)
    ok = (dsp.rds_bler == 0.0 and dsp.rds_groups_checked == 0
          and dsp.rds_groups_failed == 0 and dsp.rds_sync_losses == 0)
    print(f"[{'OK' if ok else 'FAIL'}] pipeline exposes bler attrs "
          f"(bler={dsp.rds_bler})")
    return ok


def main() -> int:
    ok = test_clean_bler_zero()
    ok &= test_corrupt_counts_failure()
    ok &= test_reset_clears()
    ok &= test_pipeline_exposes()
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
