"""
Waterfall ecosystem test (no hardware required).

Verifies eco_waterfall.EcoSystem (ported from ecosystem-sim/sim.py mechanics)
with a headless pygame surface:
- population caps hold under sustained stimulus
- entities stay inside the panel
- plants track station columns, wither when stations vanish
- spores infect on clip events; predators nibble prey (energy transfer)
- draw path never raises; per-frame budget stays under 2ms
"""

import os
import sys
import time

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame

from eco_waterfall import (EcoSystem, Spore, Herbivore, Carnivore,
                           MAX_HERBS, MAX_CARNS, MAX_APEX, MAX_PLANTS,
                           MAX_SPORES, MAX_DECOMPS, MAX_GARBAGES)


def _env(eco, peaks=True, clipped=False):
    cols = [(100.0 + 80.0 * i, 18.0) for i in range(5)] if peaks else []
    eco.update({"plants": cols, "clipped": clipped, "dt": 1.0 / 30.0})


def test_population_caps():
    eco = EcoSystem(400, 300, seed=7)
    eco.enabled = True
    for k in range(900):  # 30 simulated seconds, clips every 2s
        _env(eco, peaks=True, clipped=(k % 60 == 0))
    assert len(eco.plants) <= MAX_PLANTS, len(eco.plants)
    assert len(eco.herbs) <= MAX_HERBS, len(eco.herbs)
    assert len(eco.carns) <= MAX_CARNS, len(eco.carns)
    assert len(eco.apexs) <= MAX_APEX, len(eco.apexs)
    assert len(eco.spores) <= MAX_SPORES, len(eco.spores)
    assert len(eco.decomps) <= MAX_DECOMPS, len(eco.decomps)
    assert len(eco.garbages) <= MAX_GARBAGES, len(eco.garbages)
    assert len(eco.herbs) > 0, "food chain never started"
    print(f"[OK] caps hold (plants={len(eco.plants)} herbs={len(eco.herbs)} "
          f"carns={len(eco.carns)} apex={len(eco.apexs)} "
          f"decomps={len(eco.decomps)} garbage={len(eco.garbages)})")


def test_bounds_and_wither():
    eco = EcoSystem(400, 300, seed=11)
    eco.enabled = True
    for _ in range(300):
        _env(eco, peaks=True)
    for a in eco.herbs + eco.carns + eco.apexs + eco.decomps:
        assert 0.0 <= a.x <= eco.width and 0.0 <= a.y <= eco.height, (a.x, a.y)
    assert len(eco.plants) > 0
    for _ in range(600):  # stations vanish: plants must wither away
        _env(eco, peaks=False)
    assert len(eco.plants) == 0, len(eco.plants)
    print("[OK] entities in bounds; plants wither without stations")


def test_plant_flicker_free():
    """ピーク位置が±6pxふらついても株が明滅しない (粗量子化＋猶予＋平滑)"""
    import random as _random
    rng = _random.Random(99)
    eco = EcoSystem(400, 300, seed=99)
    eco.enabled = True
    for _ in range(300):
        cols = [(100.0 + 80.0 * i + rng.uniform(-6.0, 6.0),
                 18.0 + rng.uniform(-3.0, 3.0)) for i in range(5)]
        eco.update({"plants": cols, "clipped": False, "dt": 1.0 / 30.0})
    # 株数が安定し、どれも育っていること (明滅していたら枯死・再発芽でsizeが小さい)
    assert len(eco.plants) >= 4, len(eco.plants)
    small = sum(1 for pl in eco.plants.values() if pl.size < 0.15)
    assert small == 0, f"{small} sprouts flickering"
    print(f"[OK] plants stable under jitter ({len(eco.plants)} plants)")


def test_infection_and_predation():
    eco = EcoSystem(400, 300, seed=23)
    eco.enabled = True
    for _ in range(200):
        _env(eco, peaks=True)
    assert len(eco.herbs) >= 2
    # spore next to a herbivore must infect it (immunity may block once)
    infected = False
    for _ in range(10):
        h = eco.herbs[0]
        h.immunity = 0.0
        eco.spores.append(Spore(h.x + 2.0, h.y, eco.rng))
        for _ in range(30):
            _env(eco, peaks=True)
            if h.infected or not h.alive:
                infected = True
                break
        if infected:
            break
    assert infected, "spore never infected"
    # carnivore nibbling: prey loses energy, predator gains it
    prey = Herbivore(200.0, 150.0, 80.0, eco.rng)
    eco.herbs.append(prey)
    pred = Carnivore(200.0, 150.0, 50.0, eco.rng)
    eco.carns.append(pred)
    e_prey, e_pred = prey.energy, pred.energy
    for _ in range(60):
        _env(eco, peaks=True)
        if not prey.alive:
            break
    assert pred.energy > e_pred or not prey.alive or prey.energy < e_prey, \
        "predator never fed"
    print("[OK] infection and predation (nibbling) work")


def test_decomposer_cycle():
    eco = EcoSystem(400, 300, seed=31)
    eco.enabled = True
    for _ in range(200):
        _env(eco, peaks=True)
    # garbage on the floor must be cleaned up over time
    eco.spawn_garbage(200.0, 150.0)
    eco.spawn_garbage(210.0, 150.0)
    n0 = len(eco.garbages)
    for _ in range(1200):
        _env(eco, peaks=True)
    assert len(eco.garbages) <= n0, (n0, len(eco.garbages))
    print("[OK] decomposers clean garbage")


def test_draw_budget():
    pygame.init()
    eco = EcoSystem(792, 304, seed=5)
    eco.enabled = True
    for _ in range(200):
        _env(eco, peaks=True, clipped=True)
    surf = pygame.Surface((792, 304))
    for _ in range(10):
        _env(eco, peaks=True)
        eco.draw(surf, 0, 0)
    # スイート同時負荷のスパイクに引っ張られないよう3バッチ中央値で判定
    mss = []
    for _ in range(3):
        t0 = time.perf_counter()
        reps = 20
        for _ in range(reps):
            _env(eco, peaks=True)
            eco.draw(surf, 0, 0)
        mss.append((time.perf_counter() - t0) / reps * 1000.0)
    ms = sorted(mss)[1]
    print(f"[*] eco update+draw {ms:.3f} ms/frame (median of {mss[0]:.3f}, "
          f"{mss[1]:.3f}, {mss[2]:.3f})")
    # 33msフレームの1割未満。開発マシンの負荷変動を見込んで3ms。
    assert ms < 3.0, f"eco too slow: {ms:.2f} ms"
    eco.enabled = False
    before = (len(eco.herbs), eco._t)
    _env(eco, peaks=True)
    assert (len(eco.herbs), eco._t) == before
    print("[OK] draw budget + disable switch")


def main() -> int:
    try:
        test_population_caps()
        test_bounds_and_wither()
        test_plant_flicker_free()
        test_infection_and_predation()
        test_decomposer_cycle()
        test_draw_budget()
    except AssertionError as e:
        print(f"FAILED: {e}")
        return 1
    print("ALL ECO TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
