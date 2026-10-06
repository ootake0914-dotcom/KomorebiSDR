"""
Waterfall ecosystem (電波の海の人工生物)。

https://github.com/ootake0914-dotcom/ecosystem-sim の sim.py を母体に、
KomorebiSDR のウォーターフォールへ移植したゼロプレイヤー ambient 生物群。
食物連鎖・エネルギー収支・群れ・恐怖・遺伝子継承は本家ロジックを踏襲し、
環境だけを電波に置き換えた:
- 植物プランクトン (緑): ランダム湧きではなく検出局・liveピークの柱に咲く。
  SNRが日光。局が消えれば枯れる。
- 草食動物 (シアン): 植物を食べ、群れ、肉食から逃げる。元気だと繁殖。
- 肉食動物 (ピンク): 草食を齧る (一撃ではなく吸血)。頂点から逃げる。
- 頂点捕食者 (金): 肉食が増えたときだけ現れ、肉食を狩る。
- 胞子 (紫): ADCクリップ・過大入力 (放射線嵐) と稀な自然発生。触れた
  草食を発病させる。免疫遺伝子で確率ブロック。
- ゴミ/死骸 (灰): 死ぬと残る。分解者が掃除する。
- 分解者 (ライム): ゴミと胞子を食べ、満腹になると最寄りの植物へ還元する。
- 遺伝子: speed / altruism / immunity を継承＋突然変異 (Red Queen)。
- ベテラン色: 長生き個体は色が変わる (生存の証。ESP32版の流儀)。

描画予算 2ms/frame 以内を目標とし、個体数に上限を設ける。
可読性のため信号マーカーの下に描く。eco_enabled=False で完全停止。
"""

import math
import random

import pygame
import pygame.gfxdraw as gfx

# 個体数上限 (小パネル用に本家より絞る。描画・物理の予算管理)
MAX_PLANTS = 8
MAX_HERBS = 10
MAX_CARNS = 2
MAX_APEX = 1
MAX_SPORES = 3
MAX_DECOMPS = 3
MAX_GARBAGES = 12

# 本家のエネルギー収支 (1step = 1frame @30fps として踏襲)
HERB_DRAIN_BASE = 0.01
HERB_REP_THRESH = 120.0
HERB_REP_COST = 50.0
HERB_EAT_GAIN = 30.0
CARN_DRAIN = 0.06
CARN_REP_THRESH = 150.0
CARN_REP_COST = 70.0
DECOMP_EAT_GAIN = 20.0
DECOMP_FEED_THRESH = 120.0
DECOMP_FEED_COST = 80.0
APEX_EAT_GAIN = 5.0

# ベテラン年齢 (秒)
VETERAN_AGE = 30.0


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def _dist2(a, b):
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2


class BaseEntity:
    __slots__ = ("x", "y", "alive")

    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.alive = True


class Plant(BaseEntity):
    """局の柱に咲く。size 0..1 (SNRが日光)。"""

    __slots__ = ("size", "target", "age", "unseen")

    def __init__(self, x, y):
        super().__init__(x, y)
        self.size = 0.1
        self.target = 0.5
        self.age = 0.0
        self.unseen = 0.0


class Garbage(BaseEntity):
    __slots__ = ()

    def __init__(self, x, y, rng):
        super().__init__(x + rng.uniform(-3.0, 3.0), y + rng.uniform(-3.0, 3.0))


class MovingEntity(BaseEntity):
    __slots__ = ("vx", "vy", "energy", "speed_limit", "age")

    def __init__(self, x, y, energy, speed_limit, rng):
        super().__init__(x, y)
        self.vx = rng.uniform(-1.0, 1.0)
        self.vy = rng.uniform(-1.0, 1.0)
        self.energy = float(energy)
        self.speed_limit = float(speed_limit)
        self.age = 0

    def apply_boundary(self, w, h):
        if self.x < 0:
            self.x, self.vx = 0.0, -self.vx
        elif self.x > w:
            self.x, self.vx = float(w), -self.vx
        if self.y < 0:
            self.y, self.vy = 0.0, -self.vy
        elif self.y > h:
            self.y, self.vy = float(h), -self.vy


class Spore(MovingEntity):
    __slots__ = ("ttl", "trail")

    def __init__(self, x, y, rng):
        super().__init__(x, y, 0.0, 0.0, rng)
        self.vx = rng.uniform(-0.1, 0.9)
        self.vy = rng.uniform(-0.5, 0.5)
        self.ttl = 20.0
        self.trail = []


class Herbivore(MovingEntity):
    __slots__ = ("infected", "altruism", "immunity", "trail")

    def __init__(self, x, y, energy, rng, speed_limit=-1.0,
                 infected=False, altruism=-1.0, immunity=-1.0):
        if speed_limit == -1.0:
            speed_limit = 0.8
        else:
            speed_limit = max(0.3, min(2.0, speed_limit + rng.uniform(-0.1, 0.1)))
        super().__init__(x, y, energy, speed_limit, rng)
        self.infected = bool(infected)
        if altruism == -1.0:
            self.altruism = rng.random()
        else:
            self.altruism = max(0.0, min(1.0, altruism + rng.uniform(-0.1, 0.1)))
        if immunity == -1.0:
            self.immunity = rng.random()
        else:
            self.immunity = max(0.0, min(1.0, immunity + rng.uniform(-0.1, 0.1)))
        self.trail = []


class Carnivore(MovingEntity):
    __slots__ = ("trail",)

    def __init__(self, x, y, energy, rng, speed_limit=-1.0):
        if speed_limit == -1.0:
            speed_limit = 1.1
        else:
            speed_limit = max(0.5, min(2.5, speed_limit + rng.uniform(-0.1, 0.1)))
        super().__init__(x, y, energy, speed_limit, rng)
        self.trail = []


class ApexPredator(MovingEntity):
    __slots__ = ("trail",)

    def __init__(self, x, y, energy, rng, speed_limit=-1.0):
        if speed_limit == -1.0:
            speed_limit = 1.5
        else:
            speed_limit = max(0.8, min(3.0, speed_limit + rng.uniform(-0.1, 0.1)))
        super().__init__(x, y, energy, speed_limit, rng)
        self.trail = []


class Decomposer(MovingEntity):
    __slots__ = ("trail",)

    def __init__(self, x, y, energy, rng):
        super().__init__(x, y, energy, 0.7, rng)
        self.trail = []


class EcoSystem:
    """ウォーターフォール生態系の保持・更新・描画。"""

    def __init__(self, width=792, height=304, seed=1234):
        self.width = max(64, int(width))
        self.height = max(64, int(height))
        self.rng = random.Random(seed)
        self.plants: dict = {}
        self.herbs: list = []
        self.carns: list = []
        self.apexs: list = []
        self.spores: list = []
        self.decomps: list = []
        self.garbages: list = []
        # 真面目SDRのため既定OFF。HI-FIバッジ隠しスイッチ or configでON。
        self.enabled = False
        self._t = 0.0
        # 捕食/死亡演出 (本家 main.py の流儀: フラッシュ＋パーティクル)
        self.particles: list = []
        self.flash: dict = {}
        # 加算グロー層 (本家のglow_layer。SRCALPHA面へ光を描き、
        # BLEND_RGBA_ADDで合成するネオン表現)
        self._glow = None
        self._glow_size = (0, 0)

    def resize(self, width, height):
        self.width = max(64, int(width))
        self.height = max(64, int(height))
        self._glow = None

    # ----------------------------------------------------------
    # 更新 (env: {"plants": [(x, snr)], "clipped": bool, "dt": sec})
    # 本家 World.step() の tick 駆動を dt 駆動へ読み替えたもの。
    # dt はほぼ 1/30 固定のため、収支定数は本家値をそのまま使う。
    # ----------------------------------------------------------
    def update(self, env=None):
        if not self.enabled:
            return
        env = env or {}
        dt = _clamp(float(env.get("dt", 1.0 / 30.0)), 0.001, 0.25)
        self._t += dt
        # 適応fpsで15fpsに落ちてもテンポを保つ (2stepまで)。
        # 本家 tick 駆動の等価を維持し、3step以上は重いので頭打ち。
        steps = 2 if dt > 0.05 else 1
        cols = env.get("plants") or []
        clipped = bool(env.get("clipped", False))
        for _ in range(steps):
            self._step(cols, clipped, dt / steps)

    def _step(self, cols, clipped, dt):
        for e in self.herbs + self.carns + self.apexs + self.decomps + self.spores:
            if e.alive:
                e.age += dt
        self._update_plants(cols, dt)
        # 自然発生の種 (本家: 最小数維持＋確率湧き)
        if len(self.herbs) < 3 and self.plants and self.rng.random() < 0.05:
            pl = self.rng.choice(list(self.plants.values()))
            self.herbs.append(Herbivore(
                pl.x + self.rng.uniform(-30.0, 30.0),
                pl.y + self.rng.uniform(-30.0, 30.0), 80.0, self.rng))
        if (len(self.carns) < 1 and len(self.herbs) >= 4
                and self.rng.random() < 0.02):
            self.carns.append(Carnivore(
                self.rng.uniform(10, self.width - 10),
                self.rng.uniform(10, self.height - 10), 100.0, self.rng))
        if (len(self.apexs) < 1 and len(self.carns) >= 2 and len(self.herbs) >= 5
                and self.rng.random() < 0.02):
            self.apexs.append(ApexPredator(
                self.rng.uniform(10, self.width - 10),
                self.rng.uniform(10, self.height - 10), 300.0, self.rng))
        if len(self.decomps) < 2 and self.rng.random() < 0.03:
            self.decomps.append(Decomposer(
                self.rng.uniform(10, self.width - 10),
                self.rng.uniform(10, self.height - 10), 80.0, self.rng))
        # 胞子: 放射線嵐 (クリップ) と稀な自然発生
        if clipped and len(self.spores) < MAX_SPORES:
            self.spores.append(Spore(
                self.rng.uniform(10, self.width - 10),
                self.rng.uniform(10, self.height - 10), self.rng))
        elif (not self.spores and self.herbs and self.rng.random() < 0.0007
                and len(self.spores) < MAX_SPORES):
            h = self.rng.choice(self.herbs)
            self.spores.append(Spore(h.x, h.y, self.rng))

        self._update_decomps(dt)
        self._update_spores(dt)
        self._update_herbs(dt)
        self._update_carns(dt)
        self._update_apexs(dt)
        # フラッシュ減衰・パーティクル統合
        for k in list(self.flash.keys()):
            v = self.flash[k] - dt * 3.0
            if v <= 0.0:
                del self.flash[k]
            else:
                self.flash[k] = v
        alive_parts = []
        for p in self.particles:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[2] *= 0.94
            p[3] *= 0.94
            p[4] -= dt * 2.2
            if p[4] > 0.0:
                alive_parts.append(p)
        self.particles = alive_parts

        self.herbs = [e for e in self.herbs if e.alive]
        self.carns = [e for e in self.carns if e.alive]
        self.apexs = [e for e in self.apexs if e.alive]
        self.spores = [e for e in self.spores if e.alive]
        self.decomps = [e for e in self.decomps if e.alive]
        self.garbages = [e for e in self.garbages if e.alive]

    def spawn_garbage(self, x, y):
        self.garbages.append(Garbage(x, y, self.rng))
        if len(self.garbages) > MAX_GARBAGES:
            self.garbages.pop(0)

    def spawn_particles(self, x, y, color, count=6, speed=22.0):
        for _ in range(count):
            if len(self.particles) >= 60:
                self.particles.pop(0)
            a = self.rng.uniform(0.0, 2.0 * math.pi)
            sp = self.rng.uniform(0.3, 1.0) * speed
            self.particles.append([float(x), float(y),
                                   math.cos(a) * sp, math.sin(a) * sp,
                                   1.0, color])

    def _die(self, e, color):
        e.alive = False
        self.spawn_garbage(e.x, e.y)
        self.spawn_particles(e.x, e.y, color, count=8, speed=30.0)

    def _update_plants(self, cols, dt):
        seen = set()
        for x, snr in cols[:MAX_PLANTS]:
            try:
                # ピーク位置は±数pxふらつくため粗い量子化にする。
                # 細かすぎると別株扱いで明滅する。
                key = int(round(float(x) / 16.0))
            except (TypeError, ValueError):
                continue
            seen.add(key)
            pl = self.plants.get(key)
            if pl is None:
                if len(self.plants) >= MAX_PLANTS:
                    continue
                pl = Plant(_clamp(float(x), 6.0, self.width - 6.0),
                           self.rng.uniform(self.height * 0.2, self.height * 0.8))
                self.plants[key] = pl
            else:
                pl.x += (_clamp(float(x), 6.0, self.width - 6.0) - pl.x) * min(1.0, dt * 2.0)
                pl.unseen = 0.0
            try:
                tgt = _clamp((float(snr) - 4.0) / 24.0, 0.05, 1.0)
            except (TypeError, ValueError):
                tgt = 0.2
            # 目標も平滑化する (番組の緩急で明滅させない)
            pl.target += (tgt - pl.target) * min(1.0, dt * 3.0)
        for key in list(self.plants.keys()):
            pl = self.plants[key]
            # 食べ尽くされた株は除去する (柱が見えていれば再発芽する。本家同様)
            if not pl.alive:
                del self.plants[key]
                continue
            pl.age += dt
            if key not in seen:
                # 見失っても2秒は枯らさない (フェージング・検出揺らぎの猶予)
                pl.unseen += dt
                if pl.unseen > 2.0:
                    pl.target = 0.0
            rate = 0.25 if pl.target > pl.size else 0.3
            pl.size += _clamp(pl.target - pl.size, -rate * dt, rate * dt)
            if pl.size <= 0.01 and pl.target <= 0.0:
                del self.plants[key]

    def _update_decomps(self, dt):
        for d in self.decomps:
            if not d.alive:
                continue
            min_d, target = 1e18, None
            for item in self.garbages + self.spores:
                if not item.alive:
                    continue
                d_sq = _dist2(d, item)
                if d_sq <= 1600.0 and d_sq < min_d:
                    min_d, target = d_sq, item
            if target is not None:
                mag = math.sqrt(min_d)
                if mag > 0:
                    d.vx = d.vx * 0.95 + ((target.x - d.x) / mag) * 0.1
                    d.vy = d.vy * 0.95 + ((target.y - d.y) / mag) * 0.1
                if min_d < 36.0:
                    target.alive = False
                    d.energy += DECOMP_EAT_GAIN
            else:
                d.vx += self.rng.uniform(-0.1, 0.1)
                d.vy += self.rng.uniform(-0.1, 0.1)
            speed = math.hypot(d.vx, d.vy)
            if speed > 0.7:
                d.vx, d.vy = d.vx / speed * 0.7, d.vy / speed * 0.7
            d.x += d.vx
            d.y += d.vy
            d.apply_boundary(self.width, self.height)
            self._push_trail(d)
            d.energy -= 0.05
            if d.energy <= 0:
                self._die(d, (150, 255, 50))
            elif d.energy > DECOMP_FEED_THRESH:
                # 満腹還元: 最寄り植物へ返す (局の柱にしか咲かない制約を守る)
                d.energy -= DECOMP_FEED_COST
                best, bd = None, 1e18
                for pl in self.plants.values():
                    dd = (pl.x - d.x) ** 2 + (pl.y - d.y) ** 2
                    if dd < bd:
                        bd, best = dd, pl
                if best is not None:
                    best.size = min(1.0, best.size + 0.3)
                    self.flash[id(d)] = 1.0
                    self.spawn_particles(best.x, best.y,
                                         (150, 255, 50), count=3, speed=12.0)

    def _update_spores(self, dt):
        for s in self.spores:
            if not s.alive:
                continue
            s.x += s.vx
            s.y += s.vy
            s.x %= self.width
            s.y %= self.height
            s.ttl -= dt
            self._push_trail(s)
            for h in self.herbs:
                if not isinstance(h, Herbivore):
                    continue
                if h.alive and not h.infected and _dist2(h, s) < 36.0:
                    s.alive = False
                    # 免疫で確率ブロック (Red Queen)
                    if self.rng.random() >= h.immunity:
                        h.infected = True
                        self.spawn_particles(h.x, h.y,
                                             (180, 0, 255), count=5, speed=18.0)
                    break
            if s.ttl <= 0.0:
                s.alive = False

    def _update_herbs(self, dt):
        for h in self.herbs:
            if not h.alive:
                continue
            # 採餌
            nearest_p, min_d = None, 1e18
            for p in self.plants.values():
                if not p.alive or p.size < 0.05:
                    continue
                d_sq = _dist2(h, p)
                if d_sq <= 1600.0 and d_sq < min_d:
                    min_d, nearest_p = d_sq, p
            if nearest_p is not None:
                mag = math.sqrt(min_d)
                if mag > 0 and not h.infected:
                    h.vx = h.vx * 0.97 + ((nearest_p.x - h.x) / mag) * 0.05
                    h.vy = h.vy * 0.97 + ((nearest_p.y - h.y) / mag) * 0.05
                if min_d < 25.0:
                    nearest_p.alive = False
                    h.energy += HERB_EAT_GAIN
                    self.flash[id(h)] = 1.0
                    self.spawn_particles(nearest_p.x, nearest_p.y,
                                         (150, 255, 150), count=3, speed=12.0)
            else:
                h.vx += self.rng.uniform(-0.1, 0.1)
                h.vy += self.rng.uniform(-0.1, 0.1)
            # 群れ (整列・結合・分離)＋利他分与
            ax, ay, cx, cy, n_flock = 0.0, 0.0, 0.0, 0.0, 0
            for o in self.herbs:
                if not isinstance(o, Herbivore) or o is h or not o.alive:
                    continue
                dx, dy = o.x - h.x, o.y - h.y
                d_sq = dx * dx + dy * dy
                if d_sq < 1600.0 and not h.infected and not o.infected:
                    ax += o.vx
                    ay += o.vy
                    cx += o.x
                    cy += o.y
                    n_flock += 1
                    if 0.0 < d_sq < 3.0:
                        h.vx -= (dx / d_sq) * 2.0
                        h.vy -= (dy / d_sq) * 2.0
                if (d_sq < 400.0 and not h.infected and not o.infected
                        and h.energy > 60.0 and o.energy < 30.0
                        and self.rng.random() < h.altruism):
                    h.energy -= 1.0
                    o.energy += 1.0
            if n_flock > 0 and not h.infected:
                ax /= n_flock
                ay /= n_flock
                am = math.hypot(ax, ay)
                if am > 0:
                    h.vx += (ax / am) * 0.04
                    h.vy += (ay / am) * 0.04
                cx, cy = cx / n_flock - h.x, cy / n_flock - h.y
                cm = math.hypot(cx, cy)
                if cm > 0:
                    h.vx += (cx / cm) * 0.015
                    h.vy += (cy / cm) * 0.015
            # 恐怖 (肉食・頂点から逃げる)
            for c in self.carns:
                if not c.alive:
                    continue
                dx, dy = h.x - c.x, h.y - c.y
                d_sq = dx * dx + dy * dy
                if d_sq < 4000.0:
                    mag = math.sqrt(d_sq)
                    if mag > 0:
                        h.vx += (dx / mag) * 0.1
                        h.vy += (dy / mag) * 0.1
            for a in self.apexs:
                if not a.alive:
                    continue
                dx, dy = h.x - a.x, h.y - a.y
                d_sq = dx * dx + dy * dy
                if d_sq < 6000.0:
                    mag = math.sqrt(d_sq)
                    if mag > 0:
                        h.vx += (dx / mag) * 0.12
                        h.vy += (dy / mag) * 0.12
            # 発病 (利他個体は自己隔離、他は錯乱)
            if h.infected:
                if h.altruism > 0.6:
                    ex = -1.0 if h.x < self.width / 2 else 1.0
                    ey = -1.0 if h.y < self.height / 2 else 1.0
                    h.vx = h.vx * 0.9 + ex * 0.1
                    h.vy = h.vy * 0.9 + ey * 0.1
                else:
                    h.vx += self.rng.uniform(-0.5, 0.5)
                    h.vy += self.rng.uniform(-0.5, 0.5)
                h.energy -= 0.15
            speed = math.hypot(h.vx, h.vy)
            limit = h.speed_limit + (0.4 if h.infected else 0.0)
            if speed > limit:
                h.vx, h.vy = h.vx / speed * limit, h.vy / speed * limit
            h.x += h.vx
            h.y += h.vy
            h.apply_boundary(self.width, self.height)
            self._push_trail(h)
            # 代謝 (高速・高免疫は燃費が悪い)
            h.energy -= (0.01 + 0.03 * h.speed_limit + 0.02 * h.immunity)
            if h.energy <= 0:
                self._die(h, (180, 0, 255) if h.infected else (0, 255, 255))
                if h.infected and len(self.spores) < MAX_SPORES - 1:
                    self.spores.append(Spore(h.x, h.y, self.rng))
                    self.spores.append(Spore(h.x, h.y, self.rng))
            elif (h.energy > HERB_REP_THRESH and not h.infected
                    and len(self.herbs) < MAX_HERBS):
                h.energy -= HERB_REP_COST
                self.herbs.append(Herbivore(
                    h.x, h.y, 80.0, self.rng, h.speed_limit,
                    False, h.altruism, h.immunity))

    def _update_carns(self, dt):
        for c in self.carns:
            if not c.alive:
                continue
            nearest_h, min_d = None, 10000.0
            for h in self.herbs:
                if not h.alive:
                    continue
                d_sq = _dist2(c, h)
                if d_sq < min_d:
                    min_d, nearest_h = d_sq, h
            if nearest_h is not None:
                mag = math.sqrt(min_d)
                if mag > 0:
                    c.vx = c.vx * 0.96 + ((nearest_h.x - c.x) / mag) * 0.08
                    c.vy = c.vy * 0.96 + ((nearest_h.y - c.y) / mag) * 0.08
                if min_d < 36.0:
                    # 齧りつき (吸血＋減速)。一撃死ではない
                    nearest_h.energy -= 2.5
                    c.energy += 2.5
                    c.vx *= 0.5
                    c.vy *= 0.5
                    nearest_h.vx *= 0.2
                    nearest_h.vy *= 0.2
                    self.flash[id(c)] = 1.0
                    if nearest_h.energy <= 0:
                        self._die(nearest_h, (0, 255, 255))
            else:
                c.vx += self.rng.uniform(-0.1, 0.1)
                c.vy += self.rng.uniform(-0.1, 0.1)
            # 頂点からのアドレナリン逃避
            escaping = False
            for a in self.apexs:
                if not a.alive:
                    continue
                dx, dy = c.x - a.x, c.y - a.y
                d_sq = dx * dx + dy * dy
                if d_sq < 8000.0:
                    escaping = True
                    mag = math.sqrt(d_sq)
                    if mag > 0:
                        c.vx += (dx / mag) * 0.3
                        c.vy += (dy / mag) * 0.3
            speed = math.hypot(c.vx, c.vy)
            limit = c.speed_limit + (0.6 if escaping else 0.0)
            if speed > limit:
                c.vx, c.vy = c.vx / speed * limit, c.vy / speed * limit
            c.x += c.vx
            c.y += c.vy
            c.apply_boundary(self.width, self.height)
            self._push_trail(c)
            c.energy -= CARN_DRAIN * (c.speed_limit / 1.1)
            if c.energy <= 0:
                self._die(c, (255, 50, 150))
            elif c.energy > CARN_REP_THRESH and len(self.carns) < MAX_CARNS:
                c.energy -= CARN_REP_COST
                self.carns.append(Carnivore(c.x, c.y, 100.0, self.rng, c.speed_limit))

    def _update_apexs(self, dt):
        for a in self.apexs:
            if not a.alive:
                continue
            nearest_c, min_d = None, 14400.0
            for c in self.carns:
                if not c.alive:
                    continue
                d_sq = _dist2(a, c)
                if d_sq < min_d:
                    min_d, nearest_c = d_sq, c
            if nearest_c is not None:
                mag = math.sqrt(min_d)
                if mag > 0:
                    a.vx = a.vx * 0.98 + ((nearest_c.x - a.x) / mag) * 0.12
                    a.vy = a.vy * 0.98 + ((nearest_c.y - a.y) / mag) * 0.12
                if min_d < 64.0:
                    nearest_c.energy -= APEX_EAT_GAIN
                    a.energy += APEX_EAT_GAIN
                    a.vx *= 0.6
                    a.vy *= 0.6
                    nearest_c.vx *= 0.1
                    nearest_c.vy *= 0.1
                    if nearest_c.energy <= 0:
                        self._die(nearest_c, (255, 50, 150))
            else:
                a.vx += self.rng.uniform(-0.1, 0.1)
                a.vy += self.rng.uniform(-0.1, 0.1)
            speed = math.hypot(a.vx, a.vy)
            if speed > a.speed_limit:
                a.vx, a.vy = a.vx / speed * a.speed_limit, a.vy / speed * a.speed_limit
            a.x += a.vx
            a.y += a.vy
            a.apply_boundary(self.width, self.height)
            self._push_trail(a)
            a.energy -= 0.15 * (a.speed_limit / 1.5)
            if a.energy <= 0:
                self._die(a, (255, 215, 0))

    def _push_trail(self, e, cap=28):
        tr = getattr(e, "trail", None)
        if tr is None:
            return
        # ワープ (胞子の端回り込み) では線を引かない (本家と同様に履歴破棄)
        if tr:
            lx, ly = tr[-1]
            if (e.x - lx) ** 2 + (e.y - ly) ** 2 > 1600.0:
                del tr[:]
        tr.append((e.x, e.y))
        if len(tr) > cap:
            del tr[0]

    # ----------------------------------------------------------
    # 描画 (本家 main.py の語彙: グロー・うねり尻尾・回転花弁・
    # 進行方向コア・パーティクル・フラッシュ・ベテラン色)。
    # 加算レイヤは使わず、同心円の濃淡でグローを偽装する (高速化)。
    # ----------------------------------------------------------
    @staticmethod
    def _lerp(c1, c2, t):
        return (int(c1[0] + (c2[0] - c1[0]) * t),
                int(c1[1] + (c2[1] - c1[1]) * t),
                int(c1[2] + (c2[2] - c1[2]) * t))

    def _vet_t(self, age):
        return _clamp((age - 10.0) / 20.0, 0.0, 1.0)

    def _glow_dot(self, glow, x, y, r, col, alpha=40):
        """加算グロー層への発光 (本家のglow_layerと同義)。
        重なり飽和で白飛びするため、半径・濃度は控えめにする。"""
        try:
            gfx.filled_circle(glow, x, y, int(r * 1.7),
                              (col[0], col[1], col[2], alpha))
        except Exception:
            pass

    def _body(self, screen, glow, ex, ey, ox, oy, r, col, vx=0.0, vy=0.0,
              fur=0, seed=0.0, fl=0.0):
        x, y = ox + int(ex), oy + int(ey)
        lx, ly = int(ex), int(ey)
        # グローは素の色で (フラッシュで白ハローにしない。本家と同一)
        self._glow_dot(glow, lx, ly, r, col)
        if fl > 0.0:
            col = self._lerp(col, (255, 255, 255), min(1.0, fl))
        try:
            gfx.filled_circle(screen, x, y, r, col)
        except Exception:
            pygame.draw.circle(screen, col, (x, y), r)
        # 毛並み (放射ストランド＋揺らぎ。物理シェーダではなく
        # 小サイズ用の様式化モフ。seedで個体ごとに毛束を固定する)
        if fur > 0:
            dark = (col[0] // 2, col[1] // 2, col[2] // 2)
            lean = math.atan2(vy, vx) if vx or vy else 0.0
            for k in range(fur):
                h1 = math.sin(k * 12.9898 + seed * 78.233) * 43758.5453
                h1 -= math.floor(h1)
                h2 = math.sin(k * 39.425 + seed * 11.123) * 24634.6345
                h2 -= math.floor(h2)
                ang = lean + (k / fur) * 2.0 * math.pi + (h1 - 0.5) * 0.9
                ang += math.sin(self._t * 3.0 + k * 1.3 + seed) * 0.12
                ln = r + 1.5 + h2 * 2.5
                x1 = x + int(math.cos(ang) * (r - 1))
                y1 = y + int(math.sin(ang) * (r - 1))
                x2 = x + int(math.cos(ang) * ln)
                y2 = y + int(math.sin(ang) * ln)
                try:
                    pygame.draw.line(screen, dark, (x1, y1), (x2, y2), 1)
                except Exception:
                    pass
        # 進行方向コア (白いハイライト)
        sp = math.hypot(vx, vy)
        cr = max(1, int(r * 0.4))
        if sp > 1e-6:
            cx = x + int(vx / sp * 1.5)
            cy = y + int(vy / sp * 1.5)
        else:
            cx, cy = x, y
        pygame.draw.circle(screen, (255, 255, 255), (cx, cy), cr)

    def _draw_trail(self, screen, ox, oy, e, col, r_body=3):
        """うねり＋先細り尻尾 (本家のdrawWedgeLine風)。
        履歴28点のうち直近16点を、補間で滑らかに・太く・明るめに描く。
        遅い個体でも尻尾が見えるよう、フェードは線形寄り (t^0.7) にする。
        太さは体サイズに比例させる (頂点ほど太い尾)。"""
        trail = e.trail
        if len(trail) < 3:
            return
        pts = trail[-16:]
        n = len(pts)
        vx, vy = getattr(e, "vx", 0.0), getattr(e, "vy", 0.0)
        sp = math.hypot(vx, vy)
        px_ang = 0.0
        if sp > 0.05:
            px_ang = math.atan2(vy, vx) + math.pi / 2
        # 隣点間を補間して点線化を防ぐ (本家のdense_pts)
        dense = []
        for i in range(n - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            dist = math.hypot(x2 - x1, y2 - y1)
            steps = max(1, min(6, int(dist / 1.5) + 1))
            for j in range(steps):
                t = j / steps
                dense.append((x1 + (x2 - x1) * t, y1 + (y2 - y1) * t))
        dense.append(pts[-1])
        m = len(dense)
        wscale = max(0.5, r_body / 3.0)
        for i in range(m - 1):
            t = i / max(1, m - 1)
            w = max(1, int((1 + 3 * t) * wscale))
            fade = self._lerp((10, 18, 34), col, t ** 0.7)
            x1, y1 = dense[i]
            x2, y2 = dense[i + 1]
            if sp > 0.05:
                wave = math.sin(self._t * 9.0 - t * 5.0) * 2.0 * (1.0 - t)
                x1 += math.cos(px_ang) * wave
                y1 += math.sin(px_ang) * wave
                x2 += math.cos(px_ang) * wave
                y2 += math.sin(px_ang) * wave
            try:
                pygame.draw.line(screen, fade,
                                 (ox + int(x1), oy + int(y1)),
                                 (ox + int(x2), oy + int(y2)), w)
            except Exception:
                pass

    def draw(self, screen, ox, oy):
        if not self.enabled:
            return
        # 加算グロー層の確保 (パネル同寸・透過クリア)
        if self._glow is None or self._glow_size != (self.width, self.height):
            self._glow = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            self._glow_size = (self.width, self.height)
        glow = self._glow
        glow.fill((0, 0, 0, 0))
        idx = 0
        for pl in self.plants.values():
            # 放射状の花 (中心で交差させない。芯＋5弁＋脈動で花に見せる)
            pulse = (math.sin(self._t * 2.0 + idx) + 1.0) / 2.0
            idx += 1
            r = 3.0 + pulse * 1.5 + 2.5 * pl.size
            cx, cy = ox + int(pl.x), oy + int(pl.y)
            lx, ly = int(pl.x), int(pl.y)
            leaf = (120, 255, 160) if pl.age >= VETERAN_AGE else (150, 255, 150)
            self._glow_dot(glow, lx, ly, r, leaf, alpha=60)
            rot = self._t * 0.9 + idx * 2.4
            for k in range(5):
                ang = rot + k * (2.0 * math.pi / 5.0)
                L0 = 3.0
                L1 = L0 + r * (0.9 + 0.3 * math.sin(self._t * 3.0 + k * 2.1 + idx))
                x0, y0 = int(cx + math.cos(ang) * L0), int(cy + math.sin(ang) * L0)
                x1, y1 = int(cx + math.cos(ang) * L1), int(cy + math.sin(ang) * L1)
                pygame.draw.line(screen, leaf, (x0, y0), (x1, y1), 1)
                pygame.draw.circle(screen, leaf, (x1, y1), 1)
            pygame.draw.circle(screen, (235, 255, 235), (cx, cy), 2)
        for gb in self.garbages:
            # 死骸は小さめ・暗めの× (背景に溶かし、主役にしない)
            x, y = ox + int(gb.x), oy + int(gb.y)
            col = (150, 85, 85)
            pygame.draw.line(screen, col, (x - 3, y - 3), (x + 3, y + 3), 1)
            pygame.draw.line(screen, col, (x + 3, y - 3), (x - 3, y + 3), 1)
        for s in self.spores:
            pulse = (math.sin(self._t * 5.0 + s.x * 0.1) + 1.0) / 2.0
            r = 2 + int(pulse * 2.0)
            self._draw_trail(screen, ox, oy, s, (150, 90, 220), 3)
            self._glow_dot(glow, int(s.x), int(s.y), r, (255, 100, 255))
            pygame.draw.circle(screen, (255, 130, 255),
                               (ox + int(s.x), oy + int(s.y)), r)
        for h in self.herbs:
            if h.infected:
                col = (180, 0, 255)
            elif h.age >= VETERAN_AGE:
                col = self._lerp((0, 255, 255), (50, 255, 50),
                                 self._vet_t(h.age))
            else:
                col = (0, 255, 255)
            fl = self.flash.get(id(h), 0.0)
            self._draw_trail(screen, ox, oy, h, col, 3)
            self._body(screen, glow, h.x, h.y, ox, oy, 3, col,
                       h.vx, h.vy, fur=16, seed=float(id(h) % 1024), fl=fl)
        for c in self.carns:
            if c.age >= VETERAN_AGE:
                col = self._lerp((255, 50, 150), (255, 0, 0),
                                 self._vet_t(c.age))
            else:
                col = (255, 50, 150)
            fl = self.flash.get(id(c), 0.0)
            self._draw_trail(screen, ox, oy, c, col, 4)
            self._body(screen, glow, c.x, c.y, ox, oy, 4, col,
                       c.vx, c.vy, fur=18, seed=float(id(c) % 1024), fl=fl)
        for a in self.apexs:
            if a.age >= VETERAN_AGE:
                col = self._lerp((255, 215, 0), (255, 255, 200),
                                 self._vet_t(a.age))
            else:
                col = (255, 215, 0)
            fl = self.flash.get(id(a), 0.0)
            self._draw_trail(screen, ox, oy, a, col, 5)
            self._body(screen, glow, a.x, a.y, ox, oy, 5, col,
                       a.vx, a.vy, fur=22, seed=float(id(a) % 1024), fl=fl)
        for d in self.decomps:
            col = (150, 255, 50)
            fl = self.flash.get(id(d), 0.0)
            self._draw_trail(screen, ox, oy, d, col, 2)
            self._body(screen, glow, d.x, d.y, ox, oy, 2, col,
                       d.vx, d.vy, fur=10, seed=float(id(d) % 1024), fl=fl)
        for p in self.particles:
            a = _clamp(p[4], 0.0, 1.0)
            try:
                gfx.filled_circle(glow, int(p[0]), int(p[1]), 2,
                                  (p[5][0], p[5][1], p[5][2], int(255 * a)))
            except Exception:
                pass
        # 加算合成 (本家と同一フラグ)
        try:
            screen.blit(glow, (ox, oy), special_flags=pygame.BLEND_RGBA_ADD)
        except Exception:
            screen.blit(glow, (ox, oy))
