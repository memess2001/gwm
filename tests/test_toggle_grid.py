"""Toggle-grid ground truth, checked against an independent pure-Python reference simulator."""
import itertools

import numpy as np
import pytest

import zoo
from zoo.base import SPLIT_SEEDS
from zoo.toggle_grid import DOWN, LEFT, RIGHT, UP, ToggleGrid


class Ref:
    """Reference implementation written from the rules in the module docstring (char map + dicts)."""

    def __init__(self, k=1, rw=5, rh=5, switches=None):
        W, H = (k + 1) * (rw + 1) + 1, rh + 2
        self.g = [["#"] * W for _ in range(H)]
        for j in range(k + 1):
            for y in range(1, rh + 1):
                for x in range(j * (rw + 1) + 1, j * (rw + 1) + rw + 1):
                    self.g[y][x] = "."
        self.doors = {(i * (rw + 1), 1 + rh // 2): i - 1 for i in range(1, k + 1)}
        sw = switches or [((i - 1) * (rw + 1) + 2, 2) for i in range(1, k + 1)]
        self.switches = {tuple(s): i for i, s in enumerate(sw)}

    def step(self, s, a):
        x, y, bits = s[0], s[1], list(s[2:])
        nx, ny = x + (0, 0, -1, 1)[a], y + (-1, 1, 0, 0)[a]
        door = self.doors.get((nx, ny))
        entered = None
        if self.g[ny][nx] == "." or (door is not None and bits[door] == 1):
            x, y = nx, ny
            if (x, y) in self.switches:
                entered = self.switches[(x, y)]
                bits[entered] ^= 1
        return (x, y, *bits), entered

    def run(self, s, actions):
        states, entries = [tuple(int(v) for v in s)], []
        for a in actions:
            nxt, e = self.step(states[-1], int(a))
            states.append(nxt)
            entries.append(e)
        return states, entries


def walkable(w):
    return [(x, y) for y in range(w.H) for x in range(w.W) if w.free[y, x] or w.door_id[y, x] >= 0]


@pytest.mark.parametrize("k", [1, 2, 3])
def test_dynamics_match_reference_simulator(k):
    w, ref = ToggleGrid(n_doors=k), Ref(k)
    for policy in ("random", "loops"):
        tr = w.sample(k, 300, 20, "pos", policy=policy, init_doors="random", start="any")
        for i in range(20):
            states, entries = ref.run(tr.state[i, 0], tr.actions[i])
            assert np.array_equal(np.array(states), tr.state[i])
            assert [-1 if e is None else e for e in entries] == tr.info["switch_entered"][i].tolist()
    assert ToggleGrid(n_doors=k).ascii().count("D") == k


def test_holonomy_by_brute_force_simulation():
    rng = np.random.default_rng(0)
    for k in (1, 2, 3):
        w, ref = ToggleGrid(n_doors=k), Ref(k)
        for trial in range(150):
            s0 = w._initial(rng, 1, dict(w.cfg, start="any", init_doors="random"))[0]
            if trial % 2 == 0:
                acts = rng.integers(0, 4, 15)                                  # arbitrary (open) action sequence
            else:
                constraint = (None, True, False)[trial % 3]
                if constraint is False and w.switch_id[s0[1], s0[0]] >= 0:
                    constraint = None
                try:
                    acts = w.random_loop(s0, rng, length=(1, 12), switch=constraint)
                except ValueError:                                             # behind a closed door, no switch
                    assert constraint is True and w.room[s0[1], s0[0]] == k   # reachable: only in the last room
                    continue
            states, entries = ref.run(s0, acts)
            hol = w.holonomy(s0, acts)
            flips = np.array(states[-1][2:]) ^ np.array(states[0][2:])
            parity = np.array([sum(e == i for e in entries) % 2 for i in range(k)])
            assert np.array_equal(hol["flips"], flips) and np.array_equal(flips, parity)
            assert hol["position_closed"] == (states[-1][:2] == states[0][:2]) == hol["obs_closed"]
            if trial % 2:
                assert hol["position_closed"]


def test_hidden_observation_is_identical_for_all_door_bits():
    for k in (1, 3):
        hid, vis = ToggleGrid(n_doors=k), ToggleGrid(n_doors=k, door_visible=True)
        for x, y in walkable(hid):
            st = np.array([[x, y, *b] for b in itertools.product((0, 1), repeat=k)])
            for mode in ("grid", "pos"):
                o = hid.observe(st, mode)
                assert (o == o[:1]).all()                                  # exactly identical
                assert len(np.unique(vis.observe(st, mode).reshape(len(st), -1), axis=0)) == len(st)
        assert hid.ground_truth()["hidden_state_bits"] == k and vis.ground_truth()["hidden_state_bits"] == 0
        g = hid.observe(np.array([1, 1] + [0] * k), "grid")
        assert g.shape == hid.spec("grid")["obs_shape"] and g[3].sum() == k and g[2].sum() == k and g[1].sum() == 1


@pytest.mark.parametrize("k", [1, 2, 3])
def test_hidden_bits_by_room_by_reachability(k):
    """Breadth-first search over the true state graph from every start in R0 (any door bits)."""
    w, ref = ToggleGrid(n_doors=k), Ref(k)
    frontier = {(x, y, *b) for x, y in walkable(w) if w.room[y, x] == 0 for b in itertools.product((0, 1), repeat=k)}
    seen = set(frontier)
    while frontier:
        frontier = {ref.step(s, a)[0] for s in frontier for a in range(4)} - seen
        seen |= frontier
    for j, bits in enumerate(w.ground_truth()["hidden_bits_by_room"]):
        configs = {s[2:] for s in seen if w.room[s[1], s[0]] == j}
        assert len(configs) == 2 ** bits and all(all(c[:j]) for c in configs)   # D_1..D_j open in R_j


def test_doors_block_until_opened_from_the_switch_side():
    w = ToggleGrid()
    (dx, dy), (sx, sy) = w.doors[0], w.switches[0]
    left = np.array([dx - 1, dy, 0])
    assert np.array_equal(w.step(left, RIGHT), left)                         # closed door blocks
    assert np.array_equal(w.step([dx + 1, dy, 0], LEFT), [dx + 1, dy, 0])    # from both sides
    opened = np.array([dx - 1, dy, 1])
    assert np.array_equal(w.step(opened, RIGHT), [dx, dy, 1])                # open door cell is walkable
    assert np.array_equal(w.step([dx, dy, 1], RIGHT), [dx + 1, dy, 1])
    assert np.array_equal(w.step([dx, dy, 1], UP), [dx, dy, 1])              # walls around the door cell
    path = [UP] * 4 + [LEFT] * 6 + [DOWN] + [RIGHT]                          # wander into the switch, then back
    st = w.trajectory(left, path)
    assert st[-1, 2] == 1 and (st[:, :2] == (sx, sy)).all(1).sum() == 1
    assert w.ground_truth()["bottleneck_cells"] == w.doors == w.ground_truth()["door_cells"]


def test_toggle_only_on_entering_switch():
    w = ToggleGrid(switches=[(1, 1)])                                        # corner switch: blocked moves possible
    on = np.array([1, 1, 0])
    for a in (UP, LEFT):
        assert np.array_equal(w.step(on, a), on)                             # failed move on a switch: no toggle
    assert np.array_equal(w.step([2, 1, 0], LEFT), [1, 1, 1])                # entering toggles
    assert np.array_equal(w.trajectory([2, 1, 0], [LEFT, RIGHT, LEFT])[-1], [1, 1, 0])  # twice -> back
    with pytest.raises(ValueError):
        ToggleGrid(switches=[(5, 3)])                                        # adjacent to the door
    with pytest.raises(ValueError):
        ToggleGrid(switches=[(8, 2)])                                        # not in R0


@pytest.mark.parametrize("k", [1, 2, 3])
def test_plaquette_curvature_map(k):
    w, ref = ToggleGrid(n_doors=k), Ref(k)
    cm = w.curvature_map()
    P, par = cm["corners"], cm["parity"]
    assert len(P) == (k + 1) * (w.cfg["room_w"] - 1) * (w.cfg["room_h"] - 1)
    for (x, y), p in zip(P, par):
        corners = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]
        touch = np.array([tuple(s) in corners for s in w.switches])
        assert np.array_equal(p.astype(bool), touch)                          # non-zero iff touching a switch
        for bits in itertools.product((0, 1), repeat=k):                      # brute force: all door states,
            for r in range(4):                                                # all start corners, both orientations
                for loop in (cm["loop"], cm["reverse"]):
                    start = corners[r] if loop is cm["loop"] else corners[(4 - r) % 4]
                    acts = np.roll(loop, -r)
                    states, _ = ref.run((*start, *bits), acts)
                    assert states[-1][:2] == states[0][:2]
                    assert np.array_equal(np.array(states[-1][2:]) ^ np.array(bits), p)
    assert (par.any(1)).sum() == 4 * k                                        # 4 plaquettes around each switch
    assert (cm["map"] >= 0).sum() == len(P) * k


def test_random_loops_respect_switch_constraint():
    rng = np.random.default_rng(3)
    w, ref = ToggleGrid(n_doors=2), Ref(2)
    for _ in range(200):
        s0 = w._initial(rng, 1, dict(w.cfg, init_doors="random"))[0]
        if w.switch_id[s0[1], s0[0]] >= 0:
            continue
        for constraint in (None, True, False):
            acts = w.random_loop(s0, rng, switch=constraint)
            states, entries = ref.run(s0, acts)
            n_in = sum(e is not None for e in entries)
            assert states[-1][:2] == states[0][:2]
            if constraint is True:
                assert n_in >= 1
            if constraint is False:
                assert n_in == 0 and states[-1] == states[0]


def test_loops_policy_closes_loops():
    w = ToggleGrid(n_doors=2, policy="loops")
    tr = w.sample(5, 400, 16, "grid")
    assert tr.obs.shape == (16, 401) + w.spec("grid")["obs_shape"]
    for i in range(16):
        ends = np.flatnonzero(tr.info["loop_closed"][i])
        assert len(ends) >= 10 and ends[0] == 0
        for a, b in zip(ends[:-1], ends[1:]):                                 # every completed loop
            assert np.array_equal(tr.state[i, a, :2], tr.state[i, b, :2])
            hol = w.holonomy(tr.state[i, a], tr.actions[i, a:b])
            e = tr.info["switch_entered"][i, a:b]
            assert hol["position_closed"] and np.array_equal(hol["flips"], [(e == j).sum() % 2 for j in range(2)])
            assert (tr.info["loop_id"][i, a:b] == tr.info["loop_id"][i, a]).all()


def test_visible_world_obs_closed_loops_have_no_holonomy():
    rng = np.random.default_rng(4)
    hid, vis = ToggleGrid(), ToggleGrid(door_visible=True)
    n_flip = 0
    for _ in range(100):
        s0 = hid._initial(rng, 1, hid.cfg)[0]
        if hid.switch_id[s0[1], s0[0]] >= 0:
            continue
        acts = hid.random_loop(s0, rng, switch=True)
        h, v = hid.holonomy(s0, acts), vis.holonomy(s0, acts)
        assert np.array_equal(h["flips"], v["flips"]) and h["obs_closed"]
        assert v["obs_closed"] == (not v["flips"].any())                      # visible: closed in obs <=> no flip
        n_flip += int(v["flips"].any())
        st, ob = vis.rollout(s0, acts)
        assert np.array_equal(ob[0], ob[-1]) == bool(v["obs_closed"])
    assert n_flip > 10


def test_commutator_examples():
    w, ref = ToggleGrid(n_doors=2), Ref(2)
    ex = w.ground_truth()["commutator_examples"]
    assert sum(e["hidden_differs"] for e in ex) == 4 and not ex[-1]["hidden_differs"]
    for e in ex:
        ea, eb = ref.run(e["start"], e["seq_a"])[0][-1], ref.run(e["start"], e["seq_b"])[0][-1]
        assert e["same_position"] and ea[:2] == eb[:2]
        assert (ea[2:] != eb[2:]) == e["hidden_differs"]
        assert np.array_equal(e["end_a"], ea) and np.array_equal(e["end_b"], eb)


def test_seed_determinism_splits_and_registry():
    w = zoo.make("toggle_grid", n_doors=2)
    for policy in ("random", "loops"):
        a, b = w.sample(9, 50, 4, "grid", policy=policy), w.sample(9, 50, 4, "grid", policy=policy)
        assert np.array_equal(a.obs, b.obs) and np.array_equal(a.actions, b.actions)
    seqs = {s: w.sample(zoo.split_seed(s), 40, 32, "pos").actions for s in SPLIT_SEEDS}
    for s1, s2 in itertools.combinations(seqs, 2):
        assert not (seqs[s1][:, None] == seqs[s2][None]).all(-1).any()        # no shared trajectory
    assert w.sample(0, 5, 2, "pos", door_visible=True).obs.shape[-1] == 4
    assert w.sample(0, 5, 2, "pos", n_doors=3).state.shape[-1] == 5        # layout override rebuilds
    assert w.cfg["n_doors"] == 2
    with pytest.raises(TypeError):
        zoo.make("toggle_grid", doors=2)
