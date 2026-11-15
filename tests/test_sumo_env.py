"""Small checks for stage 2; the SUMO run is in smoke_test.py."""

from types import SimpleNamespace

import pytest

from sumo_env import SumoEnv, yellow_state


@pytest.mark.parametrize("name,actions,n_in,n_out", [
    ("cologne1", tuple(range(8)), 8, 8),
    ("ingolstadt1", (0, 2, 4), 7, 6),
])
def test_network_mapping_and_actions(name, actions, n_in, n_out):
    env = SumoEnv(name)
    assert env.actions == actions
    assert len(env.incoming_lanes) == n_in
    assert len(env.outgoing_lanes) == n_out
    for lanes in (env.incoming_lanes, env.outgoing_lanes):
        assert lanes == tuple(sorted(set(lanes)))
        assert all(not lane.startswith(":") for lane in lanes)
    assert all("y" not in state for state in env.green_states.values())
    assert all(lane.startswith(":" + env.tls_id + "_") for lane in env.internal_lanes)
    if name == "ingolstadt1":
        assert env.original_green_states[0] == "GGgGrGGG"
        assert env.green_states[0] == "GGrGrGGG"
        assert all(not lane.endswith("_0") for lane in env.incoming_lanes)
    else:
        assert env.original_green_states[0] == "srrrrrrrrrGGGGGrrrrr"
        assert env.green_states[0] == "rrrrrrrrrrGGGGGrrrrr"


def test_yellow_closes_every_open_movement_including_shared_and_yielding():
    assert yellow_state("GgsrrG") == "yyyrry"
    assert yellow_state("GGgGrGGG") == "yyyyryyy"
    assert yellow_state("srrrrrrrrrGGGGGrrrrr") == "yrrrrrrrrryyyyyrrrrr"


@pytest.mark.parametrize("name", ["cologne1", "ingolstadt1"])
def test_runtime_closes_yielding_movements_without_losing_any_link(name):
    env = SumoEnv(name)
    assert all(set(state) <= {"G", "r"} for state in env.green_states.values())
    assert all(not pairs for pairs in env.phase_foes.values())
    for index, _, _, _ in env.links:
        assert any(state[index] == "G" for state in env.green_states.values())


class ClockConnection:
    """Deterministic clock/queues: SUMO itself is checked separately."""

    def __init__(self, begin, queues):
        self.time = begin
        self.begin = begin
        self.queues = queues
        self.signal = ""
        self.simulation = SimpleNamespace(
            getTime=lambda: self.time,
            getCollisions=lambda: (),
            getStartingTeleportIDList=lambda: (),
        )
        self.trafficlight = SimpleNamespace(
            setRedYellowGreenState=self.set_signal,
            getRedYellowGreenState=lambda _: self.signal,
        )
        self.lane = SimpleNamespace(getLastStepHaltingNumber=self.queue)

    def set_signal(self, tls_id, signal):
        self.signal = signal

    def simulationStep(self):
        self.time += 1

    def queue(self, lane):
        if lane != "-32038056#3_0":
            return 0
        index = max(0, int(self.time - self.begin) - 1)
        return self.queues[min(index, len(self.queues) - 1)]


def clock_env(queues=(0,)):
    env = SumoEnv("cologne1")
    env.conn = ClockConnection(env.begin, queues)
    env.current_phase = 0
    env.conn.signal = env.green_states[0]
    env._junction_vehicles = lambda: ()
    return env


def test_same_action_no_transition_and_reward_uses_all_samples():
    env = clock_env((2, 4, 0))
    state, reward, done, info = env.step(0)
    assert reward == pytest.approx(-6 / 30)
    assert info["elapsed"] == 30
    assert info["durations"] == {"yellow": 0, "all_red": 0, "green": 30}
    assert state["phase"] == 0 and sum(state["incoming"]) == 0
    assert not done


def test_changed_action_reward_includes_yellow_red_and_green():
    env = clock_env((2,) * 8 + (4,) * 2 + (1,) * 30)
    state, reward, done, info = env.step(1)
    assert reward == pytest.approx(-54 / 40)
    assert info["durations"] == {"yellow": 8, "all_red": 2, "green": 30}
    assert state["phase"] == 1 and not done
    assert [s["signal"] for s in info["samples"][:8]] == ["rrrrrrrrrryyyyyrrrrr"] * 8
    assert all(s["signal"] == "r" * 20 for s in info["samples"][8:10])


def test_all_red_waits_for_clearance_instead_of_opening_after_two_seconds():
    env = clock_env()
    env._junction_vehicles = lambda: ("tail",) if env.conn.time < env.begin + 13 else ()
    _, _, _, info = env.step(1)
    assert info["durations"] == {"yellow": 8, "all_red": 5, "green": 30}


@pytest.mark.parametrize("end_offset,stage,phase", [
    (2, "yellow", None), (9, "all_red", None),
    (10, "all_red", None), (11, "green", 1),
])
def test_end_cuts_transition_or_green_without_advancing_past_end(end_offset, stage, phase):
    env = clock_env()
    env.end = env.begin + end_offset
    state, _, done, info = env.step(1)
    assert done and env.conn.time == env.end
    assert info["elapsed"] == end_offset
    assert info["samples"][-1]["stage"] == stage
    assert state["phase"] == phase
    with pytest.raises(RuntimeError):
        env.step(0)


def test_blocked_junction_stays_red_until_episode_end():
    env = clock_env()
    env.end = env.begin + 12
    env._junction_vehicles = lambda: ("blocked",)
    state, _, done, info = env.step(1)
    assert done and state["phase"] is None
    assert info["durations"]["green"] == 0
    assert info["durations"]["all_red"] == 4


def test_invalid_action_does_not_advance_simulation():
    env = clock_env()
    for action in (-1, 8, True, 1.0):
        with pytest.raises(ValueError):
            env.step(action)
    assert env.conn.time == env.begin
    env.conn = None
    with pytest.raises(RuntimeError):
        env.step(0)


def test_bus_tail_is_tracked_after_front_passes_short_outgoing_edge():
    env = SumoEnv("ingolstadt1")
    lane = ":cluster_274083968_cluster_1200364014_1200364088_8_0"
    position = 12.19  # 1 m before the last internal segment ends.
    distance = 100.0
    env.conn = SimpleNamespace(
        lane=SimpleNamespace(getLastStepVehicleIDs=lambda l: ("bus",) if l == lane else ()),
        vehicle=SimpleNamespace(
            getIDList=lambda: ("bus",),
            getLanePosition=lambda _: position,
            getLength=lambda _: 12.0,
            getDistance=lambda _: distance,
        ),
    )
    assert env._junction_vehicles() == ("bus",)
    lane = "uncontrolled_next_internal_lane"
    distance = 111.0  # Front travelled 11 m; rear still 2 m inside.
    assert env._junction_vehicles() == ("bus",)
    distance = 113.1
    assert env._junction_vehicles() == ()
