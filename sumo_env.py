"""Shared SUMO environment, stage-2 prototype (not a safety certification).

Action IDs are the ORIGINAL green-phase indices: 0..7 / 0,2,4.
G=priority green, g=yielding green, s=stop then yield, y=yellow, r=red.
See reports/stage2/README.md for timing, sampling and safety limitations.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import sumolib
import traci


ROOT = Path(__file__).resolve().parent


def yellow_state(green):
    """Close all open movements; s->y temporarily replaces its stop rule.

    This provisional transition also closes shared links before G/g changes.
    """
    return "".join("y" if light in "Ggs" else "r" for light in green)


class SumoEnv:
    def __init__(self, map_name, output_dir=None):
        if map_name not in ("cologne1", "ingolstadt1"):
            raise ValueError("map_name must be cologne1 or ingolstadt1")
        self.map_name = map_name
        self.config = ROOT / "data" / map_name / f"{map_name}.sumocfg"
        config = ET.parse(self.config).getroot()
        self.begin = float(config.find("time/begin").get("value"))
        self.end = float(config.find("time/end").get("value"))
        net_path = self.config.parent / config.find("input/net-file").get("value")
        net_xml = ET.parse(net_path).getroot()
        tls, = net_xml.findall("tlLogic")
        self.tls_id = tls.get("id")
        self.green_states = {
            i: phase.get("state") for i, phase in enumerate(tls.findall("phase"))
            if "y" not in phase.get("state") and "G" in phase.get("state")
        }
        self.actions = tuple(self.green_states)
        self.links = tuple(
            (int(c.get("linkIndex")),
             f'{c.get("from")}_{c.get("fromLane")}',
             f'{c.get("to")}_{c.get("toLane")}', c.get("via"))
            for c in net_xml.findall("connection") if c.get("tl") == self.tls_id
        )
        net = sumolib.net.readNet(str(net_path), withInternal=True)
        for index, incoming, outgoing, via in self.links:
            connection, = (
                c for c in net.getLane(incoming).getOutgoing()
                if c.getToLane().getID() == outgoing and c.getViaLaneID() == via
            )
            if connection.getJunctionIndex() != index:
                raise ValueError("TLS index does not match junction request index")
        for _, incoming, outgoing, _ in self.links:
            for lane_id in (incoming, outgoing):
                lane = net.getLane(lane_id)
                if lane_id.startswith(":") or not any(lane.allows(c) for c in ("passenger", "bus")):
                    raise ValueError(f"Not a controlled motor-vehicle boundary lane: {lane_id}")
        self.incoming_lanes = tuple(sorted({link[1] for link in self.links}))
        self.outgoing_lanes = tuple(sorted({link[2] for link in self.links}))
        # Follow only this TLS's internal paths, including second yielding segments.
        self.distance_to_exit = {}
        for _, _, _, via in self.links:
            path = []
            lane = net.getLane(via)
            while lane.getEdge().getFunction() == "internal":
                if lane in path:
                    raise ValueError("Cycle in internal connection path")
                path.append(lane)
                connection, = lane.getOutgoing()
                lane = net.getLane(connection.getViaLaneID()) if connection.getViaLaneID() else connection.getToLane()
            remaining = 0
            for lane in reversed(path):
                remaining += lane.getLength()
                self.distance_to_exit[lane.getID()] = remaining
        self.internal_lanes = tuple(sorted(self.distance_to_exit))
        junction = next(j for j in net_xml.findall("junction") if j.get("id") == self.tls_id)
        requests = {int(r.get("index")): r for r in junction.findall("request")}
        count = len(requests)
        if {link[0] for link in self.links} != set(range(count)):
            raise ValueError("Incomplete signal/request mapping")
        self.phase_foes = {}
        for phase, state in self.green_states.items():
            if len(state) != count or set(state) - set("Ggsr"):
                raise ValueError(f"Invalid green phase {phase}")
            pairs = []
            for i in range(count):
                for j in range(i + 1, count):
                    # SUMO bitstrings start with index 0 at the RIGHT.
                    foes = requests[i].get("foes")[-1-j] == "1" or requests[j].get("foes")[-1-i] == "1"
                    if not foes or state[i] == "r" or state[j] == "r":
                        continue
                    if state[i] == state[j] == "G":
                        raise ValueError(f"Conflicting protected greens: phase {phase}, {i}/{j}")
                    yields_i = state[i] in "gs" and requests[i].get("response")[-1-j] == "1"
                    yields_j = state[j] in "gs" and requests[j].get("response")[-1-i] == "1"
                    if not (yields_i or yields_j):
                        raise ValueError(f"Foes without a yielding rule: phase {phase}, {i}/{j}")
                    pairs.append((i, j))
            self.phase_foes[phase] = tuple(pairs)
        self.green_seconds = 30
        # shortcut: provisional yellow durations, calibrate before claiming safety.
        self.yellow_seconds = 4 if map_name == "cologne1" else 3
        self.minimum_red_seconds = 2
        self.output_dir = Path(output_dir).resolve() if output_dir is not None else None
        self.conn = None
        self.current_phase = None
        self._clear_after = {}

    def reset(self):
        """Start an independent process at the original begin; return its state."""
        self.close()
        self._clear_after = {}
        command = [
            sumolib.checkBinary("sumo"), "-c", str(self.config),
            "--step-length", "1", "--seed", "23423",
            "--collision.check-junctions", "true", "--collision.action", "teleport",
            "--no-step-log", "true", "--duration-log.disable", "true",
        ]
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            command += [
                "--log", str(self.output_dir / "sumo.log"),
                "--error-log", str(self.output_dir / "warnings.log"),
                "--statistic-output", str(self.output_dir / "statistics.xml"),
                "--collision-output", str(self.output_dir / "collisions.xml"),
            ]
            (self.output_dir / "command.txt").write_text(
                "\n".join(command) + "\n", encoding="utf-8")
        label = f"stage2-{id(self)}"
        traci.start(command, label=label, doSwitch=False)
        self.conn = traci.getConnection(label)
        try:
            if self.conn.simulation.getTime() != self.begin or self.conn.simulation.getDeltaT() != 1:
                raise ValueError("Unexpected SUMO begin/step-length")
            if tuple(self.conn.trafficlight.getIDList()) != (self.tls_id,):
                raise ValueError("Unexpected traffic lights")
            actual = self.conn.trafficlight.getControlledLinks(self.tls_id)
            actual_links = {
                (index, incoming, outgoing, via)
                for index, group in enumerate(actual) for incoming, outgoing, via in group
            }
            if actual_links != set(self.links):
                raise ValueError("TraCI controlled links differ from XML")
            self.current_phase = self.actions[0]
            self.conn.trafficlight.setRedYellowGreenState(self.tls_id, self.green_states[self.current_phase])
            return self.get_state()
        except Exception:
            # Close the owned process on validation failure, then expose the error.
            self.close()
            raise

    def close(self):
        """Close TraCI and wait for the owned SUMO process to exit."""
        if self.conn is not None:
            try:
                self.conn.close()
            finally:
                self.conn = None
                self.current_phase = None

    def get_state(self):
        """Queues after the last SUMO step, in sorted boundary-lane order."""
        if self.conn is None:
            raise RuntimeError("Call reset() before reading state")
        return {
            "incoming": tuple(self.conn.lane.getLastStepHaltingNumber(lane) for lane in self.incoming_lanes),
            "outgoing": tuple(self.conn.lane.getLastStepHaltingNumber(lane) for lane in self.outgoing_lanes),
            "phase": self.current_phase,
        }

    def _junction_vehicles(self):
        """Vehicles on internal lanes OR with a rear still over the exit."""
        vehicle_api = self.conn.vehicle
        for lane in self.internal_lanes:
            for vehicle in self.conn.lane.getLastStepVehicleIDs(lane):
                self._clear_after[vehicle] = (
                    vehicle_api.getDistance(vehicle) + self.distance_to_exit[lane]
                    - vehicle_api.getLanePosition(vehicle) + vehicle_api.getLength(vehicle)
                )
        for lane in self.outgoing_lanes:
            for vehicle in self.conn.lane.getLastStepVehicleIDs(lane):
                rear_distance = vehicle_api.getLength(vehicle) - vehicle_api.getLanePosition(vehicle)
                if rear_distance > 0:
                    self._clear_after[vehicle] = vehicle_api.getDistance(vehicle) + rear_distance
        # Odometer tracking also covers a bus longer than the outgoing edge.
        active = set(vehicle_api.getIDList())
        self._clear_after = {
            vehicle: distance for vehicle, distance in self._clear_after.items()
            if vehicle in active and vehicle_api.getDistance(vehicle) < distance
        }
        return tuple(sorted(self._clear_after))

    def _run_signal(self, signal, seconds, stage, samples):
        """Advance one second at a time; sample queues at each interval's end."""
        if self.conn.simulation.getTime() >= self.end:
            return
        self.conn.trafficlight.setRedYellowGreenState(self.tls_id, signal)
        for _ in range(seconds):
            before = self.conn.simulation.getTime()
            if before >= self.end:
                break
            occupied = self._junction_vehicles()
            self.conn.simulationStep()
            after = self.conn.simulation.getTime()
            actual_signal = self.conn.trafficlight.getRedYellowGreenState(self.tls_id)
            if after - before != 1 or after > self.end or actual_signal != signal:
                raise RuntimeError("SUMO timing or signal differs from the requested interval")
            samples.append({
                "start": before, "end": after, "stage": stage,
                "signal": actual_signal,
                "queue": sum(self.conn.lane.getLastStepHaltingNumber(lane) for lane in self.incoming_lanes),
                "junction_before": occupied,
                "collisions": [vars(c) for c in self.conn.simulation.getCollisions()],
                "teleports": self.conn.simulation.getStartingTeleportIDList(),
            })

    def step(self, action):
        """Execute one green action; return state, reward, done, info.

        A changed action closes every open movement, waits for clearance,
        then requests 30 s of green. The fixed horizon may cut ANY stage.
        """
        if self.conn is None:
            raise RuntimeError("Call reset() before step()")
        if type(action) is not int or action not in self.actions:
            raise ValueError(f"action must be one of {self.actions}")
        start = self.conn.simulation.getTime()
        if start >= self.end:
            raise RuntimeError("Episode ended; call reset()")
        previous_phase = self.current_phase
        samples = []
        if action != previous_phase:
            self.current_phase = None
            self._run_signal(yellow_state(self.green_states[previous_phase]),
                             self.yellow_seconds, "yellow", samples)
            self._run_signal("r" * len(self.green_states[action]),
                             self.minimum_red_seconds, "all_red", samples)
            # A blocked junction stays red, even if this uses the rest of the episode.
            while self.conn.simulation.getTime() < self.end and self._junction_vehicles():
                self._run_signal("r" * len(self.green_states[action]), 1, "all_red", samples)
        if self.conn.simulation.getTime() < self.end:
            self.current_phase = action
            self._run_signal(self.green_states[action], self.green_seconds, "green", samples)
        finish = self.conn.simulation.getTime()
        elapsed = finish - start
        queue_seconds = sum(s["queue"] * (s["end"] - s["start"]) for s in samples)
        reward = -queue_seconds / elapsed
        durations = {stage: sum(s["end"] - s["start"] for s in samples if s["stage"] == stage)
                     for stage in ("yellow", "all_red", "green")}
        info = {
            "start": start, "end": finish, "elapsed": elapsed,
            "previous_phase": previous_phase, "action": action,
            "durations": durations, "queue_seconds": queue_seconds,
            "mean_queue": -reward, "samples": samples,
        }
        return self.get_state(), reward, finish >= self.end, info
