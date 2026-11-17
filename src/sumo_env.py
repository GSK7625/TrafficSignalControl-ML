"""Moi truong SUMO dung chung (khong phai chung nhan an toan).

Action ID la chi so pha xanh GOC: 0..7 / 0,2,4.
G=xanh uu tien, g=xanh nhuong duong, s=dung roi nhuong, y=vang, r=do.
Luc chay se tat g/s; moi movement van duoc phuc vu co bao ve o pha khac.
Xem reports/stage3/README.md ve thoi gian, lay mau va gioi han an toan.
"""

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import sumolib
import traci


ROOT = Path(__file__).resolve().parent.parent
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def yellow_state(green):
    """Dong tat ca movement dang mo, ke ca cac link dung chung voi pha tiep theo."""
    return "".join("y" if light in "Ggs" else "r" for light in green)


class SumoEnv:
    def __init__(self, map_name, output_dir=None, seed=23423, *, tripinfo=False):
        if map_name not in ("cologne1", "ingolstadt1"):
            raise ValueError("map_name must be cologne1 or ingolstadt1")
        if type(seed) is not int or not 0 <= seed < 2**31:
            raise ValueError("seed must be an integer in [0, 2**31)")
        self.seed = seed
        if tripinfo and output_dir is None:
            raise ValueError("tripinfo requires output_dir")
        self.tripinfo = tripinfo
        self.map_name = map_name
        self.config = ROOT / "data" / map_name / f"{map_name}.sumocfg"
        config = ET.parse(self.config).getroot()
        self.begin = float(config.find("time/begin").get("value"))
        self.end = float(config.find("time/end").get("value"))
        net_path = self.config.parent / config.find("input/net-file").get("value")
        net_xml = ET.parse(net_path).getroot()
        tls, = net_xml.findall("tlLogic")
        self.tls_id = tls.get("id")
        self.original_green_states = {
            i: phase.get("state") for i, phase in enumerate(tls.findall("phase"))
            if "y" not in phase.get("state") and "G" in phase.get("state")
        }
        # Xe nhuong duong co the dung cho trong giao lo; phuc vu chung bang G o pha khac.
        self.green_states = {
            i: "".join("G" if light == "G" else "r" for light in state)
            for i, state in self.original_green_states.items()
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
        # Chi di theo duong noi bo cua TLS nay, bao gom ca doan nhuong duong thu hai.
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
        for index in range(count):
            if not any(state[index] == "G" for state in self.green_states.values()):
                raise ValueError(f"Movement {index} has no protected green service")
        self.phase_foes = {}
        for phase, state in self.green_states.items():
            if len(state) != count or set(state) - set("Gr") or "G" not in state:
                raise ValueError(f"Invalid green phase {phase}")
            pairs = []
            for i in range(count):
                for j in range(i + 1, count):
                    # Chuoi bit cua SUMO danh chi so 0 tu ben PHAI.
                    foes = requests[i].get("foes")[-1-j] == "1" or requests[j].get("foes")[-1-i] == "1"
                    if not foes or state[i] == "r" or state[j] == "r":
                        continue
                    raise ValueError(f"Conflicting protected greens: phase {phase}, {i}/{j}")
            self.phase_foes[phase] = tuple(pairs)
        self.green_seconds = 30
        # ceil(1 + (2 * toc do tiep can toi da) / (2 * 3 m/s^2)); kiem tra trong stage3.
        self.yellow_seconds = 8 if map_name == "cologne1" else 6
        self.minimum_red_seconds = 2
        self.output_dir = Path(output_dir).resolve() if output_dir is not None else None
        self.conn = None
        self.current_phase = None
        self._clear_after = {}

    def reset(self):
        """Khoi dong mot tien trinh doc lap tai moc begin goc; tra ve state ban dau."""
        self.close()
        self._clear_after = {}
        command = [
            sumolib.checkBinary("sumo"), "-c", str(self.config),
            "--step-length", "1", "--seed", str(self.seed),
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
            if self.tripinfo:
                command += [
                    "--tripinfo-output", str(self.output_dir / "tripinfo.xml"),
                    "--tripinfo-output.write-unfinished", "true",
                    "--tripinfo-output.write-undeparted", "true",
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
            # Dong tien trinh so huu khi kiem tra that bai, roi nem loi ra ngoai.
            self.close()
            raise

    def close(self):
        """Dong TraCI va cho tien trinh SUMO so huu thoat han."""
        if self.conn is not None:
            try:
                self.conn.close()
            finally:
                self.conn = None
                self.current_phase = None

    def get_state(self):
        """Hang doi sau buoc SUMO cuoi cung, theo thu tu boundary lane da sap xep."""
        if self.conn is None:
            raise RuntimeError("Call reset() before reading state")
        return {
            "incoming": tuple(self.conn.lane.getLastStepHaltingNumber(lane) for lane in self.incoming_lanes),
            "outgoing": tuple(self.conn.lane.getLastStepHaltingNumber(lane) for lane in self.outgoing_lanes),
            "phase": self.current_phase,
        }

    def _junction_vehicles(self):
        """Xe dang tren lane noi bo HOAC co duoi xe van con nam tren diem ra."""
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
        # Theo doi quang duong di duoc de xu ly ca xe buyt dai hon doan edge dau ra.
        active = set(vehicle_api.getIDList())
        self._clear_after = {
            vehicle: distance for vehicle, distance in self._clear_after.items()
            if vehicle in active and vehicle_api.getDistance(vehicle) < distance
        }
        return tuple(sorted(self._clear_after))

    def _run_signal(self, signal, seconds, stage, samples):
        """Cho mo phong tien tung giay mot; lay mau hang doi o cuoi moi khoang 1 giay."""
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
        """Thuc thi mot pha xanh; tra ve state, reward, done, info.

        Neu doi action thi dong moi movement dang mo, cho giai toa giao lo,
        roi yeu cau 30s xanh. Moc ket thuc co dinh co the cat BAT KY giai doan nao.
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
            # Giao lo bi ket van giu do, ke ca khi dung het phan con lai cua episode.
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
