"""Seeded traffic simulation, baseline controllers, and tabular Q-learning."""
from dataclasses import dataclass, field
from statistics import mean

LENGTH = 180
MIN_GREEN, MAX_GREEN, CLEARANCE = 6, 30, 2
SCENARIOS = ("rush", "balanced", "surge")
CONTROLLERS = ("ai", "timer", "queue")
TRAINING_EPISODES = 2400


class RandomStream:
    """Reproducible generator shared by all traffic scenarios."""

    def __init__(self, seed):
        self.state = seed

    def next(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state / 2**32


def arrivals(scenario, seed, interventions=()):
    random = RandomStream(seed)
    stream = []
    for second in range(LENGTH):
        ns, ew = (0.18, 0.66) if scenario == "rush" else (0.36, 0.36)
        if scenario == "surge" and 60 <= second < 120:
            ns, ew = 0.91, 0.20
        for at, direction in interventions:
            if at <= second < at + 25:
                if direction == "NS":
                    ns = 0.97
                else:
                    ew = 0.97
        stream.append(tuple(int(random.next() < rate) for rate in (ns, ew, ns, ew)))
    return stream


@dataclass
class Junction:
    queues: list = field(default_factory=lambda: [0, 0, 0, 0])
    phase: int = 0
    age: int = 0
    clearance: int = 0
    wait: int = 0
    peak: int = 0
    served: int = 0
    arrived: int = 0
    decision: dict | None = None

    def totals(self):
        n, e, s, w = self.queues
        return n + s, e + w

    def allowed(self):
        if self.clearance or self.age < MIN_GREEN:
            return (0,)
        return (1,) if self.age >= MAX_GREEN else (0, 1)

    def tick(self, incoming, action):
        if action:
            self.clearance = CLEARANCE
            self.age = 0
        for lane, vehicles in enumerate(incoming):
            self.queues[lane] += vehicles
            self.arrived += vehicles
        if self.clearance:
            self.clearance -= 1
            if not self.clearance:
                self.phase = 1 - self.phase
        else:
            for lane in (self.phase, self.phase + 2):
                if self.queues[lane]:
                    self.queues[lane] -= 1
                    self.served += 1
            self.age += 1
        self.peak = max(self.peak, max(self.queues))
        queued = sum(self.queues)
        self.wait += queued
        return -queued

    def frame(self, all_red=False):
        return dict(queues=self.queues.copy(), phase=self.phase, age=self.age,
                    wait=self.wait, peak=self.peak, served=self.served,
                    arrived=self.arrived, decision=self.decision, allRed=all_red)


def queue_bin(size):
    return sum(size >= boundary for boundary in (2, 5, 9, 15))


def state_index(junction):
    ns, ew = junction.totals()
    age = junction.age
    if junction.clearance:
        age_bin = 0
    elif age < MIN_GREEN:
        age_bin = 1
    elif age >= MAX_GREEN:
        age_bin = 6
    else:
        age_bin = 2 + (age - MIN_GREEN) // 6
    return (((junction.phase * 5 + queue_bin(ns)) * 5 + queue_bin(ew)) * 7 + age_bin) * 2


def train():
    values = [0.0] * (2 * 5 * 5 * 7 * 2)
    random = RandomStream(1953)
    for episode in range(TRAINING_EPISODES):
        stream = arrivals(SCENARIOS[episode % len(SCENARIOS)], 0x80000000 + episode)
        junction = Junction()
        epsilon = 0.28 * (1 - episode / TRAINING_EPISODES) + 0.04
        for second, incoming in enumerate(stream):
            key = state_index(junction)
            allowed = junction.allowed()
            if len(allowed) == 1:
                action = allowed[0]
            elif random.next() < epsilon:
                action = int(random.next() < 0.5)
            else:
                action = int(values[key + 1] > values[key])
            reward = junction.tick(incoming, action)
            next_key = state_index(junction)
            future = 0.0 if second == LENGTH - 1 else max(
                values[next_key + choice] for choice in junction.allowed()
            )
            current = key + action
            values[current] += 0.10 * (reward + 0.93 * future - values[current])
    return tuple(values)


def choose(junction, controller, second, values):
    allowed = junction.allowed()
    if len(allowed) == 1:
        return allowed[0]
    ns, ew = junction.totals()
    if controller == "timer":
        return int(junction.age >= 18)
    if controller == "queue":
        return int(ew > ns if junction.phase == 0 else ns > ew)
    key = state_index(junction)
    action = int(values[key + 1] > values[key])
    junction.decision = dict(second=second, ns=ns, ew=ew, phase=junction.phase,
                             age=junction.age, action=action,
                             hold=values[key], switch=values[key + 1])
    return action


def simulate(stream, controller, values, record=True):
    junction = Junction()
    frames = [junction.frame()] if record else None
    for second, incoming in enumerate(stream, 1):
        action = choose(junction, controller, second, values)
        all_red = bool(action or junction.clearance)
        junction.tick(incoming, action)
        if record:
            frames.append(junction.frame(all_red))
    return frames if record else junction


def benchmark(scenario, seed, runs, values, interventions):
    datasets = {controller: [] for controller in CONTROLLERS}
    for offset in range(1, runs + 1):
        stream = arrivals(scenario, seed + offset, interventions)
        results = [simulate(stream, controller, values, record=False)
                   for controller in CONTROLLERS]
        best = min(result.wait for result in results)
        for controller, result in zip(CONTROLLERS, results):
            datasets[controller].append((result.wait, result.peak, result.served, result.wait == best))
    return [
        dict(wait=mean(waits), peak=mean(peaks), served=mean(served), wins=sum(wins))
        for waits, peaks, served, wins in (zip(*data) for data in datasets.values())
    ]
