from __future__ import annotations

import argparse
import json
import math
import os
import random
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from typing import List, Optional, Tuple


@dataclass
class IzhikevichParams:
    a: float = 0.02
    b: float = 0.2
    c: float = -65.0
    d: float = 8.0


@dataclass
class NeuronState:
    v: float = -65.0
    u: float = 0.0
    params: IzhikevichParams = field(default_factory=IzhikevichParams)
    is_inhibitory: bool = False
    bias_current: float = 0.0
    kind: str = "regular"

    def reset_recovery(self) -> None:
        self.u = self.params.b * self.v


@dataclass
class Synapse:
    target: int
    weight: float
    delay_steps: int


@dataclass
class SimulationConfig:
    neuron_count: int
    dt_ms: float = 1.0
    connection_prob: float = 0.03
    inhibitory_ratio: float = 0.2
    weight_exc: float = 3.0
    weight_inh: float = -4.0
    delay_ms_range: Tuple[float, float] = (1.0, 20.0)
    noise_current: Tuple[float, float] = (0.0, 0.6)
    pulse_every_steps: int = 0
    pulse_magnitude: float = 8.0
    pulse_fraction: float = 0.03
    delay_max_ms: float = 20.0
    seed: Optional[int] = None


class Network:
    def __init__(
        self,
        neurons: List[NeuronState],
        synapses: List[List[Synapse]],
        dt_ms: float,
        max_delay_steps: int,
    ) -> None:
        self.neurons = neurons
        self.synapses = synapses
        self.dt_ms = dt_ms
        self.max_delay_steps = max_delay_steps
        self._ring_size = max_delay_steps + 1
        self._input_ring = [
            [0.0 for _ in range(self._ring_size)] for _ in range(len(neurons))
        ]
        self._ring_index = 0

    def step(self, external_inputs: Optional[List[float]] = None) -> List[int]:
        if external_inputs is None:
            external_inputs = [0.0 for _ in self.neurons]
        if len(external_inputs) != len(self.neurons):
            raise ValueError("external_inputs must match neuron count")

        spikes: List[int] = []
        ring_index = self._ring_index
        for i, neuron in enumerate(self.neurons):
            input_current = external_inputs[i] + self._input_ring[i][ring_index] + neuron.bias_current
            self._input_ring[i][ring_index] = 0.0

            dv = 0.04 * neuron.v * neuron.v + 5 * neuron.v + 140 - neuron.u + input_current
            neuron.v += dv * (self.dt_ms / 1.0)
            neuron.u += neuron.params.a * (neuron.params.b * neuron.v - neuron.u)

            if not math.isfinite(neuron.v) or not math.isfinite(neuron.u):
                neuron.v = neuron.params.c
                neuron.reset_recovery()
                continue

            if neuron.v >= 30.0:
                neuron.v = neuron.params.c
                neuron.u += neuron.params.d
                spikes.append(i)

        for source_index in spikes:
            for synapse in self.synapses[source_index]:
                target_index = synapse.target
                target_slot = (ring_index + synapse.delay_steps) % self._ring_size
                self._input_ring[target_index][target_slot] += synapse.weight

        self._ring_index = (self._ring_index + 1) % self._ring_size
        return spikes


class SimulationRunner:
    def __init__(self, config: SimulationConfig, network: Network) -> None:
        self.config = config
        self.network = network
        self.step_index = 0
        self._lock = threading.Lock()
        self._latest_state = {
            "step": 0,
            "mean_v": 0.0,
            "spike_count": 0,
            "firing_rate_hz": 0.0,
            "sample_v": [],
            "sample_inhibitory": [],
            "sample_spikes": [],
            "timestamp": time.time(),
            "pulse_glow": 0.0,
        }
        self._sample_indices = self._pick_sample_indices()
        self._sample_edges = self._build_sample_edges(limit_per_node=6)
        self._paused = False
        self._pending_pulse = 0.0
        self._last_pulse_time: Optional[float] = None
        self._exc_count = sum(1 for n in self.network.neurons if not n.is_inhibitory)
        self._inh_count = len(self.network.neurons) - self._exc_count
        self._neuron_presets = {
            "regular": {"params": IzhikevichParams(0.02, 0.2, -65.0, 8.0), "bias": 0.0, "inh": False},
            "pacemaker": {"params": IzhikevichParams(0.02, 0.2, -65.0, 8.0), "bias": 7.0, "inh": False},
            "bursting": {"params": IzhikevichParams(0.02, 0.2, -50.0, 2.0), "bias": 2.0, "inh": False},
            "fast_inhibitory": {
                "params": IzhikevichParams(0.1, 0.2, -65.0, 2.0),
                "bias": 2.0,
                "inh": True,
            },
        }

    def _build_sample_edges(self, limit_per_node: int) -> List[Tuple[int, int]]:
        sample_lookup = {full_idx: i for i, full_idx in enumerate(self._sample_indices)}
        edges: List[Tuple[int, int]] = []
        for src_full in self._sample_indices:
            src_sample = sample_lookup[src_full]
            links = [
                sample_lookup[syn.target]
                for syn in self.network.synapses[src_full]
                if syn.target in sample_lookup
            ]
            random.shuffle(links)
            for dst_sample in links[:limit_per_node]:
                edges.append((src_sample, dst_sample))
        return edges

    def rebuild_network(self) -> None:
        new_network = build_network(self.config)
        self.network = new_network
        self._sample_indices = self._pick_sample_indices()
        self._sample_edges = self._build_sample_edges(limit_per_node=6)
        self._exc_count = sum(1 for n in self.network.neurons if not n.is_inhibitory)
        self._inh_count = len(self.network.neurons) - self._exc_count
        for neuron in self.network.neurons:
            neuron.kind = "regular"
            neuron.bias_current = 0.0

    def apply_neuron_profile(self, sample_index: int, profile_name: str) -> None:
        if sample_index < 0 or sample_index >= len(self._sample_indices):
            return
        profile = self._neuron_presets.get(profile_name)
        if profile is None:
            return
        neuron_index = self._sample_indices[sample_index]
        neuron = self.network.neurons[neuron_index]
        neuron.params = profile["params"]
        neuron.bias_current = profile["bias"]
        neuron.is_inhibitory = profile["inh"]
        neuron.kind = profile_name
        neuron.v = neuron.params.c
        neuron.reset_recovery()

    def clear_circuit(self) -> None:
        for i in range(len(self._sample_indices)):
            self.apply_neuron_profile(i, "regular")

    def _pick_sample_indices(self, sample_size: int = 264) -> List[int]:
        indices = list(range(len(self.network.neurons)))
        random.shuffle(indices)
        return indices[: min(sample_size, len(indices))]

    def _noise_inputs(self) -> List[float]:
        low, high = self.config.noise_current
        return [random.uniform(low, high) for _ in self.network.neurons]

    def _pulse_inputs(self, base_inputs: List[float]) -> None:
        if self._pending_pulse > 0:
            pulse_count = max(1, int(len(base_inputs) * self.config.pulse_fraction))
            for idx in random.sample(range(len(base_inputs)), pulse_count):
                base_inputs[idx] += self._pending_pulse
            self._last_pulse_time = time.time()
            self._pending_pulse = 0.0
            return

        if self.config.pulse_every_steps <= 0:
            return
        if self.step_index % self.config.pulse_every_steps != 0:
            return
        pulse_count = max(1, int(len(base_inputs) * self.config.pulse_fraction))
        for idx in random.sample(range(len(base_inputs)), pulse_count):
            base_inputs[idx] += self.config.pulse_magnitude
        self._last_pulse_time = time.time()

    def step(self) -> None:
        if self._paused:
            return
        inputs = self._noise_inputs()
        self._pulse_inputs(inputs)
        spikes = self.network.step(inputs)

        finite_vs = [n.v for n in self.network.neurons if math.isfinite(n.v)]
        mean_v = sum(finite_vs) / len(finite_vs) if finite_vs else 0.0
        firing_rate = len(spikes) / len(self.network.neurons) * (1000.0 / self.config.dt_ms)
        sample_v = [
            self.network.neurons[i].v if math.isfinite(self.network.neurons[i].v) else -65.0
            for i in self._sample_indices
        ]
        sample_inhibitory = [self.network.neurons[i].is_inhibitory for i in self._sample_indices]
        sample_types = [self.network.neurons[i].kind for i in self._sample_indices]
        spike_set = set(spikes)
        sample_spikes = [i in spike_set for i in self._sample_indices]
        exc_spikes = sum(1 for i in spikes if not self.network.neurons[i].is_inhibitory)
        inh_spikes = len(spikes) - exc_spikes
        exc_rate = (
            exc_spikes / self._exc_count * (1000.0 / self.config.dt_ms) if self._exc_count else 0.0
        )
        inh_rate = (
            inh_spikes / self._inh_count * (1000.0 / self.config.dt_ms) if self._inh_count else 0.0
        )
        pulse_glow = 0.0
        if self._last_pulse_time is not None:
            pulse_glow = max(0.0, 1.0 - (time.time() - self._last_pulse_time) / 1.5)

        with self._lock:
            self._latest_state = {
                "step": self.step_index,
                "mean_v": mean_v,
                "spike_count": len(spikes),
                "firing_rate_hz": firing_rate,
                "exc_rate_hz": exc_rate,
                "inh_rate_hz": inh_rate,
                "sample_v": sample_v,
                "sample_inhibitory": sample_inhibitory,
                "sample_spikes": sample_spikes,
                "sample_types": sample_types,
                "sample_edges": self._sample_edges,
                "timestamp": time.time(),
                "pulse_glow": pulse_glow,
            }
        self.step_index += 1

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._latest_state)

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def is_paused(self) -> bool:
        return self._paused

    def step_once(self) -> None:
        was_paused = self._paused
        self._paused = False
        self.step()
        self._paused = was_paused

    def trigger_pulse(self, magnitude: Optional[float] = None) -> None:
        if magnitude is None:
            magnitude = self.config.pulse_magnitude
        self._pending_pulse = float(magnitude)

    def update_config(
        self,
        noise_min: Optional[float] = None,
        noise_max: Optional[float] = None,
        pulse_mag: Optional[float] = None,
        pulse_fraction: Optional[float] = None,
        connection_prob: Optional[float] = None,
        weight_exc: Optional[float] = None,
        weight_inh: Optional[float] = None,
        delay_max_ms: Optional[float] = None,
    ) -> None:
        if noise_min is not None or noise_max is not None:
            low, high = self.config.noise_current
            if noise_min is not None:
                low = noise_min
            if noise_max is not None:
                high = noise_max
            self.config.noise_current = (low, high)
        if pulse_mag is not None:
            self.config.pulse_magnitude = pulse_mag
        if pulse_fraction is not None:
            self.config.pulse_fraction = pulse_fraction
        if connection_prob is not None:
            self.config.connection_prob = max(0.0, min(1.0, connection_prob))
        if weight_exc is not None:
            self.config.weight_exc = weight_exc
        if weight_inh is not None:
            self.config.weight_inh = weight_inh
        if delay_max_ms is not None:
            self.config.delay_max_ms = delay_max_ms


def build_network(config: SimulationConfig) -> Network:
    if config.seed is not None:
        random.seed(config.seed)

    neurons: List[NeuronState] = []
    for _ in range(config.neuron_count):
        neuron = NeuronState()
        neuron.is_inhibitory = random.random() < config.inhibitory_ratio
        neuron.reset_recovery()
        neurons.append(neuron)

    delay_max = config.delay_max_ms
    max_delay_steps = max(1, int(delay_max / config.dt_ms))
    synapses: List[List[Synapse]] = [[] for _ in neurons]

    for i, neuron in enumerate(neurons):
        for j in range(len(neurons)):
            if i == j:
                continue
            if random.random() > config.connection_prob:
                continue
            weight = config.weight_inh if neuron.is_inhibitory else config.weight_exc
            delay_ms = random.uniform(config.delay_ms_range[0], delay_max)
            delay_steps = max(1, int(delay_ms / config.dt_ms))
            synapses[i].append(Synapse(target=j, weight=weight, delay_steps=delay_steps))

    return Network(neurons, synapses, config.dt_ms, max_delay_steps)


def auto_neuron_count(max_neurons: int = 2000) -> int:
    cpu_count = os.cpu_count() or 1
    return max(100, min(max_neurons, cpu_count * 250))


class SimulationServer(BaseHTTPRequestHandler):
    runner: SimulationRunner

    def _write_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        parsed = urlparse(self.path)
        if parsed.path == "/state":
            snapshot = self.runner.snapshot()
            snapshot["paused"] = self.runner.is_paused()
            self._write_json(snapshot)
            return
        if parsed.path == "/control":
            params = parse_qs(parsed.query)
            action = (params.get("action") or [""])[0]
            if action == "pause":
                self.runner.pause()
            elif action == "resume":
                self.runner.resume()
            elif action == "step":
                self.runner.step_once()
            elif action == "pulse":
                mag_value = (params.get("magnitude") or [None])[0]
                magnitude = float(mag_value) if mag_value else None
                self.runner.trigger_pulse(magnitude)
            elif action == "config":
                noise_min = (params.get("noise_min") or [None])[0]
                noise_max = (params.get("noise_max") or [None])[0]
                pulse_mag = (params.get("pulse_mag") or [None])[0]
                pulse_fraction = (params.get("pulse_fraction") or [None])[0]
                connection_prob = (params.get("connection_prob") or [None])[0]
                weight_exc = (params.get("weight_exc") or [None])[0]
                weight_inh = (params.get("weight_inh") or [None])[0]
                delay_max = (params.get("delay_max") or [None])[0]
                self.runner.update_config(
                    noise_min=float(noise_min) if noise_min is not None else None,
                    noise_max=float(noise_max) if noise_max is not None else None,
                    pulse_mag=float(pulse_mag) if pulse_mag is not None else None,
                    pulse_fraction=float(pulse_fraction) if pulse_fraction is not None else None,
                    connection_prob=float(connection_prob) if connection_prob is not None else None,
                    weight_exc=float(weight_exc) if weight_exc is not None else None,
                    weight_inh=float(weight_inh) if weight_inh is not None else None,
                    delay_max_ms=float(delay_max) if delay_max is not None else None,
                )
                self.runner.rebuild_network()
            elif action == "neuron":
                sample_index = (params.get("sample_index") or [None])[0]
                profile = (params.get("type") or [None])[0]
                if sample_index is not None and profile is not None:
                    self.runner.apply_neuron_profile(int(sample_index), profile)
            elif action == "clear_circuit":
                self.runner.clear_circuit()
            self._write_json({"ok": True, "paused": self.runner.is_paused()})
            return
        if parsed.path == "/":
            body = DASHBOARD_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()


DASHBOARD_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Brain Simulator</title>
    <style>
      :root {
        --ink: #1a1a1a;
        --paper: #f7efe2;
        --accent: #f05a3f;
        --accent-dark: #7c3b2b;
        --cool: #0f5a6b;
        --glow: #ffd27a;
      }
      body {
        font-family: "Palatino Linotype", "Book Antiqua", Palatino, serif;
        margin: 24px;
        color: var(--ink);
        background:
          radial-gradient(circle at 10% 10%, #fff8ef 0%, #f4e8d6 40%, #eadcc9 100%);
      }
      body::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: 0.15;
        background-image: repeating-linear-gradient(45deg, #ffffff 0px, #ffffff 2px, transparent 2px, transparent 6px);
      }
      h1 { margin: 0 0 4px 0; letter-spacing: 0.5px; }
      .subtitle { margin: 0 0 16px 0; color: #5b5b5b; font-size: 0.95rem; }
      .grid { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
      .panel {
        position: relative;
        background: #fffaf4;
        border-radius: 14px;
        box-shadow: 0 10px 26px rgba(52, 34, 20, 0.12);
        padding: 12px;
      }
      .panel h3 {
        margin: 0 0 8px 0;
        font-size: 0.95rem;
        letter-spacing: 0.4px;
        color: #4a3a2f;
      }
      .panel small {
        display: block;
        color: #6b5c51;
        margin-bottom: 8px;
        line-height: 1.3;
      }
      .panel canvas {
        width: 100%;
        height: 160px;
        border-radius: 10px;
        background: transparent;
      }
      .builder-buttons { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 8px; }
      .builder-buttons button {
        border: 1px solid #d4c7b7;
        border-radius: 999px;
        padding: 6px 12px;
        background: #fffaf4;
        cursor: pointer;
        font-weight: 600;
        color: #4a3a2f;
      }
      .builder-buttons button.active {
        background: var(--cool);
        color: #fff;
        border-color: var(--cool);
      }
      #mapCanvas { height: 300px; }
      #rasterCanvas { height: 200px; }
      #topologyCanvas { height: 220px; }
      #traceCanvas { height: 200px; }
      #eiCanvas { height: 160px; }
      .stats { display: flex; gap: 16px; flex-wrap: wrap; align-items: center; }
      .stat { background: #ffffff; padding: 12px 16px; border-radius: 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.08); }
      .controls { display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0; align-items: center; }
      .controls button {
        border: none;
        border-radius: 999px;
        padding: 10px 18px;
        cursor: pointer;
        background: var(--cool);
        color: #fff;
        font-weight: 600;
        letter-spacing: 0.3px;
      }
      .controls button.secondary { background: var(--accent); }
      .status { font-size: 0.9rem; color: #5b5b5b; }
      .control-group { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
      .control-group label { font-size: 0.85rem; color: #5b5b5b; }
      .control-group input {
        border-radius: 10px;
        border: 1px solid #d4c7b7;
        padding: 6px 8px;
        font-family: inherit;
        width: 80px;
        background: #fffaf4;
      }
      .context {
        margin-top: 16px;
        background: #ffffff;
        padding: 14px 16px;
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        line-height: 1.4;
      }
      .legend { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; font-size: 0.85rem; color: #4a4a4a; }
      .legend span { display: inline-flex; align-items: center; gap: 6px; }
      .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
    </style>
  </head>
  <body>
    <h1>Brain Simulator</h1>
    <p class="subtitle">High-fidelity Izhikevich spiking model with synaptic delays and stochastic input.</p>
    <div class="stats">
      <div class="stat"><strong>Step</strong> <span id="step">0</span></div>
      <div class="stat"><strong>Mean V</strong> <span id="mean">0</span></div>
      <div class="stat"><strong>Spikes</strong> <span id="spikes">0</span></div>
      <div class="stat"><strong>Firing Hz</strong> <span id="rate">0</span></div>
      <div class="stat"><strong>Exc Hz</strong> <span id="excRate">0</span></div>
      <div class="stat"><strong>Inh Hz</strong> <span id="inhRate">0</span></div>
      <div class="stat status" id="status">Connecting...</div>
    </div>
    <div class="controls">
      <button id="toggle">Pause</button>
      <button class="secondary" id="stepOnce">Step</button>
      <button id="pulse">Pulse</button>
      <button id="modeToggle">Spike Heat</button>
      <div class="control-group">
        <label>Noise</label>
        <input id="noiseMin" type="number" step="0.1" value="0.0" />
        <input id="noiseMax" type="number" step="0.1" value="0.6" />
      </div>
      <div class="control-group">
        <label>Pulse</label>
        <input id="pulseMag" type="number" step="0.5" value="8.0" />
        <input id="pulseFrac" type="number" step="0.01" value="0.03" />
        <button id="applyConfig">Apply</button>
      </div>
      <div class="control-group">
        <label>Scenario</label>
        <button id="scenarioBaseline">Baseline</button>
        <button id="scenarioBurst">Burst/Quench</button>
        <button id="scenarioOsc">Oscillator</button>
        <button id="scenarioRipple">Ripple</button>
      </div>
    </div>
    <div class="grid">
      <section class="panel">
        <h3>Circuit Builder</h3>
        <small>Select a neuron type, then click a cell in the neural field to place it.</small>
        <div class="builder-buttons">
          <button data-type="regular" class="active">Regular</button>
          <button data-type="pacemaker">Pacemaker</button>
          <button data-type="bursting">Bursting</button>
          <button data-type="fast_inhibitory">Fast Inhibitory</button>
          <button id="clearCircuit">Clear Selections</button>
        </div>
        <div class="legend">
          <span><i class="dot" style="background: #f05a3f;"></i> pacemaker</span>
          <span><i class="dot" style="background: #c77dff;"></i> bursting</span>
          <span><i class="dot" style="background: #0f5a6b;"></i> fast inhibitory</span>
        </div>
      </section>
      <section class="panel">
        <h3>Neural Field</h3>
        <small>Sampled membrane potentials across the network. Warm tones = depolarized, cool = hyperpolarized.</small>
        <canvas id="mapCanvas" width="320" height="280"></canvas>
        <div class="legend">
          <span><i class="dot" style="background: var(--glow);"></i> spike glow</span>
          <span><i class="dot" style="background: var(--accent);"></i> higher V</span>
          <span><i class="dot" style="background: #2a6f78;"></i> lower V</span>
          <span><i class="dot" style="background: #0f5a6b;"></i> inhibitory</span>
          <span><i class="dot" style="background: #f05a3f;"></i> excitatory</span>
        </div>
      </section>
      <section class="panel">
        <h3>Topology Sketch</h3>
        <small>Force-directed sample of synaptic wiring. Edge density hints at local connectivity.</small>
        <canvas id="topologyCanvas" width="320" height="220"></canvas>
        <div class="legend">
          <span><i class="dot" style="background: #f05a3f;"></i> excitatory</span>
          <span><i class="dot" style="background: #0f5a6b;"></i> inhibitory</span>
          <span><i class="dot" style="background: var(--glow);"></i> spike halo</span>
        </div>
      </section>
      <section class="panel">
        <h3>Spike Raster</h3>
        <small>Spike trains over time (each row = sampled neuron, columns = recent steps).</small>
        <canvas id="rasterCanvas" width="320" height="200"></canvas>
      </section>
      <section class="panel">
        <h3>Membrane Trace</h3>
        <small>Shift-click a cell to pin its membrane potential trace (V).</small>
        <canvas id="traceCanvas" width="320" height="200"></canvas>
      </section>
      <section class="panel">
        <h3>Mean Membrane Potential</h3>
        <small>Network-wide average membrane potential (V) over time.</small>
        <canvas id="meanChart" width="320" height="160"></canvas>
      </section>
      <section class="panel">
        <h3>Spike Count</h3>
        <small>Number of spikes per step (higher = more synchronous activity).</small>
        <canvas id="spikeChart" width="320" height="160"></canvas>
      </section>
      <section class="panel">
        <h3>Excitatory vs Inhibitory Rate</h3>
        <small>Firing rate split by neuron type (Hz).</small>
        <canvas id="eiCanvas" width="320" height="160"></canvas>
        <div class="legend">
          <span><i class="dot" style="background: #f05a3f;"></i> excitatory Hz</span>
          <span><i class="dot" style="background: #0f5a6b;"></i> inhibitory Hz</span>
        </div>
      </section>
    </div>
    <div class="context">
      <strong>What you are seeing</strong>
      <p>
        Each cell in the neural field is a sample of membrane potential (V) from the simulated network.
        Warm tones indicate higher membrane potential, and glowing cells mark spiking events (V &gt; 0 mV).
        The model uses Izhikevich dynamics, capturing biologically plausible bursting and spiking with low compute cost.
      </p>
    </div>
    <script>
      const meanCanvas = document.getElementById('meanChart');
      const spikeCanvas = document.getElementById('spikeChart');
      const mapCanvas = document.getElementById('mapCanvas');
      const rasterCanvas = document.getElementById('rasterCanvas');
      const topologyCanvas = document.getElementById('topologyCanvas');
      const traceCanvas = document.getElementById('traceCanvas');
      const eiCanvas = document.getElementById('eiCanvas');
      const meanCtx = meanCanvas.getContext('2d');
      const spikeCtx = spikeCanvas.getContext('2d');
      const mapCtx = mapCanvas.getContext('2d');
      const rasterCtx = rasterCanvas.getContext('2d');
      const topologyCtx = topologyCanvas.getContext('2d');
      const traceCtx = traceCanvas.getContext('2d');
      const eiCtx = eiCanvas.getContext('2d');
      const meanData = [];
      const spikeData = [];
      const excRateData = [];
      const inhRateData = [];
      const rasterHistory = [];
      const traceHistory = [];
      let pinnedIndex = null;
      let nodePositions = [];
      let lastSampleV = [];
      let lastSampleInh = [];
      let lastSampleSpikes = [];
      let lastSampleTypes = [];
      let mapMode = 'voltage';
      let selectedType = 'regular';

      const tooltip = document.createElement('div');
      tooltip.style.position = 'absolute';
      tooltip.style.pointerEvents = 'none';
      tooltip.style.background = 'rgba(26, 26, 26, 0.85)';
      tooltip.style.color = '#fff';
      tooltip.style.padding = '6px 8px';
      tooltip.style.borderRadius = '8px';
      tooltip.style.fontSize = '0.8rem';
      tooltip.style.opacity = '0';
      document.body.appendChild(tooltip);

      function drawChart(ctx, data, color) {
        const w = ctx.canvas.width;
        const h = ctx.canvas.height;
        ctx.clearRect(0, 0, w, h);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        const max = Math.max(...data, 1);
        const min = Math.min(...data, 0);
        const span = Math.max(max - min, 1);
        data.forEach((value, idx) => {
          const x = (idx / (data.length - 1)) * w;
          const y = h - ((value - min) / span) * h;
          if (idx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      }

      function drawDualChart(ctx, seriesA, seriesB, colorA, colorB) {
        const w = ctx.canvas.width;
        const h = ctx.canvas.height;
        ctx.clearRect(0, 0, w, h);
        ctx.lineWidth = 2;
        const combined = seriesA.concat(seriesB);
        const max = Math.max(...combined, 1);
        const min = Math.min(...combined, 0);
        const span = Math.max(max - min, 1);

        function drawLine(series, color) {
          ctx.strokeStyle = color;
          ctx.beginPath();
          series.forEach((value, idx) => {
            const x = (idx / (series.length - 1)) * w;
            const y = h - ((value - min) / span) * h;
            if (idx === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          });
          ctx.stroke();
        }

        drawLine(seriesA, colorA);
        drawLine(seriesB, colorB);
      }

      function drawMap(values, inhibitors, spikes, types, tick, pulseGlow) {
        const w = mapCtx.canvas.width;
        const h = mapCtx.canvas.height;
        const cols = 22;
        const rows = Math.max(1, Math.ceil(values.length / cols));
        const cellW = w / cols;
        const cellH = h / rows;
        mapCtx.clearRect(0, 0, w, h);
        const gradient = mapCtx.createLinearGradient(0, 0, w, h);
        gradient.addColorStop(0, '#fdf8f0');
        gradient.addColorStop(1, '#efe2cf');
        mapCtx.fillStyle = gradient;
        mapCtx.fillRect(0, 0, w, h);

        const total = values.length;
        for (let i = 0; i < total; i++) {
          const v = values[i];
          const x = (i % cols) * cellW;
          const y = Math.floor(i / cols) * cellH;
          const norm = Math.min(1, Math.max(0, (v + 80) / 110));
          const glow = Math.max(0, v - 5) / 30;
          const flicker = 0.4 + 0.6 * Math.sin((tick / 6) + i / 3);
          let r = Math.floor(30 + 200 * norm + 40 * glow);
          let g = Math.floor(70 + 60 * (1 - norm) + 25 * glow);
          let b = Math.floor(90 + 140 * (1 - norm));
          if (mapMode === 'spike') {
            const spikeBoost = spikes[i] ? 1 : 0;
            r = Math.floor(40 + 200 * spikeBoost);
            g = Math.floor(60 + 90 * (1 - spikeBoost));
            b = Math.floor(80 + 60 * (1 - spikeBoost));
          }
          mapCtx.fillStyle = `rgb(${r}, ${g}, ${b})`;
          mapCtx.fillRect(x + 1, y + 1, cellW - 2, cellH - 2);

          if (glow > 0.1) {
            mapCtx.globalAlpha = Math.min(0.9, glow * flicker);
            mapCtx.fillStyle = '#ffd27a';
            mapCtx.beginPath();
            mapCtx.arc(
              x + cellW / 2,
              y + cellH / 2,
              Math.min(cellW, cellH) * (0.35 + 0.15 * Math.sin(tick / 8)),
              0,
              Math.PI * 2
            );
            mapCtx.fill();
            mapCtx.globalAlpha = 1;
          }

          if (spikes[i]) {
            mapCtx.strokeStyle = '#f05a3f';
            mapCtx.lineWidth = 2;
            mapCtx.strokeRect(x + 2, y + 2, cellW - 4, cellH - 4);
          }

          if (inhibitors[i]) {
            mapCtx.strokeStyle = 'rgba(15, 90, 107, 0.6)';
            mapCtx.lineWidth = 1.5;
            mapCtx.beginPath();
            mapCtx.moveTo(x + 3, y + 3);
            mapCtx.lineTo(x + cellW - 3, y + cellH - 3);
            mapCtx.stroke();
          }

          if (types[i] === 'pacemaker') {
            mapCtx.fillStyle = '#f05a3f';
            mapCtx.beginPath();
            mapCtx.arc(x + cellW / 2, y + cellH / 2, 3.2, 0, Math.PI * 2);
            mapCtx.fill();
          }
          if (types[i] === 'bursting') {
            mapCtx.fillStyle = '#c77dff';
            mapCtx.fillRect(x + cellW / 2 - 3, y + cellH / 2 - 3, 6, 6);
          }
          if (types[i] === 'fast_inhibitory') {
            mapCtx.strokeStyle = '#0f5a6b';
            mapCtx.lineWidth = 2;
            mapCtx.beginPath();
            mapCtx.arc(x + cellW / 2, y + cellH / 2, 4, 0, Math.PI * 2);
            mapCtx.stroke();
          }
        }
        mapCtx.strokeStyle = 'rgba(60, 40, 20, 0.08)';
        mapCtx.lineWidth = 1;
        for (let c = 1; c < cols; c++) {
          mapCtx.beginPath();
          mapCtx.moveTo(c * cellW, 0);
          mapCtx.lineTo(c * cellW, h);
          mapCtx.stroke();
        }
        for (let r = 1; r < rows; r++) {
          mapCtx.beginPath();
          mapCtx.moveTo(0, r * cellH);
          mapCtx.lineTo(w, r * cellH);
          mapCtx.stroke();
        }

        if (pulseGlow > 0) {
          const radius = Math.min(w, h) * (0.25 + 0.25 * pulseGlow);
          const gradient = mapCtx.createRadialGradient(w / 2, h / 2, 10, w / 2, h / 2, radius);
          gradient.addColorStop(0, `rgba(255, 210, 122, ${0.35 * pulseGlow})`);
          gradient.addColorStop(1, 'rgba(255, 210, 122, 0)');
          mapCtx.fillStyle = gradient;
          mapCtx.beginPath();
          mapCtx.arc(w / 2, h / 2, radius, 0, Math.PI * 2);
          mapCtx.fill();
        }
      }

      function drawTopology(values, inhibitors, spikes, edges, tick) {
        const w = topologyCtx.canvas.width;
        const h = topologyCtx.canvas.height;
        topologyCtx.clearRect(0, 0, w, h);
        topologyCtx.fillStyle = '#fffaf4';
        topologyCtx.fillRect(0, 0, w, h);

        if (nodePositions.length === 0 && values.length > 0) {
          nodePositions = values.map(() => ({
            x: Math.random() * w,
            y: Math.random() * h,
            vx: 0,
            vy: 0,
          }));
        }

        const center = { x: w / 2, y: h / 2 };
        const targetRadius = Math.min(w, h) * 0.35;
        for (let iter = 0; iter < 2; iter++) {
          nodePositions.forEach((node, i) => {
            let fx = (center.x - node.x) * 0.0005;
            let fy = (center.y - node.y) * 0.0005;
            nodePositions.forEach((other, j) => {
              if (i === j) return;
              const dx = node.x - other.x;
              const dy = node.y - other.y;
              const distSq = dx * dx + dy * dy + 0.1;
              const repulse = 25 / distSq;
              fx += dx * repulse * 0.001;
              fy += dy * repulse * 0.001;
            });
            edges.forEach(([src, dst]) => {
              if (src !== i && dst !== i) return;
              const other = nodePositions[src === i ? dst : src];
              if (!other) return;
              const dx = other.x - node.x;
              const dy = other.y - node.y;
              fx += dx * 0.00008;
              fy += dy * 0.00008;
            });
            const dxC = node.x - center.x;
            const dyC = node.y - center.y;
            const dist = Math.sqrt(dxC * dxC + dyC * dyC) || 1;
            if (dist < targetRadius) {
              const push = (targetRadius - dist) * 0.0006;
              fx += (dxC / dist) * push;
              fy += (dyC / dist) * push;
            }
            node.vx = (node.vx + fx) * 0.92;
            node.vy = (node.vy + fy) * 0.92;
            node.x = Math.min(w - 6, Math.max(6, node.x + node.vx));
            node.y = Math.min(h - 6, Math.max(6, node.y + node.vy));
          });
        }

        topologyCtx.strokeStyle = 'rgba(60, 40, 20, 0.15)';
        topologyCtx.lineWidth = 1;
        edges.forEach(([src, dst]) => {
          const a = nodePositions[src];
          const b = nodePositions[dst];
          if (!a || !b) return;
          topologyCtx.beginPath();
          topologyCtx.moveTo(a.x, a.y);
          topologyCtx.lineTo(b.x, b.y);
          topologyCtx.stroke();
        });

        values.forEach((v, i) => {
          const node = nodePositions[i];
          if (!node) return;
          const glow = Math.max(0, v - 5) / 30;
          const r = inhibitors[i] ? 40 : 240;
          const g = inhibitors[i] ? 110 : 90;
          const b = inhibitors[i] ? 130 : 70;
          topologyCtx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.9)`;
          const radius = 3 + glow * 2;
          topologyCtx.beginPath();
          topologyCtx.arc(node.x, node.y, radius, 0, Math.PI * 2);
          topologyCtx.fill();
          if (spikes[i]) {
            topologyCtx.strokeStyle = '#ffd27a';
            topologyCtx.lineWidth = 2;
            topologyCtx.beginPath();
            topologyCtx.arc(node.x, node.y, radius + 3, 0, Math.PI * 2);
            topologyCtx.stroke();
          }
        });
      }

      function drawRaster(history) {
        const w = rasterCtx.canvas.width;
        const h = rasterCtx.canvas.height;
        rasterCtx.clearRect(0, 0, w, h);
        rasterCtx.fillStyle = '#fffaf4';
        rasterCtx.fillRect(0, 0, w, h);
        const rows = history.length > 0 ? history[0].length : 0;
        const maxCols = 120;
        const colWidth = w / maxCols;
        const rowHeight = rows > 0 ? h / rows : h;
        rasterCtx.fillStyle = 'rgba(15, 90, 107, 0.6)';
        history.forEach((col, colIndex) => {
          col.forEach((spike, row) => {
            if (!spike) return;
            rasterCtx.fillRect(
              Math.floor(colIndex * colWidth),
              Math.floor(row * rowHeight),
              Math.max(1, colWidth - 1),
              Math.max(1, rowHeight - 1)
            );
          });
        });
      }

      function drawTrace(history) {
        const w = traceCtx.canvas.width;
        const h = traceCtx.canvas.height;
        traceCtx.clearRect(0, 0, w, h);
        traceCtx.fillStyle = '#fffaf4';
        traceCtx.fillRect(0, 0, w, h);
        if (history.length < 2) {
          traceCtx.fillStyle = 'rgba(26, 26, 26, 0.6)';
          traceCtx.font = '14px Palatino, serif';
          traceCtx.fillText('Click a cell to pin its trace.', 12, 24);
          return;
        }
        const max = Math.max(...history, 30);
        const min = Math.min(...history, -90);
        const span = Math.max(max - min, 1);
        traceCtx.strokeStyle = '#f05a3f';
        traceCtx.lineWidth = 2;
        traceCtx.beginPath();
        history.forEach((value, idx) => {
          const x = (idx / (history.length - 1)) * w;
          const y = h - ((value - min) / span) * h;
          if (idx === 0) traceCtx.moveTo(x, y);
          else traceCtx.lineTo(x, y);
        });
        traceCtx.stroke();
      }

      async function poll() {
        try {
          const response = await fetch('/state');
          const data = await response.json();
          document.getElementById('step').textContent = data.step ?? 0;
          document.getElementById('mean').textContent = (data.mean_v ?? 0).toFixed(2);
          document.getElementById('spikes').textContent = data.spike_count ?? 0;
          document.getElementById('rate').textContent = (data.firing_rate_hz ?? 0).toFixed(2);
          document.getElementById('excRate').textContent = (data.exc_rate_hz ?? 0).toFixed(2);
          document.getElementById('inhRate').textContent = (data.inh_rate_hz ?? 0).toFixed(2);
          document.getElementById('status').textContent = data.paused ? 'Paused' : 'Running';
          document.getElementById('toggle').textContent = data.paused ? 'Resume' : 'Pause';

          meanData.push(data.mean_v ?? 0);
          spikeData.push(data.spike_count ?? 0);
          excRateData.push(data.exc_rate_hz ?? 0);
          inhRateData.push(data.inh_rate_hz ?? 0);
          if (meanData.length > 120) meanData.shift();
          if (spikeData.length > 120) spikeData.shift();
          if (excRateData.length > 120) excRateData.shift();
          if (inhRateData.length > 120) inhRateData.shift();

          drawChart(meanCtx, meanData, '#bc3a1a');
          drawChart(spikeCtx, spikeData, '#0d5c63');
          const sampleV = data.sample_v ?? [];
          const sampleInh = data.sample_inhibitory ?? [];
          const sampleSpikes = data.sample_spikes ?? [];
          const sampleTypes = data.sample_types ?? [];
          const sampleEdges = data.sample_edges ?? [];
          lastSampleV = sampleV;
          lastSampleInh = sampleInh;
          lastSampleSpikes = sampleSpikes;
          lastSampleTypes = sampleTypes;
          drawMap(sampleV, sampleInh, sampleSpikes, sampleTypes, data.step ?? 0, data.pulse_glow ?? 0);
          drawTopology(sampleV, sampleInh, sampleSpikes, sampleEdges, data.step ?? 0);
          rasterHistory.push(sampleSpikes);
          if (rasterHistory.length > 120) rasterHistory.shift();
          drawRaster(rasterHistory);

          if (pinnedIndex !== null && sampleV[pinnedIndex] !== undefined) {
            traceHistory.push(sampleV[pinnedIndex]);
            if (traceHistory.length > 120) traceHistory.shift();
          }
          drawTrace(traceHistory);
          drawDualChart(eiCtx, excRateData, inhRateData, '#f05a3f', '#0f5a6b');
        } catch (err) {
          document.getElementById('status').textContent = 'Disconnected';
        }
      }

      document.getElementById('toggle').addEventListener('click', async () => {
        const paused = document.getElementById('toggle').textContent === 'Resume';
        await fetch(`/control?action=${paused ? 'resume' : 'pause'}`);
        poll();
      });

      document.getElementById('stepOnce').addEventListener('click', async () => {
        await fetch('/control?action=step');
        poll();
      });

      document.getElementById('pulse').addEventListener('click', async () => {
        await fetch('/control?action=pulse');
        poll();
      });

      document.getElementById('modeToggle').addEventListener('click', () => {
        mapMode = mapMode === 'voltage' ? 'spike' : 'voltage';
        document.getElementById('modeToggle').textContent =
          mapMode === 'voltage' ? 'Spike Heat' : 'Voltage Map';
      });

      document.getElementById('applyConfig').addEventListener('click', async () => {
        const noiseMin = parseFloat(document.getElementById('noiseMin').value);
        const noiseMax = parseFloat(document.getElementById('noiseMax').value);
        const pulseMag = parseFloat(document.getElementById('pulseMag').value);
        const pulseFrac = parseFloat(document.getElementById('pulseFrac').value);
        await fetch(
          `/control?action=config&noise_min=${noiseMin}&noise_max=${noiseMax}` +
          `&pulse_mag=${pulseMag}&pulse_fraction=${pulseFrac}`
        );
      });

      async function applyScenario(config) {
        document.getElementById('noiseMin').value = config.noiseMin;
        document.getElementById('noiseMax').value = config.noiseMax;
        document.getElementById('pulseMag').value = config.pulseMag;
        document.getElementById('pulseFrac').value = config.pulseFrac;
        await fetch(
          `/control?action=config&noise_min=${config.noiseMin}&noise_max=${config.noiseMax}` +
          `&pulse_mag=${config.pulseMag}&pulse_fraction=${config.pulseFrac}` +
          `&connection_prob=${config.connectionProb}&weight_exc=${config.weightExc}` +
          `&weight_inh=${config.weightInh}&delay_max=${config.delayMax}`
        );
      }

      document.getElementById('scenarioBaseline').addEventListener('click', () => {
        applyScenario({
          noiseMin: 0.0,
          noiseMax: 0.3,
          pulseMag: 6.0,
          pulseFrac: 0.02,
          connectionProb: 0.02,
          weightExc: 2.5,
          weightInh: -3.5,
          delayMax: 16.0,
        });
      });

      document.getElementById('scenarioBurst').addEventListener('click', () => {
        applyScenario({
          noiseMin: 0.0,
          noiseMax: 1.2,
          pulseMag: 14.0,
          pulseFrac: 0.06,
          connectionProb: 0.04,
          weightExc: 4.5,
          weightInh: -6.5,
          delayMax: 18.0,
        });
      });

      document.getElementById('scenarioOsc').addEventListener('click', () => {
        applyScenario({
          noiseMin: 0.2,
          noiseMax: 0.9,
          pulseMag: 10.0,
          pulseFrac: 0.04,
          connectionProb: 0.05,
          weightExc: 3.8,
          weightInh: -4.2,
          delayMax: 12.0,
        });
      });

      document.getElementById('scenarioRipple').addEventListener('click', () => {
        applyScenario({
          noiseMin: 0.0,
          noiseMax: 0.6,
          pulseMag: 12.0,
          pulseFrac: 0.05,
          connectionProb: 0.03,
          weightExc: 3.6,
          weightInh: -4.8,
          delayMax: 28.0,
        });
      });

      mapCanvas.addEventListener('mousemove', (event) => {
        const rect = mapCanvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        const cols = 22;
        const rows = Math.max(1, Math.ceil(lastSampleV.length / cols));
        const cellW = rect.width / cols;
        const cellH = rect.height / rows;
        const col = Math.floor(x / cellW);
        const row = Math.floor(y / cellH);
        const idx = row * cols + col;
        if (idx >= 0 && idx < lastSampleV.length) {
          const v = lastSampleV[idx].toFixed(2);
          const kind = lastSampleInh[idx] ? 'inhibitory' : 'excitatory';
          const spike = lastSampleSpikes[idx] ? 'spike' : 'quiet';
          const type = lastSampleTypes[idx] || 'regular';
          tooltip.textContent = `Sample ${idx} | V ${v} | ${kind} | ${type} | ${spike}`;
          tooltip.style.left = `${event.clientX + 12}px`;
          tooltip.style.top = `${event.clientY + 12}px`;
          tooltip.style.opacity = '1';
        } else {
          tooltip.style.opacity = '0';
        }
      });

      mapCanvas.addEventListener('mouseleave', () => {
        tooltip.style.opacity = '0';
      });

      mapCanvas.addEventListener('click', (event) => {
        const rect = mapCanvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        const cols = 22;
        const rows = Math.max(1, Math.ceil(lastSampleV.length / cols));
        const cellW = rect.width / cols;
        const cellH = rect.height / rows;
        const col = Math.floor(x / cellW);
        const row = Math.floor(y / cellH);
        const idx = row * cols + col;
        if (idx >= 0 && idx < lastSampleV.length) {
          if (event.shiftKey) {
            pinnedIndex = idx;
            traceHistory.length = 0;
          } else {
            fetch(`/control?action=neuron&sample_index=${idx}&type=${selectedType}`);
          }
        }
      });

      poll();
      setInterval(poll, 200);

      document.querySelectorAll('.builder-buttons button[data-type]').forEach((button) => {
        button.addEventListener('click', () => {
          document.querySelectorAll('.builder-buttons button[data-type]').forEach((btn) => {
            btn.classList.remove('active');
          });
          button.classList.add('active');
          selectedType = button.dataset.type;
        });
      });

      document.getElementById('clearCircuit').addEventListener('click', () => {
        selectedType = 'regular';
        document.querySelectorAll('.builder-buttons button[data-type]').forEach((btn) => {
          btn.classList.toggle('active', btn.dataset.type === 'regular');
        });
        fetch('/control?action=clear_circuit');
      });
    </script>
  </body>
</html>
"""


def run_server(runner: SimulationRunner, host: str, port: int) -> None:
    SimulationServer.runner = runner
    httpd = ThreadingHTTPServer((host, port), SimulationServer)
    httpd.serve_forever()


def run_simulation(runner: SimulationRunner, sleep_s: float = 0.0) -> None:
    sleep_s = max(sleep_s, 0.001)
    while True:
        runner.step()
        if sleep_s > 0:
            time.sleep(sleep_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="High-fidelity brain simulator")
    parser.add_argument("--neuron-count", type=int, default=0)
    parser.add_argument("--auto-scale", action="store_true")
    parser.add_argument("--dt-ms", type=float, default=1.0)
    parser.add_argument("--connection-prob", type=float, default=0.05)
    parser.add_argument("--inhibitory-ratio", type=float, default=0.2)
    parser.add_argument("--weight-exc", type=float, default=5.0)
    parser.add_argument("--weight-inh", type=float, default=-5.0)
    parser.add_argument("--delay-min-ms", type=float, default=1.0)
    parser.add_argument("--delay-max-ms", type=float, default=20.0)
    parser.add_argument("--noise-min", type=float, default=0.0)
    parser.add_argument("--noise-max", type=float, default=5.0)
    parser.add_argument("--pulse-every", type=int, default=0)
    parser.add_argument("--pulse-mag", type=float, default=20.0)
    parser.add_argument("--pulse-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    neuron_count = args.neuron_count
    if args.auto_scale or neuron_count <= 0:
        neuron_count = auto_neuron_count()

    config = SimulationConfig(
        neuron_count=neuron_count,
        dt_ms=args.dt_ms,
        connection_prob=args.connection_prob,
        inhibitory_ratio=args.inhibitory_ratio,
        weight_exc=args.weight_exc,
        weight_inh=args.weight_inh,
        delay_ms_range=(args.delay_min_ms, args.delay_max_ms),
        noise_current=(args.noise_min, args.noise_max),
        pulse_every_steps=args.pulse_every,
        pulse_magnitude=args.pulse_mag,
        pulse_fraction=args.pulse_fraction,
        seed=args.seed,
    )

    network = build_network(config)
    runner = SimulationRunner(config, network)

    if args.server:
        sleep_s = args.sleep if args.sleep > 0 else (config.dt_ms / 1000.0)
        sim_thread = threading.Thread(target=run_simulation, args=(runner, sleep_s), daemon=True)
        sim_thread.start()
        host_display = "127.0.0.1" if args.host == "0.0.0.0" else args.host
        print(f"Brain simulator server running at http://{host_display}:{args.port}/")
        run_server(runner, args.host, args.port)
        return

    steps = args.steps if args.steps > 0 else 1000
    for _ in range(steps):
        runner.step()
        if runner.step_index % 50 == 0:
            snapshot = runner.snapshot()
            print(
                f"step={snapshot.get('step')} mean_v={snapshot.get('mean_v'):.2f} "
                f"spikes={snapshot.get('spike_count')} firing_hz={snapshot.get('firing_rate_hz'):.2f}"
            )
        if args.sleep > 0:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
