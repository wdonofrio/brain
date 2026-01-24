from brain.neuron import IzhikevichParams, NeuronState, Network, Synapse


def test_izhikevich_spike_reset():
    params = IzhikevichParams(a=0.02, b=0.2, c=-65.0, d=8.0)
    neuron = NeuronState(v=30.0, u=0.0, params=params)
    network = Network([neuron], [[]], dt_ms=1.0, max_delay_steps=1)

    spikes = network.step([0.0])

    assert spikes == [0]
    assert neuron.v == params.c
    assert neuron.u >= params.d


def test_synapse_delay_scheduling():
    n0 = NeuronState(v=30.0)
    n1 = NeuronState(v=-65.0)
    synapses = [[Synapse(target=1, weight=10.0, delay_steps=2)], []]
    network = Network([n0, n1], synapses, dt_ms=1.0, max_delay_steps=2)

    network.step([0.0, 0.0])

    scheduled = network._input_ring[1][2]
    assert scheduled == 10.0
