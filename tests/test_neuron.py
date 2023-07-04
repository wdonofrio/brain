from brain.neuron import Neuron, Brain

def test_neuron_update():
    n1 = Neuron(1, 0, [])
    n2 = Neuron(2, 0, [])
    n3 = Neuron(3, 0, [])
    n1.links = [n2, n3]

    n1.update()

    assert n1.value == (n2.value + n3.value) / 2


def test_brain_build():
    n1 = Neuron(1, 0, [])
    n2 = Neuron(2, 0, [])
    n3 = Neuron(3, 0, [])
    the_brain = Brain(0, [n1, n2, n3])

    the_brain.build(2)

    assert len(n1.links) <= 25
    assert len(n2.links) <= 25
    assert len(n3.links) <= 25


def test_brain_ping():
    n1 = Neuron(1, 0, [])
    the_brain = Brain(0, [n1])

    the_brain.ping(n1)

    assert -100 <= n1.value <= 100


def test_brain_touch():
    n1 = Neuron(1, 0, [])
    the_brain = Brain(0, [n1])

    the_brain.touch(5)

    assert -100 <= n1.value <= 100
