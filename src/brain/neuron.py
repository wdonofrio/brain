import random
import time
from typing import List
from math import floor


class Neuron:
    i: int = 0
    value: int = 0
    links: List[int] = []

    def __init__(self):
        self.value = 0
        self.links = []

    def update(self):
        """Get value from average of connections"""
        self.value = floor(
            (sum(x.value for x in self.links) + self.value) / (len(self.links) + 1)
        )

    def display(self):
        print("ID:%s Value:%s" % (self.i, self.value))
        for n in self.links:
            print("Linked to ID: %s Val: %s" % (n.i, n.value))


class Brain:
    neurons: List[Neuron] = []
    amount = 1

    def __init__(self, amount, neurons):
        self.neurons = neurons
        for _ in range(amount):
            self.neurons.append(Neuron())

    def build(self, amount):
        temp = []
        for neuron in self.neurons:
            if abs(neuron.value) > 50:
                temp.append(neuron)
        for _ in range(amount):
            if temp:
                x = random.choice(temp)
            else:
                x = random.choice(self.neurons)
            if len(x.links) < 25:
                x.links.append(random.choice(self.neurons))

    def update(self):
        for neuron in self.neurons:
            neuron.update()

    def display(self):
        for neuron in self.neurons:
            neuron.display()
            time.sleep(0.01)

    def displayVals(self):
        store = ""
        for neuron in self.neurons:
            store += f"{neuron.value} "
        print(store)

    def ping(self, neuron):
        neuron.value += random.randint(-20, 20)
        if neuron.value > 100:
            neuron.value = 100
        elif neuron.value < -100:
            neuron.value = -100

    def touch(self, amount):
        for _ in range(amount):
            self.ping(random.choice(self.neurons))


def main():
    """
    The entry function for pyproject to execute.
    """
    the_brain = Brain(50, [])
    while True:
        time.sleep(0.05)
        the_brain.build(random.randint(1, 5))
        the_brain.touch(random.randint(10, 30))
        the_brain.update()
        the_brain.displayVals()
        # the_brain.neurons[20].display()


if __name__ == "__main__":
    main()
