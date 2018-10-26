#!/usr/bin/python

import random
import time

class neuron:
	i 		= 0
	value 	= 0
	links	= []

	def __init__(self, i, val, links):
		self.i			= i
		self.value 		= val
		self.links 		= links

	def update(self):
		"""Get value from average of connections"""
		try:
			avg = (sum(x.value for x in self.links) + self.value)/(len(self.links)+1)
			self.value = avg
			self.degrade()
			for neuron in self.links:
				neuron.value = avg;
		except:
			pass

	def degrade(self):
		if self.value > 20:
			self.value -= 10
		if self.value < -20:
			self.value += 10

	def display(self):
		print("ID:%s Value:%s" % (self.i, self.value))
		for n in self.links:
			print("Linked to ID: %s Val: %s" % (n.i, n.value))

class brain:
	neurons = []
	amount  = 1
	def __init__(self, amount):
		self.neurons = [0] * amount
		for i in range(amount):
		    self.neurons[i] = [0] * amount

		for i in range(amount):
			for j in range(amount):
				self.neurons[i][j] = (neuron([i,j], 0, []))

	def build(self):
		length = len(self.neurons) -1
		for x in range(length):
			for y in range(length):
				if abs(self.neurons[x][y].value) > 50:
					self.neurons[x][y].links.append(self.neurons[x+random.randint(-1,1)][y+random.randint(-1,1)])

	def update(self):
		length = len(self.neurons) -1
		for x in range(length):
			for y in range(length):
				self.neurons[x][y].update()

	def display(self):
		length = len(self.neurons) -1
		for x in range(length):
			for y in range(length):
				self.neurons[x][y].display()

	def displayVals(self):
		length = len(self.neurons) -1
		print("   "),
		for x in range(length):
			print(str('{:3d}'.format(x))),
		print("")
		for x in range(length):
			print(str('{:3d}'.format(x))),
			for y in range(length):
				print(str('{:3d}'.format(self.neurons[x][y].value))),
			print("")
		print("   ------------------------------------")

	def ping(self, neuron):
		neuron.value += random.randint(-20, 20)
		if neuron.value > 100:
			neuron.value = 100
		elif neuron.value < -100:
			neuron.value = -100

	def touch(self):
		length = len(self.neurons) -1
		x = random.randint(0,length)
		y = random.randint(0,length)
		t = self.neurons[x][y]
		self.ping(t)


if __name__ == "__main__":
	ai = brain(100)
	while True:
		# time.sleep(1)
		ai.build()
		ai.touch()
		ai.update()
		ai.displayVals()
		# ai.neurons[20].display()