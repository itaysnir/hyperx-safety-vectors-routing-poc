import sys
from functools import reduce
import random
import hyperx
# returns the distance between src and all the dest
def dijkstra(adjacency_list, src):
	nnodes = len(adjacency_list.keys())
	visited = [False] * nnodes
	distance = [100] * nnodes
	visited[src] = True
	distance[src] = 0
	stack = [src]
	while len(stack) > 0:
		curr_node = stack.pop()
		visited[curr_node] = True
		print('LIST:{}'.format(adjacency_list))
		for neighbor in adjacency_list[curr_node]:
			if not visited[neighbor]:
				stack.append(neighbor)
			distance[neighbor] = min(distance[neighbor], distance[curr_node] + 1)
	return distance

class Xpander:
	def __init__(self, num_metanodes, tor_radix):
		self.nmetanodes = num_metanodes
		self.ntors_per_metanode = num_metanodes - 1
		self.tor_radix = tor_radix
		self.adjacency_list = {}
		self.create_switches()
		self.connect_switches()
		print("diameter of this network is : {}".format(self.diameter()))

	# checks the diameter of the graph
	def diameter(self):
		max_dist = 0
		for i in self.adjacency_list.keys():
			dist_array = dijkstra(self.adjacency_list, i)
			max_dist = max(max_dist, max(dist_array))
		return max_dist

	# generates all of the switches
	def create_switches(self):
		for metanode in range(self.nmetanodes):
			offset = metanode * self.ntors_per_metanode
			for tor in range(self.ntors_per_metanode):
				self.adjacency_list[offset + tor] = []
		return

	# connects and wires the entire topology, employs k-lifting
	def connect_switches(self):
		metanode_offset = [0] * self.nmetanodes
		r = 0
		group_pairings = []
		for src_metanode in range(self.nmetanodes - 1):
			for dst_metanode in range(src_metanode + 1, self.nmetanodes, 1):
				group_pairings.append((src_metanode, dst_metanode, ))
		while r < self.tor_radix:
			# perform the lift
			# starting_metanode = random.randint(0, self.nmetanodes - 1)
			scrambled = [0] * self.nmetanodes
			for metanode in range(self.nmetanodes):
				vect = range(self.ntors_per_metanode)
				random.shuffle(vect)
				scrambled[metanode] = vect
			for metanode_pair in group_pairings:
				src = metanode_pair[0]
				dst = metanode_pair[1]
				offset_src = scrambled[src].pop()
				offset_dst = scrambled[dst].pop()
				sw1_index = src * self.ntors_per_metanode + offset_src
				sw2_index = dst * self.ntors_per_metanode + offset_dst
				self.adjacency_list[sw1_index].append(sw2_index)
				self.adjacency_list[sw2_index].append(sw1_index)
			r += 1


	def cheeger_constant(self):
		trials = 1000
		cc = [0] * trials
		for i in range(trials):
			A, B = hyperx.splitset(self.adjacency_list.keys())
			boundary_links = 0
			for elem in A:
				for neighbor in self.adjacency_list[elem]:
					if neighbor in B:
						boundary_links += 1
			cc[i] = (float(boundary_links) / len(A))
		return min(cc)

	def adjacency_matrix(self):
		num_switches = len(self.adjacency_list.keys())
		mat = [0] * num_switches
		for i in range(num_switches):
			mat[i] = [0] * num_switches
		for sw in self.adjacency_list.keys():
			for neighbor in self.adjacency_list[sw]:
				mat[sw][neighbor] += 1
		return mat
