import sys
from functools import reduce
import random
import copy


def cart_product(set1, set2):
    cart_prod = []
    for element1 in set1:
        for element2 in set2:
            cart_prod.append(element1 + element2)

    return cart_prod

# finds the number of switches in the topology
def num_switches(L, S):
    result = 1
    if (len(S) == 1):
        result = pow(S[0], L)
    else:
        for dim in S:
            result *= dim

    return result

# calculates the beta parameter of the HyperX, assuming constant K
def calculate_beta(S, K, T):

    return (min(S) * K / (2 * T))

# calculates the number of neighbors each switch connected to
def num_neighbors(S):
    num_neighbors = 0
    for i in S:
        num_neighbors += (i - 1)

    return num_neighbors

# splits the universe set into 2 subsets, A and B, such that A union B equals universe.
# need to receive copy
def splitset(universe1):
    universe = list(universe1)
    size_A = random.randint(1, len(universe) / 2 + 1)
    A = []
    B = []
    for i in range(size_A):
        index = random.randint(0, len(universe) - 1)
        if index >= len(universe):
            print
            "universe : {}   ,   index : {}".format(len(universe), index)
        assert (index < len(universe))
        A.append(universe[index])
        del universe[index]
    B = universe
    return A, B


class HyperX:
    def __init__(self, L_arg, S_arg, K_arg, T_arg, faulty_ratio = 0):
        self.L = L_arg
        self.S = S_arg  ## Note that
        assert (len(self.S) == 1 or len(self.S) == L_arg)
        if len(self.S) == 1:
            self.S = [self.S[0]] * self.L
        self.K = K_arg
        self.T = T_arg
        self.num_switches = num_switches(L_arg, S_arg)
        self.d = num_neighbors(S_arg)
        self.num_links = int(self.num_switches * self.d / 2)
        self.beta = calculate_beta(S_arg, K_arg, T_arg)
        self.adjacency_list = {}
        self.safety_vectors = {}
        self.safety_vectors_dims = {}
        self.links_list = {}
        self.faulty_links = []
        self.id_to_coordinates = {}
        self.coordinates_to_id = {}
        self.link_capacity = 100  # in Gbps
        # print self.S
        # first generate all the switches
        self.create_switches()
        self.wire_network()
        self.create_faulty_links(faulty_ratio)
        self.create_safety_vectors()
        return

    def create_switches(self):
        # num_switches = product = reduce((lambda x, y: x * y), self.S)
        # dims = [0] * self.L
        # First step is to create all of the switches, this can be done by just generating the cartesian product of all the coordinates
        all_coords = []
        for i in range(self.S[-1]):
            all_coords.append((i,))
        for dim2 in range(self.L - 2, -1, -1):
            coords1 = []
            for i in range(self.S[dim2]):
                coords1.append((i,))
            all_coords = cart_product(coords1, all_coords)
        swid = 0
        for coord in all_coords:
            self.adjacency_list[coord] = []
            self.safety_vectors_dims[coord] = []
            self.id_to_coordinates[swid] = coord
            self.coordinates_to_id[coord] = swid
            swid += 1
        # print all_coords
        #print("total number of switches = " + str(len(all_coords)))
        return

    def wire_network(self):
        # Next step is to wire all of them together
        num_neighbors = 0
        for i in self.S:
            num_neighbors += (i - 1)
        link_id = 0
        for src in self.adjacency_list.keys():
            for dst in self.adjacency_list.keys():
                if (not self.share_dimension(src, dst)) or (dst in self.adjacency_list[src]):
                    continue
                else:
                    self.adjacency_list[src].append(dst)
                    self.links_list[link_id] = (src, dst)
                    link_id += 1
            assert (len(self.adjacency_list[src]) == num_neighbors)
        return

    # terminating 'ratio' (number between 0 to 1) percentage of random links
    def create_faulty_links(self, faulty_ratio):
        fallen_links_no = int(faulty_ratio * self.num_links)
        fallen_links_ids = random.sample(range(self.num_links), fallen_links_no)
        #print('{} Faulty links!'.format(fallen_links_no))
        for link_id in fallen_links_ids:
            (src, dst) = self.links_list[link_id]
            if (dst in self.adjacency_list[src]):
                self.adjacency_list[src].remove(dst)
                self.faulty_links.append((src, dst))
            if (src in self.adjacency_list[dst]):
                self.adjacency_list[dst].remove(src)
                self.faulty_links.append((dst, src))

        return

    # checks to see if there is at least one dimension, returns True is so, and False otherwise
    def share_dimension(self, coord1, coord2):
        see_diff = False
        for i in range(len(coord1)):
            if coord1[i] == coord2[i]:
                continue
            elif coord1[i] != coord2[i] and see_diff:
                return False
            else:
                see_diff = True
        return see_diff

    def cheeger_constant(self):
        min_so_far = 10E10
        for i in range(1000):
            A, B = splitset(self.adjacency_list.keys())
            boundary_links = 0
            for elem in A:
                for neighbor in self.adjacency_list[elem]:
                    if neighbor in B:
                        boundary_links += 1
            min_so_far = min(float(boundary_links) / len(A), min_so_far)
        return min_so_far

    def dijkstra(self, adjacency_list, src):
        nnodes = self.num_switches
        visited = [False] * nnodes
        distance = [100] * nnodes
        visited[src] = True
        distance[src] = 0
        stack = [src]
        while len(stack) > 0:
            curr_node = stack.pop()
            visited[curr_node] = True
            #for neighbor in adjacency_list[curr_node]:
            potential_neighbors = adjacency_list[curr_node]
            potential_neighbors_count = len(potential_neighbors)
            for i in range(potential_neighbors_count):
                if (potential_neighbors[i] == 1):
                    neighbor = i
                    if not visited[neighbor]:
                        stack.append(neighbor)
                    distance[neighbor] = min(distance[neighbor], distance[curr_node] + 1)
                else:
                    continue
        return distance

    def adjacency_matrix(self):
        num_switches = self.num_switches
        mat = [0] * num_switches
        offset = 0
        coord_to_index = {}
        for sw_coord in self.adjacency_list.keys():
            coord_to_index[sw_coord] = offset
            mat[offset] = [0] * num_switches
            offset += 1
        for sw in self.adjacency_list.keys():
            for neighbor in self.adjacency_list[sw]:
                mat[coord_to_index[sw]][coord_to_index[neighbor]] += 1
        return mat

    def convert_to_cords(self, path):
        result = []
        for sw_id in path:
            result.append(self.id_to_coordinates[sw_id])

        return result

    def path_recur(self, offset_dims):
        if len(offset_dims) == 0:
            return []
        elif len(offset_dims) == 1:
            return [offset_dims]
        else:
            all_paths = []
            for i in range(len(offset_dims)):
                copied_list = copy.copy(offset_dims)
                copied_list.pop(i)
                collection = self.path_recur(copied_list)
                for item in collection:
                    item.append(offset_dims[i])
                    all_paths.append(item)
            return all_paths

    def find_offset_dimensions(self, src_coord, dst_coord):
        offset_dimensions = []
        for i in range(len(src_coord)):
            if src_coord[i] != dst_coord[i]:
                offset_dimensions.append(i)

        return offset_dimensions
    # returns a set of paths (list of vertices which are id of switches) which are shortest paths
    # between a src and dst
    def shortest_path_set(self, adj_matrix, src, dst):
        src_coord = self.id_to_coordinates[src]
        dst_coord = self.id_to_coordinates[dst]
        offset_dimensions = self.find_offset_dimensions(src_coord, dst_coord)

        all_dim_sequences = self.path_recur(offset_dimensions)
        # print('Dimension Sequences: {}'.format(all_dim_sequences))
        all_paths = []
        all_paths_cords = []
        # the path_recur function merely returns all permutations of sequences of dimensions to take
        # we still need to transform them into id's of the switches
        for dim_seq in all_dim_sequences:
            path = [src, ]
            path_cords = [self.id_to_coordinates[src], ]
            curr_coord = list(self.id_to_coordinates[src])
            '''
            link = (tuple(curr_coord), dst_coord)
            if link in self.faulty_links or link[::-1] in self.faulty_links:
                print('Link {} is invalid'.format(link))
                continue
            '''
            for dim_ind in dim_seq:
                curr_coord[dim_ind] = dst_coord[dim_ind]
                path.append(self.coordinates_to_id[tuple(curr_coord)])
                path_cords.append(tuple(curr_coord))
            all_paths.append(path)
            all_paths_cords.append(path_cords)

        result = []
        for path in all_paths:
            for i in range(len(path) - 1):
                if (self.id_to_coordinates[path[i]], self.id_to_coordinates[path[i+1]]) not in self.faulty_links:
                    result.append(path)

        return result

    '''
    def shortest_path_set_cords(self, adj_matrix, src, dst):
        src_coord = self.id_to_coordinates[src]
        dst_coord = self.id_to_coordinates[dst]
        offset_dimensions = []
        for i in range(len(src_coord)):
            if src_coord[i] != dst_coord[i]:
                offset_dimensions.append(i)
        all_dim_sequences = self.path_recur(offset_dimensions)
        # print('Dimension Sequences: {}'.format(all_dim_sequences))
        all_paths = []
        # the path_recur function merely returns all permutations of sequences of dimensions to take
        # we still need to transform them into id's of the switches
        for dim_seq in all_dim_sequences:
            path = [self.id_to_coordinates[src], ]
            curr_coord = list(self.id_to_coordinates[src])
            for dim_ind in dim_seq:
                curr_coord[dim_ind] = dst_coord[dim_ind]
                path.append(tuple(curr_coord))
            all_paths.append(path)
        return all_paths
    '''

    # routes a traffic matrix using wcmp
    def route_wcmp(self, traffic_matrix):
        adj_matrix = self.adjacency_matrix()
        num_switches = len(adj_matrix)
        traffic_load = [0] * num_switches  # traffic load on each link between links
        for i in range(num_switches):
            traffic_load[i] = [0] * num_switches
        for src in range(num_switches):
            for dst in range(num_switches):
                if src == dst:
                    continue
                path_total_weight = 0
                path_set = self.shortest_path_set(adj_matrix, src, dst)
                path_weight = []
                for path in path_set:
                    path_min_capacity = self.link_capacity
                    curr_node = path[0]
                    for i in range(1, len(path), 1):
                        intermediate_swid = path[i]
                        path_min_capacity = min(float(path_min_capacity),
                                                adj_matrix[curr_node][intermediate_swid] * self.link_capacity)
                        curr_node = intermediate_swid
                    path_weight.append(path_min_capacity)
                    path_total_weight += path_min_capacity
                if (0 == path_total_weight):
                    break
                path_index = 0
                for path in path_set:
                    curr_node = path[0]
                    for i in range(1, len(path), 1):
                        intermediate_swid = path[i]
                        traffic_load[curr_node][intermediate_swid] += (
                                    traffic_matrix[src][dst] * (path_weight[path_index] / path_total_weight))
                        curr_node = intermediate_swid
                    path_index += 1
        # compute the mlu
        mlu = 0
        for src in range(num_switches):
            for dst in range(num_switches):
                if src == dst:
                    continue
                if adj_matrix[src][dst] <= 0:
                    continue
                mlu = max(mlu, traffic_load[src][dst] / adj_matrix[src][dst] / self.link_capacity)

        return mlu / (1. / float(self.num_switches * (self.num_switches - 1)))

    # returns a neighbor that shares offset dimension 'dim'
    def dal_find_neighbor(self, curr_coord, dim):
        result = None
        for neighbor_cord in self.adjacency_list[curr_coord]:
            if dim in self.find_offset_dimensions(curr_coord, neighbor_cord):
                result = neighbor_cord
                break

        return result

    def dal_find_better_neighbor(self, curr_coord, dst_coord):
        result = None
        curr_delta = len(self.find_offset_dimensions(curr_coord, dst_coord))
        for neighbor_cord in self.adjacency_list[curr_coord]:
            neighbor_delta = len(self.find_offset_dimensions(neighbor_cord, dst_coord))
            if (neighbor_delta < curr_delta): # found a neighbor that is closer to destination
                result = neighbor_cord
                break

        return result

    def dal_find_neighbor_deroute(self, curr_coord, dst_coord, dim):
        result = None
        curr_delta = len(self.find_offset_dimensions(curr_coord, dst_coord))
        for neighbor_cord in self.adjacency_list[curr_coord]:
            offset_dims = self.find_offset_dimensions(neighbor_cord, dst_coord)
            if dim in offset_dims:
                neighbor_delta = len(offset_dims)
                if (neighbor_delta == curr_delta): # found an neighbor within the same distance, not buffered
                    result = neighbor_cord
                    break

        return result

    def further_deroute_neighbor(self, curr_coord, dst_coord):
        result = None
        curr_delta = len(self.find_offset_dimensions(curr_coord, dst_coord))
        for neighbor_cord in self.adjacency_list[curr_coord]:
            second_neighbor_coord = self.dal_find_better_neighbor(neighbor_cord, dst_coord)
            if second_neighbor_coord != None:
                second_neighbor_delta = len(self.find_offset_dimensions(second_neighbor_coord, dst_coord))
                if (second_neighbor_delta < curr_delta):  # found a neighbor that is closer to destination
                    result = second_neighbor_coord
                    break

        return result

    # dal deroute
    def dal_deroute(self, is_routable_dims, curr_coord, dst_coord, hops):
        for dim in range(len(is_routable_dims)):
            if is_routable_dims[dim] == 'derouteable':
                neighbor_coord = self.dal_find_neighbor_deroute(curr_coord, dst_coord, dim)
                if neighbor_coord != None:
                    is_routable_dims[dim] = 'unrouteable'
                    hops += 1
                    return neighbor_coord  # found a good neighbor, after derouting.
                else:
                    if dim == (len(is_routable_dims) - 1):
                        return None
                    continue  # try other dim
        return None

    # returns the cost of virtual channel routing
    def virtual_channel_route(self, curr_coord, dst_coord):
        curr_delta = len(self.find_offset_dimensions(curr_coord, dst_coord))
        cost = curr_delta * 4

        return cost

        # returns the cost of virtual channel routing
    def virtual_channel_route_safety(self, curr_coord, dst_coord):
        curr_delta = len(self.find_offset_dimensions(curr_coord, dst_coord))
        cost = curr_delta * 2

        return cost

    def do_dal_routing(self, src, dst):
        hops = 0
        if src != dst:
            src_coord = self.id_to_coordinates[src]
            dst_coord = self.id_to_coordinates[dst]
            is_routable_dims = ['derouteable'] * self.L

            curr_coord = src_coord
            while (curr_coord != dst_coord):
                '''
                offset_dimensions = self.find_offset_dimensions(curr_coord, dst_coord)
                for dim in offset_dimensions:
                    neighbor_coord = self.dal_find_neighbor(curr_coord, dim)
                    if neighbor_coord != None:
                        curr_coord = neighbor_coord
                        break # only executed if the inner loop DID break
                    if is_routable_dims[dim] == 'derouteable':
                '''
                neighbor_coord = self.dal_find_better_neighbor(curr_coord, dst_coord)
                if neighbor_coord != None:
                    curr_coord = neighbor_coord
                else:
                    '''
                    for dim in range(len(is_routable_dims)):
                        if is_routable_dims[dim] == 'derouteable':
                            neighbor_coord = self.dal_find_neighbor_deroute(curr_coord, dst_coord, dim)
                            if neighbor_coord != None:
                                curr_coord = neighbor_coord
                                is_routable_dims[dim] = 'unrouteable'
                                hops += 1
                                break # found a good neighbor, after derouting.
                            else:
                                continue # try other dim
                    '''
                    neighbor_coord = self.dal_deroute(is_routable_dims, curr_coord, dst_coord, hops)
                    if neighbor_coord != None:
                        curr_coord = neighbor_coord
                        hops += 1

                    second_neighbor_coord = self.further_deroute_neighbor(curr_coord, dst_coord)
                    if second_neighbor_coord != None:
                        curr_coord = second_neighbor_coord
                        hops += 2
                    else:
                        hops += self.virtual_channel_route(curr_coord, dst_coord)
                        break

                hops += 1

        return hops

    def route_dal(self, traffic_matrix):
        adj_matrix = self.adjacency_matrix()
        hop_count = 0
        optimal_routes = 0
        num_switches = self.num_switches
        traffic_load = [0] * num_switches  # traffic load on each link between links
        for i in range(num_switches):
            traffic_load[i] = [0] * num_switches
        for src in range(num_switches):
            distance_vector = self.dijkstra(adj_matrix, src)
            for dst in range(num_switches):
                if src == dst:
                    continue
                hops = self.do_dal_routing(src, dst)
                hop_count += hops
                # path_set = self.shortest_path_set(adj_matrix, src, dst)
                minimal_distance = distance_vector[dst]
                if (hops == minimal_distance):
                    optimal_routes += 1

        average_hop_count = float(hop_count) / (num_switches * (num_switches - 1))
        optimal_route_ratio = float(optimal_routes) / (num_switches * (num_switches - 1))
        #print('DAL - optimal route ratio: {}'.format(optimal_route_ratio))

        return average_hop_count, optimal_route_ratio

    # return 'True' if one of the switch's links has fallen
    def is_an_switch_of_faulty_link(self, switch_coord):
        optimal_neighbors_no = self.d
        neighbors_no = len(self.adjacency_list[switch_coord])
        return optimal_neighbors_no != neighbors_no

    def safety_vectors_induction(self, switch_coord, k):
        if len(self.adjacency_list[switch_coord]) == 0:
            return

        u_k_i = 1
        sum = 0
        for i in range(self.L):
            u_k_i = 1
            for neighbor in self.adjacency_list[switch_coord]:
                if neighbor[i] != switch_coord[i]:
                    u_k_i = min(u_k_i, self.safety_vectors[neighbor][k-1])
            self.safety_vectors_dims[switch_coord][i] = u_k_i
            sum += u_k_i

        if sum < (self.L - k):
            self.safety_vectors[switch_coord][k] = 0
        else:
            self.safety_vectors[switch_coord][k] = 1


    def create_safety_vectors(self):
        for switch in self.adjacency_list.keys():
            self.safety_vectors_dims[switch] = self.L * [0]
            if self.is_an_switch_of_faulty_link(switch):
                self.safety_vectors[switch] = [0] + (self.L - 1) * [0]
            else: # note the shifted index offset by 1. index k means k+1 possible distance
                self.safety_vectors[switch] = [1] + (self.L - 1) * [0]

        for k in range(1, self.L):
            for switch in self.adjacency_list.keys():
                self.safety_vectors_induction(switch, k)

        for sw_coord in self.safety_vectors:
            self.safety_vectors[sw_coord] = [0] + self.safety_vectors[sw_coord] + [1]

        return

    def safety_vectors_find_neighbor(self, curr_coord, dst_coord, src_coord):
        if dst_coord in self.adjacency_list[curr_coord]:
            return dst_coord
        result = None
        preferred_dimensions = []
        preferred_neighbors = []
        semi_preferred_neighbors = []
        sparse_neighbors = []
        curr_delta = len(self.find_offset_dimensions(curr_coord, dst_coord))
        if (curr_coord == src_coord): # source node routing
            for neighbor_coord in self.adjacency_list[curr_coord]:
                neighbor_delta = len(self.find_offset_dimensions(neighbor_coord, dst_coord))
                if ((neighbor_delta < curr_delta and ((self.safety_vectors[src_coord][curr_delta - 1] == 1) or self.safety_vectors[src_coord][curr_delta] == 1))):
                    return neighbor_coord

            for neighbor_coord in self.adjacency_list[curr_coord]:
                neighbor_delta = len(self.find_offset_dimensions(neighbor_coord, dst_coord))
                if (neighbor_delta == curr_delta and ((self.safety_vectors[neighbor_coord][curr_delta] == 1) or (self.safety_vectors[src_coord][curr_delta + 1] == 1))):
                    return neighbor_coord

            for neighbor_coord in self.adjacency_list[curr_coord]:
                neighbor_delta = len(self.find_offset_dimensions(neighbor_coord, dst_coord))
                if (neighbor_delta == curr_delta):
                    for second_neighbor in self.adjacency_list[neighbor_coord]:
                        second_neighbor_delta = len(self.find_offset_dimensions(second_neighbor, dst_coord))
                        if (second_neighbor_delta < neighbor_delta): # debug
                            return neighbor_coord


            return None # No available path, faulty network - virtual channel

        else: # intermediate node routing
            for neighbor_coord in self.adjacency_list[curr_coord]:
                neighbor_delta = len(self.find_offset_dimensions(neighbor_coord, dst_coord))
                if (neighbor_delta < curr_delta):  # found a preferred neighbor
                    preferred_neighbors.append(neighbor_coord)
                    preferred_dim = self.find_offset_dimensions(curr_coord, neighbor_coord)[0]
                    preferred_dimensions.append(preferred_dim)

            for neighbor_coord in self.adjacency_list[curr_coord]:
                dim_diff = self.find_offset_dimensions(curr_coord, neighbor_coord)[0]
                if dim_diff in preferred_dimensions and neighbor_coord not in preferred_neighbors:
                    semi_preferred_neighbors.append(neighbor_coord)

            for neighbor_coord in self.adjacency_list[curr_coord]:
                if neighbor_coord not in semi_preferred_neighbors and neighbor_coord not in preferred_dimensions:
                    sparse_neighbors.append(neighbor_coord)

            for neighbor_coord in preferred_neighbors:
                if (self.safety_vectors[neighbor_coord][curr_delta - 1] == 1):
                    return neighbor_coord

            for neighbor_coord in preferred_neighbors: # check this
                return neighbor_coord

            for neighbor_coord in semi_preferred_neighbors:
                neighbor_delta = len(self.find_offset_dimensions(neighbor_coord, dst_coord))
                if (self.safety_vectors[neighbor_coord][curr_delta] == 1 and neighbor_delta == curr_delta):
                    for second_neighbor in self.adjacency_list[neighbor_coord]:
                        second_neighbor_delta = len(self.find_offset_dimensions(second_neighbor, dst_coord))
                        if (second_neighbor_delta < neighbor_delta):

                            return neighbor_coord

            for neighbor_coord in semi_preferred_neighbors:
                neighbor_delta = len(self.find_offset_dimensions(neighbor_coord, dst_coord))
                if (neighbor_delta == curr_delta):
                    for second_neighbor in self.adjacency_list[neighbor_coord]:
                        second_neighbor_delta = len(self.find_offset_dimensions(second_neighbor, dst_coord))
                        if (second_neighbor_delta < neighbor_delta):
                            return neighbor_coord

            for neighbor_coord in sparse_neighbors:
                neighbor_delta = len(self.find_offset_dimensions(neighbor_coord, dst_coord))
                if (self.safety_vectors[neighbor_coord][curr_delta] == 1 and neighbor_delta == curr_delta):
                        for second_neighbor in self.adjacency_list[neighbor_coord]:
                            second_neighbor_delta = len(self.find_offset_dimensions(second_neighbor, dst_coord))
                            if (second_neighbor_delta < neighbor_delta):
                                return neighbor_coord

            return result

    def do_safety_routing(self, src, dst):
        hops = 0
        if src != dst:
            src_coord = self.id_to_coordinates[src]
            dst_coord = self.id_to_coordinates[dst]

            curr_coord = src_coord
            while (curr_coord != dst_coord):
                neighbor_coord = self.safety_vectors_find_neighbor(curr_coord, dst_coord, src_coord)
                if neighbor_coord != None:
                    curr_coord = neighbor_coord
                    hops += 1
                else:
                    hops += self.virtual_channel_route_safety(curr_coord, dst_coord)
                    break

        return hops


    def safety_vectors_route(self):
        adj_matrix = self.adjacency_matrix()
        hop_count = 0
        optimal_routes = 0
        num_switches = self.num_switches
        traffic_load = [0] * num_switches  # traffic load on each link between links
        for i in range(num_switches):
            traffic_load[i] = [0] * num_switches
        for src in range(num_switches):
            distance_vector = self.dijkstra(adj_matrix, src)
            for dst in range(num_switches):
                if src == dst:
                    continue
                hops = self.do_safety_routing(src, dst)
                hop_count += hops
                # path_set = self.shortest_path_set(adj_matrix, src, dst)
                minimal_distance = distance_vector[dst]
                if (hops == minimal_distance):
                    optimal_routes += 1

        average_hop_count = float(hop_count) / (num_switches * (num_switches - 1))
        optimal_route_ratio = float(optimal_routes) / (num_switches * (num_switches - 1))
        # print('Safety Vectors - optimal route ratio: {}'.format(optimal_route_ratio))

        return average_hop_count, optimal_route_ratio
