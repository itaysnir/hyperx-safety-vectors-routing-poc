import hyperx
import traffic_generator as tg
import routing_simulator

def print_hi():

    # hx = hyperx.HyperX(4, [3,3,3,6], 1, 5, 0.6)
    # hx = hyperx.HyperX(3, [2,2,3], 1, 5, 0.2)
    hx = hyperx.HyperX(3, [4, 2, 3], 1, 5, 0.15)
    adj_matrix = hx.adjacency_matrix()
    traffic_mat = tg.generate_uniform_traffic(hx.num_switches)
    # maximal_link_usage = hx.route_wcmp(traffic_mat)
    # print('MLU: {}'.format(maximal_link_usage))

    dal = hx.route_dal(traffic_mat)
    safety = hx.safety_vectors_route()
    if dal > safety:
        winner = 'safety'
    else:
        winner = 'dal'
    print('Average DAL hop count: {} Safety vectors hop count: {} Winner: {}'.format(dal, safety, winner))
    return winner


def main():
    counter = 0
    safety_won_times = 0
    total_run_times = 50
    while (counter < total_run_times):
        counter+=1
        if (print_hi() == 'safety'):
            safety_won_times += 1

    print('Safety win percantage: {}'.format(safety_won_times / total_run_times))

    routing_simulator.run()

if __name__ == '__main__':
    main()
