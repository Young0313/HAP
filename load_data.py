import networkx as nx
from networkx import LFR_benchmark_graph
from cdlib import NodeClustering
from utils import Community
import community_enhance
import Link_Pred


def LFR_generator(n=1000, tau1=3, tau2=1.1, mu=0.1, average_degree=20, max_degree=80,
                  min_community=10, max_community=50, maximum=50):
    counter = 0
    graph_list = []
    ground_truth_list = []
    while counter < maximum:
        try:
            G = LFR_benchmark_graph(n=n, tau1=tau1, tau2=tau2, mu=mu, average_degree=average_degree,
                                    max_degree=max_degree, min_community=min_community, max_community=max_community)
            community = {frozenset(G.nodes[v]["community"]) for v in G}
            ground_truth = []
            for i in community:
                ground_truth.append(list(i))
            edges = nx.edges(G)
            for edge in edges:
                if edge[0] == edge[1]:
                    G.remove_edge(edge[0], edge[0])
            graph_list.append(G)
            ground_truth_list.append(ground_truth)
            counter += 1
        except nx.exception.ExceededMaxIterations:
            continue
    return graph_list, ground_truth_list


def batch_graph_generator(mu=None, n=1000, tau1=3, tau2=1.1, average_degree=10, max_degree=50,
                          min_community=10, max_community=50, maximum=1):
    if mu is None:
        mu = [0.1, 0.2, 0.3, 0.4]
        # mu = [0.1, 0.2, 0.3, 0.4, 0.5]
    graph_list_result = []
    ground_true_result = []
    for i in range(len(mu)):
        graph_list, ground_truth_list = LFR_generator(n, tau1, tau2, mu[i], average_degree, max_degree,
                                                      min_community, max_community, maximum)
        graph_list_result.append(graph_list[0])
        ground_true_result.append(Community(NodeClustering(ground_truth_list[0], graph = graph_list[0], method_name='ground_truth')
                                            ,graph_list[0]))
    return graph_list_result, ground_true_result


def load_data(filename):
    root = './datasets/'
    dataset = nx.read_gml(root + filename + '.gml', label='id', destringizer=int)
    dataset_cluster = community_enhance.construct_ground_truth(dataset)
    output = Link_Pred.LinkPrediction(dataset_cluster)
    return output
