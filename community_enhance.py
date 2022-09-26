import copy
import random
import networkx as nx
from cdlib.evaluation import normalized_mutual_information as nmi
from cdlib import NodeClustering
import utils
from Link_Pred import LinkPrediction
import numpy as np


def delete_edges(g, proportion, iter_num=10):
    result = []
    edge_delete = []
    for _ in range(iter_num):
        temp_Graph = copy.deepcopy(g)
        edges = list(g.edges())
        single_time_delete = []
        random.shuffle(edges)
        num = len(edges)
        num_of_del = int(num * proportion)
        counter, i = 0, 0
        # make sure that the graph is still connected
        while counter < num_of_del:
            link = [edges[i][0], edges[i][1]]
            temp_Graph.remove_edge(edges[i][0], edges[i][1])
            if nx.is_connected(temp_Graph):
                single_time_delete.append(link)
                counter += 1
                i += 1
            else:
                temp_Graph.add_edge(edges[i][0], edges[i][1])
                i += 1
        result.append(temp_Graph)
        edge_delete.append(single_time_delete)
    return result, edge_delete


def algorithm_application(graph_list, algorithm):
    algorithm_result = []
    for delta in graph_list:
        temp_delta = []
        for mu in delta:
            temp_mu = []
            for graph in mu:
                temp_mu.append(utils.Community(algorithm(graph), graph))
            temp_delta.append(temp_mu)
        algorithm_result.append(temp_delta)
    return algorithm_result


def add_edge(Link_pred, func, proportion):
    # Link_pred is supposed to be Link_pred.LinkPrediction data type
    # func is supposed to be a similarity-based link prediction function implemented in Link_Pred.LinkPrediction
    temp_graph = copy.deepcopy(Link_pred.graph)
    new_edge_list = []
    missing_edges = [i for i in nx.non_edges(Link_pred.graph)]
    L = int(len(missing_edges)*proportion)
    result = Link_pred.link_prediction_result(missing_edges, func)
    sorted_result = sorted(result, key=(lambda x: [x[1]]))[::-1]
    for i in range(L):
        new_edge_list.append(sorted_result[i][0])
    temp_graph.add_edges_from(new_edge_list)
    return temp_graph


def adding_edges(Link_pred, func, L, num_of_graph):
    # Link_pred is supposed to be Link_pred.LinkPrediction data type
    new_graph_list = []
    missing_edges = [i for i in nx.non_edges(Link_pred.graph)]
    link_prediction_result = Link_pred.normalized_link_prediction_result(missing_edges, func)
    edges_weight = []
    for i in link_prediction_result:
        edges_weight.append(i[1])
    J, q = utils.alias_setup(edges_weight)
    draft = utils.batch_alias_draw(J, q, L, num_of_graph)
    for i in draft:
        temp = copy.deepcopy(Link_pred.graph)
        edges_append = []
        for j in i:
            edges_append.append(missing_edges[j])
        temp.add_edges_from(edges_append)
        new_graph_list.append(temp)
    return new_graph_list


def co_community_matrix_generator(community_class_list):
    num_of_nodes = community_class_list[0].number_of_nodes
    co_matrix = np.zeros((num_of_nodes, num_of_nodes))
    for i in range(num_of_nodes):
        for j in range(i, num_of_nodes):
            temp = 0
            for k in range(len(community_class_list)):
                if community_class_list[k].community_list[i] == community_class_list[k].community_list[j]:
                    temp += 1
            co_matrix[i, j] = temp
            co_matrix[j, i] = temp
    return co_matrix/len(community_class_list)


def single_algorithm_application(graph, algorithm):
    return utils.Community(algorithm(graph), graph)


def update_graph(Link_pred, func, add_num, algorithm_community):
    temp_graph = copy.deepcopy(Link_pred.graph)
    important = 0
    new_edge_list = []
    missing_edges = [i for i in nx.non_edges(Link_pred.graph)]
    L = add_num
    result = Link_pred.link_prediction_result(missing_edges, func, algorithm_community)
    sorted_result = sorted(result, key=(lambda x: [x[1]]))[::-1]
    for i in range(L):
        x,y = sorted_result[i][0][0], sorted_result[i][0][1]
        new_edge_list.append(sorted_result[i][0])
        if algorithm_community.community_list[x] != algorithm_community.community_list[y] and\
                Link_pred.gt_community.community_list[x] == Link_pred.gt_community.community_list[y]:
            important += 1
    temp_graph.add_edges_from(new_edge_list)
    return temp_graph, important, new_edge_list


def early_stopping_cluster(cluster_result_list, rounds, threshold):
    # according to the difference among several latest clustering results
    # cluster_result_list is a python list contain all algorithm output, element is utils.Community datatype
    # threshold is a hyperparameter to decide when to stop the algorithm
    if len(cluster_result_list) <= rounds:
        return False
    relativity_list = [normalized_mutual_information(cluster_result_list[i], cluster_result_list[-1]) \
                       for i in range(len(cluster_result_list)-1)]
    if min(relativity_list[-rounds:]) >= threshold:
        return True
    else:
        return False


def early_stopping_reinforce_edges(reinforce_num_list, edge_num, rounds, threshold):
    # according to what type of edges are added to the graph
    # if the lp method majorly adds reinforcing edges for a user given round, deploy early-stopping
    # threshold is a hyperparameter to decide when to stop the algorithm
    reinforce_ratio_list = [reinforce_num_list[i]/edge_num for i in reinforce_num_list]
    if min(reinforce_ratio_list[-rounds:]) >= threshold:
        return True
    else:
        return False


def early_stopping_revise_edges(revise_num_list, edge_num, rounds, threshold):
    # according to what type of edges are added to the graph
    # if the lp method adds revising edges less than expected for a user given round, deploy early-stopping
    # threshold is a hyperparameter to decide when to stop the algorithm
    reinforce_ratio_list = [revise_num_list[i]/edge_num for i in revise_num_list]
    if min(reinforce_ratio_list[-rounds:]) <= threshold:
        return True
    else:
        return False


def single_graph_updates(Link_pred, detection, func, links, rounds=10, early_stop=None, stopping_rounds=None):
    if early_stop is not None:
        important_total = []
        communities = []
    Link_pred_copy = copy.deepcopy(Link_pred)
    community = single_algorithm_application(Link_pred_copy.graph, detection)
    result, important, new_edge_list = update_graph(Link_pred_copy, func, links, community)
    if early_stop == 'NMI':
        communities.append(community)
    elif early_stop == 'revise':
        important_total.append(important)
    Link_pred_copy.graph = result
    NMI = [normalized_mutual_information(community, Link_pred_copy.gt_community)]
    print('Initial NMI value is:',np.round(NMI[0], 4), end='\t')
    for _ in range(rounds-1):
        algorithm_output = single_algorithm_application(result, detection)
        result, important, new_edge_list = update_graph(Link_pred_copy, func, links, algorithm_output)
        if early_stop == 'NMI':
            communities.append(algorithm_output)
            if early_stopping_cluster(communities, stopping_rounds, threshold=0.95):
                print('Early stopping!')
                break
        elif early_stop == 'revise':
            important_total.append(important)
            if early_stopping_revise_edges(important_total, stopping_rounds, threshold=0.30):
                print('Early stopping!')
                break
        NMI.append(normalized_mutual_information(algorithm_output, Link_pred.gt_community))
    print('NMI value after iteration is:', np.round(NMI[-1], 4), end='\n')


# estimation methods
def relative_error(coms1, coms2):
    return (coms1.number_of_community - coms2.number_of_community)/coms2.number_of_community


def normalized_mutual_information(coms1, coms2):
    return nmi(coms1.clustering, coms2.clustering).score


def estimation_application(algorithm_result, ground_truth, estimation_method):
    scores = []
    for delta in range(len(algorithm_result)):
        score_delta = []
        for mu in range(len(algorithm_result[0])):
            score_mu = []
            for graph in range(len(algorithm_result[0][0])):
                score_mu.append(estimation_method(algorithm_result[delta][mu][graph], ground_truth[mu]))
            score_delta.append(score_mu)
        scores.append(score_delta)
    return scores


def convert_estimation(algorithm_result):
    result = []
    for i in range(len(algorithm_result)):
        sub_result = []
        for j in algorithm_result[i]:
            sub_result.append([sum(k) / len(k) for k in j])
        result.append(sub_result)
    return result


def extract_class_label(gml_file):
    result_dict = {}
    for i in gml_file.nodes:
        result_dict[i] = gml_file.nodes[i]['gt']
    return result_dict


def cluster_convert(clusters_dict):
    clusters = list(set(clusters_dict.values()))
    length = len(clusters)
    result = []
    for _ in range(length):
        result.append([])
    for u in clusters_dict.keys():
        result[clusters.index(clusters_dict[u])].append(u)
    return result


def construct_ground_truth(gml_file):
    class_labels = extract_class_label(gml_file)
    clusters = cluster_convert(class_labels)
    return utils.Community(NodeClustering(clusters, graph = gml_file, method_name='Ground Truth'), graph=gml_file)
