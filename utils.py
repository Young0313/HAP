import numpy as np
import numpy.random as npr
from cdlib import algorithms, NodeClustering


class Community: # define a data container, which will be used in link pred
    def __init__(self, clustering, graph):
        """
            self.clustering type cdlib.classes.node_clustering.NodeClustering.
            self.graph requires networkx.graph datatype.
            self.community type two dimension python list.
            self.community_list one dimension python list.
        """
        self.clustering = clustering
        self.graph = graph
        self.community = self.community_transfer()
        self.community_list = self.to_community_list()
        self.number_of_community = len(self.community)
        self.size_of_community = [len(i) for i in self.community]
        self.community_link_matrix = self.calculate_community_link_matrix()

    def community_transfer(self):
        communities = [[]]
        community_dict = self.clustering.to_node_community_map()
        for i in community_dict.keys():
            temp = community_dict[i][0]
            while temp + 1 > len(communities):
                communities.append([])
            communities[temp].append(i)
        return communities

    def to_community_list(self):
        total = 0
        for i in self.community:
            total += len(i)
        temp = [0] * total
        community_index = 0
        for i in self.community:
            for j in i:
                temp[j] = community_index
            community_index += 1
        return temp

    def calculate_community_link_matrix(self):
        matrix = np.zeros((self.number_of_community, self.number_of_community))
        for edge in self.graph.edges():
            community1 = self.community_list[edge[0]]
            community2 = self.community_list[edge[1]]
            matrix[community1, community2] += 1
            matrix[community2, community1] += 1
        return matrix


def alias_setup(probs):
    '''
    input： a discrete probability distribution
    return: Alias list and Prob list for future sampling work
    '''
    K = len(probs)
    q = np.zeros(K)  # Prob list
    J = np.zeros(K, dtype=int)  # Alias list
    # Sort the data into the outcomes with probabilities
    # that are larger and smaller than 1/K.
    smaller = []  # store value that is smaller than 1
    larger = []  # store value that is larger than 1
    for kk, prob in enumerate(probs):
        q[kk] = K * prob  # probability value
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    # Loop though and create little binary mixtures that
    # appropriately allocate the larger outcomes over the
    # overall uniform mixture.

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large  # filling the Alias list
        q[large] = q[large] - (1.0 - q[small])

        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    input: Prob list and Alias list provided by alias_setup function
    output: one time sampling results
    '''
    K = len(J)
    # Draw from the overall uniform mixture.
    kk = int(np.floor(npr.rand() * K))  # randomly select one column

    # Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
    if npr.rand() < q[kk]:  # comparison for decision
        return kk
    else:
        return J[kk]


def onetime_alias_draw(J, q, num_of_draw):
    # for sampling num_of_draw non-repeated element
    sub_result = []
    j = 0
    while j < num_of_draw:
        draft = alias_draw(J, q)
        if draft in sub_result:
            continue
        else:
            sub_result.append(draft)
            j += 1
    return sub_result


def batch_alias_draw(J, q, num_of_draw, num_of_iter):
    # for sampling num_of_draw non-repeated element for each batch
    # no guarantee that samples will not repeat among different batches
    result = []
    for i in range(num_of_iter):
        result.append(onetime_alias_draw(J, q, num_of_draw))
    return result



