import networkx as nx
import numpy as np
import utils
import math

'''
community:utils.Community data type
implemented with networkx library
'''


class LinkPrediction():
    def __init__(self, community):
        # require a utils.Community datatype as input
        # LinkPrediction class store a graph and its ground-truth community labels
        self.graph = community.graph
        self.gt_community = community
        self.algorithm_community = None
        self.motif_index = nx.triangles(self.graph)

    def neighbors(self, x):
        return set(self.graph.neighbors(x))

    def joint_neighbors(self, x, y):
        return self.neighbors(x) & self.neighbors(y)

    # following are all kinds of similarity-based link prediction method
    def AA(self, x, y):
        jn = self.joint_neighbors(x, y)
        similarity_value = 0
        for i in jn:
            similarity_value += 1 / np.log(len(self.neighbors(i)))
        return similarity_value

    def CN(self, x, y):
        return len(self.joint_neighbors(x, y))

    def JC(self, x, y):
        try:
            return self.CN(x, y) / len(self.neighbors(x) | self.neighbors(y))  # union for two sets
        except ZeroDivisionError:  # no common neighbours
            return 0

    def PA(self, x, y):
        return len(self.neighbors(x)) * len(self.neighbors(y))

    def SA(self, x, y):
        try:
            return self.CN(x, y) / np.sqrt(self.PA(x, y))
        except ZeroDivisionError:
            return 0

    def HDI(self, x, y):
        try:
            if len(self.neighbors(x)) < len(self.neighbors(y)):
                return self.CN(x, y) / len(self.neighbors(x))
            else:
                return self.CN(x, y) / len(self.neighbors(y))
        except ZeroDivisionError:
            return 0

    def HPI(self, x, y):
        try:
            if len(self.neighbors(x)) > len(self.neighbors(y)):
                return self.CN(x, y) / len(self.neighbors(x))
            else:
                return self.CN(x, y) / len(self.neighbors(y))
        except ZeroDivisionError:
            return 0

    def LLHN(self, x, y):
        try:
            return self.CN(x, y) / self.PA(x, y)
        except ZeroDivisionError:
            return 0

    def RA(self, x, y):
        jn = self.joint_neighbors(x, y)
        total = 0
        for i in jn:
            total += 1 / len(self.neighbors(i))
        return total

    def CI_zero(self, x, y):
        return self.graph.degree(x) + self.graph.degree(y) - 1

    def CI_one(self, x, y):
        temp_1 = self.CI_zero(x, y)
        iter_x = [i for i in self.neighbors(x)]
        iter_y = [i for i in self.neighbors(y)]
        temp_2 = 0
        for i in iter_x:
            temp_2 += self.CI_zero(x, i)
            temp_2 -= 1
        temp_3 = 0
        for i in iter_y:
            temp_3 += self.CI_zero(y, i)
            temp_3 -= 1
        return temp_1 * (temp_2 + temp_3 - 2 * temp_1 + 2)

    def IM1(self, x, y):
        temp_1 = len(self.neighbors(x) & self.neighbors(y))
        if temp_1 + self.motif_index[x] + self.motif_index[y] == 0:
            return 0
        elif self.motif_index[x] + self.motif_index[y] == 0:
            return 1e-5
        else:
            return temp_1 / (temp_1 + self.motif_index[x] + self.motif_index[y])

    def IM2(self, x, y):
        temp = self.neighbors(x) & self.neighbors(y)
        temp_1 = 0
        for i in temp:
            temp_1 += self.motif_index[i]
        return len(temp) + self.motif_index[x] + self.motif_index[y] + temp_1

    def CC(self, x):
        temp = self.graph.degree(x)
        if (temp * (temp - 1)) == 0:
            return 0
        else:
            return 2 * self.motif_index[x] / (temp * (temp - 1))

    def CCLP(self, x, y):
        temp = self.neighbors(x) & self.neighbors(y)
        result = 0
        for i in temp:
            result += self.CC(i)
        return result

    def LCL(self, x, y):
        temp = self.neighbors(x) & self.neighbors(y)
        total = 0
        for i in temp:
            for j in temp:
                total += self.graph.number_of_edges(i, j)
        return total

    def CAR(self, x, y):
        temp_1 = self.neighbors(x)
        temp_2 = self.neighbors(y)
        return len(temp_1 & temp_2) * self.LCL(x, y)

    def community_RA(self, x, y):
        if self.algorithm_community is None:
            print('Community detection has not been done! Require algorithm results!')
            raise ValueError
        common_neighbor = self.joint_neighbors(x, y)
        class_x = self.algorithm_community.community_list[x]
        class_y = self.algorithm_community.community_list[y]
        if len(common_neighbor) == 0:
            return 0
        total = 0
        for i in common_neighbor:
            if self.algorithm_community.community_list[i] == class_x and\
                    self.algorithm_community.community_list[i] == class_y:
                total += (2 / nx.degree(self.graph, i))
            else:
                total += (1 / nx.degree(self.graph, i))
        return total

    def neighbor_class(self, x):
        neighbor_nodes = self.neighbors(x)
        node_class = []
        for i in neighbor_nodes:
            node_class.append(self.algorithm_community.community_list[i])
        return node_class

    def Harmony_value(self, x, y):
        common_neighbor = self.joint_neighbors(x, y)
        if len(common_neighbor) == 0:
            return 0
        summary = 0
        for i in common_neighbor:
            summary += centrality(self.neighbor_class(i))
        return summary / len(common_neighbor)

    def CSA_value(self, x, y):
        class_x = self.algorithm_community.community_list[x]
        class_y = self.algorithm_community.community_list[y]
        number_class_x = len(self.algorithm_community.community[class_x])
        number_class_y = len(self.algorithm_community.community[class_y])
        max_class = 0
        for i in range(self.algorithm_community.number_of_community):
            if len(self.algorithm_community.community[i]) > max_class:
                max_class = len(self.algorithm_community.community[i])
        temp = min(
            [self.algorithm_community.community_link_matrix[class_x, class_x],
             self.algorithm_community.community_link_matrix[class_y, class_y]])
        # temp = min([number_class_x, number_class_y])
        if class_x == class_y:
            return math.sqrt(number_class_x) / max_class
        if self.algorithm_community.community_link_matrix[class_x, class_y] == 0:
            return 0
        else:
            return self.algorithm_community.community_link_matrix[class_x, class_y] / temp

    def HAP(self, x, y):
        return self.Harmony_value(x,y) * self.CSA_value(x, y)

    def link_prediction_result(self, edge_list, function, algorithm_community):
        self.algorithm_community = algorithm_community
        result = []
        for i in edge_list:
            result.append([i, function(self, i[0], i[1])])
        return result


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  # 样本数
    labelCounts = {}  # 该数据集每个类别的频数
    for featVec in dataSet:  # 对每一行样本
        currentLabel = featVec  # 该样本的标签
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # 计算p(xi)
        shannonEnt -= prob * math.log(prob, 2)  # log base 2
    return shannonEnt


def centrality(dataset):
    if len(dataset) == 1:
        return 1
    shannonEnt = calcShannonEnt(dataset)
    return 1 - shannonEnt / math.log(len(dataset), 2)


def normalized_link_prediction_result(edge_list, function):
    # prepared for future reference if we want to involve alias sampling methods
    normalized_result = []
    values = []
    total = 0
    for i in edge_list:
        temp = function(i[0], i[1])
        values.append(temp)
        total += temp
    for i in range(len(edge_list)):
        if values[i] != 0:
            normalized_result.append([edge_list[i], values[i] / total])
        else:
            continue
    return normalized_result
