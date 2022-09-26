import load_data
import utils
import Link_Pred
import community_enhance
import community_detec
from cdlib import algorithms
import networkx as nx


# Load dataset, toy examples
names = ['dolphins', 'karate']
dolphins_dataset = load_data.load_data('dolphins')
karate_dataset = load_data.load_data('karate')

dataset = [dolphins_dataset, karate_dataset]
# dataset = [load_data.load_data(i) for i in names]

# element in dataset: Link_Pred.LinkPrediction
detection_method = [algorithms.louvain, community_detec.infomap_implementation, algorithms.label_propagation]
algorithm_name = ['Louvain', 'Infomap', 'Label Propagation']
lp_method = [Link_Pred.LinkPrediction.AA, Link_Pred.LinkPrediction.CN, Link_Pred.LinkPrediction.RA,
             Link_Pred.LinkPrediction.HAP]
lp_names = ['AA', 'CN', 'RA', 'HAP']
edge_number = [10, 10]

for i in range(len(dataset)):
    for j in range(len(detection_method)):
        for k in range(len(lp_method)):
            print('Result for graph ' + names[i] + ' under ' + algorithm_name[j] + ' community detection method and '
                  + lp_names[k] + ' Link Prediction method is following:')
            community_enhance.single_graph_updates(dataset[i], detection_method[j], lp_method[k], edge_number[i])
        print('********************************************************')
    print('--------------------------------------------------------')
