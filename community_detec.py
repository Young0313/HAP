from cdlib import algorithms, NodeClustering
import infomap


# self-defined community detection method should yield cdlib.NodeClustering datatype for future estimation
# demo
def infomap_implementation(G):
    """
    Partition network with the Infomap algorithm.
    Annotates nodes with 'community' id and return number of communities found.
    """
    infomapWrapper = infomap.Infomap("--two-level --silent")

    #print("Building Infomap network from a NetworkX graph...")
    for e in G.edges():
        infomapWrapper.addLink(*e)

    #print("Find communities with Infomap...")
    infomapWrapper.run()

    tree = infomapWrapper.tree

    #print("Found %d modules with codelength: %f" % (infomapWrapper.num_top_modules, infomapWrapper.codelength))

    communities = {}
    for node in tree:
        if node.is_leaf:
            communities[node.node_id] = node.module_id

    infomap_communities = [[]]
    for i in communities.keys():
        temp = communities[i]
        while communities[i] > len(infomap_communities):
            infomap_communities.append([])
        infomap_communities[temp - 1].append(i)
    return NodeClustering(infomap_communities, graph=G, method_name="infomap")
