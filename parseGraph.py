import networkx as nx
import json
from pprint import pprint
import math
import matplotlib.pyplot as plt
import collections

allMoneyStaked = 0
allWeights = []

def main():
    G=defineGraph(readFile())
    basicStatistics(G)
    #degreeDistribution(G)
    #drawGraph(G)

#degreeCentrality

def basicStatistics(G):
    print("Number of LN nodes : ", G.order())
    print("Number of LN payment channels: ", G.size())
    print("Amount of BTC in all payment channels:", allMoneyStaked)
    print("Number of connected components: ", len(list(nx.connected_components(G))))
    print("Maximal independent set: ", len(nx.algorithms.maximal_independent_set(G)))
    print("Number of bridges: ", len(list(nx.algorithms.bridges(G))))
    print("Size of the dominating set: ", len(nx.algorithms.dominating_set(G)))
    print("LN is Chordal graph: ", nx.algorithms.chordal.is_chordal(G))
    print("LN degree assortativity",nx.algorithms.assortativity.degree_assortativity_coefficient(G))
    #print("LN rich-club coefficient: ", nx.algorithms.richclub.rich_club_coefficient(G))
    #print("LN rich-club normalized coefficient: ", nx.algorithms.richclub.rich_club_coefficient(G, normalized=True))
    #print(list(nx.connected_components(G))[1])
    G.remove_nodes_from(list(nx.connected_components(G))[1]) #there is a small second component
    #print("LN diameter: ", nx.algorithms.distance_measures.diameter(G)) #6
    #print("LN radius", nx.algorithms.distance_measures.radius(G)) #3
    #print("LN Wiener index", nx.algorithms.wiener_index(G)) #7686159.0
    print("LN is Eulerian: ",nx.algorithms.is_eulerian(G))
    print("LN is planar: ", nx.algorithms.planarity.check_planarity(G))
    print("Number of isolates in LN: ", list(nx.isolates(G)))

def degreeDistribution(G):

    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    # print "Degree sequence", degree_sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)

    plt.yscale("log")
    n = 20
    invisible = [202,213,226,291,323,407,415,267,269] ##disturbing degrees
    for index, label in enumerate(ax.xaxis.get_ticklabels()):
        if deg[index] % n != 0 and deg[index]<200 or (deg[index] in invisible):
            label.set_visible(False)
    plt.show()

#draw graph in inset
def drawGraph(G):
     plt.axes([0.4, 0.4, 0.5, 0.5])
     Gcc = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0]
     pos = nx.spring_layout(G)
     plt.axis('off')
     nx.draw_networkx_nodes(G, pos, node_size=20)
     nx.draw_networkx_edges(G, pos, alpha=0.4)

if __name__ == "__main__":
    main()

def defineGraph(data) -> object:
    global allMoneyStaked
    G = nx.Graph()
    for x in range(len(data)):
        G.add_edge(data[x]['node2_pub'], data[x]['node1_pub'], weight=data[x]['capacity'])
        allMoneyStaked+=int(data[x]['capacity'])
    return G


##https://graph.lndexplorer.com/api/graph
def readFile() -> object:
    with open('graph.json') as f:
        data = json.load(f)
    return data['edges']
