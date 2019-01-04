import networkx as nx
import json
from pprint import pprint
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import collections
import numpy as np
import heapq
from networkx.algorithms import approximation


allMoneyStaked = 0
allWeights = []

def main():
    G=defineGraph(readFile())
    basicStatistics(G)
    #degreeDistribution(G)
    #weightsDistribution(G)
    #shortestPaths(G)
    #drawGraph(G)
    #robustness(G)

#degreeCentrality

def basicStatistics(G):
    print("Number of LN nodes : ", G.order())
    print("Number of LN payment channels: ", G.size())
    print("Amount of BTC, denominated in Satoshis, in all payment channels:", allMoneyStaked)
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
    #print("LN's S-metric: ", smetric(G)) #0.6879664061934981
    print("LN average clustering coefficient", approximation.clustering_coefficient.average_clustering(G))
    #print("LN's largest clique size: ", nx.algorithms.approximation.clique.max_clique(G))

def robustness(G):
    G.remove_nodes_from(list(nx.connected_components(G))[1])  # there is a small second component
    sortedNodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    print("what?",len(list(nx.connected_components(G))))
    for x in range(30):
        G.remove_node(sortedNodes[x][0])
        print(x+1,len(list(nx.connected_components(G))))

def weightsDistribution(G):
    global allWeights
    allWeights.sort(reverse=True)
    edgeCount = collections.Counter(allWeights)
    weight, cnt = zip(*edgeCount.items())

    fig, ax = plt.subplots()
    #plt.bar(weight, cnt, width=0.80, color='b')

    plt.hist(allWeights,100)
    plt.title("Weight Histogram")
    plt.ylabel("Count")
    plt.xlabel("Weights")
    ax.set_xticks([d  for d in weight])
    ax.set_xticklabels(weight)

    for index, label in enumerate(ax.xaxis.get_ticklabels()):
        if weight[index] not in [16777216,9000000,5000000,2000000,70000]:
            label.set_visible(False)

    plt.show()

def degreeDistribution(G):

    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    print(degree_sequence)
    # print "Degree sequence", degree_sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')
    plt.bar(deg, powerList(cnt,-1), color='r', width=0.80)


    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)



    plt.yscale("log")
    plt.xscale("log")
    n = 20
    invisible = [202,213,226,291,323,407,415,267,269] ##disturbing degrees
    for index, label in enumerate(ax.xaxis.get_ticklabels()):
        if deg[index] % n != 0 and deg[index]<200 or (deg[index] in invisible):
            label.set_visible(False)
    plt.show()

def shortestPaths(G):
    shortPaths=nx.algorithms.shortest_paths.generic.shortest_path(G)
    print(shortPaths)

#draw graph in inset
def drawGraph(G):
     #plt.axes([0.4, 0.4, 0.5, 0.5])
     Gcc = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0]
     pos = nx.spring_layout(G)
     #pos = nx.drawing.nx_pydot.graphviz_layout(G)
     fig, ax = plt.subplots()
     plt.axis('off')

     degrees = G.degree()
     nodes = G.nodes()
     n_color = np.asarray([degrees[n] for n in nodes])

     #nx.draw(G, nodelist=d.keys(), node_size=[v * 100 for v in d.values()])


     sc=nx.draw_networkx_nodes(G, pos, nodelist=nodes, ax=ax, node_size=n_color, node_color=n_color, cmap='viridis')
     nx.draw_networkx_edges(G, pos, alpha=0.2)

     sc.set_norm(mcolors.LogNorm())
     fig.colorbar(sc)

if __name__ == "__main__":
    main()

def defineGraph(data) -> object:
    global allMoneyStaked
    G = nx.Graph()
    for x in range(len(data)):
        G.add_edge(data[x]['node2_pub'], data[x]['node1_pub'], weight=data[x]['capacity'])
        allWeights.append(int(data[x]['capacity']))
        allMoneyStaked+=int(data[x]['capacity'])
    return G


##https://graph.lndexplorer.com/api/graph
def readFile() -> object:
    with open('graph.json') as f:
        data = json.load(f)
    return data['edges']

def smetric(G) -> object:
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    Gmax = li_smax_graph(list(degree_sequence))
    return smetricNonNormalized(G)/smetricNonNormalized(Gmax)

def smetricNonNormalized(G) -> object:
    return float(sum([G.degree(u) * G.degree(v) for (u, v) in G.edges()]))

def powerList(my_list, exponent):
    return [ x**exponent for x in my_list ]


def li_smax_graph(degree_seq):
    """Generates a graph based with a given degree sequence and maximizing
    the s-metric.  Experimental implementation.
    Maximum s-metrix  means that high degree nodes are connected to high
    degree nodes.

    - `degree_seq`: degree sequence, a list of integers with each entry
       corresponding to the degree of a node.
       A non-graphical degree sequence raises an Exception.
    Reference::

      @unpublished{li-2005,
       author = {Lun Li and David Alderson and Reiko Tanaka
                and John C. Doyle and Walter Willinger},
       title = {Towards a Theory of Scale-Free Graphs:
               Definition, Properties, and  Implications (Extended Version)},
       url = {http://arxiv.org/abs/cond-mat/0501169},
       year = {2005}
      }
    The algorithm::
     STEP 0 - Initialization
     A = {0}
     B = {1, 2, 3, ..., n}
     O = {(i; j), ..., (k, l),...} where i < j, i <= k < l and
             d_i * d_j >= d_k *d_l
     wA = d_1
     dB = sum(degrees)
     STEP 1 - Link selection
     (a) If |O| = 0 TERMINATE. Return graph A.
     (b) Select element(s) (i, j) in O having the largest d_i * d_j , if for
             any i or j either w_i = 0 or w_j = 0 delete (i, j) from O
     (c) If there are no elements selected go to (a).
     (d) Select the link (i, j) having the largest value w_i (where for each
             (i, j) w_i is the smaller of w_i and w_j ), and proceed to STEP 2.
     STEP 2 - Link addition
     Type 1: i in A and j in B.
             Add j to the graph A and remove it from the set B add a link
             (i, j) to the graph A. Update variables:
             wA = wA + d_j -2 and dB = dB - d_j
             Decrement w_i and w_j with one. Delete (i, j) from O
     Type 2: i and j in A.
         Check Tree Condition: If dB = 2 * |B| - wA.
             Delete (i, j) from O, continue to STEP 3
         Check Disconnected Cluster Condition: If wA = 2.
             Delete (i, j) from O, continue to STEP 3
         Add the link (i, j) to the graph A
         Decrement w_i and w_j with one, and wA = wA -2
     STEP 3
         Go to STEP 1
    The article states that the algorithm will result in a maximal s-metric.
    This implementation can not guarantee such maximality. I may have
    misunderstood the algorithm, but I can not see how it can be anything
    but a heuristic. Please contact me at sundsdal@gmail.com if you can
    provide python code that can guarantee maximality.
    Several optimizations are included in this code and it may be hard to read.
    Commented code to come.
    """

    if not is_valid_degree_sequence(degree_seq):
        raise nx.NetworkXError('Invalid degree sequence')
    degree_seq.sort()  # make sure it's sorted
    degree_seq.reverse()
    degrees_left = degree_seq[:]
    A_graph = nx.Graph()
    A_graph.add_node(0)
    a_list = [False] * len(degree_seq)
    b_set = set(range(1, len(degree_seq)))
    a_open = set([0])
    O = []
    for j in b_set:
        heapq.heappush(O, (-degree_seq[0] * degree_seq[j], (0, j)))
    wa = degrees_left[0]  # stubs in a_graph
    db = sum(degree_seq) - degree_seq[0]  # stubs in b-graph
    a_list[0] = True  # node 0 is now in a_Graph
    bsize = len(degree_seq) - 1  # size of b_graph
    selected = []
    weight = 0
    while O or selected:
        if len(selected) < 1:
            firstrun = True
            while O:
                (newweight, (i, j)) = heapq.heappop(O)
                if degrees_left[i] < 1 or degrees_left[j] < 1:
                    continue
                if firstrun:
                    firstrun = False
                    weight = newweight
                if not newweight == weight:
                    break
                heapq.heappush(selected, [-degrees_left[i], \
                                          -degrees_left[j], (i, j)])
            if not weight == newweight:
                heapq.heappush(O, (newweight, (i, j)))
            weight *= -1
        if len(selected) < 1:
            break

        [w1, w2, (i, j)] = heapq.heappop(selected)
        if degrees_left[i] < 1 or degrees_left[j] < 1:
            continue
        if a_list[i] and j in b_set:
            # TYPE1
            a_list[j] = True
            b_set.remove(j)
            A_graph.add_node(j)
            A_graph.add_edge(i, j)
            degrees_left[i] -= 1
            degrees_left[j] -= 1
            wa += degree_seq[j] - 2
            db -= degree_seq[j]
            bsize -= 1
            newweight = weight
            if not degrees_left[j] == 0:
                a_open.add(j)
                for k in b_set:
                    if A_graph.has_edge(j, k): continue
                    w = degree_seq[j] * degree_seq[k]
                    if w > newweight:
                        newweight = w
                    if weight == w and not newweight > weight:
                        heapq.heappush(selected, [-degrees_left[j], \
                                                  -degrees_left[k], (j, k)])
                    else:
                        heapq.heappush(O, (-w, (j, k)))
                if not weight == newweight:
                    while selected:
                        [w1, w2, (i, j)] = heapq.heappop(selected)
                        if degrees_left[i] * degrees_left[j] > 0:
                            heapq.heappush(O, [-degree_seq[i] * degree_seq[j], (i, j)])
            if degrees_left[i] == 0:
                a_open.discard(i)

        else:
            # TYPE2
            if db == (2 * bsize - wa):
                # tree condition
                # print "removing because tree condition    "
                continue
            elif db < 2 * bsize - wa:
                raise networkx.NetworkXError("THIS SHOULD NOT HAPPEN! - not graphable")
                continue
            elif wa == 2 and bsize > 0:
                # print "removing because disconnected  cluster"
                # disconnected cluster condition
                continue
            elif wa == db - (bsize) * (bsize - 1):
                # print "MYOWN removing because disconnected  cluster"
                continue
            A_graph.add_edge(i, j)
            degrees_left[i] -= 1
            degrees_left[j] -= 1
            if degrees_left[i] < 1:
                a_open.discard(i)
            if degrees_left[j] < 1:
                a_open.discard(j)
            wa -= 2
            if not degrees_left[i] < 0 and not degrees_left[j] < 0:
                selected2 = (selected)
                selected = []
                while selected2:
                    [w1, w1, (i, j)] = heapq.heappop(selected2)
                    if degrees_left[i] * degrees_left[j] > 0:
                        heapq.heappush(selected, [-degrees_left[i], \
                                                  -degrees_left[j], (i, j)])
    return A_graph


def is_valid_degree_sequence(deg_sequence):
    """Return True if deg_sequence is a valid sequence of integer degrees
    equal to the degree sequence of some simple graph.

      - `deg_sequence`: degree sequence, a list of integers with each entry
         corresponding to the degree of a node (need not be sorted).
         A non-graphical degree sequence (i.e. one not realizable by some
         simple graph) will raise an exception.

    See Theorem 1.4 in [chartrand-graphs-1996]. This algorithm is also used
    in havel_hakimi_graph()
    References:
    [chartrand-graphs-1996] G. Chartrand and L. Lesniak, "Graphs and Digraphs",
                            Chapman and Hall/CRC, 1996.
    """
    # some simple tests
    if deg_sequence == []:
        return True  # empty sequence = empty graph
    if not nx.utils.is_list_of_ints(deg_sequence):
        return False  # list of ints
    if min(deg_sequence) < 0:
        return False  # each int not negative
    if sum(deg_sequence) % 2:
        return False  # must be even

    # successively reduce degree sequence by removing node of maximum degree
    # as in Havel-Hakimi algorithm

    s = deg_sequence[:]  # copy to s
    while s:
        s.sort()  # sort in non-increasing order
        if s[0] < 0:
            return False  # check if removed too many from some node

        d = s.pop()  # pop largest degree
        if d == 0: return True  # done! rest must be zero due to ordering

        # degree must be <= number of available nodes
        if d > len(s):   return False

        # remove edges to nodes of next higher degrees
        s.reverse()  # to make it easy to get at higher degree nodes.
        for i in range(d):
            s[i] -= 1

    # should never get here b/c either d==0, d>len(s) or d<0 before s=[]
    return False