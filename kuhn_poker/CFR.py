from treelib import Node, Tree
import numpy as np
import igraph as ig
from igraph import plot
import plotly.graph_objects as go
import cairo
import matplotlib.pyplot as plt

class GameTree:
    def __init__(self):
        self.gtree = Tree()
        self.gtree.create_node("ROOT", "root")

        ### LEVEL 1
        self.gtree.create_node("KQ", "KQ", parent="root", data=GTNode(2/3, 1/3, False, 1))
        self.gtree.create_node("KJ", "KJ", parent="root", data=GTNode(2/3, 1/3, False, 1))
        self.gtree.create_node("QK", "QK", parent="root", data=GTNode(1/2, 1/2, False, 1))
        self.gtree.create_node("QJ", "QJ", parent="root", data=GTNode(1/2, 1/2, False, 1))
        self.gtree.create_node("JK", "JK", parent="root", data=GTNode(1/3, 2/3, False, 1))
        self.gtree.create_node("JQ", "JQ", parent="root", data=GTNode(1/3, 2/3, False, 1))

        ### LEVEL 2
        ## KQ-KJ cb/pf Information Set
        self.gtree.create_node("Qb", "KQb", parent="KQ", data=GTNode(1/2, 1/2, True, 2))
        self.gtree.create_node("Qp", "KQp", parent="KQ", data=GTNode(2/3, 1/3, False, 2))
        
        self.gtree.create_node("Jb", "KJb", parent="KJ", data=GTNode(0, 1, True, 2))
        self.gtree.create_node("Jp", "KJp", parent="KJ", data=GTNode(1/3, 2/3, False, 2))

        ## QK-QJ cb/pf Information Set
        self.gtree.create_node("Qb", "QKb", parent="QK", data=GTNode(1, 0, True, 2))
        self.gtree.create_node("Qp", "QKp", parent="QK", data=GTNode(1, 0, False, 2))
        
        self.gtree.create_node("Qb", "QJb", parent="QJ", data=GTNode(0, 1, True, 2))
        self.gtree.create_node("Qp", "QJp", parent="QJ", data=GTNode(1/3, 2/3, False, 2))

        ## JK-JQ cb/pf Information Set
        self.gtree.create_node("Kb", "JKb", parent="JK", data=GTNode(1, 0, True, 2))
        self.gtree.create_node("Kp", "JKp", parent="JK", data=GTNode(1, 0, False, 2))
        
        self.gtree.create_node("Qb", "JQb", parent="JQ", data=GTNode(1/2, 1/2, True, 2))
        self.gtree.create_node("Qp", "JQp", parent="JQ", data=GTNode(2/3, 1/3, False, 2))

        ### LEVEL 2 bc/bf Utility Nodes
        self.gtree.create_node("util", "KQb_uc", parent="KQb", data=TNode(2, -2))
        self.gtree.create_node("util", "KQb_uf", parent="KQb", data=TNode(1, -1))
        self.gtree.create_node("util", "KJb_uc", parent="KJb", data=TNode(2, -2))
        self.gtree.create_node("util", "KJb_uf", parent="KJb", data=TNode(1, -1))

        self.gtree.create_node("util", "QKb_uc", parent="QKb", data=TNode(-2, 2))
        self.gtree.create_node("util", "QKb_uf", parent="QKb", data=TNode(1, -1))
        self.gtree.create_node("util", "QJb_uc", parent="QJb", data=TNode(2, -2))
        self.gtree.create_node("util", "QJb_uf", parent="QJb", data=TNode(1, -1))

        self.gtree.create_node("util", "JKb_uc", parent="JKb", data=TNode(-2, 2))
        self.gtree.create_node("util", "JKb_uf", parent="JKb", data=TNode(1, -1))
        self.gtree.create_node("util", "JQb_uc", parent="JQb", data=TNode(-2, 2))
        self.gtree.create_node("util", "JQb_uf", parent="JQb", data=TNode(1, -1))

        ### LEVEL 2 pp Utility Nodes
        self.gtree.create_node("util", "KQp_up", parent="KQp", data=TNode(1, -1))
        self.gtree.create_node("util", "KJp_up", parent="KJp", data=TNode(1, -1))
        self.gtree.create_node("util", "QKp_up", parent="QKp", data=TNode(-1, 1))
        self.gtree.create_node("util", "QJp_up", parent="QJp", data=TNode(1, -1))
        self.gtree.create_node("util", "JKp_up", parent="JKp", data=TNode(-1, 1))
        self.gtree.create_node("util", "JQp_up", parent="JQp", data=TNode(-1, 1))

        ### LEVEL 3
        ## KQpb KJpb Information Set
        self.gtree.create_node("Kpb", "KQpb", parent="KQp", data=GTNode(1, 0, True, 1))
        self.gtree.create_node("Kpb", "KJpb", parent="KJp", data=GTNode(1, 0, True, 1))

        ## QKpb QJpb Information Set
        self.gtree.create_node("Qpb", "QKpb", parent="QKp", data=GTNode(1/2, 1/2, True, 1))
        self.gtree.create_node("Qpb", "QJpb", parent="QJp", data=GTNode(1/2, 1/2, True, 1))

        ## JKpb JQpb Information Set
        self.gtree.create_node("Jpb", "JKpb", parent="JKp", data=GTNode(0, 1, True, 1))
        self.gtree.create_node("Jpb", "JQpb", parent="JQp", data=GTNode(0, 1, True, 1))

        ### LEVEL 3 pb Utility Nodes
        self.gtree.create_node("util", "KQpb_uc", parent="KQpb", data=TNode(2, -2))
        self.gtree.create_node("util", "KQpb_uf", parent="KQpb", data=TNode(-1, 1))
        self.gtree.create_node("util", "KJpb_uc", parent="KJpb", data=TNode(2, -2))
        self.gtree.create_node("util", "KJpb_uf", parent="KJpb", data=TNode(-1, 1))

        self.gtree.create_node("util", "QKpb_uc", parent="QKpb", data=TNode(-2, 2))
        self.gtree.create_node("util", "QKpb_uf", parent="QKpb", data=TNode(-1, 1))
        self.gtree.create_node("util", "QJpb_uc", parent="QJpb", data=TNode(2, -2))
        self.gtree.create_node("util", "QJpb_uf", parent="QJpb", data=TNode(-1, 1))

        self.gtree.create_node("util", "JKpb_uc", parent="JKpb", data=TNode(-2, 2))
        self.gtree.create_node("util", "JKpb_uf", parent="JKpb", data=TNode(-1, 1))
        self.gtree.create_node("util", "JQpb_uc", parent="JQpb", data=TNode(-2, 2))
        self.gtree.create_node("util", "JQpb_uf", parent="JQpb", data=TNode(-1, 1))

        ### BELIEFS
        self.beliefs = {"K": [1/2, 1/2], "Q": [1/2, 1/2], "J": [1/2, 1/2], 
                        "Kpb": [None, None], "Qpb": [None, None], "Jpb": [None, None],
                        "Kp": [None, None], "Kb": [None, None], "Qp": [None, None],
                        "Qb": [None, None], "Jp": [None, None], "Jb": [None, None]}

        self.utils = {"K": [None, None], "Q": [None, None], "J": [None, None], 
                        "Kpb": [2, -1], "Qpb": [None, -1], "Jpb": [-2, -1],
                        "Kp": [None, 1], "Kb": [2, -1], "Qp": [None, None],
                        "Qb": [None, -1], "Jp": [None, -1], "Jb": [-2, -1]}

    def print_tree(self):
        print(self.gtree.show(stdout=False))
        '''
        # DFS Traversal
        print(','.join([self.gtree[node].tag for node in \
            self.gtree.expand_tree(mode=Tree.DEPTH)]))
        '''

    '''
    ### BELIEF CALCULATION
    def DFS(self, start_nid):
        # DFS Traversal Order
        #for _ in self.gtree.expand_tree(mode=Tree.DEPTH):
            #print(_)
        if start_nid == None:
            return

        node = self.gtree.get_node(start_nid)
        if type(node.data) == GTNode:
            left = node.data.cb_strat
            right = node.data.pf_strat
        elif type(node.data) == TNode:
            left = node.data.p1_util
            right = node.data.p2_util

        print(start_nid, left, right)

        children = self.gtree.children(start_nid)

        if len(children) == 0:
            return

        cb = children[0]
        pf = children[1]

        cbbelief = cb.tag
        pfbelief = pf.tag

        if "util" in cbbelief:
            return 

        self.beliefs[cbbelief][0] = self.cb_strat * self.DFS(cb._identifier)
        self.beliefs[cbbelief][1] = 1 - self.beliefs[cbbelief][0]

        self.beliefs[pf.tag][0] = self.DFS(pf._identifier)
        self.beliefs[pf.tag][1] = 1 - self.beliefs[pfbelief][0]
        '''
    def get_edge(self, action, p1_card, p2_card):
        node = self.gtree.get_node(p1_card + p2_card)
        print(node.data)
        assert type(node.data) == GTNode

        if action == "cb":
            return node.data.cb_strat
        if action == "pf":
            return node.data.pf_strat

    def get_l3_edge(self, action, p1_card, p2_card, prev_action):
        node = self.gtree.get_node(p1_card + p2_card + prev_action)
        print(p1_card+p2_card+prev_action, node)
        assert type(node.data) == GTNode

        if action == "cb":
            return node.data.cb_strat
        if action == "p":
            return node.data.pf_strat

    def calculate_beliefs(self):
        K = "K"
        Q = "Q"
        J = "J"

        ### LEVEL 2 Beliefs
        ## QbH Belief
        num = self.get_edge("cb", K, Q) * 1/2
        den = num + (self.get_edge("cb", J, Q) * 1/2)

        val = num / den

        self.beliefs["Qb"][0] = val 
        self.beliefs["Qb"][1] = 1 -val 

        ## QpH Belief
        num = self.get_edge("pf", K, Q) * 1/2
        den = num + (self.get_edge("pf", J, Q) * 1/2)

        val =  num / den

        self.beliefs["Qp"][0] = val
        self.beliefs["Qp"][1] = 1 - val

        ## JbH Belief
        num = self.get_edge("cb", K, J) * 1/2
        den = num + (self.get_edge("cb", Q, J) * 1/2)

        val =  num / den

        self.beliefs["Jb"][0] = val
        self.beliefs["Jb"][1] = 1 - val

        ## JpH Belief
        num = self.get_edge("pf", K, J) * 1/2
        den = num + (self.get_edge("pf", Q, J) * 1/2)

        val =  num / den

        self.beliefs["Jp"][0] = val
        self.beliefs["Jp"][1] = 1 - val

        ## KbH Belief
        num = self.get_edge("cb", Q, K) * 1/2
        den = num + (self.get_edge("cb", J, K) * 1/2)

        val =  num / den

        self.beliefs["Kb"][0] = val
        self.beliefs["Kb"][1] = 1 - val

        ## KpH Belief
        num = self.get_edge("pf", Q, K) * 1/2
        den = num + (self.get_edge("pf", J, K) * 1/2)

        val =  num / den

        self.beliefs["Kp"][0] = val
        self.beliefs["Kp"][1] = 1 - val

        ### LEVEL 3 Beliefs
        ## KpbH Belief
        num = self.get_edge("pf", K, Q) * self.get_l3_edge("cb", K, Q, "p") * 1/2
        den = num + (self.get_edge("pf", K, J) * self.get_l3_edge("cb", K, J, "p") * 1/2)

        print("NUM", den)

        val = num / den
        self.beliefs["Kpb"][0] = val
        self.beliefs["Kpb"][1] = 1 - val

        ## QpbH Belief
        num = self.get_edge("pf", Q, K) * self.get_l3_edge("cb", Q, K, "p") * 1/2
        den = num + (self.get_edge("pf", Q, J) * self.get_l3_edge("cb", Q, J, "p") * 1/2)

        val = num / den

        self.beliefs["Qpb"][0] = val
        self.beliefs["Qpb"][1] = 1 - val
        
        ## JpbH Belief
        num = self.get_edge("pf", J, K) * self.get_l3_edge("cb", J, K, "p") * 1/2
        den = num + (self.get_edge("pf", J, Q) * self.get_l3_edge("cb", J, Q, "p") * 1/2)

        val = num / den
        self.beliefs["Jpb"][0] = val
        self.beliefs["Jpb"][1] = 1 - val

    def get_utils(self, nid, player, action):
        card_hash = {"K": ["Q", "J"], "Q": ["K", "J"], "J": ["K", "Q"]}

        h, l = card_hash[nid[0]]

        nidH = h + nid
        nidL = l + nid

        if action == "cb":
            adder = "_uc"
            nidH += adder
            nidL += adder
        if action == "pf":
            adder = "_uf"
            nidH += adder
            nidL += adder
        if action == "pp":
            adder = "_up"
            nidH += adder
            nidL += adder

        print("H & L", nidH, nidL)

        nodeH = self.gtree.get_node(nidH)
        nodeL = self.gtree.get_node(nidL)
        assert type(nodeH.data) == TNode

        h_util = [nodeH.data.p1_util, nodeH.data.p2_util][player-1]
        l_util = [nodeL.data.p1_util, nodeL.data.p2_util][player-1]

        return h_util, l_util

    def calculate_deviation_payoffs(self):
        ### LEVEL 3 deviation payoffs
        ## Qpb Utilities
        h = self.beliefs["Qpb"][0]
        l = self.beliefs["Qpb"][1]

        h_util, l_util = self.get_utils("Qpb", 2, "cb")
        print(h_util, l_util)

        self.utils["Qpb"][0] = (h_util * h) + (l_util * l)
        ## Qb Utilities
        h = self.beliefs["Qb"][0]
        l = self.beliefs["Qb"][1]

        h_util, l_util = self.get_utils("Qb", 2, "cb")
        print(h_util, l_util)

        self.utils["Qb"][0] = (h_util * h) + (l_util * l)
        print(self.utils["Qb"][0])

    def visualize_tree(self):
        nodes = list(self.gtree.nodes.keys())
        edges = [(node.identifier, child.identifier) for node in self.gtree.all_nodes() for child in self.gtree.children(node.identifier)]

        # Step 3: Convert to igraph
        graph = ig.Graph()
        graph.add_vertices(nodes)
        graph.add_edges(edges)


        print(nodes)
        print()
        print(edges)
        
        layout = graph.layout("tree")

        '''
        coords = [(x[0], -x[1]) for x in layout]  # Flip y-coordinates for better visualization

        # Plot using matplotlib
        plt.figure(figsize=(8, 6))
        for edge in edges:
            src, tgt = edge
            src_idx = nodes.index(src)
            tgt_idx = nodes.index(tgt)
            x = [coords[src_idx][0], coords[tgt_idx][0]]
            y = [coords[src_idx][1], coords[tgt_idx][1]]
            plt.plot(x, y, color='gray')

        for idx, (x, y) in enumerate(coords):
            plt.scatter(x, y, color='blue', s=100)
            plt.text(x, y + 0.1, nodes[idx], ha='center')

        plt.axis('off')
        plt.show()
        '''

        scaling_factor = 1
        x_coords = [coord[0] * scaling_factor for coord in layout]
        y_coords = [-coord[1] for coord in layout]  # Flip Y for better visualization

        edge_x = []
        edge_y = []
        for edge in edges:
            src, tgt = edge
            src_idx = nodes.index(src)
            tgt_idx = nodes.index(tgt)
            edge_x.extend([x_coords[src_idx], x_coords[tgt_idx], None])
            edge_y.extend([y_coords[src_idx], y_coords[tgt_idx], None])

        node_x = x_coords
        node_y = y_coords

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='gray'),
            hoverinfo='none',
            mode='lines'
        )

        '''
        ## adding beliefs in textbox for each node
        node_text = ["root"]
        for node_name in nodes[1:]:
            try:
                node_text.append(" ".join(self.beliefs[node_name]))
            except KeyError:
                util_node = self.gtree.get_node(node_name)

                print(type(util_node.data))
                assert type(util_node.data) == TNode
                node_text.append(f"{util_node.data.p1_util} {util_node.data.p2_util}")
        '''

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=nodes,  # Display node names as text
            textposition="top center",
            hoverinfo='text',
            marker=dict(size=5, color='red')
        )

        # Step 6: Visualize with Plotly
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title="Counterfactual Regret Minimization Game Tree",
                            showlegend=False,
                            xaxis=dict(showgrid=True, zeroline=False),
                            yaxis=dict(showgrid=True, zeroline=False),
                            hovermode='closest'
                        ))

        fig.show()


class GTNode:
    def __init__(self, cb_strat, pf_strat, from_cb, player_iset):
        self.cb_strat = cb_strat
        self.pf_strat = pf_strat
        self.from_cb = from_cb
        self.player_iset = player_iset 

class TNode:
    def __init__(self, p1_util, p2_util):
        self.p1_util = p1_util
        self.p2_util = p2_util

if __name__ == '__main__':
    gt = GameTree()
    # gt.print_tree()
    gt.calculate_beliefs()
    gt.calculate_deviation_payoffs()
    gt.visualize_tree()

    print(gt.beliefs)
    print()
    print(gt.utils)
