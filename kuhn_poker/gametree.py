import numpy as np

class GameTree:
    def __init__(self, root_node: GTNode):
        self.root = root_node
        # 3 is the number of top level information sets
        roots = [RootGTNode(), RootGTNode(), RootGTNode()]
        self.proliferate(roots)

    def proliferate(self, roots):
        cards_hash = {"K": 3, "Q": 2, "J": 1}
        cards = ["K", "Q", "J"]
        for i in range(len(cards)):
            curr_info_sets = []
            for j in range(len(cards)):
                if cards[i] != cards[j]:
                    info_set = cards[i] + cards[j]
                    curr_info_sets.append(info_set)

            if cards_hash[curr_info_sets[0][-1]] > cards_hash[curr_info_sets[1][-1]]:
                roots[i].card_H = GTNode(curr_info_sets[0], 0, 0, None, None




class RootGTNode:
    def __init__(self, num_children):
        self.card_H = None
        self.card_L = None

class GTNode:
    def __init__(self, info_set, cb_strat, pf_strat, cb_next, pf_next, prev):
        self.info_set = info_set
        self.cb_strat = None
        self.pf_strat = None
        self.cb_next = None
        self.pf_next = None
        self.prev = None

class TerminalGTNode:
    def __init__(self, p1_util, p2_util):
        self.p1_util = p1_util
        self.p2_util = p2_util
        self.prev = None

if __name__ == '__main__':
