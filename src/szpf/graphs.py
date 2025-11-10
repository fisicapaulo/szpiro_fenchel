import networkx as nx

def cycle_graph(n:int):
    return nx.cycle_graph(n)

def star_graph(k:int):
    return nx.star_graph(k)  # 1 centro + k folhas
