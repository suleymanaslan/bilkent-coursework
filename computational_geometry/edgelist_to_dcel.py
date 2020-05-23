class Node():
    def __init__(self, value, next_node):
        self.value = value
        self.next_node = next_node
        
class Vertex():
    def __init__(self, node):
        self.node = node
        
class EdgeNode():
    def __init__(self, v1 , v2, f1, f2, p1, p2):
        self.v1 = v1
        self.v2 = v2
        self.f1 = f1
        self.f2 = f2
        self.p1 = p1
        self.p2 = p2
        
    def __repr__(self):
        return f"v1:{self.v1}, v2:{self.v2}, f1:{self.f1}, f2:{self.f2}, p1:{self.p1}, p2:{self.p2}"

vertices = [Vertex(Node(9, Node(1, Node(7, None)))),
            Vertex(Node(1, Node(13, Node(2, None)))),
            Vertex(Node(2, Node(4, Node(3, None)))),
            Vertex(Node(3, Node(5, Node(8, None)))),
            Vertex(Node(8, Node(7, None))),
            Vertex(Node(11, Node(6, Node(5, None)))),
            Vertex(Node(6, Node(10, Node(9, None)))),
            Vertex(Node(12, Node(13, Node(10, None)))),
            Vertex(Node(4, Node(12, Node(11, None))))
           ]

number_of_edges = 0
for i in range(len(vertices)):
    vertex = vertices[i]
    cur_edge = vertex.node
    while True:
        if cur_edge.value > number_of_edges:
            number_of_edges = cur_edge.value
        if cur_edge.next_node is None:
            break
        cur_edge = cur_edge.next_node
edge_nodes = []
for _ in range(number_of_edges):
    edge_nodes.append(EdgeNode(None,None,None,None,None,None))

for i in range(len(vertices)):
    vertex = vertices[i]
    cur_edge = vertex.node
    first_edge = cur_edge.value
    while True:
        if edge_nodes[cur_edge.value-1].v1 is None:
            edge_nodes[cur_edge.value-1].v1 = i+1
            if cur_edge.next_node is not None:
                edge_nodes[cur_edge.value-1].p1 = cur_edge.next_node.value
            else:
                edge_nodes[cur_edge.value-1].p1 = first_edge
        else:
            edge_nodes[cur_edge.value-1].v2 = i+1
            if cur_edge.next_node is not None:
                edge_nodes[cur_edge.value-1].p2 = cur_edge.next_node.value
            else:
                edge_nodes[cur_edge.value-1].p2 = first_edge

        if cur_edge.next_node is None:
            break
        cur_edge = cur_edge.next_node
        
for edge_node in edge_nodes:
    print(edge_node)
