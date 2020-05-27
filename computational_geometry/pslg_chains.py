class Vertex():
    def __init__(self, x, y, incoming_edges=None, outgoing_edges=None):
        self.x = x
        self.y = y
        self.incoming_edges = incoming_edges
        self.outgoing_edges = outgoing_edges

class Edge():
    def __init__(self, v1, v2, weight=1):
        self.v1 = v1
        self.v2 = v2
        self.weight = weight
        
    def __repr__(self):
        return f"({self.v2.y}, {self.v1.y})"
        
vertices = [Vertex(1,0),
            Vertex(2,1),
            Vertex(0,2),
            Vertex(1,3),
            Vertex(2,4),
            Vertex(1,5),
            Vertex(2,6),
            Vertex(0,7),
            Vertex(1,8)]

edges = [Edge(vertices[0], vertices[1], 1),
         Edge(vertices[0], vertices[2], 3),
         Edge(vertices[0], vertices[3], 1),
         Edge(vertices[0], vertices[4], 1),
         Edge(vertices[1], vertices[4], 1),
         Edge(vertices[2], vertices[3], 1),
         Edge(vertices[2], vertices[5], 1),
         Edge(vertices[2], vertices[7], 1),
         Edge(vertices[3], vertices[4], 1),
         Edge(vertices[3], vertices[5], 1),
         Edge(vertices[4], vertices[5], 2),
         Edge(vertices[4], vertices[6], 1),
         Edge(vertices[5], vertices[6], 1),
         Edge(vertices[5], vertices[8], 3),
         Edge(vertices[6], vertices[8], 2),
         Edge(vertices[7], vertices[8], 1)]

vertices[8].incoming_edges = [edges[15], edges[13], edges[14]]
vertices[7].incoming_edges = [edges[7]]
vertices[6].incoming_edges = [edges[12], edges[11]]
vertices[5].incoming_edges = [edges[6], edges[9], edges[10]]
vertices[4].incoming_edges = [edges[8], edges[3], edges[4]]
vertices[3].incoming_edges = [edges[5], edges[2]]
vertices[2].incoming_edges = [edges[1]]
vertices[1].incoming_edges = [edges[0]]

vertices[7].outgoing_edges = [edges[15]]
vertices[6].outgoing_edges = [edges[14]]
vertices[5].outgoing_edges = [edges[12], edges[13]]
vertices[4].outgoing_edges = [edges[11], edges[10]]
vertices[3].outgoing_edges = [edges[8], edges[9]]
vertices[2].outgoing_edges = [edges[5], edges[6], edges[7]]
vertices[1].outgoing_edges = [edges[4]]
vertices[0].outgoing_edges = [edges[0], edges[3], edges[2], edges[1]]

chains = []
for i in reversed(range(len(vertices))):
    cur_vertex = vertices[i]
    if i == len(vertices)-1:
        for _ in range(sum(edge.weight for edge in cur_vertex.incoming_edges)):
            chains.append([])
        count = 0
        for j in range(len(cur_vertex.incoming_edges)):
            cur_edge = cur_vertex.incoming_edges[j]
            for _ in range(cur_edge.weight):
                chains[count].append(cur_edge)
                count += 1
                
    elif i != 0:
        for j in range(len(cur_vertex.incoming_edges)):
            count = 0
            for k in range(len(chains)):
                if chains[k][-1] in cur_vertex.outgoing_edges:
                    chains[k].append(cur_vertex.incoming_edges[j])
                    count += 1
                    if count == cur_vertex.incoming_edges[j].weight:
                        break

for chain in chains:
    print(chain)
