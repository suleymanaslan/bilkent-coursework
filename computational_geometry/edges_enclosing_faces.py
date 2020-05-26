class EdgeNode():
    def __init__(self, v1 , v2, f1, f2, p1, p2):
        self.v1 = v1
        self.v2 = v2
        self.f1 = f1
        self.f2 = f2
        self.p1 = p1
        self.p2 = p2

edge_nodes = [EdgeNode(1, 2, 6, 1, 7, 13),
              EdgeNode(2, 3, 6, 2, 1, 14),
              EdgeNode(3, 4, 6, 3, 2, 15),
              EdgeNode(3, 9, 3, 2, 3, 12),
              EdgeNode(4, 6, 5, 3, 8, 11),
              EdgeNode(6, 7, 5, 4, 5, 10),
              EdgeNode(1, 5, 5, 6, 9, 8),
              EdgeNode(4, 5, 6, 5, 3, 7),
              EdgeNode(1, 7, 1, 5, 1, 6),
              EdgeNode(7, 8, 1, 4, 9, 12),
              EdgeNode(6, 9, 4, 3, 6, 4),
              EdgeNode(9, 8, 4, 2, 11, 13),
              EdgeNode(2, 8, 2, 1, 2, 10)]

faces = [1, 6]

number_of_faces = 0
for edge in edge_nodes:
    if edge.f1 > number_of_faces:
        number_of_faces = edge.f1
    if edge.f2 > number_of_faces:
        number_of_faces = edge.f2

face_lists = []
for _ in range(number_of_faces):
    face_lists.append([])

for i in range(len(edge_nodes)):
    face_lists[edge_nodes[i].f1-1].append(i+1)
    face_lists[edge_nodes[i].f2-1].append(i+1)

edge_counts = [0] * len(edge_nodes)
for face in faces:
    face_list = face_lists[face-1]
    for edge in face_list:
        edge_counts[edge-1] += 1

found_edges = []
for i in range(len(edge_counts)):
    if edge_counts[i] == 1:
        found_edges.append(i+1)
        
print(found_edges)
