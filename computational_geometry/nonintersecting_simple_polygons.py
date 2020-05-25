class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __repr__(self):
        return f"({self.x}, {self.y})"

class Edge():
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2
        
    def __repr__(self):
        return f"{self.v1}-{self.v2}"

def point_x(point):
    return point.x

def point_y(point):
    return point.y

def position(edge, point):
    sign = lambda a: (a>0) - (a<0)
    return sign((edge.v2.x - edge.v1.x) * (point.y - edge.v1.y) 
                - (edge.v2.y - edge.v1.y) * (point.x - edge.v1.x))

points = [Point(0,1), Point(1,3), Point(2,4), Point(3,3), Point(4,0), Point(5,4), Point(5,1), Point(7,2), Point(1,0), Point(2,-1)]

points = sorted(points, key=point_y)
points = sorted(points, key=point_x)

polygon1_edges = []
polygon1_edges.append(Edge(points[0], points[1]))
polygon1_edges.append(Edge(points[1], points[2]))
polygon1_edges.append(Edge(points[2], points[0]))
del points[2]
del points[1]
del points[0]

point_a = points[0]
point_b = points[-1]
temporary_edge = Edge(point_a, point_b)

points_b = []
points_c = []
for point in points:
    if position(temporary_edge, point) >= 0:
        points_b.append(point)
    else:
        points_c.append(point)

polygon2_edges = []
for i in range(len(points_b)):
    if i != len(points_b)-1:
        polygon2_edges.append(Edge(points_b[i], points_b[i+1]))
for i in range(len(points_c)):
    if i != len(points_c)-1:
        polygon2_edges.append(Edge(points_c[i], points_c[i+1]))
polygon2_edges.append(Edge(points_b[0], points_c[0]))
polygon2_edges.append(Edge(points_b[-1], points_c[-1]))

polygon1_str = ""
for edge in polygon1_edges:
    polygon1_str += str(edge) + " "
polygon2_str = ""
for edge in polygon2_edges:
    polygon2_str += str(edge) + " "
print(polygon1_str)
print(polygon2_str)
