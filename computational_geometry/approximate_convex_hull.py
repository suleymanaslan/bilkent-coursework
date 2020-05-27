from math import ceil, floor
class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __repr__(self):
        return f"({self.x}, {self.y})"
        
def point_x(point):
    return point.x

def point_y(point):
    return point.y

points = [Point(0,1),
          Point(1,0),
          Point(1,2),
          Point(2,4),
          Point(3,0),
          Point(3.9,3),
          Point(5,0),
          Point(5,2),
          Point(5.9,1)]
d = 2
X = 5

start_point = points[0]
for i in range(len(points)):
    if points[i].x < start_point.x:
        start_point = points[i]
    elif points[i].x == start_point.x and points[i].y < start_point.y:
        start_point = points[i]

slices = []
for _ in range(ceil(X/d)):
    slices.append([])
    
for i in range(len(points)):
    slice_index = floor(points[i].x/d)
    slices[slice_index].append(points[i])

slice_max_points = []
slice_min_points = []
for i in range(len(slices)):
    slice_max_points.append(max(slices[i], key=point_y))
    slice_min_points.append(min(slices[i], key=point_y))

hull_vertices = []
hull_vertices.append(start_point)
for i in range(len(slices)):
    # eliminating points with Graham's scan is omitted
    hull_vertices.append(slice_min_points[i])

end_point = points[0]
for i in reversed(range(len(slices))):
    if i == len(slices) - 1:
        end_point = max(slices[i], key=point_x)
        hull_vertices.append(end_point)
    # eliminating points with Graham's scan is omitted
    hull_vertices.append(slice_max_points[i])
    
for point in hull_vertices:
    print(point)
