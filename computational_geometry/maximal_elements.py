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

points = sorted(points, key=point_y, reverse=True)
points = sorted(points, key=point_x, reverse=True)

maximal_elements = []
y_current = points[0].y
i, add_flag = 0, True
if i < len(points)-1 and points[i].y == points[i+1].y:
    add_flag = False
if add_flag:
    maximal_elements.append(points[0])
for j in range(len(points)):
    if points[j].y > y_current:
        i, add_flag = j, True
        if i < len(points)-1 and points[i].y == points[i+1].y:
            add_flag = False
        if add_flag:
            maximal_elements.append(points[j])
        y_current = points[j].y
        
for maximal_point in maximal_elements:
    print(maximal_point)
