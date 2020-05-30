class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __repr__(self):
        return f"({self.x}, {self.y})"

points_a = [Point(0,0), Point(9,-2)]
points_b = [Point(-1,5), Point(3,4), Point(9,3)]

i, j = 0, 0
flag = True
xd, yd, min_dist = float('Inf'), float('Inf'), float('Inf')

while i < len(points_a) and j < len(points_b):
    cur_xd = abs(points_a[i].x - points_b[j].x)
    cur_yd = abs(points_a[i].y - points_b[j].y)
    if (cur_xd >= xd and cur_yd >= yd) \
        or (i == len(points_a)-1 and not flag) or (j == len(points_b)-1 and flag):
        flag = not flag
    cur_dist = cur_xd + cur_yd
    xd = cur_xd
    yd = cur_yd
    if cur_dist < min_dist:
        min_dist = cur_dist
        closest_pair = (points_a[i], points_b[j])
    if flag:
        j += 1
    else:
        i += 1
        
print(closest_pair)
