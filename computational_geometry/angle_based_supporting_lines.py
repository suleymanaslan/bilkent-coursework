from math import atan2, degrees
class Point():
    def __init__(self, x, y, prev_p=None, next_p=None):
        self.x = x
        self.y = y
        self.prev_p = prev_p
        self.next_p = next_p
        
    def __repr__(self):
        return f"{(self.x, self.y)}"
        
class Line():
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        
    def __repr__(self):
        return f"{self.p1}-{self.p2}"

def GetAngleLines(l1, l2):
    p1, p2 = l1.p1, l1.p2
    angle1 = GetAngle(p1, p2)
    p1, p2 = l2.p1, l2.p2
    angle2 = GetAngle(p1, p2)
    angle = abs(angle1 - angle2)
    return angle

def GetAngle(p1, p2):
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    dX = x2 - x1
    dY = y2 - y1
    rads = atan2(-dY, dX)
    return degrees(rads)

p0 = Point(0,1,None,None)
p1 = Point(1,0,p0,None)
p2 = Point(2,1,p1,None)
p3 = Point(1,2,p2,p0)
p0.prev_p = p3
p0.next_p = p1
p1.next_p = p2
p2.next_p = p3

s0 = Point(4,1,None,None)
s1 = Point(5,0,s0,None)
s2 = Point(6,1,s1,None)
s3 = Point(5,2,s2,s0)
s0.prev_p = s3
s0.next_p = s1
s1.next_p = s2
s2.next_p = s3

p1_start = p0
p2_start = s0

supporting_lines = []
cur_p1 = p1_start
while True:
    cur_p2 = p2_start
    while True:
        cur_line = Line(cur_p1, cur_p2)
        v_prev = Line(cur_p1.prev_p, cur_p1)
        v_next = Line(cur_p1, cur_p1.next_p)
        w_prev = Line(cur_p2.prev_p, cur_p2)
        w_next = Line(cur_p2, cur_p2.next_p)
        angle1 = GetAngleLines(v_prev, cur_line)
        angle2 = GetAngleLines(cur_line, v_next)
        angle3 = GetAngleLines(w_prev, cur_line)
        angle4 = GetAngleLines(cur_line, w_next)
        supporting_line = True if (((0<angle1<90 and 0<angle2<90) 
                                   or (90<angle1<180 and 90<angle2<180))
                                   and ((0<angle3<90 and 0<angle4<90) 
                                   or (90<angle3<180 and 90<angle4<180))) else False
        if supporting_line:
            supporting_lines.append(cur_line)
        if cur_p2.next_p == p2_start:
            break
        cur_p2 = cur_p2.next_p
    if cur_p1.next_p == p1_start:
        break
    cur_p1 = cur_p1.next_p

for supporting_line in supporting_lines:
    print(supporting_line)
