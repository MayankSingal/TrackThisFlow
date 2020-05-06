import torch
from math import pi

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return Vector(self.x + v.x, self.y + v.y)

    def __sub__(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return Vector(self.x - v.x, self.y - v.y)

    def cross(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return self.x*v.y - self.y*v.x


class Line:
    # ax + by + c = 0
    def __init__(self, v1, v2):
        self.a = v2.y - v1.y
        self.b = v1.x - v2.x
        self.c = v2.cross(v1)

    def __call__(self, p):
        return self.a*p.x + self.b*p.y + self.c

    def intersection(self, other):
        # See e.g.     https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Using_homogeneous_coordinates
        if not isinstance(other, Line):
            return NotImplemented
        w = self.a*other.b - self.b*other.a
        return Vector(
            (self.b*other.c - self.c*other.b)/w,
            (self.c*other.a - self.a*other.c)/w
        )


def intersection_area_projected_to_2d(r1, r2):
    # r1 and r2 are in (center, width, height, rotation) representation
    # First convert these into a sequence of vertices

    r1 = r1.squeeze().view(-1,3)
    r2 = r2.squeeze().view(-1,3)

    rect1 =   (
        Vector(r1[0][0], r1[0][2]) ,
        Vector(r1[1][0], r1[1][2]) ,
        Vector(r1[2][0], r1[2][2]) ,
        Vector(r1[3][0], r1[3][2]) 
    )

    rect2 =   (
        Vector(r2[0][0], r2[0][2]) ,
        Vector(r2[1][0], r2[1][2]) ,
        Vector(r2[2][0], r2[2][2]) ,
        Vector(r2[3][0], r2[3][2])
    )


    # Use the vertices of the first rectangle as
    # starting vertices of the intersection polygon.
    intersection = rect1

    # Loop over the edges of the second rectangle
    for p, q in zip(rect2, rect2[1:] + rect2[:1]):
        if len(intersection) <= 2:
            break # No intersection

        line = Line(p, q)

        # Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
        # any point p with line(p) > 0 is on the "outside".

        # Loop over the edges of the intersection polygon,
        # and determine which part is inside and which is outside.
        new_intersection = []
        line_values = [line(t) for t in intersection]
        for s, t, s_value, t_value in zip(
            intersection, intersection[1:] + intersection[:1],
            line_values, line_values[1:] + line_values[:1]):
            if s_value <= 0:
                new_intersection.append(s)
            if s_value * t_value < 0:
                # Points are on opposite sides.
                # Add the intersection of the lines to new_intersection.
                intersection_point = line.intersection(Line(s, t))
                new_intersection.append(intersection_point)

        intersection = new_intersection

    # Calculate area
    if len(intersection) <= 2:
        return 0

    return 0.5 * sum(p.x*q.y - p.y*q.x for p, q in
                     zip(intersection, intersection[1:] + intersection[:1]))
    
    
def iou_projected_to_2d(r1, r2):
    
    r1 = r1.squeeze().view(-1,3)
    r2 = r2.squeeze().view(-1,3)
    
    area1 = torch.sqrt((r1[0][0] - r1[1][0])**2 + (r1[0][2] - r1[1][2])**2) * torch.sqrt((r1[1][0] - r1[2][0])**2 + (r1[1][2] - r1[2][2])**2) 
    area2 = torch.sqrt((r2[0][0] - r2[1][0])**2 + (r2[0][2] - r2[1][2])**2) * torch.sqrt((r2[1][0] - r2[2][0])**2 + (r2[1][2] - r2[2][2])**2)
    intersection = intersection_area_projected_to_2d(r1, r2)
    
    # print(intersection, area1, area2)
    
    iou = intersection / (area1 + area2 - intersection + 0.000001)
    
    return iou
    