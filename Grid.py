import numpy as np
import math

def unit(x):
    return x / np.linalg.norm(x)

def norm(x):
    return np.linalg.norm(x)

def floor(num):
    return int(math.floor(num))

def ceiling(num):
    return int(math.ceil(num))

def int_round(num):
    return int(round(num))

class Grid(object):
    def __init__(self, spacing):
        self.grid_0 = np.array([0, 0])
        
        self.dx = np.array([spacing, 0])
        self.dy = np.array([0, spacing])
    
    def set_origin(self, x_0, y_0):
        self.grid_0 = np.array([x_0, y_0])
        
    def phi(self, x):
        return np.arccos(np.dot(unit(x), unit(self.dx)))
    
    def pos(self, i, j):
        return self.grid_0 + i * self.dx + j * self.dy
    
    def order_points(self, x_1, x_2, x_3, x_4):
        # Order the given points as: Bottom, Left, Right, Top
        
        points = [x_1, x_2, x_3, x_4]
        bottom_top = sorted(points, key=lambda point: point[1])
        left_right = sorted(bottom_top[1:-1], key=lambda point: point[0])
        
        return bottom_top[0], left_right[0], left_right[-1], bottom_top[-1]
    
    def pixelate_area(self, x_1, x_2, x_3, x_4):
        bottom, left, right, top = self.order_points(x_1, x_2, x_3, x_4)
        
        bot_left, a = unit(left - bottom), norm(left - bottom)
        left_top, b = unit(top - left), norm(top - left)
        extrapolated_left = left - (a / np.tan(self.phi(left_top))) * left_top
        
        bot_right, c = unit(right - bottom), norm(right - bottom)
        right_top, d = unit(top - right), norm(top - right)
        extrapolated_right = right - (c / np.tan(np.pi - self.phi(right_top))) * right_top
        
        ranges = []
        j = ceiling((bottom[1] - self.grid_0[1])/self.dy[1])
        height = self.grid_0[1] + j * self.dy[1]
        
        while height < top[1]:
            left_bound1 = (bottom + (height - bottom[1])/np.sin(self.phi(bot_left)) * bot_left)[0]
            left_bound2 = (extrapolated_left + (height - bottom[1])/np.sin(self.phi(left_top)) * left_top)[0]
            
            right_bound1 = (bottom + (height - bottom[1])/np.sin(self.phi(bot_right)) * bot_right)[0]
            right_bound2 = (extrapolated_right + (height - bottom[1])/np.sin(self.phi(right_top)) * right_top)[0]
            
            i_min = ceiling((max(left_bound1, left_bound2) - self.grid_0[0])/self.dx[0])
            i_max = ceiling((min(right_bound1, right_bound2) - self.grid_0[0])/self.dx[0])
            
            if i_min < i_max:
                ranges.append((j, i_min, i_max))
            
            j += 1
            height += self.dy[1]
        
        return ranges
    
    def edge_scan(self, x_1, x_2):
        bottom, top = sorted([x_1, x_2], key=lambda vect: vect[1])
        bot_top, c = unit(top - bottom), norm (top - bottom)
        x_iter_sign = (1 if bot_top[0] > 0 else -1)
        
        i, j = int_round((bottom[0] - self.grid_0[0])/self.dx[0]), int_round((bottom[1] - self.grid_0[1])/self.dy[1])
        
        s = 0
        position = bottom
        indices = []
        y_line = self.grid_0[1] + (j + 0.5)*self.dy[1]
        x_line = self.grid_0[0] + (i + x_iter_sign*0.5)*self.dx[0]
        
        while s < c:
            indices.append((i, j))
            
            x_intersect = (y_line - position[1])/np.tan(self.phi(bot_top))
            s_x = np.sqrt(x_intersect**2 + (y_line - position[1])**2)
            y_intersect = (x_line - position[0])*np.tan(self.phi(bot_top))
            s_y = np.sqrt(y_intersect**2 + (x_line - position[0])**2)
            
            if s_x < s_y:
                s += s_x
                position = np.array([position[0] + x_intersect, y_line])
                j += 1
                y_line += self.dy[1]
            elif s_y < s_x:
                s += s_y
                position = np.array([x_line, position[1] + y_intersect])
                i += x_iter_sign
                x_line +=  x_iter_sign * self.dy[1]
            else:
                raise Exception
        
        return indices
