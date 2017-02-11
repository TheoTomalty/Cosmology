from __future__ import division
import numpy as np
import math
import Constants as const

class SimpleWake(object):
    def __init__(self, angle, phi, theta, total_width, length, orientation):
        self.angle = angle
        self.phi = phi
        self.theta = theta
        self.length = length
        self.total_width = total_width
        self.span = 1*const.deg
        self.smooth = 0.5*const.deg
        
        assert abs(orientation) == 1, "The attribute orientation must be +/- 1"
        self.orientation = orientation
    
    @property
    def slope(self):
        return np.tan(self.angle)
    
    @property
    def inverse_slope(self):
        return -1/self.slope
    
    def linear_coords(self, x, y):
        # [rho, line] conversion
        rho = x*math.sin(self.angle) - y*math.cos(self.angle)
        line = x*math.cos(self.angle) + y*math.sin(self.angle)
        
        return rho, line
    
    def width(self, rho, line):
        rho = self.orientation * rho
        line = abs(line)
        if line < self.length/2 - self.smooth:
            scaling = 1
        elif line < self.length/2:
            scaling = (self.length/2 - line)/self.smooth
        else:
            scaling = 0
        
        if abs(rho) > self.span/2:
            return 0
        
        return scaling*(rho + self.span/2)/self.span * self.total_width
    
    def width_at_pixel(self, x, y, i, j, size):
        x = x - self.phi + j*size
        y = y - self.theta + i*size
        
        linear = self.linear_coords(x, y)
        return 10*const.uK#self.width(*linear)
    
    def pixelation(self, step):
        assert 0 < self.angle < math.pi
        
        sign = (1 if self.angle < math.pi/2 else -1)
        slope = abs(self.slope)
        inverse_slope = -abs(self.inverse_slope)
        if sign > 0:
            angle = self.angle
        else:
            angle = math.pi - self.angle
        
        side1 = - np.cos(angle) * self.length/2
        side2 =  abs(np.cos(math.pi/2 - angle) * self.span/2)
        bottom = slope * side1 - abs(inverse_slope) * side2
        
        height_step = step
        ranges = []
        while True:
            left_range1 = - height_step / abs(inverse_slope)
            left_range2 = height_step / abs(slope) - self.span / np.sin(angle)
            
            right_range1 = height_step / abs(slope)
            right_range2 = -height_step / abs(inverse_slope) + self.length / np.cos(angle)
            
            index_left = int(math.ceil(max(left_range1, left_range2) / step))
            index_right = int(math.floor(min(right_range1, right_range2) / step))
            
            if index_left > index_right:
                break
            else:
                height_step += step
                ranges.append((index_left, index_right))
        
        if sign > 0:
            return self.phi + side1 + side2, self.theta + bottom, ranges
        
        inverted_ranges = []
        for bounds in ranges:
            inverted_ranges.append((-bounds[1], -bounds[0]))
        
        return self.phi - (side1 + side2), self.theta + bottom, inverted_ranges
    
    def edge_scan(self, x_pixel, y_pixel, size):
        sign = (-1 if self.angle < np.pi/2 else 1)
        centre_line = (- sign*np.sin(self.angle) * (self.span/2), sign*np.cos(self.angle) * (self.span/2))
        centre_index = (int(round((centre_line[0]-x_pixel)/size)), int(round((centre_line[1]-y_pixel)/size)))
        centre_pixel = (centre_index[0] * size + x_pixel, centre_index[1] * size + y_pixel)
        print centre_line, centre_index, centre_pixel
        
        s = 0
        position = centre_line
        index = centre_index
        indices = [index]
        y_line = centre_pixel[1] + 0.5 * size
        x_line = centre_pixel[0] + 0.5 * size
        
        while s < self.length + self.smooth/2:
            x_intersect = abs((y_line - position[1])/self.slope)
            s_x = np.sqrt(x_intersect**2 + (y_line - position[1])**2)
            y_intersect = abs((x_line - position[0])*self.slope)
            s_y = np.sqrt(y_intersect**2 + (x_line - position[0])**2)
            print x_intersect, y_intersect
            #print s, position, index, y_line, x_line, x_intersect, y_intersect, self.slope
            
            if s_x < s_y:
                s += s_x
                position = (position[0] + x_intersect, y_line)
                index = (index[0], index[1] + 1)
                indices.append(index)
                y_line += size
            elif s_y < s_x:
                s += s_y
                position = (x_line, position[1] + y_intersect)
                index = (index[0] - sign, index[1])
                indices.append(index)
                x_line += size
            else:
                print "WRONG"
            #print s, position, index, y_line, x_line
        
        #print indices
        
        return indices

def gamma(v):
    # Reletivistic gamma factor for a velocity v
    return 1/np.sqrt(1 - (v)**2)

def time(z):
    # The time (measured from Big Bang) corresponding to the redshift z
    return const.t_0/(1.0 + z)**(3/2)

def redshift(t):
    # The redshift associated with the given time t
    return (const.t_0/t)**(2/3) - 1

def a(t):
    # Scale factor of the universe at the given time t
    # Set to unity at present time
    return (t/const.t_0)**(2/3)

def hubble_length(t):
    # Hubble length at the given time t using scale factor definition
    return 3*t/2

def comoving(x, t):
    # Comoving distance associated with physical distance x at time t
    return x/a(t)

def physical(x_c, t):
    # Physical distance associated with comoving distance x_c at time t
    return x_c*a(t)

class String(object):
    def __init__(self, tension=3e-7):
        self.tension = tension
        self.theta = 7*const.deg
        self.phi = 0*const.deg
        self.z = 1000.0
        self.v_s = 0.8 # light-years per year, perpendicular to us
    
    @property
    def t_i(self):
        return time(self.z)
    
    @property
    def r_c(self):
        return comoving(1/2*hubble_length(self.t_i), self.t_i)

class Wake(String):
    def __init__(self, tension=3e-7):
        String.__init__(self, tension)
    
    @property
    def t_cross(self):
        return ((3*const.t_0 + self.r_c)/(3*const.t_0**(2/3)))**3
    
    @property
    def dims(self):
        return const.c_1*hubble_length(self.t_cross), self.t_i*self.v_s*gamma(self.v_s)
    
    @property
    def psi_0(self):
        return 24*const.pi/5*self.tension*self.v_s*gamma(self.v_s)*const.t_0/(self.z + 1)**(0.5)
    
    def height(self, t):
        return self.psi_0*(self.z + 1)/(redshift(t) + 1)**2
        
    def width(self, theta, phi):
        return

