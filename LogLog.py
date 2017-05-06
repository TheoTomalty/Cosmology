from __future__ import division
import numpy as np
import math

class LogLog(object):
    def __init__(self, nodes=(), spacing=None):
        ''' Simple class to set up a piecewise liner plot in log-log space.
        
        # Define the spectrum
        >>> loglog = LogLog()
        
        # Add the nodes
        >>> loglog.append(10, 3)
        >>> loglog.append(100, 5)
        >>> loglog.append(50, 4)
        
        # Evaluate Spectrum
        >>> eval = [loglog.eval(x) for x in range(20, 80, 10)]
        '''
        
        self.nodes = []
        self.spacing = spacing
        
        for node in nodes:
            self.append(*node)
        
    def sort(self):
        # Sort by x_value
        self.nodes = sorted(self.nodes, key=lambda vertex: vertex[0])
    
    def append(self, x, y):
        # Add a point to the spectrum
        self.nodes.append((x, y))
        
    def eval(self, x):
        ''' Evaluate the power per log interval of wave number $\Delta_T^2 = \frac{l(l+1)C_l}{2\pi}$
        
        :param l: Spherical harmonic number
        :return: Power of modes per log interval in wave number
        '''
        x = float(x)
        assert len(self.nodes) > 0, "No nodes in spectrum"
        
        if x <= self.nodes[0][0]:
            return self.nodes[0][1]
        
        end = None
        start = None
        
        if self.spacing is None:
            for node, index in zip(self.nodes, range(len(self.nodes))):
                if not index:
                    continue
                if x < node[0] or index == len(self.nodes) - 1:
                    end = node
                    start = self.nodes[index - 1]
                    break
        else:
            low_index = int(math.floor(x/self.spacing) -1)
            if low_index >= len(self.nodes) - 1:
                low_index = len(self.nodes) - 2
            
            start = self.nodes[low_index]
            end = self.nodes[low_index + 1]
            
            
        # Linear in log-log between nodes
        alpha = (
            (np.log(float(start[1])) - np.log(float(end[1]))) /
            np.log(float(start[0]) / float(end[0]))
        )
        return start[1] * (x/start[0])**alpha
