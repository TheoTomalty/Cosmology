from __future__ import division
import json
import getopt
import sys
import numpy as np
import Constants as const
from algorithm.DirectoryEmbedded import *

class Analysis(DirectoryEmbedded):
    def __init__(self, directory):
        DirectoryEmbedded.__init__(self, directory)
        self.dA = (const.resolution * const.region_pixel_width)**2
        
        self.int = 0.0
        self.int_sin2 = 0.0
        self.int_sin4 = 0.0
        self.int_x = 0.0
        self.int_sin2_x = 0.0
        
        # Error Approximation
        self.N = 0
        self.sum_x = 0.0
        self.sum_x2 = 0.0
    
    @property
    def normalization(self):
        int_dev2 = self.int_sin4 + (self.int - 2) * self.int_sin2**2
        
        if int_dev2 == 0:
            return 0
        
        return 1/np.sqrt(int_dev2)
    
    @property
    def Q(self):
        return self.normalization * (self.int_sin2_x - self.int_sin2 * self.int_x)
    
    @property
    def Q_err(self):
        return np.sqrt(self.var_Q)
    
    @property
    def var_x(self):
        if not self.N:
            return 0
        
        mean_x = self.sum_x / self.N
        mean_x2 = self.sum_x2 / self.N
        return mean_x2 - mean_x**2
    
    @property
    def var_Q(self):
        return self.var_x * self.dA
    
    def print_results(self):
        print "Q: %f +/- %f, Area: %d" %(self.Q, self.Q_err, self.int/(const.deg)**2)
    
    def add_line(self, line):
        data = json.loads(line)
        
        theta = data['theta']
        scalar, labels = np.array(data['scalar']), np.array(data['labels'])
        
        for value, label in zip(scalar.reshape(-1), labels.reshape(-1)):
            self.int += self.dA
            self.int_sin2 += self.dA * np.sin(theta)**2
            self.int_sin4 += self.dA * np.sin(theta)**4
            self.int_x += self.dA * value
            self.int_sin2_x += self.dA * value * np.sin(theta)**2
            
            if not label:
                self.N += 1
                self.sum_x += value
                self.sum_x2 += value**2
    
    def load_file(self, file_name):
        with open(file_name, 'r') as file:
            for line in file:
                self.add_line(line[:-1])
                
    def load_directory(self):
        files = get_files(self.directory, "final")
        for file_name in files:
            self.load_file(file_name)

if __name__ == "__main__":
    directory = ""
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"i:",[])
    except getopt.GetoptError:
        print "Invalid inputs"
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-i":
            #Input directory to save the numbered image files, ex: "images1.txt"
            if not os.path.exists(arg):
                print "Image Directory not found."
                sys.exit()
            else:
                directory = arg
    
    a = Analysis(directory)
    a.load_directory()
    a.print_results()
