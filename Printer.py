from __future__ import division

import getopt
import json
import os
import sys

import numpy as np

import Constants as const
from Frame import Frame
from Frame import Spectrum
from algorithm.DirectoryEmbedded import DirectoryEmbedded

uK = const.uK

class Printer(DirectoryEmbedded):
    def __init__(self, image_directory):
        DirectoryEmbedded.__init__(self, os.path.join(image_directory))
        
        self.resolution = const.resolution
        self.num_pixels = const.image_pixel_width
    
    @property
    def size(self):
        return self.resolution * self.num_pixels
    
    def draw(self, exaggerated=False):
        frame = Frame(0, 0., 5*const.deg, 100)
        frame.add_noise(const.C_EE)
        frame.add_strings(int(exaggerated)*100)
    
    def scale(self, theta):
        return 3/2 * np.sin(theta)**2
    
    def train_queue(self, n):
        return [[(np.pi/2, 0)]*200]*int(round(n/200))
    
    def sphere_queues(self):
        # In south pole coordinates
        angular_length = 2*np.pi * (11/24)
        angular_width = 50*const.deg
        
        latitude_slices = int(round(angular_width / self.size))
        slice_width = angular_width / latitude_slices
        
        queue = []
        for slice in range(latitude_slices):
            theta = slice_width * (1/2 + slice) + 25*const.deg
            slice_length = angular_length * (np.sin(theta - slice_width/2) + np.sin(theta + slice_width/2))/2
            
            num_longitude = int(round(slice_length * slice_width / self.size**2))
            
            for image_num in range(num_longitude):
                phi = angular_length * (image_num/num_longitude)
                r = np.array([np.sin(theta) * np.sin(phi) , np.cos(theta), np.sin(theta) * np.cos(phi)])
                # (Theta, Phi) in CMB quadrupole coordinates
                queue.append((np.arctan(r[2]/np.sqrt(r[0]**2 + r[1]**2)), (np.pi/2 if not r[0] else np.arctan(r[1]/r[0]))))
        
        return [queue[i:i + 200] for i in range(0, len(queue), 200)]
    
    def print_to_file(self, n, train=False, string=True):
        queues = self.train_queue(n) if train else self.sphere_queues()
        counter = 0
        for queue, queue_num in zip(queues, range(1000)):
            with open(self.file('images' + str(queue_num+1) + '.txt'), 'w') as image_file:
                with open(self.file('info' + str(queue_num+1) + '.txt'), 'w') as info_file:
                    for theta, phi in queue:
                        image = Frame(theta, phi, self.size, self.num_pixels)
                        image.add_noise(const.C_EE)
                        if string:
                            image.add_strings(self.scale(theta))
                        
                        pixels = np.reshape(image.pixels, [image.num_pixels**2]) / (const.uK)
                        regions = np.reshape(image.regions, [image.num_regions**2])
                        image_file.write(", ".join(str(x) for x in pixels) + ", " +
                                         ", ".join(str(x) for x in regions)+ '\n')
                        
                        dictionary = {'theta':theta, 'phi':phi, 'scale':self.scale(theta)}
                        info_file.write(json.dumps(dictionary) + "\n")
                        
                        counter += 1
                        print str(counter) + ": ", "Theta: %d, Phi: %d" %(theta/const.deg, phi/const.deg)
                    
                    print queue_num + 1

if __name__ == "__main__":
    train = False
    directory = ""
    string = True
    n = 200
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"n:i:",["train", "nostring"])
    except getopt.GetoptError:
        print "Invalid inputs"
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-n':
            n = int(arg)
        elif opt == "--train":
            train = True
        elif opt == "--nostring":
            string = False
        elif opt == "-i":
            #Input directory to save the numbered image files, ex: "images1.txt"
            if not os.path.exists(arg):
                print "Image Directory not found."
                sys.exit()
            else:
                directory = arg
    
    p = Printer(directory)
    p.print_to_file(n, train=train, string=string)
