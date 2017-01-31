from __future__ import division
from Frame import Frame
from Frame import Spectrum
import Constants as const
import os
import sys
import getopt
from algorithm.CSinput import DirectoryEmbedded
import json
import numpy as np

uK = const.uK

class Printer(DirectoryEmbedded):
    def __init__(self, image_directory):
        DirectoryEmbedded.__init__(self, os.path.join(image_directory))
        
        self.resolution = 3*const.arcmin
        self.num_pixels = 100
        
        self.C_EE = Spectrum(
            [(2, 0.2*uK), (10, 0.05*uK), (600, 4*uK), (1100, 4*uK), (3000, 1*uK)]
        )
        self.C_TT = Spectrum(
            [(2, 30*uK), (30, 30*uK), (100, 70*uK), (700, 40*uK), (3000, 5*uK), (20000, 5*uK)]
        )
    
    @property
    def size(self):
        return self.resolution * self.num_pixels
    
    def draw(self, exaggerated=False):
        frame = Frame(0, 0., 5*const.deg, 100)
        frame.add_noise(self.C_EE)
        frame.add_strings(25, int(exaggerated)*100)
    
    def sphere_decomposition(self):
        latitude_slices = int(round(np.pi / self.size))
        slice_width = np.pi / latitude_slices
        
        decomposition = []
        for slice in range(latitude_slices):
            slice_angle = slice_width * (1/2 + slice)
            slice_length = 2*np.pi * (np.sin(slice_angle - slice_width/2) + np.sin(slice_angle + slice_width/2))/2
            
            num_longitude = int(round(slice_length * slice_width / self.size**2))
            decomposition.append((slice_angle, num_longitude))
        
        return decomposition
    
    def scale(self, theta):
        return 3/2 * np.sin(theta)**2
    
    def queues(self, train=False):
        if train:
            return [[(np.random.randint(2)*np.pi/2, i % 1) for i in range(1000)]]
        
        queue = []
        
        for theta, num_images in self.sphere_decomposition():
            for image_num in range(num_images):
                phi = 2*np.pi * (image_num/num_images)
                queue.append((theta, phi))
        
        return [queue[i:i + 200] for i in range(0, len(queue), 200)]
    
    def print_to_file(self, train=False, string=True):
        for queue, queue_num in zip(self.queues(train=train), range(1000)):
            with open(self.file('images' + str(queue_num+1) + '.txt'), 'w') as image_file:
                with open(self.file('info' + str(queue_num+1) + '.txt'), 'w') as info_file:
                    for theta, phi in queue:
                        image = Frame(theta, phi, self.size, self.num_pixels)
                        image.add_noise(self.C_EE)
                        if string:
                            image.add_strings(25, self.scale(theta))
                        watermark = ", " + str(int(round(self.scale(theta)/self.scale(np.pi/2))))
                        
                        array = np.reshape(image.pixels, [image.num_pixels**2]) / (10 * const.uK)
                        image_file.write(", ".join(str(x) for x in array) + watermark + '\n')
                        dictionary = {'theta':theta, 'phi':phi, 'scale':self.scale(theta)}
                        info_file.write(json.dumps(dictionary) + "\n")
                        
                        print "Theta: %d, Phi: %d" %(theta/const.deg, phi/const.deg)
                    
                    print queue_num + 1

if __name__ == "__main__":
    train = False
    directory = ""
    string = True
    
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
    p.print_to_file(train=train, string=string)
