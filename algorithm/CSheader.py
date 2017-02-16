import tensorflow as tf
import math

class GlobalFlags:
    '''
    Class that handles all of the algorithm global variables including:
        1) Run Directives: Which directories to use, train or test or output Tensorboard images, number of batches/files
        3) Fixed Parameters: Unchanging values specifying algorithm hyper-parameters
        3) Variable Parameters: Algorithm hyper-parameters that can be changed according to desired functionality
        
        All Parameters can be set using a single-line JSON string parameter file (inserted to CSalgorithm.py call using the -p flag).
        The dictionary should have keys corresponding to parameter names to be overwritten, and their corresponding desired values.
        It is recommended to only overwrite Variable Parameters this way, and to specify all Directives
        using the c-style flags when calling CSalgorithm.py (use "python CSalgorithm.py -h" for a list of options).
        
        FIXED PARAMETERS SHOULD NOT BE CHANGED UNLESS TO MODIFY THE ARCHITECTURE OF THE ALGORITHM
    '''
    
    def __init__(self):
        # Run Directives
        self.image_directory = "" # Directory path that contains the images numbered
        self.run_directory = "" # Directory path where training checkpoints and testing results are saved
        self.training = True # Indicates whether training or testing
        self.continue_session = False # If Training: Initialise networks with the values from most recent session
        self.__num = None # Set by set_num_iterations. Number of iterations for given task, num batches/files for training/testing resp.
        self.print_tensorboard = False # Use with <self.training = 0> to save images into logs that can be seen using Tensorboard
        
        # Fixed Parameters
        self.num_pixels = 75 # Width of square input images to network, in number of pixels. Image is num_pixels x num_pixels in total.
        self.num_regions = 5
        self.min_after_dequeue = 200 # Quantity used for shuffling batches. Higher = better randomization but slower processing speed.
        self.filter_size = 5 # Pixel width of filters used for feature identification. To change this you will need to also modify 
                             # the size of the default filters in get_initial_filters. Preferably an odd number.
        self.images_per_file = 200 
        self.num_tensorboard = 20
        
        # Variable Parameters
        self.average_decay_rate = 0. # Weight used for moving averages of network variables $avg_{i} = 0.995*avg_{i-1} + (1.0 - 0.995)*var_{i}$
        self.batch_size = 300 # Number of images in a single batch (stochastic method of machine learning)
        self.num_angles = 12 # The number of angles for which to create corresponding filters
        self.initial_learning_rate = 0.5 # Step size used by tf.train.AdamOptimizer before decay
        self.decay_rate_per_thousand = 1/math.e # Ratio that learning rate (step size) is reduced in every 1000 batch interval
    
    def set_parameters(self, parameters):
        # Save the entries of parameters dictionary into the instance of this class (load parameters from dictionary)
        for key in parameters:
              setattr(self, key, parameters[key])
    
    def get_batch_size(self):
        # Use a different batch size between training and testing/Tensorflow 
        if self.training:
          return self.batch_size
        # Testing files are set to 2000 entries when being separated in Setup.py
        # Load entire file in one batch since algorithm doesn't have to compute gradients (so stochastic method is not necessary)
        return self.images_per_file
    
    @property
    def data_size(self):
        # Number of input variables to a given network, number of pixels in the square images of rings
        return self.num_pixels**2
    
    @property
    def label_size(self):
        return self.num_regions**2
    
    @property
    def num_iterations(self):
        # Default number of iterations to do if none is given using self.set_num_iterations
        if self.__num is not None:
            return self.__num
        if self.training:
            # Default number of batches if training
            return 20
        # Default number of files to test on if testing
        return 100
      
    def set_num_iterations(self, num_iterations):
        # num_iterations setter function
        self.__num = num_iterations

######################################################
#
# Create Instance of the GlobalFlags class:
#   - Call instance in other files with CSheader.FLAGS
#   - Called by reference so all changes are global
#
#######################
FLAGS = GlobalFlags()##
######################################################


def conv2d(x, W):
    # Convolute a 2D image, x, with a set of filters, W, using one-pixel steps
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def get_initial_filters(num_angles):
    ''' Generates a set of filters that will be used to initialize the networks so that they 
        do not have to learn these shapes on their own. From experience, the algorithm was 
        never able to converge when this was not done.
    
    :param num_angles: Number of angles (evenly distributed around full circle) for the edge filters.
    :param num_zero_filters: Number of filters with uniform zero weights.
    :return: List of filters, number of filters
    '''
    
    assert FLAGS.filter_size == 5, "If you change FLAGS.filter_size you must also change the initial filter sizes in CSheader.py"
    filter_size = FLAGS.filter_size
    
    # First eval of number of filters (note that each angle will have one smooth edge and one rigid edge filter)
    num_filters = num_angles
    
    # Smooth vertical edge set to correspond with an electron ring
    initial_edge = [[0., -1., 1., 0., 0.],
                  [0., -1., 1., 0., 0.],
                  [0., -1., 1., 0., 0.],
                  [0., -1., 1., 0., 0.],
                  [0., -1., 1., 0., 0.]]
    initial_line = [[-0.25, -0.25, 1., -0.25, -0.25],
                  [-0.25, -0.25, 1., -0.25, -0.25,],
                  [-0.25, -0.25, 1., -0.25, -0.25,],
                  [-0.25, -0.25, 1., -0.25, -0.25,],
                  [-0.25, -0.25, 1., -0.25, -0.25,]]
    
    
    conv1_filters = []
    conv2_filters = []
    base_rotation = 2*math.pi/num_angles # Angle between each edge filter
    
    # Crudely rotate both vertical filters (defined above) to fill each angle with num_angles in a 2*pi rotation
    for angle_num in range(num_angles):
        filter_angle = angle_num * base_rotation
        
        # Setup filter_size x filter_size filter with None in each pixel
        edge_rot = [[None for x_0 in range(filter_size)] for y_0 in range(filter_size)]
        line_rot = [[None for x_0 in range(filter_size)] for y_0 in range(filter_size)]
        
        # Set each pixel weight in vertical filters to closest rotated conterpart
        for i in range(filter_size):
            for j in range(filter_size):
                x_0 = i - 2 # Get pixel x-position relative to centre
                y_0 = j - 2 # Get pixel y-position relative to centre
                
                # Perform 2D rotation transformation on positions
                x_rot = math.cos(filter_angle)*x_0 - math.sin(filter_angle)*y_0
                y_rot = math.sin(filter_angle)*x_0 + math.cos(filter_angle)*y_0
                
                # Find closest pixel index of rotated position
                i_rot = int(round(x_rot)) + 2
                j_rot = int(round(y_rot)) + 2
                if (i_rot in range(filter_size) and j_rot in range(filter_size)):
                    # If pixel index is in range of the filter (i.e. not rotated outside filter) set rotated pixel to value of non-rotated pixel weight
                    edge_rot[j_rot][i_rot] = initial_edge[j][i]
                    line_rot[j_rot][i_rot] = initial_line[j][i]
        
        
        # Search for pixels that were not touched using the above method and set their values to the average of nearby pixel weights.
        for i in range(filter_size):
            for j in range(filter_size):
                # Check if pixel has None value which means it was not set using above method
                if (edge_rot[j][i] == None):
                    num = 0
                    sum = 0.
                    if (j and edge_rot[j-1][i] != None):
                        # Add to average if pixel has j-1 neighbor and neighbor has value
                        num += 1
                        sum += edge_rot[j-1][i]
                    if (i and edge_rot[j][i-1] != None):
                        num += 1
                        sum += edge_rot[j][i-1]
                    if (j < 4 and edge_rot[j+1][i] != None):
                        num += 1
                        sum += edge_rot[j+1][i]
                    if (i < 4 and edge_rot[j][i+1] != None):
                        num += 1
                        sum += edge_rot[j][i+1]
                    # Set the pixel weight to the average of the adjacent weights
                    edge_rot[j][i] = sum / num
                if (line_rot[j][i] == None):
                    num = 0
                    sum = 0.
                    if (j and line_rot[j-1][i] != None):
                        # Add to average if pixel has j-1 neighbor and neighbor has value
                        num += 1
                        sum += line_rot[j-1][i]
                    if (i and line_rot[j][i-1] != None):
                        num += 1
                        sum += line_rot[j][i-1]
                    if (j < 4 and line_rot[j+1][i] != None):
                        num += 1
                        sum += line_rot[j+1][i]
                    if (i < 4 and line_rot[j][i+1] != None):
                        num += 1
                        sum += line_rot[j][i+1]
                    # Set the pixel weight to the average of the adjacent weights
                    line_rot[j][i] = sum / num
        
        # Attach both, complete, filters to the list of filters that will later be returned by this function
        conv1_filters.append(edge_rot)
        conv2_filters.append(line_rot)
        
    # Return complete list of filters with total number that were used
    return conv1_filters, conv2_filters, num_filters
