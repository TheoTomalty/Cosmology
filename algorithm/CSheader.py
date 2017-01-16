import tensorflow as tf
import math

class GlobalFlags:
    '''
    Class that handles all of the algorithm global variables including:
        1) Run Directives: Which directories to use, train or test or output Tensorboard images, number of batches/files
        2) Tensorboard Directives: details of which images to show in Tensorboard
        3) Fixed Parameters: Unchanging values specifying algorithm hyper-parameters
        4) Variable Parameters: Algorithm hyper-parameters that can be changed according to desired functionality
        
        All Parameters can be set using a single-line JSON string parameter file (inserted to SKalgorithm.py call using the -p flag).
        The dictionary should have keys corresponding to parameter names to be overwritten, and their corresponding desired values.
        It is recommended to only overwrite Variable Parameters this way, and to specify all Directives (Tensorboard or otherwise)
        using the c-style flags when calling CSalgorithm.py (use "python CSalgorithm.py -h" for a list of options).
        
        FIXED PARAMETERS SHOULD NOT BE CHANGED UNLESS TO MODIFY THE ARCHITECTURE OF THE ALGORITHM/DETECTOR/CONE
    '''
    
    def __init__(self):
        # Run Directives
        self.image_directory = "" # Directory path that contains the images numbered
        self.run_directory = "" # Directory path where training checkpoints and testing results are saved
        self.training = True # Indicates whether training or testing
        self.continue_session = False # If Training: Initialise networks with the values from most recent session
                                      # If Testing: Append test results to existing results file (use different regime_name flag for second run to not overwrite!)
        self.__num = None # Set by set_num_iterations. Number of iterations for given task, num batches/files/images for training/testing/Tensorboard resp.
        
        # Tensorboard Directives
        self.print_tensorboard = False # Use with <self.training = 0> to save images into logs that can be seen using Tensorboard
        self.show_strings = False # Save electron images (if print_tensorboard == True)
        self.show_unstringed = False # Save Muon images (if print_tensorboard == True)
        self.show_worked = False # Also Save images that the algorithm correctly predicted (default to only save faulty ones)
        self.custom_cut = False # Cut images according to non-standard information defined in-line (line 249 of SKalgorithm.py within "extra_mask.append(bool(...))")
        
        # Fixed Parameters
        self.num_pixels = 100 # Width of square input images to network, in number of pixels. Image is num_pixels x num_pixels in total.
        self.min_after_dequeue = 200 # Quantity used for shuffling batches. Higher = better randomization but slower processing speed.
        self.filter_size = 5 # Pixel width of filters used for feature identification. To change this you will need to also modify 
                             # the size of the default filters in get_initial_filters. Preferably an odd number.
        
        # Variable Parameters
        self.average_decay_rate = 0.995 # Weight used for moving averages of network variables $avg_{i} = 0.995*avg_{i-1} + (1.0 - 0.995)*var_{i}$
        self.dropout_prob = 1.0 # Neurons in the fully connected layers are randomly deactivated with probability $1.0 - dropout_prob$
        self.batch_size = 50 # Number of images in a single batch (stochastic method of machine learning)
        self.num_angles = 10 # Initial filters include one rigid edge and one smooth edge, num_angles indicates how many angles in a full circle to make edge filters for
        self.num_zero_filters = 0 # Number of initial filters with zero weights in each pixel (can be used to give network more autonomy)
        self.num_neurons = 10 # Number neurons in each fully connected layer (pairs are used to make initial network setup simpler)
        self.initial_learning_rate = 0.0000004 # Step size used by tf.train.AdamOptimizer before decay (proportion of cost function gradient to step trainable variables)
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
        return 2000 
    
    @property
    def data_size(self):
        # Number of input variables to a given network, number of pixels in the square images of rings
        return self.num_pixels**2
    
    @property
    def num_iterations(self):
        # Default number of iterations to do if none is given using self.set_num_iterations
        if self.__num is not None:
            return self.__num
        if self.training:
            # Default number of batches if training
            return 200
        if not self.print_tensorboard:
            # Default number of files to test on if testing
            return 100
         # Default number of images to save if writing Tensorboard logs
        return 50
      
    def set_num_iterations(self, num_iterations):
        # num_iterations setter function
        self.__num = num_iterations

######################################################
#
# Create Instance of the GlobalFlags class:
#   - Call instance in other files with SKheader.FLAGS
#   - Called by reference so all changes are global
#
#######################
FLAGS = GlobalFlags()##
######################################################


def conv2d(x, W):
    # Convolute a 2D image, x, with a set of filters, W, using one-pixel steps
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def get_initial_filters(num_angles, num_zero_filters):
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
    num_filters = num_angles + num_zero_filters
    
    # Smooth vertical edge set to correspond with an electron ring
    initial_edge = [[-1., -1., 1., 0.5, 0.5],
                  [-1., -1., 1., 0.5, 0.5],
                  [-1., -1., 1., 0.5, 0.5],
                  [-1., -1., 1., 0.5, 0.5],
                  [-1., -1., 1., 0.5, 0.5]]

    
    initial_filters = []
    base_rotation = 2*math.pi/num_angles # Angle between each edge filter
    
    # Crudely rotate both vertical filters (defined above) to fill each angle with num_angles in a 2*pi rotation
    for angle_num in range(num_angles):
        filter_angle = angle_num * base_rotation
        
        # Setup filter_size x filter_size filter with None in each pixel
        filter_rot = [[None for x_0 in range(filter_size)] for y_0 in range(filter_size)]
        
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
                    filter_rot[j_rot][i_rot] = initial_edge[j][i]
        
        # Search for pixels that were not touched using the above method and set their values to the average of nearby pixel weights.
        for i in range(filter_size):
            for j in range(filter_size):
                # Check if pixel has None value which means it was not set using above method
                if (filter_rot[j][i] == None):
                    num = 0
                    sum = 0.
                    if (j and filter_rot[j-1][i] != None):
                        # Add to average if pixel has j-1 neighbor and neighbor has value
                        num += 1
                        sum += filter_rot[j-1][i]
                    if (i and filter_rot[j][i-1] != None):
                        num += 1
                        sum += filter_rot[j][i-1]
                    if (j < 4 and filter_rot[j+1][i] != None):
                        num += 1
                        sum += filter_rot[j+1][i]
                    if (i < 4 and filter_rot[j][i+1] != None):
                        num += 1
                        sum += filter_rot[j][i+1]
                    # Set the pixel weight to the average of the adjacent weights
                    filter_rot[j][i] = sum / num
        
        # Attach both, complete, filters to the list of filters that will later be returned by this function
        initial_filters.append(filter_rot)

        
    # Return complete list of filters with total number that were used
    return initial_filters, num_filters
