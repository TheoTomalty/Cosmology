import tensorflow as tf
import CSheader as h
import os
import json

FLAGS = h.FLAGS

#TODO: Normalize filters
#TODO: Add neighbouring angles in second convolution (useful for curved strings)

class Tracker(object):
    ''' Simple class to keep track of value averages during training.
        Use tracker.add() to input values in each batch and print_average
        to return the average values, after some number of steps, and
        reset the variables.
    
        >>> tracker = Tracker(["test1", "test2"])
        >>> tracker.add([1.3, 2.4])
        >>> tracker.add([0.7, 0.6])
        >>> tracker.print_average(0)
        0: test1 1.0, test2 1.5
        
        >>> tracker.add([2.0, 3.0])
        >>> tracker.print_average(1)
        1: test1 2.0, test2 3.0
    '''
    
    def __init__(self, var_names):
        ''' Initialize tracker
        
        :param var_names: list of variable names to print with averages
        '''
        
        self.var_names = var_names
        self.num_var = len(var_names)
        self.numerator = [0.]*len(var_names)
        self.denominator = 0
    
    def reset(self):
        self.numerator = [0.]*self.num_var
        self.denominator = 0
    
    def print_average(self, step, reset=True):
        ''' Prints the averages of the variables being tracked.
        
        :param step: Step number in the process to be printed with averages
        :param reset: By default clears numerator and denominator values, set to False to keep values
        '''
        
        assert self.denominator, "Error: division by zero"
        
        string = str(step) + ": "
        for name, num in zip(self.var_names, self.numerator):
            string += name + " " + str(num/float(self.denominator)) + (", " if name != self.var_names[-1] else "")
        
        if reset:
            self.reset()
        print string
    
    def save_output(self, step, directory, tensor):
        info_path = os.path.join(directory, "info" + str(step) + ".txt")
        test_path = os.path.join(directory, "final" + str(step) + ".txt")
        with open(info_path, 'r') as file:
            with open(test_path, 'w') as test_file:
                for line, value in zip(file, tensor):
                    info = json.loads(line[:-1])
                    info["value"] = int(value)
                    print info
                    
                    test_file.write(json.dumps(info) + '\n')
                
    
    def add(self, vars):
        ''' Append values to the moving average
        
        :param vars: List of values corresponding to each variable name in self.var_names
        '''
        
        for var, index in zip(vars, range(100)):
            self.numerator[index] += float(var)
        self.denominator += 1

def network(images):
    ''' Inference Mask that contains the bulk of the CNN network definitions.
        
        Converts images and associated information into the 2-channel classification
        prediction for the network.
        
    :param images: Tensor containing FLAG=.get_batch_size() images of size FLAGS.num_pixels**2
    :return: Network output of the images
    '''
    
    with tf.variable_scope('vars'):
        #Initialize all the Variables in this network
        
        #Initial filters include straight/smooth edges of different angles and a few arbitrary shapes
        initial_filters, num_filters = h.get_initial_filters(FLAGS.num_angles)
        
        #First convolution layer, searches for physical edge in image
        convolution_weights_1 = tf.Variable(
            tf.transpose(
                tf.reshape(
                    tf.constant(initial_filters),
                    [num_filters, FLAGS.filter_size, FLAGS.filter_size, 1]
                ),
                [1, 2, 3, 0]
            )
        )
        
        h_conv1 = h.conv2d(images, convolution_weights_1)
        h_pool1 = tf.nn.avg_pool(h_conv1, ksize=[1, 5, 5, 1], strides=[1, 5, 5, 1], padding='VALID')
        
        # Second convolution layer using same filters as first layer (no new variables)
        unpack = tf.unpack(convolution_weights_1, axis=3)
        padding = [
            tf.pad(
                filter,
                [[0, 0], [0, 0], [filter_num, num_filters - (filter_num + 1)]])
            for filter, filter_num in zip(unpack, range(num_filters))
        ]
        convolution_weights_2 = tf.pack(padding, axis=2)
        
        h_conv2 = h.conv2d(h_pool1, convolution_weights_2)
        h_pool2 = tf.nn.avg_pool(h_conv2, ksize=[1, 5, 5, 1], strides=[1, 5, 5, 1], padding='VALID')
        
        # Compute scalar output
        scalar = tf.reduce_sum(
            tf.reduce_mean(
                tf.abs(h_pool2),
                reduction_indices=[2, 3]
            ), 1
        )
        
        return scalar

def cost(scalar, labels):
    ''' Cost function to be minimized during network training. Expect a bimodal distribution so
        desire to minimize the peak width to peak separation ratio.
    
    :param scalar: Network output from the network() function
    :param label: Tensor of True classifications in True or False format 
    :return: Characterization of bimodal distribution
    '''
    
    # Separate the scalars into individual distributions based on the corresponding labels
    string_distribution = tf.boolean_mask(
        scalar,
        labels
    )
    noise_distribution = tf.boolean_mask(
        scalar,
        tf.logical_not(tf.cast(labels, tf.bool))
    )
    
    # Compute the average and rms of distributions
    string_avg = tf.reduce_mean(string_distribution)
    string_rms = tf.sqrt(tf.reduce_mean(tf.square(
        string_distribution - string_avg
    )))
    
    noise_avg = tf.reduce_mean(noise_distribution)
    noise_rms = tf.sqrt(tf.reduce_mean(tf.square(
        noise_distribution - noise_avg
    )))
    
    separation = tf.abs(string_avg - noise_avg)
    
    return tf.add(tf.divide(string_rms, separation), tf.divide(noise_rms, separation))

def train(cost, saver, global_step):
    ''' Group elements together into an object that will perform training steps when called with sess.run(...)
    
    :param cost: Network losses that are to be minimized, returned by cost() function above
    :param saver: Tensorflow saver object that keeps track of moving averages
    :param global_step: Untrainable Variable Tensor that will be incrimented in each step
    :return: Operation object that can be called with sess.run(training_op) to process an individual training step
    '''
    
    #Exponentially reduce step size with each step to improve network convergence (automatic incrementation of global_step)
    learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate, global_step,
                                               1000, FLAGS.decay_rate_per_thousand)
    
    #Tensorflow operation that will calculate the gradient of the cost function and step network variables appropriately
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
    with tf.control_dependencies([train_step]):
        #Operation that appends network variables to the moving averages
        #The with statement ensures that the train_step is performed first whenever sess.run(training_op) is called
        training_op = tf.group(saver.maintain_averages_op)
    
    return training_op

def prediction(scalar):
    # Converts the Network output tensor from network() to a prediction in 1hot format
    average = tf.reduce_mean(scalar, keep_dims=True)
    
    compare = tf.pack(
        [tf.tile(average, [FLAGS.get_batch_size()]), scalar],
        axis = 1
    )
    
    return tf.cast(tf.argmax(compare,1), tf.bool)

def correct(prediction, label):
    ''' Checking if the networks have made the correct classifications
    
    :param scalar: Network output from the network() function
    :param label: Tensor of True classifications in True or False format
    :return: Tensor of type tf.bool with True if the correct classification was made and False otherwise
    '''
    
    return tf.equal(prediction, label)

def accuracy(correct):
    ''' Calculate the accuracy of the Network in a sample batch
    
    :param correct: Tensor with Truth when the algorithm was right, and False when the algorithm was wrong
    :return: Tensor value of type tf.float representing (#correct predictions)/(#predictions)
    '''
        
    return tf.reduce_mean(tf.cast(correct, tf.float32))

