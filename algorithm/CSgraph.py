import tensorflow as tf
import CSheader as h

FLAGS = h.FLAGS

#TODO: Try only using the same filters on the second convolution, rather than allowing mixing (reduce learnable variables)
#TODO: Investigate possibility that cost function tends towards single bias

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
    
    def add(self, vars):
        ''' Append values to the moving average
        
        :param vars: List of values corresponding to each variable name in self.var_names
        '''
        
        for var, index in zip(vars, range(100)):
            self.numerator[index] += float(var)
        self.denominator += 1

def element(value, index, size):
    return [value if index == j else 0. for j in range(size)]

def inference(images, keep_prob):
    ''' Inference Mask that contains the bulk of the CNN network definitions.
        
        Converts images and associated information into the 2-channel classification
        prediction for the network.
        
    :param images: Tensor containing FLAGS.get_batch_size() images of size FLAGS.num_pixels**2
    :param keep_prob: Probability of dropout for the neurons
    :return: Softmax output of only the images
    '''
    
    with tf.variable_scope('vars'):
        #Initialize all the Variables in this network
        
        #Initial filters include straight/smooth edges of different angles and a few arbitrary shapes
        initial_filters, num_filters = h.get_initial_filters(FLAGS.num_angles, FLAGS.num_zero_filters)
        
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
        #convolution_bias_1 = tf.Variable(
        #    tf.zeros([num_filters])
        #)
        
        h_conv1 = tf.nn.relu(h.conv2d(images, convolution_weights_1))
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 5, 5, 1], strides=[1, 5, 5, 1], padding='VALID')
        
        #Second convolution layer, treat each output map like pixel colours
        weight_array = []
        for filter, filter_num in zip(initial_filters, range(num_filters)):
            extend_filter = [[element(filter[i][j], filter_num, num_filters) for j in range(FLAGS.filter_size)] for i in range(FLAGS.filter_size)]
            weight_array += [extend_filter]
        
        convolution_weights_2 = tf.Variable(
            tf.transpose(weight_array, [1, 2, 3, 0])
        )
        #convolution_bias_2 = tf.Variable(
        #    tf.zeros([num_filters])
        #)
        
        output_data = 4*4*num_filters
        h_conv2 = tf.nn.relu(h.conv2d(h_pool1, convolution_weights_2))
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 5, 5, 1], strides=[1, 5, 5, 1], padding='VALID')
        h_pool_flat = tf.reshape(h_pool2, [-1, output_data])
        
        #Fully connected layer takes input from 5x5 pixel feature maps (after pooling)
        layer_shape = [output_data, FLAGS.num_neurons]
        bias_shape = [FLAGS.num_neurons]
        fully_connected_weights_1 = tf.Variable(tf.ones(layer_shape))
        #fully_connected_bias_1 = tf.Variable(tf.zeros(bias_shape))
        
        h_fc1 = tf.matmul(h_pool_flat, fully_connected_weights_1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        
        ##Second fully connected layer initialized to 1-1 mapping from first layer.
        #fully_connected_weights_2 = tf.Variable(
        #    [
        #        [(1. if neuron2 == neuron1 else (0.)) for neuron2 in range(FLAGS.num_neurons)]
        #        for neuron1 in range(FLAGS.num_neurons)
        #    ]
        #)
        #fully_connected_bias_2 = tf.Variable(tf.zeros(bias_shape))
        #
        #h_fc2 = tf.matmul(h_fc1_drop, fully_connected_weights_2) + fully_connected_bias_2
        #h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
        
        #Output layer initialized so that alternating neurons map to electron output and muon output.
        full_connected_output_weights = tf.Variable(
            [[0., 0.]]*FLAGS.num_neurons
        )
        
        return tf.nn.softmax(tf.matmul(h_fc1_drop, full_connected_output_weights))

def cost(logits, labels):
    ''' Cost function to be minimized during network training.
    
    :param logits: Softmax output from inference() function.
    :param labels: Tensor containing [1, 0] label for electron and [0, 1] label for muons 
    :return: Simple squared L2 norm of the 2D vector difference between logits and labels
    '''
    
    return tf.reduce_mean(tf.reduce_sum((labels - logits)*(labels - logits), reduction_indices=[1]))

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

def correct(logits, labels):
    ''' Checking if the networks have made the correct classifications
    
    :param logits: Tensor of Network predictions from inference(...)
    :param labels: Tensor of True classifications in [1, 0] or [0, 1] format
    :return: Tensor of type tf.bool with True if the correct classification was made and False otherwise
    '''
    
    return tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))

def accuracy(logits, labels):
    ''' Calculate the accuracy of the Network in a sample batch
    
    :param logits: Tensor of Network predictions from inference(...)
    :param labels: Tensor of True classifications in [1, 0] or [0, 1] format
    :return: Tensor value of type tf.float representing (#correct predictions)/(#predictions)
    '''
        
    return tf.reduce_mean(tf.cast(correct(logits, labels), tf.float32))

