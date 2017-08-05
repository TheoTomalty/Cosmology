import tensorflow as tf
import CSheader as h
import numpy as np

FLAGS = h.FLAGS

def normalize_filters(weights, num_in, num_out, grad=False):
    sobel = tf.constant([
        [1.,  2., 0.,  -2., -1.],
        [4.,  8., 0.,  -8., -4.],
        [6., 12., 0., -12., -6.],
        [4.,  8., 0.,  -8., -4.],
        [1.,  2., 0.,  -2., -1.]
    ])
    
    smooth = tf.constant([
        [2., 1., 0., -1., -2.],
        [2., 1., 0., -1., -2.],
        [2., 1., 0., -1., -2.],
        [2., 1., 0., -1., -2.],
        [2., 1., 0., -1., -2.]
        ])
    
    flat = tf.ones([FLAGS.filter_size, FLAGS.filter_size, num_in, num_out])
    smooth_x = tf.tile(tf.expand_dims(tf.expand_dims(smooth, 2), 3), [1, 1, num_in, num_out])
    smooth_y = tf.transpose(smooth_x, [1, 0, 2, 3])
    sobel_x = tf.tile(tf.expand_dims(tf.expand_dims(sobel, 2), 3), [1, 1, num_in, num_out])
    sobel_y = tf.transpose(sobel_x, [1, 0, 2, 3])
    
    amount_flat =  flat * tf.reduce_sum(flat * weights, axis=[0, 1], keep_dims=True) / tf.reduce_sum(flat * flat, axis=[0, 1], keep_dims=True)
    amount_x = smooth_x * tf.reduce_sum(sobel_x * weights, axis=[0, 1], keep_dims=True) / tf.reduce_sum(sobel_x * smooth_x, axis=[0, 1], keep_dims=True)
    amount_y = smooth_y * tf.reduce_sum(sobel_y * weights, axis=[0, 1], keep_dims=True) / tf.reduce_sum(sobel_y * smooth_y, axis=[0, 1], keep_dims=True)
    
    if grad:
        return weights.assign(weights - amount_flat - amount_x - amount_y)
    return weights.assign(weights - amount_flat)
    

def smoothing(images, n):
    pooling_size = 2*n + 1
    gauss = tf.reshape(
        tf.constant(h.gauss_filter(n)),
        [pooling_size, pooling_size, 1, 1]
    )
    
    unpack = tf.unstack(images, axis=3)
    smooth_image = tf.stack([
        tf.squeeze(h.conv2d(
            tf.expand_dims(image, 3),
            gauss, padding='SAME'
        ), axis=[3])
        for image in unpack], axis=3)
    return smooth_image

def convolution(images, summary=None):
    with tf.variable_scope('vars', reuse=summary):
        #Initial filters include straight edges and straight lines of different angles
        conv1_filters, conv2_filters, num_filters = h.get_initial_filters(FLAGS.num_angles)
        
        #First convolution layer, searches for physical edge in image
        convolution_weights_1 = tf.get_variable(
            "conv1_weights",
            initializer=tf.transpose(
                tf.reshape(
                    tf.constant(conv1_filters),
                    [num_filters, FLAGS.filter_size, FLAGS.filter_size, 1]
                ),
                [1, 2, 3, 0]
            )
        )
        
        # Second convolution layer using same filters as first layer (no new variables)
        reduced_weights = tf.transpose(
                tf.reshape(
                    tf.constant(conv2_filters),
                    [num_filters, FLAGS.filter_size, FLAGS.filter_size, 1]
                ),
                [1, 2, 3, 0]
            )
        unpack = tf.unstack(reduced_weights, axis=3)
        padding = [
            tf.pad(
                filter,
                [[0, 0], [0, 0], [filter_num, num_filters - (filter_num + 1)]]
            )
            for filter, filter_num in zip(unpack, range(num_filters))
        ]
        convolution_weights_2 = tf.get_variable(
            "conv2_weights",
            initializer=tf.stack(padding, axis=2)
        )
        
        renormalization1 = normalize_filters(convolution_weights_1, 1, num_filters)
        renormalization2 = normalize_filters(convolution_weights_2, num_filters, num_filters, grad=False)
        with tf.control_dependencies([renormalization1, renormalization2]):
            h_conv1 = tf.nn.relu(h.conv2d(images, convolution_weights_1))
            h_pool1 = tf.nn.avg_pool(smoothing(h_conv1, 1), ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='VALID')
            
            h_conv2 = tf.nn.relu(h.conv2d(h_pool1, convolution_weights_2))
            h_pool2 = tf.nn.avg_pool(
                tf.reduce_mean(h_conv2,
                    axis=3,
                    keep_dims=True
                ),
                ksize=[1, 5, 5, 1], strides=[1, 5, 5, 1], padding='VALID'
            )
        
        return h_pool2

def network(images):
    ''' Inference Mask that contains the bulk of the CNN network definitions.
        
        Converts images and associated information into the 2-channel classification
        prediction for the network.
        
    :param images: Tensor containing FLAG=.get_batch_size() images of size FLAGS.num_pixels**2
    :return: Network output of the images
    '''
    
    h_pool = convolution(images)
    scalar = tf.reduce_mean(
        h_pool,
        axis=3
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
    string_regions = tf.cast(
        tf.cast(labels, tf.bool),
        tf.float32
    )
    noise_regions = tf.cast(
        tf.logical_not(tf.cast(labels, tf.bool)),
        tf.float32
    )
    
    # Compute the average and rms of distributions
    string_avg = tf.reduce_sum(scalar * string_regions)/tf.reduce_sum(string_regions)
    string_rms = tf.sqrt(
        tf.reduce_sum(tf.square(scalar - string_avg) * string_regions) / tf.reduce_sum(string_regions)
    )
    
    noise_avg = tf.reduce_sum(scalar * noise_regions)/tf.reduce_sum(noise_regions)
    noise_rms = tf.sqrt(
        tf.reduce_sum(tf.square(scalar - noise_avg) * noise_regions) / tf.reduce_sum(noise_regions)
    )
    
    separation = string_avg - noise_avg
    rms = (noise_rms + string_rms)/2
    
    return - separation / rms

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
    
    compare = tf.stack(
        [tf.tile(average, [FLAGS.get_batch_size(), FLAGS.num_regions, FLAGS.num_regions]), scalar],
        axis = 1
    )
    
    return tf.cast(tf.argmax(compare,1), tf.bool)

def correct(prediction, label):
    ''' Checking if the networks have made the correct classifications
    
    :param scalar: Network output from the network() function
    :param label: Tensor of True classifications in True or False format
    :return: Tensor of type tf.bool with True if the correct classification was made and False otherwise
    '''
    
    return tf.equal(prediction, tf.cast(label, tf.bool))

def accuracy(correct):
    ''' Calculate the accuracy of the Network in a sample batch
    
    :param correct: Tensor with Truth when the algorithm was right, and False when the algorithm was wrong
    :return: Tensor value of type tf.float representing (#correct predictions)/(#predictions)
    '''
        
    return tf.reduce_mean(tf.cast(correct, tf.float32))

def train_output(ix, images, labels, saver, global_step):
    #Get output of the CNN with images as input
    CSscalar = network(images[ix*FLAGS.batch_size:(ix+1)*FLAGS.batch_size])
        
    #Values and Operations to evaluate in each batch
    CScost = cost(CSscalar, labels[ix*FLAGS.batch_size:(ix+1)*FLAGS.batch_size])
    CScorrect = correct(prediction(CSscalar), labels[ix*FLAGS.batch_size:(ix+1)*FLAGS.batch_size])
    CSaccuracy = accuracy(CScorrect)
    train_op = train(CScost, saver, global_step)
    
    return train_op, CScost, CSaccuracy
