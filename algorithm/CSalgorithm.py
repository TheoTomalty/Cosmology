import tensorflow as tf
import numpy as np
import os
import sys
import getopt
import CSheader as h
import CSinput
from Tracker import Tracker
import CSgraph
import math
import json

FLAGS = h.FLAGS

#TODO: Incorporate Tensorboard into training phase

help_message = 'CSalgorithm.py -p <parameter_file> --test' \
               '-i <image_directory> -o <run_directory> -n <num_batches/files/images>' \
               '--continue --tensorboard'

try:
    opts, args = getopt.getopt(sys.argv[1:],"hp:i:o:n:",["continue", "tensorboard", "test"])
except getopt.GetoptError:
    print help_message
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print help_message
        sys.exit()
    elif opt == "-p":
        #Attempt to read a parameter file, defaults used if not found.
        #Should be file with single-line dictionary string, readable
        #by JSON and with keys associated with FLAG names in h.GlobalFlags
        #object.
        try:
            FLAGS.set_parameters(CSinput.ParameterReadWrite(arg).read())
        except:
            print "Parameter file not found, using defaults for run."
            pass
    elif opt == "-i":
        #Input directory that contains numbered image files, ex: "images1.txt"
        if not os.path.exists(arg):
            print "Input Directory not found."
            sys.exit()
        else:
            FLAGS.image_directory = arg
    elif opt == "-o":
        #Run directory where network parameters and statistics are saved
        if not os.path.exists(arg):
            print "Output Directory not found."
            sys.exit()
        else:
            FLAGS.run_directory = arg
    elif opt == "--test":
        # Run testing process rather than training
        FLAGS.training = False
    elif opt == "-n":
        #Number of batches/files/images are run when training/testing/tensorboard-ing
        FLAGS.set_num_iterations(int(arg))
    elif opt == "--continue":
        #Initialize the training session that picks up from most recent save point,
        #or run another testing process without overwriting the last one (must use different regime name)
        FLAGS.continue_session = True
    elif opt == "--tensorboard":
        #Run Network testing with the purpose of getting Tensorboard output (no event information saved)
        FLAGS.print_tensorboard = True

def process():
    ''' Defines the training procedure for the CNN.
    
    :param data_set: Integer indicating which CNN to train, recall there is a separate CNN for each image set
    '''
    
    with tf.Graph().as_default():
        #Initialize global step variable that will be incrimented during training
        global_step = tf.Variable(0, trainable=False)
        
        #Get images and image_labels in random batches of size FLAGS.batch_size
        #These are just Tensor objects for now and will not actually be evaluated until sess.run(...) is called in the loop
        images, labels = CSinput.input(shuffle=FLAGS.training)
        
        #Get output of the CNN with images as input
        scalar = CSgraph.network(images)
        print_conv1, print_conv2 = CSgraph.convolution(images, summary=True)
        
        #Initialize saver object that takes care of reading and writing parameters to checkpoint files
        saver = CSinput.Saver()
        
        #Values and Operations to evaluate in each batch
        cost = CSgraph.cost(scalar, labels)
        correct = CSgraph.correct(CSgraph.prediction(scalar), labels)
        accuracy = CSgraph.accuracy(correct)
        train_op = CSgraph.train(cost, saver, global_step)
        
        #Initialize all the Tensorflow Variables defined in appropriate networks, as well as the Tensorflow session object
        initialize = tf.global_variables_initializer()
        sess = tf.InteractiveSession()
        sess.run(initialize)
        
        #Actually Begin Processing the Graph
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #!!DO NOT CHANGE THE TENSORFLOW GRAPH AFTER CALLING start_queue_runners!!
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        #Load network parameters from the most recent training session if desired
        if FLAGS.continue_session or not FLAGS.training:
            saver.restore(sess)
        
        if FLAGS.training:
            # Training protocol
            tracker = Tracker(["Cost", "Accuracy"])
            
            # Iterate over the desired number of batches
            for batch_num in range(1, FLAGS.num_iterations + 1):
                #Run the training step once and return real-number values for cost and accuracy
                _, cost_value, acc_value, real_conv1, real_conv2 = sess.run([train_op, cost, accuracy, print_conv1, print_conv2])
                #print np.transpose(variable_val, [3, 0, 1, 2])[0]
                
                assert not math.isnan(cost_value), 'Model diverged with cost = NaN'
                tracker.add([cost_value, acc_value])
                
                #Periodically print cost and accuracy values to monitor training process
                if not batch_num % 1:
                    tracker.print_average(batch_num)
                
                #Periodically save moving averages to checkpoint files
                if not batch_num % 50 or batch_num == FLAGS.num_iterations:
                    if FLAGS.print_tensorboard:
                        CSinput.print_tensorboard(sess, [real_conv1, real_conv2])
                    saver.save(sess)
        else:
            #Testing Protocol
            tracker = Tracker(["scalar", "labels"])
            
            for file_num in range(1, FLAGS.num_files + 1):
                #Track and print accuracy for monitoring purposes
                tracker.save_output(file_num, FLAGS.image_directory, *sess.run([scalar, labels]))
        
        #Wrap up
        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    process()
