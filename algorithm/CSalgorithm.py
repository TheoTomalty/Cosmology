import tensorflow as tf
from preprocessing import io
from DirectoryEmbedded import DirectoryEmbedded
import os
import sys
import getopt
import CSheader as h
import CSinput
from Tracker import Tracker
import CSgraph
import math

FLAGS = h.FLAGS

#TODO: Incorporate Tensorboard into training phase

def process():
    ''' Defines the training procedure for the CNN.
    
    :param data_set: Integer indicating which CNN to train, recall there is a separate CNN for each image set
    '''
    
    with tf.Graph().as_default():
        #Load full set of images to memory from file
        np_images, np_labels, _ = io.open_file(DirectoryEmbedded(FLAGS.image_directory).file('data.h5py'))
        images, labels = tf.constant(np_images, dtype=tf.float32), tf.constant(np_labels, dtype=tf.float32)
        
        #Initialize saver object that takes care of reading and writing parameters to checkpoint files
        global_step = tf.Variable(0, trainable=False)
        saver = CSinput.Saver()
        ix = tf.placeholder(tf.int32)
        train_output = CSgraph.train_output(ix, images, labels, saver, global_step)

        
        #Initialize all the Tensorflow Variables defined in appropriate networks, as well as the Tensorflow session object
        initialize = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(initialize)
        
        #Load network parameters from the most recent training session if desired
        if FLAGS.continue_session or not FLAGS.training:
            saver.restore(sess)
        
        if FLAGS.training:
            # Training protocol
            tracker = Tracker(["Cost", "Accuracy"])
            
            # Iterate over the desired number of batches
            for batch_num in range(FLAGS.num_iterations):
                #Run the training step once and return real-number values for cost and accuracy
                _, cost_value, acc_value = sess.run(train_output, feed_dict={ix: batch_num % int(1000/FLAGS.batch_size)})
                #print np.transpose(variable_val, [3, 0, 1, 2])[0]
                
                assert not math.isnan(cost_value), 'Model diverged with cost = NaN' 
                tracker.add([cost_value, acc_value])
                
                #Periodically print cost and accuracy values to monitor training process
                if not batch_num % 1:
                    tracker.print_average(batch_num)
                
                #Periodically save moving averages to checkpoint files
                #if (not batch_num % 50 and batch_num != 0) or batch_num == FLAGS.num_iterations:
                #    saver.save(sess)


if __name__ == "__main__":
    help_message = 'CSalgorithm.py -p <parameter_file> --test' \
                   '-i <image_directory> -o <run_directory> -n <num_batches/files/images>' \
                   '--continue --tensorboard'
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hp:i:o:n:",["continue", "tensorboard", "test"])
    except getopt.GetoptError:
        print(help_message)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_message)
            sys.exit()
        elif opt == "-p":
            #Attempt to read a parameter file, defaults used if not found.
            #Should be file with single-line dictionary string, readable
            #by JSON and with keys associated with FLAG names in h.GlobalFlags
            #object.
            try:
                FLAGS.set_parameters(CSinput.ParameterReadWrite(arg).read())
            except:
                print("Parameter file not found, using defaults for run.")
                pass
        elif opt == "-i":
            #Input directory that contains numbered image files, ex: "images1.txt"
            if not os.path.exists(arg):
                print("Input Directory not found.")
                sys.exit()
            else:
                FLAGS.image_directory = arg
        elif opt == "-o":
            #Run directory where network parameters and statistics are saved
            if not os.path.exists(arg):
                print("Output Directory not found.")
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
    
    process()
