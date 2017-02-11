import tensorflow as tf
import CSheader as h
import os
import json
from shutil import move
from os import remove
from DirectoryEmbedded import *

FLAGS = h.FLAGS

class Saver(DirectoryEmbedded):
    ''' Custom class that is responsible for saving and loading network parameters in/from checkpoint files. '''
    
    def __init__(self):
        #Initialize the class in a 'checkpoints' directory within the run_directory set by user
        DirectoryEmbedded.__init__(self, os.path.join(FLAGS.run_directory, 'checkpoints'))
        
        #Tensorflow Moving Averages object that is responcible for keeping track of averages during training
        self.ema = tf.train.ExponentialMovingAverage(decay=FLAGS.average_decay_rate)
        #Tensorflow operation to be called whenever you want the averages to be updated
        self.maintain_averages_op = self.ema.apply(tf.trainable_variables())
        
        #Saver objects of type tf.train.Saver for saving and loading to checkpoint
        self.average_variables = None
        self.graph_variables = None
        #Initialize the saver objects above
        self.set_variables()
    
    @property
    def checkpoint_name(self):
        #Get the name of the checkpoint file
        return os.path.join(self.directory, 'network')
    
    def set_variables(self):
        #Get the list of pointers to trainable variables in Tensorflow
        variables_to_restore = self.ema.variables_to_restore()
        
        #Save dictionaries between the variable name and the variable pointer (moving averages and regular variables resp.)
        average_dict = dict()
        graph_dict = dict()
        
        #Iterate over the list of trainable variables in tensorfow, and add each to the corresponding dictionaries
        for key in variables_to_restore:
            #Check that the variable is custom and is tracked my moving average
            if key.startswith('vars') and self.ema.average(variables_to_restore[key]) is not None:
                #Save the variable name in the average dictionary with the pointer to the moving average
                average_dict[key] = self.ema.average(variables_to_restore[key])
                #Save the variable name in the graph dictionary with the pointer to the graph variable
                graph_dict[key] = variables_to_restore[key]
                
        #Initialize saver objects using each of the dictionaries that will be responsible for saving and loading 
        # Tensorflow objects to/from values of the same name in checkpoint files.
        self.average_variables = tf.train.Saver(average_dict) if len(average_dict.keys()) else None
        self.graph_variables = tf.train.Saver(graph_dict) if len(graph_dict.keys()) else None

    def restore(self, session):
        ''' Restore the network variables into the current session.
        
        :param session: The current session to initialize the network in.
        '''
        
        #Set the moving averages and the graph variables in current session to the values saved in the checkpoint file
        self.average_variables.restore(session, self.checkpoint_name)
        self.graph_variables.restore(session, self.checkpoint_name)
    
    def save(self, session):
        #Save the values in the current session moving averages to the corresponding checkpoint file
        self.average_variables.save(session, self.checkpoint_name)

class ParameterReadWrite(object):
    def __init__(self, parameter_file=""):
        ''' Simple class to read single-line dictionaries from a file, typically for parameter loading.
        
        :param parameter_file: The name of the file to read.
        '''
        
        self.parameter_file = parameter_file
    
    def read(self):
        # Parse the first line of the given file with JSON and return the python dictionary output
        # Return empty dict if no file is given
        if self.parameter_file == "":
            return {}
        assert os.path.exists(self.parameter_file), "Parameter file given does not exist:" + self.parameter_file
        with open(self.parameter_file, "r") as f:
            for line in f:
                # Return the JSON-parsed line (removing the newline character if necessary)
                if line[-1] == "\n":
                    print "Warning: may be multiple dictionaries in \"" + self.parameter_file + "\", whereas only the first is loaded."
                    return json.loads(line[:-1])
                return json.loads(line)
    
    def write(self, dictionary):
        # Write a dictionary to the parameter_file
        with open(self.parameter_file, "w") as f:
            f.write(json.dumps(dictionary))

# Reading file from queue
def read_file(filename_queue):
    '''
    Inform Tensorflow of files to be read. Pack the data in rows into appropriate tensors
    
    :param filename_queue: tf.train.string_input_producer object with files that will be read
    :return: Tensors of the relevant information for running the network (files won't be read until runtime)
    '''
    
    #Initialize the reader object and read a single line from the queue
    reader = tf.TextLineReader()
    _, line = reader.read(filename_queue)
    #Each time the line is evaluated in a session at runtime, the reader progresses to the next line.
    
    # Default values, in case of empty columns. Also specifies the type of the decoded result.
    data_defaults = []
    # Each line should have one input from each pixel in the 30x30 image as well as
    # two numbers, [1, 0] or [0, 1], describing the true particle type
    for i in range(FLAGS.data_size + 1):
      data_defaults.append([0.0])
    
    # Convert a single line, in cvs format, to a list of tf.float tensors with the same size as data_defaults
    data_row = tf.decode_csv(line, record_defaults=data_defaults)
    # Pack pixels tensors together as a single image (and normalize the values)
    datum = tf.pack(data_row[:FLAGS.data_size])
    
    # Pack last two tensors into particle identification (1hot)
    label = tf.pack(data_row[FLAGS.data_size])
    
    #Return the distinct tensors associated with a single-line read of a file
    return datum, label

def input_pipeline(files, size, num_epochs=None, shuffle=True):
    ''' 
    Handles the reading of lines in text files into appropriate *batch* tensors.
    
    :param files: Python list of file names to be read.
    :param size: The size of the batch produced
    :param num_epochs: --
    :param shuffle: Whether or not to randomize the lines and files read. Otherwise reads in order of File > Line
    :return: Input information (Pixels, labels resp.) organized in batches of size $size.
    '''
    
    #Create the queue for the files for proper handling by Tensorflow
    filename_queue = tf.train.string_input_producer(
          files, num_epochs=num_epochs, shuffle=shuffle)
    #Initialize tensors associated with a single line-read of the file queue
    datum, label = read_file(filename_queue)
    #Large capacity for better shuffling
    capacity = FLAGS.min_after_dequeue + 3 * size
    #Send the single-line read operation into a Tensorflow batch object that calls it $size times in succession (with possible shuffing)
    if shuffle:
        #Generate a batch with randomized line and file numbers for each entry
        pixel_batch, label_batch = tf.train.shuffle_batch(
            [datum, label], batch_size=size, capacity=capacity,
            min_after_dequeue=FLAGS.min_after_dequeue)
    else:
        #Generate a batch with lines and files read in order (File > Line).
        pixel_batch, label_batch = tf.train.batch(
            [datum, label], batch_size=size)
    #Each batch object is a tensor of shape [$size, ...] where ... represents the shape of the objects it contains (ex. [$size, 2] for labels)
    return pixel_batch, label_batch

def input(shuffle=True):
    # Handles all the information input for the network training and testing
    file_names = get_files(FLAGS.image_directory, 'images')
    assert len(file_names), "Error: No files listed in your queue"
    
    # Get the input batches
    pipeline =  input_pipeline(file_names, FLAGS.get_batch_size(), shuffle=shuffle)
    
    # Reshape the pixels tensor into a square image, '-1 ' indicates that this dimension can be any size (to match the size of the batch)
    # while there is a fourth dimension with a length of '1' to indicate that we are dealing with a black-and-white image rather than a
    # 3-channel colour image.
    images = tf.reshape(pipeline[0], [-1, FLAGS.num_pixels, FLAGS.num_pixels, 1])
    
    return images, tf.cast(pipeline[1], tf.bool)

def get_summary_filter(labels, correct, show_worked=True, show_strings=True, show_unstringed=True):
    '''
    Reduce a batch of images to a list of images that will be added to the Tensorboard visual output.
    Filter the images based on string existence and whether or not the algorithm worked on it.
    Custom filter also available (using more detailed information) in CSalgorithm.py
    
    :param labels: The tensor describing the particle type of the image in images
    :param correct: A tensor of type tf.bool that indicates if the algorithm worked on a particular image in images
    :return: The boolean-mask type filter to be used on the corresponding batch of images 
    '''
    
    #Ensure that all the tensor sizes match up
    size = int(correct.get_shape()[0])
    assert int(labels.get_shape()[0]) == size
    
    if show_worked:
        #Show images where the algorithm worked *in addition* to the ones that didn't (i.e. do not apply a cut here)
        right_worked = tf.constant(True, dtype=tf.bool, shape=[size])
    else:
        #Contruct a boolean mask that only shows images where the algorithm failed.
        right_worked = tf.logical_not(correct)
    
    #Construct a boolean mask that shows electrons if $FLAGS.show_electrons and shows muons if $FLAGS.show_muons
    show_strings = tf.tile(tf.constant([[show_strings, show_unstringed]], dtype=tf.bool), [size, 1])
    good_strings = tf.logical_and(show_strings, tf.cast(labels, tf.bool))
    right_strings = tf.logical_or(tf.reshape(tf.slice(good_strings, [0,0], [size, 1]), [size]), tf.reshape(tf.slice(good_strings, [0,1], [size, 1]), [size]))
    
    #Group the filters together. Final mask only shows images that passed each one.
    show = tf.logical_and(right_worked, right_strings)
    return show

def mask(images, show):
    return tf.boolean_mask(images, show)

def get_summary(images):
    #Construct the summary object that sends the images tensor to Tensorboard for display (displays a maximum of $max_images images)
    return tf.summary.image("data", images, max_outputs=20)
    
def write(session, summary):
    # Function to save images in an image summary (from get_summary) to Tensorboard log files
    logdir = '/tmp/logs' # Standard directory to save the display images in Tensorboard format
    
    # Remove all of the tfevent files from previous saves to stop image mixing
    file_list = [f for f in os.listdir(logdir)]
    for f in file_list:
        # Ensure that you are only removing files that are outputted by Tensorflow (in case there are unrelated files in $logdir)
        if f.startswith("events.out.tfevents."):
            remove(os.path.join(logdir, f))
        else:
            print "Unexpected file in logdir: " + f
    # Write summary object
    writer = tf.summary.FileWriter(logdir, session.graph)
    writer.add_summary(session.run(summary), 0)
