import tensorflow as tf
import numpy as np
import CSheader as h
import os
import json
from shutil import move
from os import remove
from DirectoryEmbedded import *
#import CSgraph

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
                    print("Warning: may be multiple dictionaries in \"" + self.parameter_file + "\", whereas only the first is loaded.")
                    return json.loads(line[:-1])
                return json.loads(line)
    
    def write(self, dictionary):
        # Write a dictionary to the parameter_file
        with open(self.parameter_file, "w") as f:
            f.write(json.dumps(dictionary))

def mask(images, show):
    return tf.boolean_mask(images, show)

def print_tensorboard(session, image_packages, package_names):
    summaries = []
    for package, pk_name in zip(image_packages, package_names):
        for image_batch, j in zip(package, range(1000)):
            images = np.expand_dims(image_batch, axis=3)
            name = pk_name + "_image" + str(j + 1)
            summaries.append(get_summary(images, name))
    write(session, summaries)

def get_summary(images, name):
    #Construct the summary object that sends the images tensor to Tensorboard for display (displays a maximum of $max_images images)
    return tf.summary.image(name, images, max_outputs=FLAGS.num_tensorboard)
    
def write(session, summaries):
    # Function to save images in an image summary (from get_summary) to Tensorboard log files
    logdir = '/tmp/logs' # Standard directory to save the display images in Tensorboard format
    
    # Remove all of the tfevent files from previous saves to stop image mixing
    file_list = [f for f in os.listdir(logdir)]
    for f in file_list:
        # Ensure that you are only removing files that are outputted by Tensorflow (in case there are unrelated files in $logdir)
        if f.startswith("events.out.tfevents."):
            remove(os.path.join(logdir, f))
        else:
            print("Unexpected file in logdir: " + f)
    # Write summary object
    writer = tf.summary.FileWriter(logdir, session.graph)
    for summary in summaries:
        writer.add_summary(session.run(summary), 0)
    
