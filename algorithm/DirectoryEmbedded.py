import os

class DirectoryEmbedded(object):
    ''' Simple class that facilitates working within a (possibly previously non-existent) directory. '''
    def __init__(self, directory):
        ''' Initialize the object with the desired directory.
        
        :param directory: Directory to work in. If non-existent it will be created.
        '''
        
        self.directory = directory
        self.update()
        
    def is_current(self, directory):
        #Check if the directory is an empty string, meaning you are working in the directory from which python was run.
        return directory == ""
    
    def update(self, directory=None):
        ''' Set a given directory as the one used by class, create it if non-existent. 
        
        :param directory: Directory to initialize, if none is given initialize the directory already in use by class.
        '''
        
        #If no directory is given, use the one already saved in class.
        if directory is None:
            directory = self.directory
        #Check if directory needs to be created
        if not self.is_current(directory) and not os.path.exists(directory):
            #Create directory
            os.makedirs(directory)
            #Check that it was created successfully
            assert os.path.exists(directory), "Failed to create directory: " + directory
        #Save the directory name in class attribute "directory"
        self.directory = directory
    
    def file(self, file_name):
        ''' Path generation for files in the initialized directory.
        
        :param file_name: Name of file without path information
        :return: Path of hypothetical file of the name file_name in the directory self.directory
        '''
        
        if self.is_current(self.directory):
            return file_name
        return os.path.join(self.directory, file_name)

def get_files(directory, name, extension="txt"):
    '''
    Easy Generation of the list of files to use in the algorithm.
    Assumes format where the FLAGS.image_directory containing 
    files called "name1.txt", "name2.txt" etc.
    
    :param name: A string representing the prefix of the files in a given directory (ex. 'info' for "info1.txt", "info2.txt", etc.)
    :param extension: Optional extension if not a plain text file
    :return: The list of files with that name prefix
    '''
    
    file_names = []
    file_num = 1
    while file_num:
        #The path of the $name$file_num.txt file in the image directory
        file_name = os.path.join(directory, name + str(file_num) + '.' + extension)
        #Add the file to filenames if it exists, break the loop if not since there will be no more files of this type
        if os.path.isfile(file_name):
            file_names.append(file_name)
            file_num += 1
        else:
            # End of Loop
            file_num = 0
    
    return file_names
