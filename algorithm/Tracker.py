import os
import json

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
    
    def save_output(self, step, directory, *args):
        info_path = os.path.join(directory, "info" + str(step) + ".txt")
        test_path = os.path.join(directory, "final" + str(step) + ".txt")
        assert len(args) == len(self.var_names)
        print step
        
        with open(info_path, 'r') as file:
            with open(test_path, 'w') as test_file:
                for it in zip(file, *args):
                    line, value = it[0], it[1:]
                    info = json.loads(line[:-1])
                    for name, val in zip(self.var_names, value):
                        info[name] = val.tolist()
                    
                    test_file.write(json.dumps(info) + '\n')
                
    
    def add(self, vars):
        ''' Append values to the moving average
        
        :param vars: List of values corresponding to each variable name in self.var_names
        '''
        
        for var, index in zip(vars, range(100)):
            self.numerator[index] += float(var)
        self.denominator += 1
