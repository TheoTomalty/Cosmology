import numpy as np
from scipy import signal
import math

filter_size = 5

def conv2d(x, W, padding='valid'):
    w_k = np.reshape(W, [1, filter_size, filter_size, 1])
    return signal.convolve(x, w_k, padding)

def flatten_image(images):
    sobel_x = np.array([
        [1.,  2., 0.,  -2., -1.],
        [4.,  8., 0.,  -8., -4.],
        [6., 12., 0., -12., -6.],
        [4.,  8., 0.,  -8., -4.],
        [1.,  2., 0.,  -2., -1.]
    ], dtype='float')
    
    smooth_x = np.array([
        [2., 1., 0., -1., -2.],
        [2., 1., 0., -1., -2.],
        [2., 1., 0., -1., -2.],
        [2., 1., 0., -1., -2.],
        [2., 1., 0., -1., -2.]
        ], dtype='float')
    
    flat = np.ones([filter_size, filter_size, 1, 1])
    smooth_y = np.transpose(smooth_x, [1, 0])
    sobel_y = np.transpose(sobel_x, [1, 0])
        
    amount_flat = conv2d(images, flat, padding='same') / np.sum(flat * flat)
    remove_flat = conv2d(amount_flat, flat) / filter_size**2
    
    amount_x = conv2d(images, sobel_x, padding='same') / np.sum(sobel_x * smooth_x)
    remove_x = - conv2d(amount_x, smooth_x) / filter_size**2
    
    amount_y = conv2d(images, sobel_y, padding='same') / np.sum(sobel_y * smooth_y)
    remove_y = - conv2d(amount_y, smooth_y) / filter_size**2
    
    return images[:, 2:-2, 2:-2, :] - remove_flat - remove_x - remove_y

def flatten_image_batch(images):
    print('Flattening...')
    data = None
    
    batch_size = 50
    for i in range(int(math.floor(images.shape[0]/batch_size))):
        new_data = flatten_image(images[i*batch_size:(i + 1)*batch_size])
        
        if data is None:
            data = new_data
        else:
            data = np.concatenate((data, new_data))
            
        print('Finished ' + str((i+1)*batch_size))
    
    return 1000.0 * data
