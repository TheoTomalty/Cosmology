from algorithm.preprocessing.io import read_folder, write_folder, open_file
import algorithm.preprocessing.flatten as flatten
import numpy as np
import sys

if len(sys.argv) != 2:
    print(sys.argv[0] + ' <image_directory>')
    sys.exit()

directory = sys.argv[1]

images, labels, info = read_folder(directory)
images = flatten.flatten_image_batch(images).astype(np.int32)

write_folder(directory, images, labels, info)
