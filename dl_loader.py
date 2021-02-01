import numpy as np
import cv2
import random
import os

def shuffle_pairwise(np_array_1, np_array_2):
    # shuffles two numpy arrays or lists along their first axis in place the same way
    zipped_list = list(zip(np_array_1, np_array_2))
    random.shuffle(zipped_list)
    return zip(*zipped_list)

def get_image_paths_from_dir(dir):
    paths_to_return = set()
    paths = os.listdir(dir)
    paths.sort()  # note, this sorts paths by python's string comparator, which is different from linux filename sorted order
    for path in paths:
        if path.endswith('.png') or path.endswith('.jpg'):
            paths_to_return.add(dir + '/' + path)
    return paths_to_return

def load_and_preprocess_image_list(image_list, target_size):
    # expects image_list to be a list of absolute or relative filepaths to image files
    images_to_return = []
    for path in image_list:
        image = cv2.imread(path)[:,:,::-1]  # load and BGR to RGB
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC).astype(np.float32)  # resize and convert uint8 to float
        image = image / 255  # convert from [0,255] to [0,1]
        images_to_return.append(image)
    return np.array(images_to_return)  # new axis is 0, returned value is of shape [b, h, w, 3]

if __name__ == '__main__':
    test_paths = get_image_paths_from_dir('./train/doom')
    test_images = load_and_preprocess_image_list(test_paths, (68, 68))
    print(test_images.shape)