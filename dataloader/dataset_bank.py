from os import listdir
from os.path import join
from ipdb import set_trace as st
import glob
import sys

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataset_std(root, data_split, tasks):
    input_list = sorted(glob.glob(join(root, 'rgb', data_split, '*.jpg')))
    aif_list = sorted(glob.glob(join(root, 'aif', data_split, '*.jpg')))
    targets_list = []
    for task in tasks:
        targets_list.append(sorted(glob.glob(join(root, task, data_split, '*.png'))))
    # return list(zip(input_list, targets_list))
    return input_list, targets_list, aif_list

def dataset_target_only(root, phase):
    return sorted(glob.glob(join(root, 'depth', phase, '*.png')))


