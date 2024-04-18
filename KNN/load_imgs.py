import numpy as np
from PIL import Image, ImageOps
import os


def memory_size(obj: np.ndarray):
    return obj.nbytes

def batch_img_generator(data_dir=None, data_paths = None,  memory_limit=None, skip_files=[], shape=(500, 500), grayscale: bool = False):

    """ 
        The batch_img_generator function allows to partially/fully load the dataset in memmory, depending on ram availability.
    """ 

    batch: list = []
    total_memory = 0
    paths = sorted(data_paths) if data_paths else sorted(os.listdir(data_dir))
    for filename in paths:
        
        if (filename.endswith(".jpg") or filename.endswith(".png")) and filename not in skip_files: 

            img: Image = Image.open(os.path.join(data_dir, filename)) if data_dir else Image.open(filename)

            if(grayscale):
                img = ImageOps.grayscale(img) 

            img_resized = img.resize(shape)

            img_arr = np.asarray(img_resized).reshape(1, -1) / 255.0
            img_memory = memory_size(img_arr)
            if total_memory + img_memory > memory_limit:
                for tensor in batch:

                    yield tensor
                batch = []
                total_memory = 0
            batch.append(img_arr)
            total_memory += img_memory

    if batch:  # yield any remaining images
        for tensor in batch:
            yield tensor


def batch_img(data_dir=None, data_paths = None,  skip_files=[], shape=(500, 500), grayscale: bool = False, normalize: bool = False):
    batch: list = []
    filenames: list = []
    paths = sorted(data_paths) if data_paths else sorted(os.listdir(data_dir))
    for filename in paths:
        
        if (filename.endswith(".jpg") or filename.endswith(".png")) and filename not in skip_files: 
            filenames.append(filename)
            img: Image = Image.open(os.path.join(data_dir, filename)) if data_dir else Image.open(filename)
            
            if(grayscale):
                img = ImageOps.grayscale(img) 

            img_resized = img.resize(shape)

            img_arr = np.asarray(img_resized)
            if normalize:
                img_arr /= 255.0

            batch.append(img_arr)
    return np.array(batch), filenames
