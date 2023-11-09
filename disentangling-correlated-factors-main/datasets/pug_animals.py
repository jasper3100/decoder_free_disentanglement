import os
import subprocess
import tarfile

import numpy as np
import pandas as pd
import torchvision
import tqdm
from PIL import Image

from fastargs.decorators import param
import datasets
import datasets.base

class PUG_Animals(datasets.base.DisentangledDataset):
    """Loading the PUG_Animals Dataset from [1].
    
    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] PUG: Photorealistic and Semantically Controllable
    Synthetic Data for Representation Learning (https://arxiv.org/pdf/2308.03977.pdf)
    """

    # We define the url from which we want to download the data
    urls = {
        "train":
        "https://dl.fbaipublicfiles.com/large_objects/pug/PUG_ANIMAL.tar.gz"
    }

    # We define the file where we want to store the data in a compressed format
    # it is created in the download method
    files = {"train": "PUG_ANIMALS.npz"} 
    
    # We define the latent names, their sizes (number of values they can take) and
    # default values, based on p.19/20 of [1]
    lat_names = (
        'world_name', 
        'character_name', 
        'character_scale', 
        'camera_yaw', 
        'character_texture'
        )
    lat_sizes = np.array([64, 70, 3, 4, 4])   
    lat_values = {
        'world_name': np.arange(64)/64.,
        'character_name': np.arange(70)/70.,
        'character_scale': np.arange(3)/3., 
        'camera_yaw': np.arange(4)/4, 
        'character_texture': np.arange(4)/4.
    }

    img_size = (3, 64, 64) # Originally (3, 512, 512) --> resize to (3, 64, 64)
    # 512 x 512 can be seen manually from any of the images in the dataset
    # 3 is the default number of channels in a color image

    background_color = datasets.COLOUR_WHITE
    # Only matters if the images have transparent components --> not relevant here 

    
    @param('data.root', 'root') 
    def __init__(self, root='data/PUG_ANIMALS/', **kwargs):
        super().__init__(root, [torchvision.transforms.ToTensor()], **kwargs)

        if not os.path.exists(self.train_data):
            self.logger.info('Training data not found. Initiating download...')
            self.download()
        else:
            self.logger.info('Training data found. Loading...')

        # We load the npz (compressed numpy) file of the dataset, which was
        # created in the download method below
        dataset_zip = np.load(self.train_data)
        self.imgs = dataset_zip['images'] # images is the key of the npz file
        assert self.imgs.shape == (215040, 64, 64, 3), "self.imgs.shape is wrong"

        # For PUG_Animals, we have all combinations of latent values (as in MPI3D f.e.)
        lat_values = []
        for wrl in self.lat_values['world_name']:
            for cha in self.lat_values['character_name']:
                for scl in self.lat_values['character_scale']:
                    for yaw in self.lat_values['camera_yaw']:
                        for tex in self.lat_values['character_texture']:
                            lat_values.append([wrl, cha, scl, yaw, tex])
        self.lat_values = np.array(lat_values)

        # If self.subset < 1, we select a random subset of images (whose size corresponds to the
        # fraction specified by self.subset) and their latent values
        if self.subset < 1:
            n_samples = int(len(self.imgs) * self.subset)
            subset = np.random.choice(len(self.imgs), n_samples, replace=False)
            self.imgs = self.imgs[subset]
            self.lat_values = self.lat_values[subset]

    def download(self):
        """Download the dataset."""
        self.logger.info('Loading PUG_ANIMALS (~78GB) - this might take several hours...')
        
        # Path which contains the data
        save_path = os.path.join(self.root, 'PUG_ANIMAL.tar.gz')
        # Path where to store the data
        aug_root_dir = os.path.join(self.root, 'PUG_Animal')

        if not(os.path.exists(self.root) and os.path.isdir(self.root)):
            os.makedirs(self.root)
            
        # If the file or data already exists, we don't download it again
        if os.path.exists(save_path) or os.path.exists(aug_root_dir):
            self.logger.info("PUG_ANIMAL.tar.gz or data already exists. Skipping download.")
        else:
            self.logger.info("Downloading PUG_ANIMAL.tar.gz...")
            # We download file from the url and save it in save_path
            subprocess.check_call([
                "curl", "-L",
                type(self).urls["train"], "--output", save_path
            ])
        
        # If the data already exists, we skip the extraction
        if os.path.exists(aug_root_dir):
            self.logger.info("PUG_ANIMAL already exists. Skipping extraction.")
        else:
            # We extract the file
            with tarfile.open(save_path) as file: 
                self.logger.info("Extracting PUG_ANIMALS data...")
                file.extractall(self.root)
        
        # Once the data is extracted or exists already, we don't need the tar.gz file
        # anymore and remove it
        if os.path.exists(save_path):
            os.remove(save_path)
        
        # We collect all images and resize them to reduce memory usage. Otherwise one might 
        # get an out-of-memory error, trying to stack the images into a numpy array.
        # We only do this if the file doesn't exist yet.
        file_path = os.path.join(self.root, self.files['train'])

        if os.path.exists(file_path):
            self.logger.info("Compressed and resized images already exist. Skipping resizing.")
        else:
            self.logger.info("Resizing and compressing images...")
            images = []
            # Traverse the directory and its subdirectories, filtering only for directories
            for root, dirs, _ in os.walk(aug_root_dir):
                # Iterate through the subdirectories in the current directory
                for subdir in dirs:
                    # Construct the full path of the subdirectory
                    subdir_path = os.path.join(root, subdir)

                    # Iterate through files in the subdirectory
                    for file in os.listdir(subdir_path):
                        # Check if the file is a PNG image
                        if file.lower().endswith('.png'):
                            # Construct the full file path
                            image_file_path = os.path.join(subdir_path, file)

                            # Load the image using PIL (Python Imaging Library) and resize it
                            image = Image.open(image_file_path)
                            image = np.array(image)
                            image = torchvision.transforms.functional.to_pil_image(image)
                            image = torchvision.transforms.functional.resize(image, [64, 64])
                            image = np.array(image)
                            if image.ndim == 2:
                                image = image[..., None]
                            images.append(image)

        # We convert the list to a numpy array
        res_images = np.stack(images)
        assert res_images.shape == (215040, 64, 64, 3), "res_images.shape is wrong"
        self.logger.info("Successfully resized images. Saving compressed images...")

        np.savez_compressed(file_path, images=res_images)

 
    def __getitem__(self, idx):
        """Get the image of `idx`
        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        lat_value : np.array
            Array of length 6, that gives the value of each factor of variation.
        """
        # ToTensor transforms numpy.ndarray (H x W x C) in the range
        # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]

        return self.transforms(self.imgs[idx]), self.lat_values[idx]