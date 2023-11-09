import os
import subprocess
import tarfile

import numpy as np
import pandas as pd
import torchvision
import tqdm
from PIL import Image
import sklearn.preprocessing

from fastargs.decorators import param
import datasets
import datasets.base

class PUG_Imagenet(datasets.base.DisentangledDataset):
    """Loading the PUG_Imagenet Dataset from [1].
    
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
        "https://dl.fbaipublicfiles.com/large_objects/pug/PUG_IMAGENET.tar.gz"
    }
    
    # We define the file where we want to store the data in a compressed format
    # it is created in the download method
    files = {"train": "PUG_IMAGENET.npz"} 

    # We define the latent names, their sizes (number of values they can take) and
    # default values, based on p. 20 of [1]. We omit the latent variable 'character_name' 
    # because it can take 724 values, which is a lot and drastically slows down the computation 
    # of the evaluation metric DCI-D. This variable is slightly redundant anyways because there 
    # is a mapping of the 724 objects to the 151 classes ('character_label').
    lat_names = (
        'world_name', 
        'character_label', 
        'character_rotation_yaw', 
        'character_rotation_roll', 
        'character_rotation_pitch', 
        'character_scale', 
        'camera_roll', 
        'camera_pitch', 
        'camera_yaw',
        'character_texture', 
        'scene_light'
        )
    # 'character_rotation_yaw', 'character_rotation_roll', 'character_rotation_pitch', 
    # 'character_scale', 'camera_roll', 'camera_pitch' officially have 6 values but in the dataset
    # they have 7 (due to "Default" as additional value), likewise 'scene_light' has 8 instead of
    # the official 7 values     
    lat_sizes = np.array([64, 151, 7, 7, 7, 7, 7, 7, 6, 9, 8])  
    lat_values = {
        'world_name': np.arange(64)/64.,
        'character_label': np.arange(151)/151.,
        'character_rotation_yaw': np.arange(7)/7.,
        'character_rotation_roll': np.arange(7)/7.,
        'character_rotation_pitch': np.arange(7)/7.,
        'character_scale': np.arange(7)/7.,
        'camera_roll': np.arange(7)/7.,
        'camera_pitch': np.arange(7)/7.,
        'camera_yaw': np.arange(6)/6.,
        'character_texture': np.arange(9)/9.,
        'scene_light': np.arange(8)/8.
    }

    img_size = (3, 64, 64) # Originally (3, 512, 512) --> resize to (3, 64, 64)
    # 512 x 512 can be seen manually from any of the images in the dataset
    # 3 is the default number of channels in a color image
    
    background_color = datasets.COLOUR_WHITE 
    # Only matters if the images have transparent components --> not relevant here 

    
    @param('data.root', 'root') 
    def __init__(self, root='data/PUG_IMAGENET/', **kwargs):
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
        assert self.imgs.shape == (88328, 64, 64, 3), "self.imgs has wrong shape"

        # We get the latent values manually (since they are not part of the data) from the
        # labels.csv file which comes with the data. Basically we just delete the first column 
        # (filename of each image) and then we're left with the latent values
        # We can't just take all combinations of latent values because PUG_SPAR doesn't contain
        # all combinations of latents (as it is done for mpi3d for example)
        df = pd.read_csv(os.path.join(self.root, 'PUG_ImageNet/labels_pug_imagenet.csv'))
        first_column = df.columns[0]
        third_column = df.columns[2] # character_name
        df = df.drop([first_column, third_column], axis=1)
        
        # In order to identify which numeric value corresponds to which latent value, we create a 
        # dictionary mapping latent values to numbers and we then replace the latent values in 
        # the dataframe with the numbers (we can't use the strings directly because later we 
        # want to convert to a Torch tensor which doesn't accept a np.array of type object, i.e., 
        # if we have strings). We first, initialize a dictionary to store mappings for each column
        column_mappings = {}

        # Iterate over each column
        for column in df.columns:
            # Extract unique values from the column and sort them
            unique_values = df[column].unique()
            unique_values.sort()

            # Create a dictionary mapping unique values to [0, 1] interval
            mapping = {value: i / len(unique_values) for i, value in enumerate(unique_values)}
            column_mappings[column] = mapping
            df[column] = df[column].replace(mapping)
        self.lat_values = df.to_numpy()

        # If self.subset < 1, we select a random subset of images (whose size corresponds to the
        # fraction specified by self.subset) and their latent values
        if self.subset < 1:
            n_samples = int(len(self.imgs) * self.subset)
            subset = np.random.choice(len(self.imgs), n_samples, replace=False)
            self.imgs = self.imgs[subset]
            self.lat_values = self.lat_values[subset]

    def download(self):
        """Download the dataset."""
        self.logger.info('Loading PUG_Imagenet (~27GB) - this might take several hours...')
        
        # Path which contains the data
        save_path = os.path.join(self.root, 'PUG_IMAGENET.tar.gz')
        # Path where to store the data
        aug_root_dir = os.path.join(self.root, 'PUG_ImageNet')
        
        if not(os.path.exists(self.root) and os.path.isdir(self.root)):
            os.makedirs(self.root)
            
        # If the file or the data already exists, we don't download it again
        if os.path.exists(save_path) or os.path.exists(aug_root_dir):
            self.logger.info("PUG_IMAGENET.tar.gz or the data already exists. Skipping download.")
        else:
            self.logger.info("Downloading PUG_IMAGENET.tar.gz...")
            # We download file from the url and save it in save_path
            subprocess.check_call([
                "curl", "-L",
                type(self).urls["train"], "--output", save_path
            ])
        
        # If the data already exists, we skip the extraction
        if os.path.exists(aug_root_dir):
            self.logger.info("PUG_ImageNet already exists. Skipping extraction.")
        else:
            # We extract the file
            with tarfile.open(save_path) as file: 
                self.logger.info("Extracting PUG_Imagenet data...")
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
            # Traverse the directory and its subdirectories, filtering for PNG images
            for root, dirs, files in os.walk(aug_root_dir):
                for file in files:
                    # Check if the file is a PNG image
                    if file.lower().endswith('.png'):
                        # Construct the full file path
                        image_file_path = os.path.join(root, file)

                        # Load the image using PIL (Python Imaging Library) and resize it
                        image = Image.open(image_file_path)
                        image = np.array(image)
                        image = torchvision.transforms.functional.to_pil_image(image)
                        image = torchvision.transforms.functional.resize(image, [64, 64])
                        image = np.array(image)
                        if image.ndim == 2:
                            image = image[..., None]
                        images.append(image)

        # we convert the list to a numpy array
        res_images = np.stack(images)
        assert res_images.shape == (88328, 64, 64, 3) 
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