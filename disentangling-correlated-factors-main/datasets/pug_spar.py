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

class PUG_Spar(datasets.base.DisentangledDataset):
    """Loading the PUG_Spar Dataset from [1].
    
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
        "https://dl.fbaipublicfiles.com/large_objects/pug/PUG_SPAR.tar.gz"
    }

    # We define the file where we want to store the data in a compressed format
    # this file is created in the download method
    files = {"train": "PUG_SPAR_128.npz"} 

    # We define the latent names, their sizes (number of values they can take) and
    # default values, based on the labels.csv file of the dataset (and p. 22/23 of [1])
    lat_names = (
        'world_name', 
        'character_name', 
        'character2_name', 
        'character1_pos', 
        'character2_pos', 
        'character_texture', 
        'character2_texture'
        )
    # - 10 worlds/backgrounds
    # - 32 animals but "blank" (no animal) is possible too --> 33
    # - character1_pos: below, left --> 2
    # - character2_pos: above, right --> 2
    # - character_texture: blue, default, grass --> 3
    # - character2_texture: default, red, stone --> 3
    lat_sizes = np.array([10, 33, 33, 2, 2, 3, 3])   
    lat_values = {
        'world_name': np.arange(10)/10.,
        'character_name': np.arange(33)/33.,
        'character2_name': np.arange(33)/33., 
        'character1_pos': np.arange(2)/2, 
        'character2_pos': np.arange(2)/2,
        'character_texture': np.arange(3)/3.,
        'character2_texture': np.arange(3)/3.
    }

    img_size = (3, 128, 128) # Originally (3, 512, 512) --> resize to (3, 128, 128)
    # 512 x 512 can be seen manually from any of the images in the dataset
    # 3 is the default number of channels in a color image

    background_color = datasets.COLOUR_WHITE 
    # Only matters if the images have transparent components --> not relevant here 

    
    @param('data.root', 'root')
    def __init__(self, root='data/PUG_SPAR/', **kwargs):
        super().__init__(root, [torchvision.transforms.ToTensor()], **kwargs)

        if not os.path.exists(self.train_data):
            self.logger.info('Training data not found. Initiating download...')
            self.download()
        else:
            self.logger.info('Training data found. Loading...')

        # We load the npz (compressed numpy) file of the dataset, which was
        # created in the download method below
        dataset_zip = np.load(self.train_data)
        self.imgs = dataset_zip['images'] 
        assert self.imgs.shape == (43560, 128, 128, 3), "self.imgs.shape is wrong"

        # We get the latent values manually (since they are not part of the data) from the
        # labels.csv file which comes with the data. Basically we just delete the first column 
        # (filename of each image) and then we're left with the latent values
        # We can't just take all combinations of latent values because PUG_SPAR doesn't contain
        # all combinations of latents (as it is done for mpi3d for example)
        df = pd.read_csv(os.path.join(self.root, 'PUG_SPAR/labels.csv'))
        first_column = df.columns[0]
        df = df.drop([first_column], axis=1)
        
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
        self.logger.info('Loading PUG_SPAR (~16GB) - this might take several hours...')
        
        # Path which contains the data
        save_path = os.path.join(self.root, 'PUG_SPAR.tar.gz')
        # Path where to store the data
        aug_root_dir = os.path.join(self.root, 'PUG_SPAR')

        if not(os.path.exists(self.root) and os.path.isdir(self.root)):
            os.makedirs(self.root)
            
        # If the file or the data already exists, we don't download it again
        if os.path.exists(save_path) or os.path.exists(aug_root_dir):
            self.logger.info("PUG_SPAR.tar.gz or data already exists. Skipping download.")
        else:
            self.logger.info("Downloading PUG_SPAR.tar.gz...")
            # We download file from the url and save it in save_path
            subprocess.check_call([
                "curl", "-L",
                type(self).urls["train"], "--output", save_path
            ])
        
        # If the data already exists, we skip the extraction
        if os.path.exists(aug_root_dir):
            self.logger.info("PUG_SPAR already exists. Skipping extraction.")
        else:
            # We extract the file
            with tarfile.open(save_path) as file: 
                self.logger.info("Extracting PUG_SPAR data...")
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
                        image = torchvision.transforms.functional.resize(image, [128, 128])
                        image = np.array(image)
                        if image.ndim == 2:
                            image = image[..., None]
                        images.append(image)

            # We convert the list to a numpy array
            res_images = np.stack(images)
            assert res_images.shape == (43560, 128, 128, 3), "res_images.shape is wrong"      
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