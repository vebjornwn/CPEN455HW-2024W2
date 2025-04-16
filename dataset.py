import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize
from bidict import bidict
from tqdm import tqdm
import pandas as pd
import pdb

rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5
replicate_color_channel = lambda x : x.repeat(3,1,1)

my_bidict = bidict({'Class0': 0, 
                    'Class1': 1,
                    'Class2': 2,
                    'Class3': 3})

class CPEN455Dataset(Dataset):
    def __init__(self, root_dir='./data', mode='train', transform=None):
        """
        Args:
            root_dir (str): Directory with all the images and CSV files.
            mode (str): One of 'train', 'validation', or 'test'. The CSV filename will
                        be "<mode>.csv".
            transform (callable, optional): Optional transform to be applied on an image sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Read the CSV file – now expecting three columns: csv_index, path, label.
        csv_path = os.path.join(self.root_dir, mode + '.csv')
        # Specify column names; adjust these if your CSV has headers already.
        import pandas as pd
        df = pd.read_csv(csv_path, header=None, names=['csv_index', 'path', 'label'])
        
        # Convert DataFrame rows into a list of tuples: (csv_index, path, label)
        # Prepend root_dir to the image path if needed
        self.samples = list(df.itertuples(index=False, name=None))
        self.samples = [
            (csv_index, os.path.join(self.root_dir, path), label)
            for csv_index, path, label in self.samples
        ]
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Unpack the tuple: csv_index from the CSV, image file path, and category.
        csv_index, img_path, category = self.samples[idx]
        # If the category is one of the known values, map it to its string name.
        if category in my_bidict.values():
            category_name = my_bidict.inverse[category]
        else:
            category_name = "Unknown"
            
        # Read the image (as a tensor) and normalize
        image = read_image(img_path)  # returns a tensor of shape (C, H, W)
        image = image.float() / 255.0  # Normalize to [0, 1]
        if image.shape[0] == 1:
            image = replicate_color_channel(image)
        if self.transform:
            image = self.transform(image)
        
        # Return image, category string, and the original CSV index.
        return image, category_name, csv_index

if __name__ == '__main__':
    
    transform_32 = Compose([
        Resize((32, 32)),  # Resize images to 32 * 32
        rescaling
    ])
    dataset_list = ['train', 'validation', 'test']
    
    for mode in dataset_list:
        print(f"Mode: {mode}")
        dataset = CPEN455Dataset(root_dir='./data', transform=transform_32, mode=mode)
        data_loader = DataLoader(dataset, batch_size = 4, shuffle=True)
        # Sample from the DataLoader
        for images, categories in tqdm(data_loader):
            print(images.shape, categories)
            images = torch.round(rescaling_inv(images) * 255).type(torch.uint8)
            show_images(images, categories, mode)
            break  # We only want to see one batch of 4 images in this example
        