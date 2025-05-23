import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class make_dataset():
    def __init__(self, opt):
        self.opt = opt
        
    def main(self):
        self.load_all_images()
        self.to_tensor()
        return self.train_dl, self.test_dl, self.valid_dl
    
    def load_all_images(self):
        self.train_x = self.load_image_array(self.opt.train_x_img_dir)
        self.train_y = self.load_image_array(self.opt.train_y_img_dir)
        self.test_x  = self.load_image_array(self.opt.test_x_img_dir)
        self.test_y  = self.load_image_array(self.opt.test_y_img_dir)
        self.valid_x = self.load_image_array(self.opt.valid_x_img_dir)
        self.valid_y = self.load_image_array(self.opt.valid_y_img_dir)

    def load_image_array(self, folder_path, filename_filter=lambda name: name.count('_') != 1):
        image_list = []
        for filename in os.listdir(folder_path):
            if filename_filter(filename):
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path).convert('L')
                img_array = np.array(img)
                image_list.append(img_array)
        return image_list

    def to_tensor(self):
        train_x = torch.tensor(np.array(self.train_x))
        train_y = torch.tensor(np.array(self.train_y))
        test_x = torch.tensor(np.array(self.test_x))
        test_y = torch.tensor(np.array(self.test_y))
        valid_x = torch.tensor(np.array(self.valid_x))
        valid_y = torch.tensor(np.array(self.valid_y))

        train = MyDataset(self.opt, train_x, train_y)
        test = MyDataset(self.opt, test_x, test_y)
        valid = MyDataset(self.opt, valid_x, valid_y)

        self.train_dl = DataLoader(train, self.opt.batch_size, shuffle=True)
        self.test_dl = DataLoader(test, self.opt.batch_size, shuffle=False)
        self.valid_dl = DataLoader(valid, self.opt.batch_size, shuffle=False)

class MyDataset(Dataset):
    def __init__(self, opt, x, y):
        self.opt = opt
        self.x = x.float()
        self.y = y.float()
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x = self.x[idx].unsqueeze(0)  # [1, H, W]
        y = self.y[idx].unsqueeze(0)
        return x.to(self.opt.device), y.to(self.opt.device)