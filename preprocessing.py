
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np


class classify_hr_lr():
    def __init__(self, opt, raw_img_dir, train_x_img_dir, train_y_img_dir, test_x_img_dir, test_y_img_dir, valid_x_img_dir, valid_y_img_dir):
        self.opt = opt
        self.raw_img_dir = raw_img_dir
        self.train_x_img_dir = train_x_img_dir
        self.train_y_img_dir = train_y_img_dir
        self.test_x_img_dir = test_x_img_dir
        self.test_y_img_dir = test_y_img_dir
        self.valid_x_img_dir = valid_x_img_dir
        self.valid_y_img_dir = valid_y_img_dir
        self.hr_data, self.lr_data = [], []
    
    def split_save_data(self): #hr: y, lr: x
        for filename in os.listdir(self.raw_img_dir):
            if filename.count('HR') == 1:
                self.hr_data.append(os.path.join(self.raw_img_dir, filename))
            else:
                self.lr_data.append(os.path.join(self.raw_img_dir, filename))
        
        lr_temp, lr_test, hr_temp, hr_test = train_test_split(self.lr_data, self.hr_data, test_size=0.2, random_state=42)
        lr_train, lr_valid, hr_train, hr_valid = train_test_split(lr_temp, hr_temp, test_size=0.25, random_state=42)
        
        for idx, (train_x, train_y, test_x, test_y, valid_x, valid_y) in enumerate(zip(lr_train, hr_train, lr_test, hr_test, lr_valid, hr_valid)):
            self.save_patch(train_x, self.train_x_img_dir, idx, self.opt.patch_size, self.opt.stride)
            self.save_patch(train_y, self.train_y_img_dir, idx, self.opt.patch_size, self.opt.stride)
            self.save_patch(test_x, self.test_x_img_dir, idx, self.opt.patch_size, self.opt.stride)
            self.save_patch(test_y, self.test_y_img_dir, idx, self.opt.patch_size, self.opt.stride)
            self.save_patch(valid_x, self.valid_x_img_dir, idx, self.opt.patch_size, self.opt.stride)
            self.save_patch(valid_y, self.valid_y_img_dir, idx, self.opt.patch_size, self.opt.stride)
            
    def save_patch(self, data, img_dir, sample_idx, patch_size, stride):
        img = Image.open(data)
        if img_dir.count('x') == 1:
            img = img.resize((img.size[0]*2, img.size[1]*2), resample=Image.BICUBIC)
        img = np.array(img)

        patch_idx = 0
        for i in range(0, img.shape[0] - patch_size + 1, stride):
            for j in range(0, img.shape[1] - patch_size + 1, stride):
                patch = img[i:i+patch_size, j:j+patch_size]
                Image.fromarray(patch).save(f"{img_dir}/img_{sample_idx:03d}_{patch_idx:02d}.png")
                #plt.imsave(f"{img_dir}/img_{sample_idx:03d}_{patch_idx:02d}.png", patch, cmap='gray')
                
                patch_idx += 1
