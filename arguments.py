# 라이브러리 선언
import argparse
import torch


def get_args():

    parser = argparse.ArgumentParser()
    
    # DATA
    parser.add_argument('--train_x_img_dir',type=str, default='./data/Urban100/train/x')
    parser.add_argument('--train_y_img_dir',type=str, default='./data/Urban100/train/y')
    parser.add_argument('--test_x_img_dir',type=str, default='./data/Urban100/test/x')
    parser.add_argument('--test_y_img_dir',type=str, default='./data/Urban100/test/y')
    parser.add_argument('--valid_x_img_dir',type=str, default='./data/Urban100/valid/x')
    parser.add_argument('--valid_y_img_dir',type=str, default='./data/Urban100/valid/y')
    parser.add_argument('--img_dir',type=str, default='./data/Urban100/image_SRF_2')

    parser.add_argument('--patch_size',type=int, default=64)
    parser.add_argument('--stride',type=int, default=32)
    parser.add_argument('--batch_size',type=int, default=16)
    
    # Learning
    parser.add_argument('--epochs',type=int, default=50)
    parser.add_argument('--lr',type=float, default=1e-4)
    parser.add_argument('--early_stop',type=int, default=20, help='early stop_patience')
    
    # GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument('--device', type=str, default=device, help='device')
    
    opt = parser.parse_args('')

    return opt