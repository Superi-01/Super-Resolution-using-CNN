
import os
import torch
import torch.nn.functional as F
import importlib.util
import sys


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def calc_psnr(x, y, max_val = 255): #정규화x이면, max_val = 255
    mse = F.mse_loss(x, y)
    psnr = 10 * torch.log10(max_val**2 / mse)
    return  psnr.item()

def import_library(file_path, module_name=None):
    file_path = os.path.abspath(file_path)

    if module_name is None:
        module_name = os.path.splitext(os.path.basename(file_path))[0]

    if not os.path.exists(file_path):
        raise ImportError(f"Module file '{file_path}' does not exist.")

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
