from utils import *
import torch.nn as nn
import torch.optim as optim


get_args = import_library(f"arguments.py").get_args
preprocessing = import_library(f"preprocessing.py").classify_hr_lr
dataloader = import_library(f"dataloader.py").make_dataset
model_loader = import_library(f"model.py").SRCNN
train = import_library(f"train.py").train
test_module = import_library(f"test.py")
test, visualize_test_result = test_module.test, test_module.visualize_test_result

def main():
    
    # 파라미터 선언
    print('set parameters')
    opt = get_args()
    print(opt.device)
    print('set parameters DONE')
    
    # 시스템 설정
    print('set system')
    for path in [opt.train_x_img_dir, opt.train_y_img_dir, opt.test_x_img_dir, opt.test_y_img_dir, opt.valid_x_img_dir, opt.valid_y_img_dir]:
        make_dir(path)
    print('set system DONE')
    
    # preprocessing
    print('preprocessing')
    preprocessing(opt, opt.img_dir, opt.train_x_img_dir, opt.train_y_img_dir, opt.test_x_img_dir, opt.test_y_img_dir, opt.valid_x_img_dir, opt.valid_y_img_dir).split_save_data()
    print('preprocessing DONE')
    
    # dataloader
    print('data load')
    train_dl, test_dl, valid_dl = dataloader(opt).main()
    print('data load DONE')
    
    # model
    print('model load')
    print(opt.device)
    model = model_loader()
    print('model load DONE')
    
    # train
    print('train')
    print(opt.device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, train_dl, valid_dl, optimizer, loss_fn, opt, 1)
    print('train DONE')
        
    # test
    print('test')
    test(model, test_dl, opt, loss_fn)
    visualize_test_result(model, test_dl, opt)
    print('test DONE')
    
if __name__ == "__main__":
    main()    

    
    
    
    
    
    
    


