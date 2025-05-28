import torch
import matplotlib.pyplot as plt
from utils import calc_psnr

def test(model, test_dl, opt, loss_fn):
    model.eval()
    model.to(opt.device)
    
    total_loss = 0
    total_psnr = 0
    with torch.no_grad():
        for inputs, targets in test_dl:
            inputs = inputs.to(opt.device)
            targets = targets.to(opt.device)
            n_batch = len(test_dl)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            total_psnr += calc_psnr(outputs, targets) 

    print(f"Test loss: {total_loss/n_batch:.4f}  Test PSNR: {total_psnr/n_batch:.4f}")
    
def visualize_test_result(model, test_dl, opt):
    model.eval()
    model.to(opt.device)

    with torch.no_grad():
        for inputs, targets in test_dl:
            inputs = inputs.to(opt.device)
            targets = targets.to(opt.device)
            outputs = model(inputs)

            # 첫 이미지만 시각화
            input_img = inputs[0].cpu()
            target_img = targets[0].cpu()
            output_img = outputs[0].cpu()

            # 채널 수가 1이면 squeeze (흑백 이미지)
            if input_img.shape[0] == 1:
                input_img = input_img.squeeze(0)
                target_img = target_img.squeeze(0)
                output_img = output_img.squeeze(0)

            # 시각화
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.title(f"Input (LR)\nPSNR: {calc_psnr(input_img, target_img):.2f} dB")
            plt.imshow(input_img.numpy(), cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title(f"Output (SR)\nPSNR: {calc_psnr(output_img, target_img):.2f} dB")
            plt.imshow(output_img.numpy(), cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title("Target (HR)")
            plt.imshow(target_img.numpy(), cmap='gray')
            plt.axis('off')

            plt.tight_layout()
            plt.show()
            print(calc_psnr(output_img, target_img))

            break