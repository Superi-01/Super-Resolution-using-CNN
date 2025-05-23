
from tqdm import tqdm
import torch
from utils import calc_psnr

import matplotlib.pyplot as plt
def train(model, train_dl, valid_dl, optimizer, loss_fn, opt, num_epochs):
    model = model.to(opt.device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_psnr = 0.0

        train_bar = tqdm(train_dl, desc="Training", leave=False)

        for inputs, targets in train_bar:
            inputs = inputs.to(opt.device).float()
            targets = targets.to(opt.device).float()
            n_batch = len(train_dl)

            # Forward
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_psnr += calc_psnr(outputs, targets)
            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / n_batch
        avg_train_psnr = total_psnr / n_batch

        # Validation
        if valid_dl:
            model.eval()
            n_batch = len(valid_dl)
            val_loss = 0.0
            total_psnr = 0.0
            val_bar = tqdm(valid_dl, desc="Validating", leave=False)
            with torch.no_grad():
                for inputs, targets in val_bar:
                    inputs = inputs.to(opt.device)
                    targets = targets.to(opt.device)

                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                    val_loss += loss.item()
                    total_psnr += calc_psnr(outputs, targets)
                    val_bar.set_postfix(loss=loss.item())

            avg_val_loss = val_loss / n_batch
            avg_val_psnr = total_psnr / n_batch
            print(f"\nEpoch {epoch+1}/{num_epochs}  -  Train Loss: {avg_train_loss:.4f}  Train PSNR: {avg_train_psnr:.4f}  /  Valid Loss: {avg_val_loss:.4f}  Valid PSNR: {avg_val_psnr:.4f}")