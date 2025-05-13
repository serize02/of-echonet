import torch
import numpy as np
from tqdm import tqdm
from utils import load_model, load_device
from dataset import get_dataloaders


def evaluate_model(model, dataloader, device):
    model.eval()
    dice_scores = []
    
    with torch.no_grad():

        for images, masks in tqdm(dataloader, desc='Testing'):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float()
            
            intersection = (preds * masks).sum()
            dice = (2 * intersection) / (preds.sum() + masks.sum() + 1e-5)
            dice_scores.append(dice.item())
    
    return np.mean(dice_scores)


if __name__ == '__main__':

    model = load_model('models/u-net-2.pth')

    _, __, test_loader = get_dataloaders()

    device = load_device()

    dice_test = evaluate_model(model, test_loader, device)
    print(f"Dice Score en test: {dice_test:.4f}")