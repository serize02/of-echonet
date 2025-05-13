import os
import cv2
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from model import UNet

def dice_loss(pred, target, smooth=1e-5):
    intersection = (pred * target).sum()
    return 1 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def bce_dice_loss(pred, target):
    bce = torch.nn.BCELoss()(pred, target)
    return bce + dice_loss(pred, target)

def load_model(path):
    model = UNet()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model


def load_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_ef(input_path, model, device,
                            threshold=0.5, mag_threshold=2.5, 
                            pixel_spacing=0.1, max_frames=100):

    def estimate_L(mask: np.ndarray, pixel_spacing: float) -> float:
        if np.sum(mask) == 0:
            return 0.0
        rows = np.where(np.any(mask == 1, axis=1))[0]
        row_base = rows[0]
        row_apex = rows[-1]
        return (row_apex - row_base) * pixel_spacing

    cap = cv2.VideoCapture(input_path)
    
    model.eval()
    frame_count = 0
    prev_mask = None
    valid_frames = []
    validation_metrics = []
    mask_areas = []
    mask_lengths = []

    val_transforms = transforms.Compose([
        transforms.ToTensor(),    
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    with torch.no_grad():
        
        while cap.isOpened() and frame_count < max_frames:
            
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_tensor = val_transforms(frame_gray).unsqueeze(0).float().to(device)
            pred = model(frame_tensor)
            curr_mask = (pred > threshold).squeeze().cpu().numpy().astype(np.uint8)
            
            current_area = np.sum(curr_mask)
            current_length = estimate_L(curr_mask, pixel_spacing)
            
            mask_areas.append(current_area)
            mask_lengths.append(current_length)
            
            metrics = {
                'frame': frame_count,
                'mean_magnitude': 0,
                'is_valid': False,
                'mask_area': current_area,
                'mask_length': current_length
            }
            
            if prev_mask is not None:
                prev_mask_flow = (prev_mask * 255).astype(np.uint8)
                curr_mask_flow = (curr_mask * 255).astype(np.uint8)
                
                flow = cv2.calcOpticalFlowFarneback(
                    prev_mask_flow,
                    curr_mask_flow,
                    None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                y, x = np.where(curr_mask == 1)
                if len(y) > 0:
                    flow_vectors = flow[y, x]
                    magnitude, _ = cv2.cartToPolar(flow_vectors[...,0], flow_vectors[...,1])
                    mean_mag = np.mean(magnitude)
                    
                    metrics.update({
                        'mean_magnitude': mean_mag,
                        'is_valid': (mean_mag <= mag_threshold) 
                    })
                    
                    if metrics['is_valid']:
                        valid_frames.append(frame_count)
                
                validation_metrics.append(metrics)
            
            prev_mask = curr_mask.copy()
            frame_count += 1
    
    cap.release()
    
    if len(valid_frames) >= 2:
        volumes = []
        for area_px, length_cm in zip(mask_areas, mask_lengths):
            area_cm2 = area_px * (pixel_spacing ** 2)
            volume = area_cm2 * length_cm
            volumes.append(volume)

        edv = max(volumes)
        esv = min(volumes)
        ef = ((edv - esv) / edv) * 100

    else:
        edv, esv, ef = 0, 0, 0

    valid_count = len(valid_frames)
    total_compared = max(1, len(validation_metrics))

    return ef