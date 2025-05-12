import os
import cv2
import numpy as np
import pandas as pd

meta = pd.read_csv('data/echonet/A4C/VolumeTracings_new.csv')
split = pd.read_csv('data/echonet/A4C/FileList_new.csv')

grouped = meta.groupby(['FileName', 'Frame'])

sp = ['train', 'val', 'test']

for s in sp:
    os.makedirs(f'data/raw/{s}/images', exist_ok=True)
    os.makedirs(f'data/raw/{s}/masks', exist_ok=True)

for (filename, frame_num), group in grouped:

    video_path = f'data/echonet/A4C/videos/{filename}'
    cap = cv2.VideoCapture(video_path)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num-1)

    ret, frame = cap.read()

    if not ret:
        print(f"Failed to read frame {frame_num} from {filename}")
        continue

    sp = str.lower(split[split['FileName'] == filename[:-4]]['Split'].values[0])


    image_path = f'data/raw/{sp}/images/{filename}_frame{frame_num}.png'

    cv2.imwrite(image_path, frame)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8) 
    points = group[["X", "Y"]].values.astype(int)
    cv2.fillPoly(mask, [points], color=255)
    
    mask_path = f'data/raw/{sp}/masks/{filename}_frame{frame_num}.png'
    cv2.imwrite(mask_path, mask)
    
    cap.release()