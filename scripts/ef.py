import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error as mae
from src.utils import load_model, load_device, get_ef

if __name__ == '__main__':
    
    meta = pd.read_csv('data/echonet/A4C/FileList_new.csv')
    meta = meta[meta['Split'] == 'TEST']
    y = meta['EF']

    model = load_model('models/u-net-2.pth')

    device = load_device()

    y_preds = [get_ef('data/echonet/A4C/videos/' + filename + '.avi', model, device) 
           for filename in tqdm(meta['FileName'], desc='Ejection Fraction Estimation')]

    print(f'MAE: {mae(y, y_preds)}')