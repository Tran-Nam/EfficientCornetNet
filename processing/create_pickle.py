import numpy as np
import pickle
import pandas as pd
from PIL import Image
import sys
sys.path.append('..')

from helper.generate_gt import gen_gt
from helper.utils import resize

image_dir = '../data/images' # for test
labels_csv = '../../data/labels.csv'
out_pkl = '../../data/data.pkl'
class CreateData():
    def __init__(self, image_dir=image_dir, labels_csv=labels_csv, out=out_pkl):
        self.out_pkl = out_pkl
        self.image_dir = image_dir
        self.labels = pd.read_csv(labels_csv)

        n_img = len(list(set(self.labels['filename'])))
        print('>>> Num images: %d'%n_img)

    def get_corner(self, file_path):
        filename = file_path.split('/')[-1]
        pts = np.zeros((4, 2))
        rows = self.labels[self.labels['filename']==file_path]
        for idx, row in rows.iterrows():
            row_c = row['class']
            if row_c=='topleft':
                pts_loc = 0
            elif row_c=='topright':
                pts_loc = 1
            elif row_c=='bottomright':
                pts_loc = 2
            elif row_c=='bottomleft':
                pts_loc = 3
            pts[pts_loc] = np.array([row['x'], row['y']])
        return pts

    def create_data(self):
        imgs = []
        heatmaps = []
        offsets = []
        masks = []
        all_file_path = list(set(self.labels['filename']))
        for file_path in all_file_path:
            img = Image.open(os.path.join(self.image_dir, file_path))
            img = np.array(img)
            pts = self.get_corner(file_path)

            img, pts = resize(img, pts, side=512)
            _, __, heatmap, offset = gen_gt(img, pts)
            img = img.astype('float32') * 1./255

            imgs.append(img)
            heatmaps.append(heatmap)
            offsets.append(offset)
            masks.append(mask)
        
        imgs = np.array(imgs)
        heatmaps = np.array(heatmaps)
        offsets = np.array(offsets)
        masks = np.array(masks)

        print(imgs.shape, heatmaps.shape, offsets.shape, masks.shape)

if __name__=='__main__':
    cd = CreateData()
    cd.create_data()

