import torch
import pickle
import numpy as np
from PIL import Image
import pickle
from BaseDataset import BaseDataset


class Fitz17k(BaseDataset):
    def __init__(self, dataframe, path_to_pickles, sens_name, sens_classes, transform):
        super(Fitz17k, self).__init__(dataframe, path_to_pickles, sens_name, sens_classes, transform)
        
        self.A = self.set_A(sens_name)  
        
        self.Y = self.dataframe.label

        self.AY_proportion = None
        
    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        import os
        img_id = str(item['md5hash'])
        img_path = os.path.join(self.path_to_images, img_id + '.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(self.path_to_images, img_id)
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        label = torch.FloatTensor([self.Y[idx]])
        
        sensitive = self.get_sensitive(self.sens_name, self.sens_classes, item)
                               
        return idx, img, label, sensitive
    

class Fitz17kTest(BaseDataset):
    '''
        Only used for evaluation!
    '''
    def __init__(self, dataframe, path_to_pickles, sens_name, sens_classes, transform, sens_attr):
        super(Fitz17kTest, self).__init__(dataframe, path_to_pickles, sens_name, sens_classes, transform)
    
        # add by zk, select samples with specific sensitive_attributes
        self.dataframe = self.dataframe[self.dataframe['skin_tone'] == sens_attr]
        self.dataframe = self.dataframe.reset_index(drop=True)
        
        self.A = self.set_A(sens_name)  
        
        self.Y = self.dataframe.label
        self.Y = self.Y.reset_index(drop=True)

        self.AY_proportion = None
        
    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        import os
        img_id = str(item['md5hash'])
        img_path = os.path.join(self.path_to_images, img_id + '.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(self.path_to_images, img_id)
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        label = torch.FloatTensor([self.Y[idx]])
        
        sensitive = self.get_sensitive(self.sens_name, self.sens_classes, item)
                               
        return idx, img, label, sensitive
    
    def __len__(self):
        return len(self.dataframe)