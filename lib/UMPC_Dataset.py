from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch

import os
import pandas as pd
from PIL import Image
import re
import random

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        modality1, modality2, label = self.data[index]
        if modality1 is None:
            modality1 = torch.zeros(3, 224, 224)
        if modality2 is None:
            modality2 = {
                'input_ids': torch.zeros((40,), dtype=torch.long),
                'token_type_ids': torch.zeros((40,), dtype=torch.long),
                'attention_mask': torch.zeros((40,), dtype=torch.long)
            }
        return modality1, modality2, label
    
def find_classes(directory) :
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}
    return classes, class_to_idx,idx_to_class

# 1. Subclass torch.utils.data.Dataset
class UMPC_FoodDataset(Dataset):
    
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir="./datasets/Food101" ,phase="test",mode ="all",transform=None):
        
        # 3. Create class attributes
        self.dataset_root = targ_dir
        self.csv_file_path = '%s/texts/%s_titles.csv' % (self.dataset_root, phase)
        self.img_dir='%s/images/%s' % (self.dataset_root, phase)
        
        self.data = pd.read_csv(self.csv_file_path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx,self.idx_to_class = find_classes(self.img_dir)
        self.mode=mode

    # 4. Make function to load images
    def load_image(self, index,img_path): #-> Image.Image:
        "Opens an image via a path and returns it."
        return Image.open(img_path).convert('RGB')
                
    def clean_text(self,raw_text):
        t = re.sub(r'^RT[\s]+', '', raw_text)# remove old style retweet text "RT"
        t = re.sub(r'https?:\/\/.*[\r\n]*', '', t)# remove hyperlinks
        t = re.sub(r'#', '', t) # remove hashtags
        return t
    
    def tokenize(self, sentence):
        ids = self.tokenizer(self.clean_text(sentence) ,
                             padding='max_length', max_length=40, truncation=True).items()
        return {k: torch.tensor(v) for k, v in ids}
    
    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self):# -> int:
        "Returns the total number of samples."
        return len(self.data)
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index): #returns  Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        sample=self.data.iloc[index]
        
        txt=sample[1] 
        text_tokens= self.tokenize(txt)
        
        class_name  = sample[2] 
        class_idx = self.class_to_idx[class_name]
        
        if self.mode =="all":
            img_path = os.path.join(self.img_dir,sample[2] ,sample[0] )
            img = self.load_image(index,img_path)
            
            # Transform if necessary
            if self.transform:
                return self.transform(img), text_tokens, txt, class_idx 
            else:
                return img, text_tokens, txt, class_idx 
        elif self.mode =="Text_only":
            return  text_tokens, txt, class_idx            
