import torch
from torch.utils import data
import numpy
import pandas
from cv2 import cv2
import torchvision.transforms as transforms
class Pnet_dataset(data.Dataset):
    def __init__(self):
        torch.set_default_tensor_type(torch.DoubleTensor)
        self.path='/media/jonny/data1/data/face/mtcnn/img/'
        f=open('/media/jonny/data1/data/face/mtcnn/label.txt')
        self.data=pandas.read_csv(f,sep='\s+',header=0)
        self.Label=self.data.drop(['img_path','type'],axis=1).values
        self.dets=self.data['type'].values
        
    def __getitem__(self,index):
        det=torch.tensor([self.dets[index]],dtype=torch.long)
        label=torch.tensor(self.Label[index])
        transform= transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
        )
        img=cv2.imread(self.path+str(index)+'.jpg')
        imgdata=transform(img)
        return imgdata,det,label
    
    def __len__(self):
        return 850511

class RNet_dataset(data.Dataset):
    def __init__(self):
        lnum=0
        f=open('/media/jonny/data1/data/face/mtcnn/PNet_out/label'+str(lnum)+'.txt')
        self.data=pandas.read_csv(f,sep='\s+',header=0)
        self.path=self.data['img_path'].values
        self.Label=self.data.drop(['img_path','type'],axis=1).values
        self.dets=self.data['type'].values 

    def __getitem__(self,index):
        det=torch.tensor([self.dets[index]],dtype=torch.long)
        label=torch.tensor(self.Label[index])
        transform= transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
        )
        img=cv2.imread(self.path[index])
        imgdata=transform(img)
        return imgdata,det,label   

    def __len__(self):
        return 400019
