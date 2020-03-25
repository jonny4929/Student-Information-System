import torch
import torch.nn as nn
import torch.nn.functional as functional

class ONet(nn.Module):
    def __init__(self):
        super(ONet,self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(3,32,3),
            nn.ReLU(),
            nn.MaxPool2d(3,2,ceil_mode=True),
            nn.Conv2d(32,64,3),
            nn.ReLU(),
            nn.MaxPool2d(3,2,ceil_mode=True),
            nn.Conv2d(64,64,3),
            nn.ReLU(),
            nn.MaxPool2d(3,2,ceil_mode=True),
            nn.Conv2d(64,128,2),
            nn.ReLU()
        )
        self.fc=nn.Linear(128*3*3,256)
        self.fc1=nn.Linear(256,2)
        self.fc2=nn.Linear(256,4)
        self.fc3=nn.Linear(256,10)

    def forward(self,x):
        out=self.layer(x)
        print(out.shape)
        out.view_(out.shape[0],128*3*3)
        out=self.fc(out)
        out1=self.fc1(out)
        out2=self.fc2(out)
        out3=self.fc3(out)
        return out1,out2,out3