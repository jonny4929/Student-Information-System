import torch
import torch.nn as nn
import torch.nn.functional as functional

class RNet(nn.Module):
    def __init__(self):
        super(RNet,self).__init__()
        self.net=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=28,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=28,out_channels=48,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True),
            nn.Conv2d(in_channels=48,out_channels=64,kernel_size=2),
            nn.ReLU()
        )
        self.fc=nn.Linear(3*3*64,128)
        self.fc1=nn.Linear(128,2)
        self.fc2=nn.Linear(128,4)
        self.fc3=nn.Linear(128,10)

    def forward(self,x):
        out=self.net(x)
        out=out.view(out.shape[0],64*3*3)
        out=self.fc(out)
        out1=self.fc1(out)
        out2=self.fc2(out)
        out3=self.fc3(out)
        return out1,out2,out3