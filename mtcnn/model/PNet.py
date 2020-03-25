# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
import torch.nn as nn
import torch.nn.functional as functional


# %%
class PNet(nn.Module):
    def __init__(self):
        super(PNet,self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=10,kernel_size=3,stride=1)
        self.pool1=nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True)
        self.conv2=nn.Conv2d(in_channels=10,out_channels=16,kernel_size=3,stride=1)
        self.conv3=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1)
        self.conv4a=nn.Conv2d(in_channels=32,out_channels=2,kernel_size=1,stride=1)
        self.conv4b=nn.Conv2d(in_channels=32,out_channels=4,kernel_size=1,stride=1)
        self.conv4c=nn.Conv2d(in_channels=32,out_channels=10,kernel_size=1,stride=1)
        
    def forward(self,x):
        out=functional.relu(self.conv1(x))
        out=self.pool1(out)
        out=functional.relu(self.conv2(out))
        out=functional.relu(self.conv3(out))
        out1=functional.relu(self.conv4a(out))
        out2=self.conv4b(out)
        out3=self.conv4c(out)
        return out1,out2,out3


# %%


