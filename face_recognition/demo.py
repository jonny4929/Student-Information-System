import torch
from torch.utils import data
import numpy
import pandas
import cv2
import torchvision.transforms as transforms
import torch.nn as nn
from model.model import SSH
from model.nms import NMS,IOU_cal
from model.model import Onet,Rnet_v2
from utils.utils import clip_boxes
from model.model import resnet50
torch.set_default_tensor_type(torch.FloatTensor)

net=SSH()
device=torch.device('cuda:0')
net.load_state_dict(torch.load('./pretrained_model/SSH.pth'))
net=net.to(device).eval() 

img=cv2.imread("./pic/.jpg")
img=cv2.resize(img,(1600,1200))
transform= transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
        )
imgtensor=transform(img)
imgtensor.unsqueeze_(0)
imgtensor=imgtensor.to(device)
x=net(imgtensor,(1200,1600),training=False)
arg=[]
for i in range(x.shape[1]):
    if x[0,i,4]>0.2:
        arg+=[i]
arg_nms=NMS(x[0][arg],0.2)
mat=x[0][arg][arg_nms]
onet=Onet()
rnet=Rnet_v2()
onet.load_state_dict(torch.load('./pretrained_model/Onet.pth'))
rnet.load_state_dict(torch.load('./pretrained_model/newRnet.pth'))
rnet=rnet.to(device).eval()
onet=onet.to(device).eval()
imagetensor=torch.zeros(mat.shape[0],3,112,96)
for m in range(mat.shape[0]):
    image=img[int(mat[m][1]):int(mat[m][3]),int(mat[m][0]):int(mat[m][2])]
    image=cv2.resize(image,(96,112))
    transform= transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
        )
    imagetensor[m]=transform(image)
oout=onet(imagetensor.to(device))
delta=oout[1].to(torch.device('cpu'))
mat[:,:4]=delta*torch.cat((mat[:,2:4]-mat[:,:2],mat[:,2:4]-mat[:,:2]),dim=1)+mat[:,:4]
mat_nms=NMS(mat,0.2)
mat=mat[mat_nms]
mat=clip_boxes(mat,(1600,1200))
imagetensor=torch.zeros(mat.shape[0],3,112,96)
for m in range(mat.shape[0]):
    image=img[int(mat[m][1]):int(mat[m][3]),int(mat[m][0]):int(mat[m][2])]
    image=cv2.resize(image,(96,112))
    transform= transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
        )
    imagetensor[m]=transform(image)
rout=rnet(imagetensor.to(device))
delta=rout[1].to(torch.device('cpu'))
landmark=rout[2].to(torch.device('cpu'))
mat[:,:4]=delta*torch.cat((mat[:,2:4]-mat[:,:2],mat[:,2:4]-mat[:,:2]),dim=1)+mat[:,:4]
mat_nms=NMS(mat,0.3)
mat=mat[mat_nms]
landmark=landmark[mat_nms]
mat=clip_boxes(mat,(1600,1200))
imgtensor=torch.zeros((mat.shape[0],3,112,96),device=device)
for m in range(mat.shape[0]):
    print(mat[m])
    image=img[int(mat[m][1]):int(mat[m][3]),int(mat[m][0]):int(mat[m][2])]
    image=cv2.resize(image,(96,112))
    imgtensor[m]=transform(image)
    cv2.imwrite('./img_done/'+str(m)+'.jpg',image)
facenet=resnet50(10)
facenet.res.load_state_dict(torch.load('pretrained_model/feature.pth'))
facenet=facenet.res
facenet.eval()
facenet=facenet.to(device)
feature=facenet(imgtensor)
feature=nn.functional.normalize(feature,dim=1)
f=open('feature.txt','w')
st=str(feature.shape[0])+'\n'
for i in range(128):
    st+='x'+str(i)
    st+=' '
st+='\n'
for m in range(feature.shape[0]):
    for i in range(feature.shape[1]):
        st+=str(round(feature[m,i].item(),3))+' '
    st+='\n'
f.write(st)
