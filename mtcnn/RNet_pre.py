import torch
from torch.utils import data
import numpy
import pandas
import cv2
import torchvision.transforms as transforms
from datasets import datasets
import torch.nn as nn
torch.set_default_tensor_type(torch.DoubleTensor)
f=open('/home/jonny/dataset/face/CelebA/Anno/list_bbox_celeba.txt')
data=pandas.read_csv(f,sep='\s+',header=1)
Label=data.drop('image_id',axis=1).values
path='/media/jonny/data1/data/face/CelebA/Img/img_celeba/img_celeba/'
out_path='/media/jonny/data1/data/face/mtcnn/PNet_out'
from model.PNet import PNet
net=PNet()
device=torch.device('cuda:0')
net.load_state_dict(torch.load('./pretrained_model/new_PNet7.pth'))
net=net.to(device)
net.eval()
def IOU_cal(cell,ref):
    x1=max(cell[0],ref[0])
    y1=max(cell[1],ref[1])
    x2=min(cell[2],ref[0]+ref[2])
    y2=min(cell[3],ref[1]+ref[3])
    s1=(x2-x1)*(y2-y1)
    s2=ref[2]*ref[3]
    s3=(cell[2]-cell[0])*(cell[3]-cell[1])
    if x1>=x2 or y1>=y2:
        return 0
    return s1/(s2+s3-s1)
def NMS(in_array):
    out_array=[]
    while in_array.shape[0]!=0:
        bbox=in_array[0]
        out_array+=[[in_array[0]]]
        in_array=numpy.delete(in_array,[0],axis=0)
        dellist=[]
        tt=bbox
        tt[2:4]-=tt[:2]
        for k in range(in_array.shape[0]):
            IOU=IOU_cal(in_array[k],tt)
            if(IOU>0.6):
                dellist+=[k]
        in_array=numpy.delete(in_array,dellist,axis=0)
    out_array=numpy.concatenate(out_array,axis=0)
    return out_array
lnum=0
wf=open(out_path+'/label'+str(lnum)+'.txt','w')
wf.write('img_path type dx1 dy1 dx2 dy2\n')
num=0
fnum=0
for index in range(202599):
    print(index)
    label=Label[index]
    transform= transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
        )
    img=cv2.imread(path+data['image_id'][index])
    while max(img.shape[0],img.shape[1])>2000:
        img=cv2.resize(img,(0,0),fx=0.5,fy=0.5)
        label=numpy.int_(label*0.5)
    rawimg=img
    minl=min(img.shape[0],img.shape[1])
    scale=0.709
    P_out=[]
    k=0
    while minl>12:
        imgdata=transform(img)
        imgdata=imgdata.unsqueeze(0)
        imgdata=imgdata.type(torch.DoubleTensor)
        imgdata=imgdata.to(device)
        out1,out2,_=net(imgdata)
        imgdata.to(torch.device('cpu'))
        ff=out2
        out1=out1.to(torch.device('cpu'))
        out1.squeeze_(0)
        out1=out1.detach().numpy()
        fout1=numpy.array([out1[i].flatten() for i in range(2)])
        stride=2
        cellsize=12
        offset=numpy.array([
        [[stride*i/scale**k for i in range(out1.shape[2])] for _ in range(out1.shape[1])],
        [[stride*j/scale**k for _ in range(out1.shape[2])] for j in range(out1.shape[1])],
        [[(stride*i+cellsize)/scale**k for i in range(out1.shape[2])] for _ in range(out1.shape[1])],
        [[(stride*j+cellsize)/scale**k for _ in range(out1.shape[2])] for j in range(out1.shape[1])]
        ])
        out2=out2.to(torch.device('cpu'))
        out2.squeeze_(0)
        out2=out2.detach().numpy()
        out2=out2*cellsize/(scale**k)
        out2=offset+out2
        fout2=numpy.array([out2[i].flatten() for i in range(4)])
        for i in range(out1.shape[1]):
            prob=fout1[:,i]
            if prob[1]<prob[0]:
                continue
            prob=prob[1]-prob[0]
            P_out+=[[numpy.concatenate((fout2[:,i],numpy.array([prob])),axis=0)]]
        img=cv2.resize(img,(0,0),fx=scale,fy=scale)
        minl=min(img.shape[0],img.shape[1])
        k+=1
        torch.cuda.empty_cache()
    if P_out==[]:
        fnum+=1
        continue
    P_out=numpy.concatenate(P_out,axis=0)
    P_out=numpy.concatenate((numpy.int_(P_out[:,0:4]),P_out[:,4:]),axis=1)
    ind=(P_out[:,4]).argsort()
    P_out=P_out[ind[::-1]]
    P_out=NMS(P_out)
    img=rawimg
    for mem in numpy.int_(P_out[:,0:4]):
        if mem[1]>=mem[3] or mem[0]>=mem[2]:
            continue
        elif mem[1]<0 or mem[0]<0:
            continue
        elif mem[3]>img.shape[0] or mem[2]>img.shape[1]:
            continue
        newimg=img[mem[1]:mem[3],mem[0]:mem[2]]
        newimg=cv2.resize(newimg,(24,24))
        cv2.imwrite('/media/jonny/data1/data/face/mtcnn/PNet_out/img/'+str(num)+'.jpg',newimg)
        IOU=IOU_cal(mem,label)
        if IOU>0.6:
            wf.write('/media/jonny/data1/data/face/mtcnn/PNet_out/img/'+str(num)+'.jpg'+' 1 '+str(round(label[0]/mem[0]-1,3))+' '+str(round(label[1]/mem[1]-1,3))+' '+str(round((label[0]+label[2])/mem[2]-1,3))+' '+str(round((label[1]+label[3])/mem[3]-1,3))+'\n')
        elif IOU>0.4:
            wf.write('/media/jonny/data1/data/face/mtcnn/PNet_out/img/'+str(num)+'.jpg'+' 2 '+str(round(label[0]/mem[0]-1,3))+' '+str(round(label[1]/mem[1]-1,3))+' '+str(round((label[0]+label[2])/mem[2]-1,3))+' '+str(round((label[1]+label[3])/mem[3]-1,3))+'\n')
        else:
            wf.write('/media/jonny/data1/data/face/mtcnn/PNet_out/img/'+str(num)+'.jpg'+' 0'+' 0'+' 0'+' 0'+' 0\n')
        num+=1
        if num>(lnum+1)*400000:
            lnum+=1
            wf.close()
            wf=open(out_path+'/label'+str(lnum)+'.txt','w')
            wf.write('img_path type dx1 dy1 dx2 dy2\n')
print(fnum)
