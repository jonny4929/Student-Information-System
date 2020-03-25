import numpy
import pandas
import cv2
f=open('/home/jonny/dataset/face/CelebA/Anno/list_bbox_celeba.txt')
datasets=pandas.read_csv(f,sep='\s+',header=1)
Label=datasets.drop('image_id',axis=1).values
path='/media/jonny/data1/data/face/CelebA/Img/img_celeba/img_celeba/'
mtcnn_path='/media/jonny/data1/data/face/mtcnn/'
mtcnn_img='/media/jonny/data1/data/face/mtcnn/img/'
wf=open('/media/jonny/data1/data/face/mtcnn/label.txt','w')
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
num=0
wf.write('img_path'+' type'+' dx1'+' dy1'+' dx2'+' dy2\n')
for i in range(202599):
    img=cv2.imread(path+datasets['image_id'][i])
    label=Label[i]
    for j in range(6):
        x1=label[0]
        y1=label[1]
        x2=label[2]
        y2=label[3]
        offset=numpy.random.rand(4)*(0.1*j+0.15)
        nlabel=[x2,y2,x2,y2]*offset
        nlabel+=[x1,y1,x1+x2,x1+y2]   
        offset=[(nlabel[0]-x1)/(nlabel[2]-nlabel[0]),(nlabel[1]-y1)/(nlabel[3]-nlabel[1]),(nlabel[2]-x1-x2)/(nlabel[2]-nlabel[0]),(nlabel[3]-y1-y2)/(nlabel[3]-nlabel[1])]
        nlabel=numpy.int_(nlabel)
        if nlabel[0]>=nlabel[2] or nlabel[1]>=nlabel[3]:
            continue
        elif nlabel[2]>img.shape[1] or nlabel[3]>img.shape[0]:
            continue
        IOU=IOU_cal(nlabel,label)
        if IOU>0.65:
            wf.write(mtcnn_img+str(num)+'.jpg'+' 1 '+str(round(offset[0],3))+' '+str(round(offset[1],3))+' '+str(round(offset[2],3))+' '+str(round(offset[3],3))+'\n')
        elif IOU>0.4:
            wf.write(mtcnn_img+str(num)+'.jpg'+' 2 '+str(round(offset[0],3))+' '+str(round(offset[1],3))+' '+str(round(offset[2],3))+' '+str(round(offset[3],3))+'\n')
        else:
            wf.write(mtcnn_img+str(num)+'.jpg'+' 0'+' 0'+' 0'+' 0'+' 0\n')
        newimg=img[nlabel[1]:nlabel[3],nlabel[0]:nlabel[2]]
        newimg=cv2.resize(newimg,(12,12))
        cv2.imwrite(mtcnn_img+str(num)+'.jpg',newimg)
        num+=1
        if num%10000==9999:
            print(num+1)
    if num>=1000000:
        break
print(num)


