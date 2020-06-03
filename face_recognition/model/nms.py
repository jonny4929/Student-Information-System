import torch
import numpy
def IOU_cal(cell,ref):
    bbox=torch.tensor([0,0,0,0])
    bbox[:2],_=torch.max(torch.cat((cell[:2].unsqueeze(0),ref[:2].unsqueeze(0)),0),0)
    bbox[2:],_=torch.min(torch.cat((cell[2:].unsqueeze(0),ref[2:].unsqueeze(0)),0),0)
    s1=(bbox[2]-bbox[0])*(bbox[3]-bbox[1])
    s2=(ref[2]-ref[0])*(ref[3]-ref[1])
    s3=(cell[2]-cell[0])*(cell[3]-cell[1])
    if bbox[0]>=bbox[2] or bbox[1]>=bbox[3]:
        return 0
    return s1/(s2+s3-s1)

def NMS(in_array,threshold):
    _,order=torch.sort(in_array,0,True)
    order=order[:,4]
    out_arg=[]
    while order.shape[0]!=0:
        out_arg+=[order[0]]
        remain_list=[]
        for k in range(1,order.shape[0]):
            IOU=IOU_cal(in_array[order[k]][0:4],in_array[order[0]][0:4])
            if(IOU<threshold):
                remain_list+=[k]
        order=order[remain_list]
    return torch.from_numpy(numpy.array(out_arg))