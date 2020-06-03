import torch
import pandas
f=open('feature.txt')
f1=open('standard.txt')
feature=torch.from_numpy(pandas.read_csv(f,header=1,sep='\s+').values)
standard=torch.from_numpy(pandas.read_csv(f1,header=1,sep='\s+').values)
f.close()
f1.close()
f=open('ans.txt','w')
l=[]
for m in range(feature.shape[0]):
    a=torch.norm(standard-feature[m:m+1],dim=1)
    _,arg=torch.min(a,dim=0)
    l+=[arg.item()]
    standard[torch.arange(standard.shape[0])==arg]=2
st=''
for item in l:
    st+=str(item)+' '
f.write(st)