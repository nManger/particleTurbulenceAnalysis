import pyAthena as ath
import numpy as np
import sys

def distance(x,y,boxl):
    #calculate the distance form point x=(x1,x2,x3) to point y=(y1,y2,y3)
    #boxl = 1.0
    sep = np.abs(x - y)
    dist = np.sqrt(np.sum((np.where(sep>boxl/2.,np.abs(sep-boxl),sep))**2,axis=-1))
    return dist

def S3longitudinal(data,r):
    z,y,x=np.meshgrid(data['x3v'],data['x2v'],data['x1v'],indexing='ij')
    pos3D=np.stack((x,y,z),axis=-1)
    dx=data['x3f'][1]-data['x3f'][0]
    
    #build initial distance mask
    xc = pos3D[0,0,0]
    dist = distance(pos3D,xc,1.0)
    cond = ((dist>r-dx*0.6) & (dist<r+dx*0.6))
    where = np.where(cond)
    
    s3sum = 0.0
    nitems = 0
    n3,n2,n1=data["RootGridSize"]
    for k in range(0,n3):
        for j in range(0,n2):
            for i in range(0,n1):
                #
                xc = pos3D[k,j,i]
                vc=np.array([data['vel1'][k,j,i],data['vel2'][k,j,i],data['vel3'][k,j,i]])
                
                #roll mask to new position
                mask = ((where[0]+k)%n3,(where[1]+j)%n2 ,(where[2]+i)%n1)
                
                vcond = np.stack((data['vel1'][mask],data['vel2'][mask],data['vel3'][mask]),axis=-1)
                poscond = pos3D[mask]

                
                rel = poscond - xc
                lens = np.sqrt(np.sum(rel**2,axis=-1))
                erel = rel/lens[:,None]

                s3sum += np.sum(np.power(np.sum((vcond-vc)*erel,axis=-1),3.0))
                nitems +=poscond.shape[0]
                #print(nitems,"\n")
    
    return s3sum/float(nitems)


if len(sys.argv) == 1:
    r = 0.05
else:
    r = float(sys.argv[1])


data = ath.athdf("TurbPar.out2.00010.athdf")

s3=S3longitudinal(data,r)

with open(f"diffusionRate_r{r}.txt",'w') as f:
    f.write(f"Third order structure function for r = {r} : {s3}\n")
    f.write(f"diffusionRate = {s3/(-4./5.*r)} \n")
