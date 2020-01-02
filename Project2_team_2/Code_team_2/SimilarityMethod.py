
# coding: utf-8

# In[2]:

import numpy as np


# In[3]:

import math
def cosine_similarity(v1,v2):
    sumxx, sumxy, sumyy = 0., 0., 0.
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)


# In[173]:

w2v = np.load('w2v.44.npy')
#relation={k: v for v, k in enumerate(['祖孫','母子','母女','父子','父女','兄弟姊妹','夫妻','姑叔舅姨甥侄','遠親','主僕','師徒','居處'])}
relation=['祖孫','母子','母女','父子','父女','兄弟姊妹','夫妻','姑叔舅姨甥侄','遠親','主僕','師徒','居處']
train_none={'四姐':['姐'],'王嫂子':['嫂子']}
test_none={'何婆子':['婆子'],'寶玉':['賈寶玉']}


# In[161]:

loc = [line.rstrip('\n') for line in open('loc.txt')]


# In[177]:

ext=['祖孫','母子','母女','父女','兄弟姊妹','居處']
ext2=['父子']
train_dict={k:[] for k in relation}
location={'居處':[]}
loc_r={'居處':[]}
with open('train.txt', 'r') as file_:
    next(file_)
    for line in file_:
        id,p1,p2,r=line.split()
        
        if p1 not in train_none:
                vp1=w2v.item().get(p1.decode('utf-8'))
        else:
                vp1=w2v.item().get(train_none[p1][0].decode('utf-8'))
        if p2 not in train_none:
                vp2=w2v.item().get(p2.decode('utf-8'))
        else:
                vp2=w2v.item().get(train_none[p2][0].decode('utf-8'))
            
        if r in ext2:
                train_dict[r].append(vp1-vp2)
                train_dict[r].append(vp2-vp1)
        elif r not in ext:
                train_dict[r].append(vp1-vp2)
        #if r in ext or r in ext2:
                #train_dict[r].append(vp2-vp1)
                
        '''
        if r == '居處':
                location[r].append(vp2)
                loc_r[r].append(vp1-vp2)
tamp=np.array([0.]*100)
for v in loc_r['居處']:
    tamp+=v
tamp/=float(len(loc_r['居處']))
'''


# In[156]:

'''
temp={k:[np.array([0.]*100)] for k in relation}
temp2={k:[] for k in relation}
for k in train_dict:
    if k not in ext2:
        for v in train_dict[k]:
            temp[k][0]+=v
        temp[k][0]/=float(len(train_dict[k]))
    else:
        temp2[k]=train_dict[k]
train_dict = temp
for k in ext2:
    train_dict[k]=temp2[k]
'''


# In[178]:

correct=0
train_dict2={k:[] for k in relation}
with open('test.txt', 'r') as file_:
    next(file_)
    for line in file_:
        id_,p1,p2,ans=line.split()
        if p1 not in test_none:
            vp1=w2v.item().get(p1.decode('utf-8'))
        else:
            vp1=w2v.item().get(test_none[p1][0].decode('utf-8'))
        if p2 not in test_none:
            vp2=w2v.item().get(p2.decode('utf-8'))
        else:
            vp2=w2v.item().get(test_none[p2][0].decode('utf-8'))
        if vp1 is None:
            vp1=np.array([0.]*100)
        v1=vp1-vp2
        
        max_s=-100.
        most_r=''
        for r in train_dict:
            if p1[:3]==p2[:3]:
                if r not in ['祖孫','母子','母女','父子','父女','兄弟姊妹','姑叔舅姨甥侄','遠親']:
                    continue
            if r=="居處":
                '''
                d=min([minkowski_distance(vp2,v,3) for v in location['居處']])                
                if d<13.45 and cosine_similarity(vp1-vp2,tamp)>0.3:
                    print p2
                    most_r=r
                    break
                else:
                    for v2 in train_dict[r]:
                        s=abs(cosine_similarity(v1,v2))
                        if s>max_s:
                            max_s=s
                            most_r=r
                '''
                if p2 in loc:
                    #print p2
                    most_r=r
                    break
            else:
                for v2 in train_dict[r]:
                    s=(cosine_similarity(v1,v2))
                    if s>max_s:
                        max_s=s
                        most_r=r        
        
        if most_r==ans:
            correct+=1
            #print 'c:',id_,p1,p2,ans,max_s
        #else:
            #print id_,p1,p2,most_r,ans,max_s
        
        
print correct/112.


# In[ ]:



