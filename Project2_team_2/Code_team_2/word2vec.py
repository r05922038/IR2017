
# coding: utf-8

# In[1]:

from gensim.models import word2vec
import numpy as np
import sys


# In[ ]:

word2vec_len=int(sys.argv[1])
word2vec_iter=int(sys.argv[2])
sentences = word2vec.Text8Corpus("Dream_of_the_Red_Chamber_seg_preprocessing2.txt")
model = word2vec.Word2Vec(sentences, size=word2vec_len, min_count=0, window=5,iter=word2vec_iter)
model.save('Dream_of_the_Red_Chamber_seg_preprocessing2_word2vec.bin')
w2v = dict(zip(model.wv.index2word, model.wv.syn0))
np.save('w2v.44.npy',w2v)
print 'train word2vec---done'
'''
my_dict_back = np.load('w2v.npy')
print(my_dict_back.item().keys())    
print(my_dict_back.item().get('a'))
'''


# In[1]:

#import word2vec


# In[11]:

#word2vec.word2vec('Dream_of_the_Red_Chamber_seg_preprocessing2.txt', 'TEST2.bin',iter_=300,min_count=0, cbow=0,size=100, verbose=True)


# In[25]:

#model = word2vec.load('TEST2.bin')


# In[13]:

#import numpy as np
#w2v = dict(zip(model.vocab, model.vectors))
#np.save('test2.npy',w2v)


# In[16]:

#word2vec.word2clusters('Dream_of_the_Red_Chamber_seg_preprocessing2.txt', 'test2-clusters.txt', 100, verbose=True)


# In[17]:

#clusters = word2vec.load_clusters('test2-clusters.txt')


# In[30]:

#indexes, metrics = model.cosine(u'媚人')
#for w in model.vocab[indexes]:
#    print w


# In[ ]:



