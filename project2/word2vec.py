
# coding: utf-8

# In[ ]:

from gensim.models import word2vec
import sys


# In[ ]:

word2vec_len=int(sys.argv[1])
word2vec_iter=int(sys.argv[2])
sentences = word2vec.Text8Corpus("Dream_of_the_Red_Chamber_seg_preprocessing.txt")
model = word2vec.Word2Vec(sentences, size=word2vec_len, min_count=3, window=5,iter=word2vec_iter)
model.save('Dream_of_the_Red_Chamber_seg_preprocessing_word2vec.bin')
print 'train word2vec---done'

