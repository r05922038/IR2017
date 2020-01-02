Similarity method
===================================================================================================================
1.If you want to reproduce the best result of similarity method,
you can directly type "python SimilarityMethod.py" to predict test.txt and then you can see the accuracy on screen.
Not that SimilarityMethod.py directly uses word embedding "w2v.44.npy" in this folder.

2.If you want to reproduce results from Dream_of_the_Red_Chamber_seg.txt, 
you can type "python preprocessing.py" to preprocess it.
Then please type "python word2vec.py #vector_dim #iter" to generate word embedding "w2v.44.npy".
After that, you can follow the previous step to predict test.txt and show the accuracy.
Note that #vector_dim is dimention of the word vector, and #iter is iteration.
(ex. in our best result, #vector_dim=100 and #iter=300)

Classifier method
===================================================================================================================
python addRelation.py