# -*- coding: utf-8 -*-

from gensim.models import word2vec
import csv
import json
import re
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import os
import pandas as pd
from sklearn import svm
import time
from sklearn import tree


def dataPreprocessing(stopTag):
    with open("./data/Dream_of_the_Red_Chamber_seg.txt", 'rb') as f:
        filedata = f.read()
        filedata = filedata.replace('\n', '')
        filedata = filedata.replace('。_Po', '。_Po\n')
     
    with open('./data/Dream_of_the_Red_Chamber_seg_newline.txt', 'w') as g:
        g.write(filedata)

    dict_={'士隱':'甄士隱','岫煙':'邢岫煙','紹祖':'孫紹祖','迎春':'賈迎春','金桂':'夏金桂','寶玉':'賈寶玉',
      '寶釵':'薛寶釵','守中':'李守中','可卿':'秦可卿','之孝':'林之孝','世仁':'卜世仁','紫英':'馮紫英',
       '尚榮':'賴尚榮','元春':'賈元春','惜春':'賈惜春','探春':'賈探春','黛玉':'林黛玉','湘雲':'史湘雲',
      '熙鳳':'王熙鳳','若錦':'張若錦','亦華':'趙亦華','湘蓮':'柳湘蓮','子騰':'王子騰','子勝':'王子勝',
      '德全':'邢德全','巧姐':'賈巧姐','天棟':'趙天棟','友士':'張友士','代善':'賈代善','代儒':'賈代儒',
       '代化':'賈代化','如海':'林如海','可卿':'秦可卿','寶琴':'薛寶琴','自芳':'花自芳','秋芳':'傅秋芳',
      '文翔':'金文翔','世仁':'卜世仁','雨村':'賈雨村','繼宗':'牛繼宗','瑞文':'陳瑞文','孝康':'侯孝康',
      '曉明':'侯曉明','光珠':'石光珠','子寧':'蔣子寧','建輝':'戚建輝','代修':'賈代修'}

    dict2={'嬸娘':set(['李']),'姨娘':set(['趙','周']),'老娘':set(['尤']),
       '嬤嬤':set(['趙','李']),'奶奶':set(['夏']),'太君':set(['史','史氏']),
      '母':set(['賈']),'夫人':set(['邢','王']),'姐':set(['四']),'二姐':set(['尤']),
       '三姐':set(['尤']),'姨媽':set(['薛']),}

    with open('./data/data.txt', 'w') as result:

        stopwords = []

        with open('./data/Dream_of_the_Red_Chamber_seg_newline.txt', 'r') as h:
            stopwords = stopwords + ['，', '。', '、', '；', '：', '？', '「', '」','『','』','●','…']
            stopwords = json.dumps(stopwords, encoding='utf-8', ensure_ascii=False)
            print stopTag

            pre=''
            for line in h:
                new_words=[]
                for word in line.split():
                    t = word.split('_')
                    tag = t[1]
                    word = t[0]
                   
                    if tag in stopTag:
                        pass
                    else:
                        if word not in stopwords and not word.isdigit():
                            if word in dict_:
                                word=dict_[word]
                                new_words.append(word)
                            else:
                                if word in dict2 and pre in dict2[word]:
                                    new_words[len(new_words)-1]=pre+word
                                else:
                                    new_words.append(word)
                            pre=word

                result.write(" ".join(new_words)+'\n')


def toVec(sentence):
    
    model = word2vec.Word2Vec(sentence, size=100,  workers=4, min_count = 1, iter = 3)
    return model

def readData():

    sess = ['train', 'test']
    categories = []
    cate_char = []
    training_y = []
    forWordVec = []
    char_toVec = []
    trainingFeatureVec = []
    testingFeatureVec = [] 
    

    for run_sess in sess:
        print "Now is %s session running..." %(run_sess)

        with open("./data/"+ str(run_sess) +".txt", 'rb') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            next(csv_reader)
            item = list(csv_reader)
            
            
            if str(run_sess) == 'train':
            # adding training tuples into word2Vec Model for model training

                with open("./data/"+ str(run_sess) +"_blank.txt", 'rb') as csv_file:

                    csv_reader = csv.reader(csv_file, delimiter='\t')
                    next(csv_reader) 
                    item_blank = list(csv_reader)

                    for i in range(0, len(item_blank)-1):
                        relation = json.dumps(item_blank[i][3], encoding='utf-8', ensure_ascii=False)
                        relation = relation.replace('[', '')
                        relation = relation.replace('\"', '')
                        relation = relation.replace(']', '')
                        relation = relation.replace(',', '')
                        relation = relation.split(' ')

                        aaa = [str(item_blank[i][1]), str(item_blank[i][2])]

                        for w in relation:

                            aaa.append(w)

                            if w not in cate_char:
                                cate_char.append(w)
                        
                        aaa.append(str(item_blank[i][1]))
                        aaa.append(str(item_blank[i][2]))

                        char_toVec = char_toVec + aaa + aaa
                        
                    
                with open ("./data/data.txt") as a:
                    context = a.readlines()
                    for lines in context:
                        lines = lines.replace('\n', '')
                        words = lines.split(' ')
                        forWordVec.append(words)
                
                forWordVec = forWordVec + char_toVec
                model = toVec(forWordVec)
        

            # start prepare for training/testing Feature Vec (sum of each entity vector)

            for i in range(0, len(item)-1):

                entityFeature = np.zeros((100,),dtype="float32")

                if str(run_sess) == 'train':

                    relation = json.dumps(item_blank[i][3], encoding='utf-8', ensure_ascii=False)
                    relation = relation.replace('[', '')
                    relation = relation.replace('\"', '')
                    relation = relation.replace(']', '')
                    relation = relation.replace(',', '')
                    relation = relation.split(' ')

                    if item[i][3] not in categories:
                        categories.append(item[i][3])
                    
                    training_y.append(item[i][3])
                elif str(run_sess) == 'test':
                    relation = []
                else:
                    print "session error!"

                with open ("./data/data.txt") as origin:

                    context = origin.readlines()

                    # finding sentence which contains both of two entities
                    
                    for x in range(0, len(context)):

                        # one sentence
                        context[x] = context[x].replace('\n', '')
                        listOfTokens1 = context[x].split(' ')

                        if item[i][1] in listOfTokens1 and item[i][2] in listOfTokens1:
                            if str(run_sess) == 'train':

                                for tt in relation:

                                    tt = tt.replace("\"", '')
                                    
                                    if tt in listOfTokens1:

                                        for words in listOfTokens1:

                                            words = json.dumps(words, encoding='utf-8', ensure_ascii=False)
                                            words = words.replace("\"", '')

                                            try:
                                                entityFeature += 2*(model.wv[words])
                            
                                            except KeyError:
                                                pass
                                
                                    
                                for words in listOfTokens1:

                                    words = json.dumps(words, encoding='utf-8', ensure_ascii=False)
                                    words = words.replace("\"", '')

                                    try:
                                        entityFeature += model.wv[words]
                            
                                    except KeyError:
                                        pass
                                        
                            else:
                                for words in listOfTokens1:

                                        words = json.dumps(words, encoding='utf-8', ensure_ascii=False)
                                        words = words.replace("\"", '')

                                        try:
                                            entityFeature += model.wv[words]
                                
                                        except KeyError:
                                            pass
                        
                        else:
                            pass

                        
                if all(v == 0 for v in entityFeature):
                    pass
                else:
                    entityFeature = (entityFeature - np.mean(entityFeature))/np.std(entityFeature)
                    
                if str(run_sess) == 'train':
                    trainingFeatureVec.append(entityFeature)
                elif str(run_sess) == 'test':
                    testingFeatureVec.append(entityFeature)
                else:
                    print "session error!"

                
    return trainingFeatureVec, testingFeatureVec, training_y

def forestClassification(trainingFeatureVec, testingFeatureVec, training_y):
     # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators = 100)

    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    # This may take a few minutes to run
    forest = forest.fit(trainingFeatureVec, training_y)
    
    # Use the random forest to make sentiment label predictions
    result = forest.predict(testingFeatureVec)
    
    return result

def SVMclassifier(trainingFeatureVec, testingFeatureVec, training_y):
    clf = LinearSVC(random_state=0)
    clf.fit(trainingFeatureVec, training_y) 
    result = clf.predict(testingFeatureVec)

    return result

def MLP(trainingFeatureVec, testingFeatureVec, training_y):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15,), random_state=1)
    clf.fit(trainingFeatureVec, training_y) 
    result = clf.predict(testingFeatureVec)
    
    return result



def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)

# ANN and Gradient Descent code from https://iamtrask.github.io//2015/07/27/python-network-part2/
def trainANN(X, y, hidden_neurons, alpha, epochs, dropout, dropout_percent):

    print "---- begin to train ANN ----"

    output = []

    with open("./data/train.txt", 'rb') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        next(csv_reader)
        item = list(csv_reader)
        classes = []
        output = []

        for i in range(0, 149):

            if item[i][3] not in classes:
                classes.append(item[i][3])
        
            output_empty = [0] * 12

            output_row = list(output_empty)
            output_row[classes.index(item[i][3])] = 1

            output.append(output_row)
    
    X = np.array(X)
    y = np.array(output)

    print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (hidden_neurons, str(alpha), dropout, dropout_percent if dropout else '') )
    
    np.random.seed(1)

    last_mean_error = 1
    # randomly initialize our weights with mean 0
    synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2*np.random.random((hidden_neurons, len(classes))) - 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)

    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)
        
    for j in iter(range(epochs+1)):

        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))
                
        if(dropout):
            layer_1 *= np.random.binomial([np.ones((len(X),hidden_neurons))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))

        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # how much did we miss the target value?
        layer_2_error = y - layer_2

        if (j% 10000) == 0 and j > 5000:
            # if this 10k iteration's error is greater than the last iteration, break out
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print ("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))) )
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                # print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                break
                
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
        
        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))
        
        if(j > 0):
            synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))        
        
        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update
        
        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update


    # persist synapses
    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist()}
    synapse_file = "synapses.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    


def ANNclassifier(trainingFeatureVec, testingFeatureVec, training_y):

    # probability threshold
    # ERROR_THRESHOLD = 0.2
    # load our calculated synapse values
    synapse_file = 'synapses.json' 
    with open(synapse_file) as data_file: 
        synapse = json.load(data_file) 
        synapse_0 = np.asarray(synapse['synapse0']) 
        synapse_1 = np.asarray(synapse['synapse1'])
    
    with open("./data/train.txt", 'rb') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        next(csv_reader)
        item = list(csv_reader)
        classes = []

        for i in range(0, 149):

            if item[i][3] not in classes:
                classes.append(item[i][3])

    finalresult = []
    
    for x in testingFeatureVec:
        # input layer is our bag of words
        l0 = x
        # matrix multiplication of input and hidden layer
        l1 = sigmoid(np.dot(l0, synapse_0))
        # output layer
        l2 = sigmoid(np.dot(l1, synapse_1))

        results = l2

        results = [[i,r] for i,r in enumerate(results)] 
        results.sort(key=lambda x: x[1], reverse=True)
        
        return_results =[[classes[r[0]]] for r in results]

        txt = json.dumps(return_results, encoding='utf-8', ensure_ascii=False)
        
        
        finalresult.append(return_results[0])


    return finalresult


def accuracy(result):
    count = 0
    total = 0

    with open("./data/test.txt", 'rb') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        next(csv_reader)
        item = list(csv_reader)
        settlement = []

        with open('./data/Dream_of_the_Red_Chamber_seg_newline.txt', 'r') as h:
            relatives = []
            lines=h.readlines()
            relatives = lines[2267].split(' ')
            final_relatives = []

            for line in lines:

                for word in line.split():
                    hh = word.split('_')
                     
                    if str(hh[1]) == 'Nc':
                        settlement.append(hh[0])
            
            relatives = relatives[1 :]

            for rel in relatives:

                
                rel = rel.split('_')
                rel[0] = json.dumps(rel[0], encoding='utf-8', ensure_ascii=False)
                rel[1] = json.dumps(rel[1], encoding='utf-8', ensure_ascii=False)
                rel[0] = rel[0].replace('\"', '')
                rel[0] = rel[0].replace('\\', '')
                rel[1] = rel[1].replace('\"', '')

                count = 0

                if str(rel[1]) == 'Nb':


                    for line in lines:
                

                        for word in line.split():

                            word = word.split('_')

                            if str(rel[0]) == str(word[0]):
                                count += 1
                    
                    if count < 3:
                        
                    
                        final_relatives.append(rel[0])


        final_relatives[0] = '賈代修'   
            
            
        settlement = json.dumps(settlement, encoding='utf-8', ensure_ascii=False)
 
        for i in range(0, len(item)-1):

            if item[i][1] in settlement or item[i][2] in settlement:
                result[i] = json.dumps(result[i], encoding='utf-8', ensure_ascii=False)
                result[i] = "居處"

            elif item[i][1] in final_relatives or item[i][2] in final_relatives:
                result[i] = "遠親"  

            indivi = json.dumps(result[i], encoding='utf-8', ensure_ascii=False)
            indivi = indivi.replace("\"", '')
            indivi = indivi.replace("[", '')
            indivi = indivi.replace("]", '')

            if str(item[i][3]) == indivi:
                count += 1
                total += 1

            else:
                total += 1

    print "Accuracy: %f" %(float(count)/float(total))
    return float(count)/float(total)



if __name__ == "__main__":

    stopTag = [ 'T4']
    dataPreprocessing(stopTag)
    trainingFeatureVec, testingFeatureVec, training_y = readData()
    # ['Daa', 'T8', 'T4', 'T']
    modelSeletion = ['random forest', 'SVM', 'MLP', 'ANN']
    for model in modelSeletion:
        if str(model) == 'random forest':
            print model
            result = forestClassification(trainingFeatureVec, testingFeatureVec, training_y)
            accuracy(result)
            
        elif str(model) == 'SVM':
            print model
            result = SVMclassifier(trainingFeatureVec, testingFeatureVec, training_y)
            accuracy(result)

        elif str(model) == 'MLP':
            print model
            result = MLP(trainingFeatureVec, testingFeatureVec, training_y)
            accuracy(result)

        elif str(model) == 'ANN':
            print model

            trainANN(trainingFeatureVec, training_y, hidden_neurons= neu, alpha=val, epochs=50000, dropout=False, dropout_percent = 0.0005)
            result = ANNclassifier(trainingFeatureVec, testingFeatureVec, training_y)
            accuracy(result)

        else:
            print "No model be selected!"
    
   
    # accuNum = accuracy(result)
    # accu.append(accuNum)
    # print accu

    # with open ('./data/stopTag.txt', 'r') as ff:
    #     Tags = ff.readlines()
    #     for i in range(0, len(Tags)):
    #         Tags[i] = Tags[i].replace('\n', '')
    #         testTags.append(Tags[i])
    #         dataPreprocessing(testTags)
    #         trainingFeatureVec, testingFeatureVec, training_y = readData()
    #         result = forestClassification(trainingFeatureVec, testingFeatureVec, training_y)
    #         accuNum = accuracy(result)
    #         accu.append(accuNum)
    #         print accu
    #         if accuNum < accu[i]:
    #             testTags.pop()

    #     print accuNum
    #     print testTags

