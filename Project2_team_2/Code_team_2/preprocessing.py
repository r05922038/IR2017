
# coding: utf-8

# In[59]:

stopwords = [line.strip() for line in open('stopwords_pos_1.txt', 'r')]+['老','，', '。', '、', '；', '：', '？', '「', '」','『','』','●','…']
#stopwords = ['老','，', '。', '、', '；', '：', '？', '「', '」','『','』','●','…']


# In[21]:

dict_={'士隱':'甄士隱','岫煙':'邢岫煙','紹祖':'孫紹祖','迎春':'賈迎春','金桂':'夏金桂','寶玉':'賈寶玉',
      '寶釵':'薛寶釵','守中':'李守中','可卿':'秦可卿','之孝':'林之孝','世仁':'卜世仁','紫英':'馮紫英',
       '尚榮':'賴尚榮','元春':'賈元春','惜春':'賈惜春','探春':'賈探春','黛玉':'林黛玉','湘雲':'史湘雲',
      '熙鳳':'王熙鳳','若錦':'張若錦','亦華':'趙亦華','湘蓮':'柳湘蓮','子騰':'王子騰','子勝':'王子勝',
      '德全':'邢德全','巧姐':'賈巧姐','天棟':'趙天棟','友士':'張友士','代善':'賈代善','代儒':'賈代儒',
       '代化':'賈代化','如海':'林如海','可卿':'秦可卿','寶琴':'薛寶琴','自芳':'花自芳','秋芳':'傅秋芳',
      '文翔':'金文翔','世仁':'卜世仁','雨村':'賈雨村','繼宗':'牛繼宗','瑞文':'陳瑞文','孝康':'侯孝康',
      '曉明':'侯曉明','光珠':'石光珠','子寧':'蔣子寧','建輝':'戚建輝','代修':'賈代修'}


# In[51]:

#李嬸娘,趙姨娘,尤老娘,趙嬤嬤,李嬤嬤,夏奶奶,周姨娘
dict2={'嬸娘':set(['李']),'姨娘':set(['趙','周']),'老娘':set(['尤']),
       '嬤嬤':set(['趙','李']),'奶奶':set(['夏']),'太君':set(['史','史氏']),
      '母':set(['賈']),'夫人':set(['邢','王']),'姐':set(['四']),'二姐':set(['尤']),
       '三姐':set(['尤']),'姨媽':set(['薛']),'婆子':set(['何'])}


# In[57]:

'''
posSW="Nh,Ps,DE,Pd,Pe,Dfa,Po,Dfb,Dl,Dj,Daa,Dh,Di,Dg,T4,T8,T3,T5,T,SHI,Dk,I"
posSW=posSW.split(',')
'''


# In[60]:

#posset=dict()
#sw_set=set()
testfile='Dream_of_the_Red_Chamber_seg'
f = open(testfile+'_preprocessing2.txt', 'w')
with open(testfile+'.txt', 'r') as file_:
    for line in file_:
        new_words=[]
        pre=''
        for word in line.split():
            word,pos=word.split('_')
            #if pos not in posset:
            #    posset[pos]=[]
            #posset[pos].append(word)
            #if pos in posSW:
                #sw_set.add(word)
            
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
        
        f.write(" ".join(new_words)+'\n')
f.close()


# In[46]:

'''
sw = open('stopwords_pos_1.txt', 'w')
for w in sw_set:
    sw.write(w+'\n')
sw.close()
'''


# In[41]:

'''
for k in posset:
    if k not in posSW:
        continue
    print k
    for w in posset[k]:
        print w
'''


# In[58]:

'''
#posset=dict()
sw_set=set()
testfile='Dream_of_the_Red_Chamber_seg'
f = open('loc.txt', 'w')
with open(testfile+'.txt', 'r') as file_:
    for line in file_:
        new_words=[]
        pre=''
        for word in line.split():
            word,pos=word.split('_')
            #if pos not in posset:
            #    posset[pos]=[]
            #posset[pos].append(word)
            #if pos in posSW:
                #sw_set.add(word)
            
            if pos not in posSW and not word.isdigit():
                if word in dict_:
                    word=dict_[word]
                    new_words.append(word)
                else:
                    if word in dict2 and pre in dict2[word]:
                        new_words[len(new_words)-1]=pre+word
                    else:
                        new_words.append(word)
                pre=word
        
        f.write(" ".join(new_words)+'\n')
f.close()
'''


# In[62]:

#posset=dict()
#sw_set=set()
testfile='Dream_of_the_Red_Chamber_seg'
f = open('loc.txt', 'w')
with open(testfile+'.txt', 'r') as file_:
    for line in file_:
        new_words=[]
        pre=''
        for word in line.split():
            word,pos=word.split('_')
            if pos=='Nc':
                f.write(word+'\n')
f.close()


# In[ ]:



