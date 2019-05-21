#!/usr/bin/env python
# coding: utf-8

# In[109]:




import json
from pprint import pprint

def getData(data_file):
    questions = []
    dict_qaPairs = {}
    with open(data_file, 'r') as fp:
        data2 = json.load(fp)
        for qap in data2['qa_pairs']:
            #print(qap)
            dict_qaPairs[qap['ques']] = qap['ans']
            questions.append(qap['ques'])
            
    return questions, dict_qaPairs
            



# In[110]:



from nltk.corpus import wordnet as wn
import pandas as pd

def getSynonyms(word1):
    synonymList1 = []
    for data1 in word1:
        wordnetSynset1 = wn.synsets(data1)
        tempList1=[]
        tempList1.append(data1)
        #print(wordnetSynset1)
        
        for synset1 in wordnetSynset1:
            synLemmas = synset1.lemma_names()
            for i in range(len(synLemmas)):
                word = synLemmas[i].replace('_',' ')
                if word not in tempList1:
                    tempList1.append(word)
        
        spl = False            
        if(data1 == "what"):
            spl = True
            wordnetSynset1=wn.synsets("define")
        elif(data1 == "explain"):
            spl = True
            wordnetSynset1=wn.synsets("describe")            
        elif(data1 == "difference"):
            spl = True
            wordnetSynset1=wn.synsets("distinguish")
        elif(data1 == "advantage"):
            spl = True
            tempList1.extend(["pros","merits"])
            wordnetSynset1=wn.synsets("benefits")
        elif(data1 == "merits"):
            spl = True
            tempList1.extend(["pros","advantage"])
            wordnetSynset1=wn.synsets("advantage")            
        elif(data1 == "disadvantage"):
            spl = True
            tempList1.append("cons")
            wordnetSynset1=wn.synsets("demerits")
        elif(data1 == "demerits"):
            spl = True
            tempList1.append("cons","disadvantage")
            wordnetSynset1=wn.synsets("disadvantage")
            
        if(spl):    
            for synset1 in wordnetSynset1:
                synLemmas = synset1.lemma_names()
                for i in range(len(synLemmas)):
                    word = synLemmas[i].replace('_',' ')
                    if word not in tempList1:
                        tempList1.append(word)
        synonymList1.append(tempList1)
    return synonymList1

#print(getSynonyms(["demerits"]))

def getSynonymsIndex(word1, word2):
    len1 = len(word1)
    len2 = len(word2)
    
    if(len1+len2 == 0):
        return 0.5
    
    counter = 0
    d2 = getSynonyms(word2)
    #print(d2)
    
    for i in range(len(word1)):          
        for k in range(len(d2)):
            if word1[i] in d2[k]:
                counter += 1
                d2.pop(k)
                break

    syn_index = counter*2 / (len1 + len2)
    return syn_index

#print (getSynonymsIndex(word1, word2))     


def getMatchingIndex(word1, word2):
    len1 = len(word1)
    len2 = len(word2)
    
    if(len1+len2 == 0):
        return 0.5
    
    counter = 0
    for i in word1:
        for j in word2:
            if(i in j):
                counter+=1
                word2.remove(j)
                break
    match_index = counter*2 / (len1 + len2)
    return match_index    


# In[111]:


import nltk
import re
from nltk.stem import WordNetLemmatizer 
from pprint import pprint


def preProcess(sentence):
    lemmatizer = WordNetLemmatizer()
    sentence = re.sub(r'[^A-Za-z\s]+', ' ', sentence)
    sentence = re.sub(r'\s{2,}', ' ', sentence)


    #lemmatize
    tokens = nltk.word_tokenize(sentence)  
    #print(tokens)
    sentence_tmp = ""
    for word,pos in nltk.pos_tag(tokens):
        if type(word) == str:
            word = word.lower()
            
        if not pos == "DT":
            #print(pos)
            sentence_tmp+=(lemmatizer.lemmatize(word)) + " "  
    #print(sentence_tmp)
    return sentence_tmp

def extract_features(sentences):
    lemmatizer = WordNetLemmatizer()
    pre_features = [[],[]]

    counter=-1
    
    #print(sentences)
    for sentence in sentences:
        counter+=1

        sentence = preProcess(sentence)

        nouns = [] #empty to array to hold all nouns
        possesives = []
        adverbs = []
        adjectives = []
        verbs = []
        digrms = []
        questionWords = []

        tokens = nltk.word_tokenize(sentence)
        
        for word,pos in nltk.pos_tag(tokens):
            if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
                nouns.append(word)
            elif (pos == 'WDT' or pos == 'WP' or pos == 'WRB'):
                questionWords.append(word)                 
            elif (pos == 'POS'):
                possesives.append(word)
            elif (pos == 'RB' or pos == 'RBR' or pos == 'RBS'):
                adverbs.append(word)
            elif (pos == 'JJ' or pos == 'JJR' or pos == 'JJS'):
                adjectives.append(word)                
            elif (pos == 'VB' or pos == 'VBD' or pos == 'VBG' or pos == 'VBN' or pos == 'VBP' or pos == 'VBZ'):
                verbs.append(word)   

        
        bigrm = nltk.bigrams(tokens)
        for i in bigrm:
            digrms.append(" ".join(i))
        #print(digrms)
        
        
        from nltk.corpus import stopwords  
        stop_words = stopwords.words('english')
        list_words = [word for word in tokens if word not in stop_words]

        pre_features[counter].append(nouns) #0
        pre_features[counter].append(possesives) #1
        pre_features[counter].append(adverbs) #2
        pre_features[counter].append(adjectives) #3
        pre_features[counter].append(verbs) #4
        pre_features[counter].append(digrms) #5
        pre_features[counter].append(questionWords) #6
        pre_features[counter].append(list(list_words)) #7



    #print(pre_features[0])
    #print(pre_features[1])

    features = []

    for x in [0,2,3,4,6,7]:
        features.append(getSynonymsIndex(pre_features[0][x], pre_features[1][x]))

    for x in [1,5]:
        features.append(getMatchingIndex(pre_features[0][x], pre_features[1][x]))

    return(features)


# In[112]:


def loadModel(saved_model):
    import pickle
    #pickle.dump(modelLR, open(filename, 'wb'))
    # load the model from disk
    loaded_model = pickle.load(open(saved_model, 'rb'))
    
    return loaded_model


def predictOutput(q1,q2,saved_model):
    
    model = loadModel(saved_model)
    
    q=[q1,q2]
    #print("hku")
    test_x = [extract_features(q)]
    test_x = pd.DataFrame(test_x, columns = ['nouns', 'adverbs', 'adjectives',
                                   'verbs', 'words', 'possesives',
                                   'digrms', 'questionWords'])  
    
    #print(model.predict_proba(test_x)[0][1])
    
    #print(model.predict_proba(test_x)[0][1] >= 0.80)
    #return (model.predict(test_x))
    return ( model.predict_proba(test_x)[0][1] )


def predictQuestion(query,questions,top,saved_model):
    qno_prob = {}
    for ques in questions:
        qno_prob[ques] = predictOutput(query, ques,saved_model)

    sorted_x = sorted(qno_prob.items(), key=lambda kv: kv[1])
    #print(sorted_x)
    return(sorted_x[-1:-top-1:-1])

def predictAnswer(predictedQuestions,dict_qaPairs):
    result = {}
    for ques in predictedQuestions:
        result[str(ques[1])] = str(ques[0]) + " " + str(dict_qaPairs[ques[0]])
    return result
    #predictedQuestion(q1,questions=questions)
    #dict_qaPairs[predicted[0][0]]
    


# In[114]:



def getAns(query,
           top=1,
           data_file = 'ques_ans4.json',
           saved_model = 'finalizedLR_model.sav' ):
        
    questions, dict_qaPairs = getData(data_file)
    predictedQuestions = predictQuestion(query,questions,top,saved_model)
    return predictAnswer(predictedQuestions,dict_qaPairs)


# In[115]:

if __name__ == "__main__":	
	query = "define computer ?"
	print(getAns(query = query, top= 2))


# In[ ]:





# In[ ]:




