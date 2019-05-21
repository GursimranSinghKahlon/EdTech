



def loadModel():
	import pickle
	filename = 'finalized_model.sav'
	pickle.dump(modelLR, open(filename, 'wb'))
	 
	# load the model from disk
	loaded_model = pickle.load(open(filename, 'rb'))
	
	return loaded_model

q1 = "what is computer?"
q2 = "what are advantage of computer?"

def predictOutput(q1,q2):
	
	model = loadModel()
	
    q=[q1,q2]
    #print("hku")
    test_x = [extract_features(q)]
    test_x = pd.DataFrame(test_x, columns = ['nouns', 'adverbs', 'adjectives',
                                   'verbs', 'words', 'possesives',
                                   'digrms', 'questionWords'])  
    
    print(model.predict_proba(test_x)[0][1])
    
    print(model.predict_proba(test_x)[0][1] >= 0.80)
    return (model.predict(test_x))



#print(predictOutput(q1,q2,modelLR))


'''
print("done")


result = predictOutput(q1,q2,loaded_model)
print(result)
