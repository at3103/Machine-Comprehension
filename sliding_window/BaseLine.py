from ContextParsing import contextParser
import os 
import json
import sys
import io
import math
import random
from nltk import bigrams
from nltk import corpus

dir_path = os.path.dirname(os.path.realpath(__file__))

stopWords = set(corpus.stopwords.words('english'))
specialChar = set(['?',',','.',';','!','-','-LRB-','-RRB'])
stopWords.update(specialChar)

def getScores(story_id,text,questionlist,dataset):
	SRPredictionsDict = getStanfordParserResults(story_id,text,questionlist,dataset=dataset)
	#SRPredictionsDict = getPyStatParserResults(passage,questionlist)
	return 	SRPredictionsDict

def getStanfordParserResults(story_id,text,questionlist,dataset): 
	SRPredictionsDict = {}

	#Very important for baseline. To have processed json docs
	contextFilePath = dir_path+"/processed/"+story_id+".json"
	if not os.path.isfile(contextFilePath):
		#contextParser(story_id=story_id,text=text,dataset = dataset)
		print "Error. No file present. ",contextFilePath

	with open(contextFilePath) as data_file:
		contextParsed = json.load(data_file)
	
	contextParsed = contextParsed.get("sentences")

	passage_tokens = []
	
	for sentence in contextParsed:
		passage_tokens.extend(sentence.get("tokens"))

	
	questionFilePath = dir_path+"/parsed_"+dataset+"_q/"+story_id+"_q"+".json"
	if not os.path.isfile(questionFilePath):
		questionParser(story_id=story_id,questionlist=questionlist,dataset = dataset)
		
	with open(questionFilePath) as question_file:
		questionParsed = json.load(question_file)
	questionParsed = questionParsed.get("questions")
	
	for q in questionParsed:
		#print q.get("id")
		q_tokens = []
		for sentence in q.get("sentences"):
			q_tokens.extend(sentence.get("tokens")) 
		q_bi = list(bigrams(q_tokens))
		qset = set(q_tokens) - stopWords
		score = -1
		answerList = []
		all_consti = []
			
		for sentence in contextParsed:
			tokens = sentence.get("tokens")
			for constituent in sentence.get("constituents"):
				
				start = constituent.get("start")
				end = constituent.get("end")
				
				constituent["text_tokens"] = []
				for i in xrange(start,end+1):
					constituent["text_tokens"].append(tokens[i])

				constituent_str = constituent.get("text")
				remaining_tokens = []
				remaining_bi = []
				
				remaining_tokens.extend(tokens[:start]) 
				remaining_bi.extend(list(bigrams(tokens[:start])))
				remaining_tokens.extend(tokens[end+1:])
				remaining_bi.extend(list(bigrams(tokens[end+1:])))
				
				cur_score =  len(qset & set(remaining_tokens))
				bi_match = [bigram for bigram in q_bi if bigram in remaining_bi]
				
				cur_score = cur_score + len(set(bi_match))
				#print "Constituent:"+constituent_str+":"+str(cur_score)+" Remaining:"+str(remaining_tokens)
				if(score<cur_score):
					score = cur_score	
					answerList = list()
					answerList.append(constituent)
				elif(score==cur_score):
					answerList.append(constituent)
		#print "Answers:"
		#print [ans.get("text") for ans in answerList]
		finalAnswer = slidingWindow(qset = qset, ansList = answerList,passage_tokens=passage_tokens)
		
		SRPredictionsDict[q.get("id")] = finalAnswer
	return 	SRPredictionsDict

def slidingWindow(qset,ansList,passage_tokens):
	final_answer_list = []
	final_score = 0
	for ans in ansList:
		constituent_length = ans.get("end") - ans.get("start") + 1
		qaset = qset
		qaset.update(set(ans.get("text_tokens")))
		score = 0
		for j in xrange(len(passage_tokens)):
			cur_score = 0
			for w in xrange(len(qaset)):
				if (j+w<len(passage_tokens)):
					if passage_tokens[j+w] in qaset:
						cur_score+=InverseCount(passage_tokens,passage_tokens[j+w])
			if(score<cur_score):
				score = cur_score
		if(final_score<score):
			final_score = score
			final_answer_list = list()
			final_answer_list.append(ans)
		elif(final_score==score):
			final_answer_list.append(ans)

	if(len(final_answer_list)>1):
		randomIndex = random.randint(0,len(final_answer_list)-1)
	else:
		randomIndex = 0
	answer = ansList[randomIndex].get("text")
	return answer

def InverseCount(passage_tokens, word):
	count = passage_tokens.count(word)
	if(count!=0) :
		return math.log(1+(1/count))
	return 0