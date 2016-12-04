from stanford_corenlp_pywrapper import CoreNLP
from fuzzywuzzy import fuzz
from Utils import *
import nltk
import random
import os 
import math
import string
from nltk.tokenize import wordpunct_tokenize
from stat_parser import Parser, display_tree
from stat_parser.tokenizer import PennTreebankTokenizer
from collections import Counter
import string
import re
import argparse
import json
import sys
import io

# parser = Parser()
# tokenizer = PennTreebankTokenizer()
# sentence = "To the Christian Nobility of the German Nation, On the Babylonian Captivity of the Church, and On the Freedom of a Christian."
# tree = parser.parse(sentence)
# constituents = list()
# for s in tree.subtrees():
# 	display_tree(s)
# 	constituents.append(s.leaves())
# print constituents
EMCount = 0
QCount = 0
dir_path = os.path.dirname(os.path.realpath(__file__))

p = CoreNLP(configdict={
		    'annotators': "tokenize,ssplit,pos,parse",
		    'parse.model': 'edu/stanford/nlp/models/srparser/englishSR.ser.gz',
		    'tokenize.language': 'English',
		    #'tokenize.verbose': True,
		    'tokenize.options': 'strictTreebank3=true'},
		    #'tokenize.verbose':True},  
		    corenlp_jars=[dir_path+"/corenlp/*", dir_path+"/corenlp/stanford-english-corenlp-2016-10-31-models.jar"])

stopWords = set(nltk.corpus.stopwords.words('english'))
specialChar = set(['?',',','.',';','!','-'])
stopWords.update(specialChar)

def getScores(story_id,passage,questionlist):
	SRPredictionsDict = getStanfordParserResults(story_id,passage,questionlist)
	#SRPredictionsDict = getPyStatParserResults(passage,questionlist)
	return 	SRPredictionsDict

def getPyStatParserResults(passage,questionlist): 
	SRPredictionsDict = {}
	sentences = nltk.sent_tokenize(passage)
	for sentence in sentences:
		sentence_tokenizer = ' '.join(tokenizer(sentence))
		tree = parser.parse(sentence)
		constituents = set()
		for s in tree.subtrees():
			#print type(s.leaves())
			constituents.add(s.leaves())
		for q in questionlist:
			qset = set(normalize(q.question)) - stopWords
			q_bi = list(nltk.bigrams(nltk.word_tokenize(q.question)))
			score = -1
			answerList = []
		for constituent in constituents:
			remaining = sentence_tokenizer - constituent

def getStanfordParserResults(story_id,passage,questionlist): 
	SRPredictionsDict = {}
	outputFilePath = dir_path+"/parsed/"+story_id+".json"
	
	if os.path.isfile(fname):
	 	p = CoreNLP(configdict={
		    'annotators': "tokenize,ssplit,pos,parse",
		    'parse.model': 'edu/stanford/nlp/models/srparser/englishSR.ser.gz',
		    'tokenize.language': 'English',
		    #'tokenize.verbose': True,
		    'tokenize.options': 'strictTreebank3=true'},
		    #'tokenize.verbose':True},  
		    corenlp_jars=[dir_path+"/corenlp/*", dir_path+"/corenlp/stanford-english-corenlp-2016-10-31-models.jar"])

		SRParsed = p.parse_doc(passage)
		with io.FileIO(outputFilePath, "w") as file:
			json.dump(SRParsed,file)
	
	else :
		SRParsed = json.load()outputFilePath
	
	SRParsed = SRParsed.get("sentences")
	
	passage_tokens = []
	
	for result in SRParsed:
		passage_tokens.extend(normalize(' '.join(result.get("tokens"))))
	
	for q in questionlist:
		qset = set(normalize(q.question)) - stopWords
		q_bi = list(nltk.bigrams(nltk.word_tokenize(q.question)))
		#print "Q:" + q.question
		score = -1
		answerList = []
		all_consti = []
			
		for result in SRParsed:
			tokens = normalize(' '.join(result.get("tokens")))
			#print ' '.join(tokens)
			for constituents in result.get("deps_basic"):
				start = 0
				end = len(tokens)-1
				#if(constituents[1]<0):
				#	end = constituents[2]+1
				if(constituents[1]>constituents[2]):
					start = constituents[2]+1 
					end = constituents[1]+1
				else:
					start = constituents[1]+1 
					end = constituents[2]+1
				
				constituent = tokens[start:end]
				all_consti.append(constituent)
				#print "Consti"+' '.join(constituent)
				remaining = []
				remaining_bi = []
				if(start>=0):
					remaining.extend(tokens[:start]) 
					remaining_bi.extend(list(nltk.bigrams(tokens[:start])))
				remaining.extend(tokens[end:])
				remaining_bi.extend(list(nltk.bigrams(tokens[end:])))
				
				cur_score =  len(qset & set(remaining))
				bi_match = [bigram for bigram in q_bi if bigram in remaining_bi]
				
				# for bigram in remaining_bi:
				# 	for bigram2 in q_bi:
				# 		#print bigram, bigram2
				# 		if bigram == bigram2:
				# 			bi_match.append(bigram)
				# 			break
				cur_score = cur_score + len(set(bi_match))

				#print " score" + str(cur_score) + ' '.join(constituent)
				if(score<cur_score):
					score = cur_score	
					answerList = list()
					answerList.append(constituent)
				elif(score==cur_score):
					answerList.append(constituent)
		print q.properties["id"]+ ":" + q.question
		print "Ground Truths" + str(q.options)
		print "Answers:" 
		for answer in answerList:
			print ' '.join(answer)
		print "Constituents:"
		for consti in all_consti:
			print ' '.join(consti)  

		finalAnswer = slidingWindow(qset = qset, ansList = answerList,passage_tokens=passage_tokens)
		
		SRPredictionsDict[q.properties["id"]] = ' '.join(finalAnswer)
	return 	SRPredictionsDict

def slidingWindow(qset,ansList,passage_tokens):
	final_answer_list = []
	final_score = 0
	for ans in ansList:
		qaset = qset
		qaset.update(set(ans))
		qaset = qaset - stopWords
		score = 0
		for j in xrange(len(passage_tokens)):
			cur_score = 0
			for w in xrange(len(qaset)):
				if (j+w<len(passage_tokens)):
					if passage_tokens[j+w] in qaset:
						cur_score+=InverseCount(passage_tokens,passage_tokens[j+w])
			if(score<cur_score):
				score = cur_score
		#print "Answer :" + ' '.join(ans) + " Score:" + str(score)
		if(final_score<score):
			final_score = score
			final_answer_list = list()
			final_answer_list.append(ans)
		elif(final_score==score):
			final_answer_list.append(ans)

	if(len(final_answer_list)>1):
		randomIndex = random.randint(0,len(final_answer_list)-1)
	
	randomIndex = 0
	answer = [word for word in final_answer_list[randomIndex] if word not in specialChar]
	return  answer

def InverseCount(passage_tokens, word):
	count = passage_tokens.count(word)
	if(count!=0) :
		return math.log(1+(1/count))
	return 0

def checkEM(q,answerList):
	global EMCount
	global QCount
	ground_truths = [option['text'] for option in q.options]
	QCount = QCount + 1

	for answer in answerList:
		emscore = metric_max_over_ground_truths(exact_match_score, ' '.join(answer), ground_truths)
		if(emscore>0):
			EMCount = EMCount + 1
			print "Temp EMCount:"+str(EMCount)
			print answer,ground_truths
			break
	print "Temp QCount:"+str(QCount)
			
def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)

    if metric_fn == exact_match_score and max(scores_for_ground_truths)>0:
        print("EM",prediction,ground_truths) 

    return max(scores_for_ground_truths)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
