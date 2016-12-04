from ProcessDataSquad import process
from Utils import *
import nltk
import operator
import io
import json
import math
from nltk.tokenize import RegexpTokenizer
import nltk.data
import os
import SRWrapper


dataset = "dev"
directory = "data"

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

stopWords = set(nltk.corpus.stopwords.words('english'))
totalF1 = 0
stories_count = 0
questionCount = 0
totalCorrect = 0
predictionsDict = {}
SRPredictionsDict = {}

def findIndices(s, passage):
	indices = []
	for sWord in s:
		indices_sWord = [i for i, x in enumerate(passage) if x == sWord]
		indices.extend(indices_sWord)

	return indices

def wordDistance(question, answer, passage):
	s_q = set(question) & set(passage)
	s_q.discard(stopWords)
	s_a = set(answer) & set(passage)
	s_a.discard(set(question))
	s_a.discard(stopWords)

	if(len(s_q) == 0 or len(s_a) == 0):
		return 1

	# Find minimum distance between word in s_q and word in s_a
	# Find indices of all words in s_q, s_a
	s_qIndices = findIndices(s_q, passage)
	s_aIndices = findIndices(s_a, passage)

	# Find minimum distance between any two indices in the two lists
	minDist = -1
	for qIndex in s_qIndices:
		for aIndex in s_aIndices:
			if(minDist == -1 or abs(qIndex - aIndex) < minDist):
				minDist = abs(qIndex - aIndex)
				#print float(minDist /((len(passage) - 1)))

	return float(minDist / (len(passage) - 1))


def getAnswer(story):
	global questionCount
	global totalCorrect
	global totalF1
	global predictionsDict
	global SRPredictionsDict
	global stories_count

	stories_count = stories_count + 1

	passage = story.text.decode('utf8')
	SRPredictionsDict.update(SRWrapper.getScores(story.story_id,passage,story.questionlist))
	"""	
	for q in story.questionlist: 
		
		questionCount = questionCount + 1

		answers = nltk.sent_tokenize(passage)
		#answers = sent_detector.tokenize(passage.strip())

		answers = [normalize(sentence) for sentence in answers]
		question_words = normalize(q.question)
		candidateAnswers = []
		for answer in answers:
			if set(answer) & set(question_words):
				candidateAnswers.append(set(answer))
			
		score = {}
		passage_words = normalize(passage)
		
		setQA = set(question_words) - stopWords
		for ca in candidateAnswers:
			wordDistance(question_words, ca, passage_words)
			caScore = 0
			for word in ca:
				if (word in setQA):
						#print passage_words[i+j]
						caScore = caScore + 1 
			sent = ' '.join(ca)
			score[sent] = caScore

		if len(score) >= 1:
			prediction =  max(score.iteritems(), key=operator.itemgetter(1))[0]
			predictionsDict[q.properties["id"]] = prediction
			prediction = set(normalize(prediction))
			F1 = 0
			for option in q.options:
				option_words = set(normalize(option['text']))
				overlap = float(len(option_words & prediction))
				precision = overlap/len(prediction)
				recall = overlap/len(option_words)
				if (len(option_words - prediction)==0):
					totalCorrect = totalCorrect + 1
				if precision == recall == 0:
					cur_F1 = 0
				else:
					cur_F1 = 2*precision*recall/ (precision+recall)
				if(cur_F1 > F1):
					F1 = cur_F1

			totalF1 = totalF1 + F1
	"""
def InverseCount(passage, word):
	count = passage.count(word)
	if(count!=0) :
		return math.log(1+(1/count))
	return 0

if __name__ == '__main__':

	all_stories = process(dir=directory,dataset=dataset)
	questionCount = 0
	for story in all_stories[:2]:
		questionCount+=len(story.questionlist)
		getAnswer(story)
	print "Total Questions:"+str(questionCount)
	# print stories_count
	# print questionCount
	# print totalF1/questionCount
	# print "Number of predictions: " , len(predictionsDict)

	# with io.FileIO("6111_prediction_"+dataset, "w") as file:
	# 	json.dump(predictionsDict,file)

	with io.FileIO("6111_prediction_"+dataset+"_SR", "w") as file:
		json.dump(SRPredictionsDict,file)
	'''
	for story in all_stories:
		story.display()
	'''

	#os.system("python data/evaluate-v1.1.py data/"+dataset+"-v1.1.json 6111_prediction_"+dataset)
	#os.system("python data/evaluate-v1.1.py data/"+dataset+"-v1.1.json 6111_prediction_"+dataset+"_SR")
	print "SRQCount: "+str(len(SRPredictionsDict))
#	print "QCount: "+str(len(predictionsDict))