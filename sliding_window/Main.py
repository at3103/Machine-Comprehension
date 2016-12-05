from ProcessDataSquad import process
from Utils import *
import nltk
import operator
import io
import json
import math
import os
import BaseLine


dataset = "dev"
directory = "data"
totalF1 = 0
stories_count = 0
questionCount = 0
totalCorrect = 0
predictionsDict = {}
SRPredictionsDict = {}

def getAnswer(story):
	global questionCount
	global totalCorrect
	global totalF1
	global predictionsDict
	global SRPredictionsDict
	global stories_count

	stories_count = stories_count + 1

	passage = story.text.decode('utf8')
	SRPredictionsDict.update(BaseLine.getScores(story_id=story.story_id,text=passage,questionlist=story.questionlist,dataset = dataset))

if __name__ == '__main__':

	all_stories = process(dir=directory,dataset=dataset)
	questionCount = 0
	for i,story in enumerate(all_stories):
		questionCount+=len(story.questionlist)
		getAnswer(story)
		print "Processed "+str(i)+" files"
	print "Total Questions:"+str(questionCount)
	with io.FileIO("BaseLine_prediction.json", "w") as file:
		json.dump(SRPredictionsDict,file)
	'''
	for story in all_stories:
		story.display()
	'''

	#os.system("python data/evaluate-v1.1.py data/"+dataset+"-v1.1.json 6111_prediction_"+dataset)
	os.system("python data/evaluate-v1.1.py data/"+dataset+"-v1.1.json BaseLine_prediction.json")
	print "SRQCount: "+str(len(SRPredictionsDict))
#	print "QCount: "+str(len(predictionsDict))