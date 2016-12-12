from ProcessDataSquad import process
from stanford_corenlp_pywrapper import CoreNLP
import os 
import json
import sys
import io

EMCount = 0
QCount = 0
dir_path = os.path.dirname(os.path.realpath(__file__))

p = None

def questionParser(story_id,questionlist,dataset): 
	SRPredictionsDict = {}
	outputFilePath = dir_path+"/parsed_"+dataset+"_q/"+story_id+"_q"+".json"
	SRParsedAllQ = {}
	SRParsedAllQ["questions"] = []  
	if os.path.isfile(outputFilePath):
		print "Skipping "+story_id+". "+ outputFilePath+" already present"
		return
	global p
	if p == None:
		p = CoreNLP(configdict={
			'annotators': 'tokenize,ssplit,pos,lemma,ner,parse',
			'tokenize.language': 'English',
			'tokenize.options': 'strictTreebank3=true',
			'coref.algorithm' : 'neural'
			},
			corenlp_jars=[dir_path+"/corenlp/*", dir_path+"/corenlp/stanford-english-corenlp-2016-10-31-models.jar"])

	for q in questionlist:
		SRParsed = p.parse_doc(q.question)
		SRParsed['id'] = q.properties['id']
		sentences = SRParsed.get("sentences")
		for sentence in sentences:
			del sentence["char_offsets"]
		SRParsed['answers'] = []
		for ground_truth in q.options:
			AnswerParsed = p.parse_doc(ground_truth['text'])
			SRParsed['answers'].append(AnswerParsed.get("sentences")[0].get("tokens"))
		SRParsedAllQ["questions"].append(SRParsed)
	with io.FileIO(outputFilePath, "w") as file:
		json.dump(SRParsedAllQ,file)
	
	
if __name__ == '__main__':
	dataset = "dev"
	directory = "data"

	all_stories = process(dir=directory,dataset=dataset)
	questionCount = 0
	for i,story in enumerate(all_stories[:1]):
		questionCount+=len(story.questionlist)
		questionParser(story.story_id,story.questionlist,dataset=dataset)