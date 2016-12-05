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
def contextParser(story_id,text,dataset): 
	
	SRPredictionsDict = {}
	outputFilePath = dir_path+"/parsed_"+dataset+"/"+story_id+".json"
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

	SRParsedAllQ = {}
	if os.path.isfile(outputFilePath):
		print "Skipping "+story_id+". "+ outputFilePath+" already present"
		return
	SRParsed = p.parse_doc(text)
	sentences = SRParsed.get("sentences")
	for sentence in sentences:
		del sentence["char_offsets"] 
	with io.FileIO(outputFilePath, "w") as file:
		json.dump(SRParsed,file)
	
	
if __name__ == '__main__':
	dataset = "train"
	directory = "data"
	all_stories = process(dir=directory,dataset=dataset)
	questionCount = 0
	for i,story in enumerate(all_stories):
		questionCount+=len(story.questionlist)
		contextParser(story.story_id,story.text,dataset = dataset)
		