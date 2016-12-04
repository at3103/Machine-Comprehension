import os
import re
from Question import Question
from Story import Story
import json

def process(dir,dataset='dev',version='v1.1'):
	file_name = dataset+'-'+version+'.json'
	file = open(os.path.join(dir,file_name),'r')
	text_stories = json.load(file)
	all_stories = []
	for item in text_stories['data']:
		new_stories = parse_story(item)
		all_stories.extend(new_stories)
	return all_stories

def parse_story(text_stories):
	story_id = text_stories['title'].encode('utf8')
	contexts = text_stories['paragraphs']
	stories = []
	for i,context in enumerate(contexts):
		text = context['context'].encode('utf8')
		questionlist = parse_questions(context['qas'])
		properties = {}
		story = Story(story_id+"_"+str(i+1),properties,text,questionlist)
		stories.append(story)
	return stories

def parse_questions(qas):
	questionlist = []
	for qa in qas:
		questionlist.append(Question(qa['question'].encode('utf8'),qa['answers'],{'id':qa['id'].encode('utf8')}))
	return questionlist

"""
Place data in folder "data" in the current working directory to run this file standalone
"""
if __name__ == '__main__':
	all_stories = process("data")
	print len(all_stories)
	# for story in all_stories:
	# 	story.display()