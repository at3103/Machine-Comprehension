import os
import re
from Question import Question
from Story import Story

def process(dir,size=160,version='dev'):
	file_name = 'mc'+str(size)+'.'+version+'.tsv'
	file = open(os.path.join(dir,file_name),'r')
	text_stories = file.readlines()
	all_stories = []
	for item in text_stories:
		story = parse_story(item)
		all_stories.append(story)

	return all_stories

def parse_story(text_stories):
	tab_pattern = re.compile("[^\t]+")

	parts = tab_pattern.findall(text_stories)
	story_id = parts[0]
	properties = dict((key.strip(),val.strip()) for key,val in (item.split(":") for item in parts[1].strip().split(";")))
	text = parts[2]
	questionlist = []
	i=3
	while i < len(parts) :
		question = parse_question(parts[i:i+5])
		questionlist.append(question)
		i = i+5
	story = Story(story_id,properties,text,questionlist)
	return story

def parse_question(parts):
	question = Question(parts[0],parts[1:])
	return question

"""
Place data in folder "data" in the current working directory to run this file standalone
"""
if __name__ == '__main__':
	all_stories = process("data")
	for story in all_stories:
		story.display()