import Utils as utils

class Story:
	
	def __init__(self,story_id,properties,text,questionlist):
		self.story_id = story_id
		self.properties = properties
		self.text = text
		self.questionlist = questionlist
		self.normalized_text = utils.normalize(text)

	def display(self):
   		print "ID : {0}".format(self.story_id)
   		print "Properties : "
   		print "----------------------------------------------"
   		for key in self.properties.iterkeys():
   			print "{0} : {1}".format(key,self.properties[key])
   		print "----------------------------------------------" 
   		print "Story : "
   		print "----------------------------------------------"
   		print self.text
   		print "----------------------------------------------"
   		for i,question in enumerate(self.questionlist,1):
   			print "Question  {0} - ".format(i)
   			question.display()