class Question:
	
	def __init__(self,question,options,properties=None):
		self.question = question
		self.options = options
		self.properties = properties

	def display(self):
		print self.question
		for i,option in enumerate(self.options,1):
			print "{0}.{1} ".format(i,option)
		