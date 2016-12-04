import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import wordpunct_tokenize

#sanitize = RegexpTokenizer(r'[a-z0-9]\w*[\']*[a-z0-9]+').tokenize
sanitize = wordpunct_tokenize


def normalize(text):
	text = text.lower()
	words =  sanitize(text)
	new_words = ""
	new_list = []

	for w in words:
		word = w.rstrip("\'").rstrip("\'").lstrip("\'").lstrip("\'")
		new_list.append(word)
	return new_list
        

#S = "What's up? How are you? / ?"
S = "Bhavana says \"what's up\""
#S = "I haven't doesn't"
#S.replace("'", "a")
#print S
#one,two = normalize(S)