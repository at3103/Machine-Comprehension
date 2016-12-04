import os
import json
from feature_extractor import *

# def parse_json(file):
# 	with open(file + '.json') as req_data_file:
# 		data_json = json.load(req_data_file)


def parse_data(path):
	# i =0
	print "hey"
	for (root, files, filenames) in os.walk(path):


		for file in filenames:
			# if (i == 2):
			# 	break
			file = os.path.splitext(file)[0]
			if file.find('_q') == 0:
				print file
				continue
			ans_features, q_features = parse_json(os.path.join(root, file))
			# print 'Answer_features :', ans_features, "Question_features", q_features
			# i += 1



"""
Read in processed data from JSON, create features, save to CSV
"""
if __name__ == '__main__':
	parse_data("../data/processed")
