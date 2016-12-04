import os
import json

def parse_data(path):
    for (root, files, filenames) in os.walk(path):
        for file in filenames:
            parse_json(os.path.join(root, file))

def parse_json(filename):
    # print filename
    with open(filename) as json_file:
        file_data = json.load(json_file)
        sentences = file_data['sentences']
        for sentence in sentences:
            pos = sentence['pos']
            tokens = sentence['tokens']
            parse = sentence['parse']
            constituents = sentence['constituents']

"""
Read in processed data from JSON, create features, save to CSV
"""
if __name__ == '__main__':
	parse_data("../data/processed")