import os
import json

def parse_json(filename):
    # print filename
    elements = []
    ext = '.json'
    i = 0

    while i < 2:
        if i == 1:
            filename = filename + '_q'
        if os.path.isfile(filename + ext):
            with open(filename + ext) as json_file:
                file_data = json.load(json_file)
                d = []
                if i == 1:
                    for q in file_data['questions']:
                        d.append(q['sentences'])
                else:
                    d.append(file_data['sentences'])
                pos = []
                tokens = []
                parse = []
                constituents = []
                for sentences in d:
                    for sentence in sentences:
                        pos.append(sentence['pos'])
                        tokens.append(sentence['tokens'])
                        parse.append(sentence['parse'])
                        constituents.append(sentence['constituents'])
                    elements.append([pos,tokens,parse,constituents])
        else:
            elements.append([])
        i += 1
    #elements[0] --> ans | elements[1] --> qs
    return elements[0], elements[1]
