import os
import json

def parse_json(filename):
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
                        print "qs"
                        d.extend(q.get('sentences',[]))
                else:
                    d = file_data.get('sentences',[])
                pos = []
                tokens = []
                parse = []
                constituents = []
                deps_basic = []
                lemmas = []
                for sentence in d:
                    pos.append(sentence['pos'])
                    tokens.append(sentence['tokens'])
                    parse.append(sentence['parse'])
                    deps_basic.append(sentence['deps_basic'])
                    lemmas.append(sentence['lemmas'])
                    constituents.append(sentence['constituents'])
                elements.append([{'pos':pos,'tokens':tokens,'parse':parse,'constituents':constituents, 'deps_basic':deps_basic, 'lemmas':lemmas}])
        else:
            elements.append([])
        i += 1
    #elements[0] --> ans | elements[1] --> qs
    return elements[0], elements[1]
