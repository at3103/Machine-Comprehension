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
            gr_truths =[]
            qids = []
            with open(filename + ext) as json_file:
                file_data = json.load(json_file)
                d = []
                #gr_truths =[]
                if i == 1:
                    for q in file_data['questions']:
                        qids.append(q.get('id'))
                        qs_gr_truth =[]
                        d.extend(q.get('sentences',[]))
                        temp = q.get('answers',[])
                        for z in temp:
                            qs_gr_truth.append(' '.join(z))
                        gr_truths.append(qs_gr_truth)
                else:
                    d = file_data.get('sentences',[])
                pos = []
                tokens = []
                parse = []
                constituents = []
                deps_basic = []
                lemmas = []
                ners = []
                g_truths =[]
                for j in range(len(d)):
                    sentence = d[j]
                    pos.append(sentence.get('pos',[]))
                    tokens.append(sentence.get('tokens',[]))
                    parse.append(sentence.get('parse',[]))
                    deps_basic.append(sentence.get('deps_basic',[]))
                    lemmas.append(sentence.get('lemmas',[]))
                    ners.append(sentence.get('ner',[]))
                    constituents.append(sentence.get('constituents',[]))

                    if i:
                        g_truths.append(gr_truths[j])
                elements.append([{'pos':pos,'tokens':tokens,'parse':parse,'constituents':constituents, 'deps_basic':deps_basic, 'lemmas':lemmas, 'ground_truth':g_truths, 'id':qids, 'ners': ners}])
        else:
            elements.append([])
        i += 1
    #elements[0] --> ans | elements[1] --> qs
    return elements[0], elements[1]
