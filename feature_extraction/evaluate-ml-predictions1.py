""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
from os import listdir
from os.path import isfile, join
import csv
from collections import defaultdict
import ast

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        #print(prediction,ground_truth)
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def evaluate(ground_truths_dict, predictions):
    f1 = exact_match = total = 0
    for qa in ground_truths_dict.keys():
        total += 1
        if qa not in predictions:
            message = 'Unanswered question ' + qa + \
            ' will receive score 0.'
            print(message, file=sys.stderr)
            continue
        ground_truths = ground_truths_dict[qa]
        prediction = predictions[qa]
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
    print(total)
    print(exact_match)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}

predictions_file_path = "../data/predictions/" 

def get_max_predictions(prediction_dict):
    max_predict = defaultdict(str)
    for key in prediction_dict:
        max = ['N','']
        predictions = prediction_dict[key]
        for prediction in predictions:
            if prediction[0] == 'Y':
                max = prediction
            elif prediction[0] == 'M':
                max[1] += ' '+prediction[1]
        max_predict[key] = max[1]
    return max_predict

def evaluate_ml_result(file):

    print("Evaluating {0} file".format(file))
    
    data = [] 
    predictions_qid = defaultdict(list)
    qid_ground_truths = defaultdict(str) 
    with open(predictions_file_path+file, 'rb') as csvfile:
        inputreader = csv.reader(csvfile, delimiter=',')
        inputreader.next()
        for row in inputreader:
            predicted_F1_score = str(row[-1])
            ground_truths = row[-2]
            qid = row[-3]
            constituent = str(row[-4])
            prediction_list = list()
            prediction_list.append(predicted_F1_score)
            prediction_list.append(constituent)
            
            predictions_qid[str(qid)].append(prediction_list)
            #print(ground_truths)
            list_gr_tr = ast.literal_eval(ground_truths)            
            qid_ground_truths[str(qid)] = list_gr_tr

    max_predict = get_max_predictions(predictions_qid)
    print("{0} questions evaluated".format(len(predictions_qid)))
    print(evaluate(qid_ground_truths,max_predict))

    # for i,key in enumerate(max_predict.keys()):
    #     if i >5 :
    #         break
    #     print(key)
    #     print(max_predict[key])
    #     print(predictions_qid[key])

def evaluate_all_predictions():
    prediction_files = [f for f in listdir(predictions_file_path) if isfile(join(predictions_file_path, f))]
    for file in prediction_files:
        evaluate_ml_result(file)

if __name__ == '__main__':
    evaluate_all_predictions()
