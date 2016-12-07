import os
import json
import math
from feature_extractor import *

word_vectors_filename = "../data/glove.6B/glove.6B.50d.txt"
word_vectors = {}
tf_idf = {}
cosine_similarity_threshold = 0.95
# def parse_json(file):
# 	with open(file + '.json') as req_data_file:
# 		data_json = json.load(req_data_file)

def vector_magnitude(vec):
	magnitude = 0
	for val in vec:
		magnitude += val * val

	magnitude = math.sqrt(magnitude)
	return magnitude

def cosine_similarity(vec1, vec2):
	if vec1 is None or vec2 is None:
		return 0

	if(len(vec1) != len(vec2)):
		print("Vectors not of same length!")
		return 0

	vec1_magnitude = vector_magnitude(vec1)
	vec2_magnitude = vector_magnitude(vec2)
	product = 0

	for i in range(len(vec1)):
		product += vec1[i] * vec2[i]

	return product / (vec1_magnitude * vec2_magnitude)

def matching_word_frequencies_feature(tokens, q_tokens, n):
	tf_idf_sum = 0
	vec_not_found = 0

	# Find if the token is similar enough to a token in the question, add its tf_idf
	for i in range(len(tokens) - n - 1):
		curr_ngram = tokens[i : i + n - 1]
		for j in range(len(q_tokens) - n - 1):
			curr_q_ngram = q_tokens[j : j + n - 1]
			# Check similarity by seeing if the normalised inner product of the two associated word vectors is close to 1
			token_vec = [word_vectors[token] for token in curr_ngram]
			q_token_vec = [word_vectors[token] for token in curr_q_ngram]
			if None in token_vec or None in q_token_vec:
				vec_not_found += 1
			else:
				similarity = [cosine_similarity(vec1, vec2) for vec1, vec2 in zip(token_vec, q_token_vec)]
				if all(s >= cosine_similarity_threshold for s in similarity):
					tf_idf_sum += tf_idf[token_vec]

	return tf_idf_sum

def root_match_feature(deptree, tokens, question_deptree, question_tokens):
	root_match = {}
	deptree_root_token = None
	question_deptree_root_token = None

	# Find if the roots of both trees are similar
	for dep in deptree:
		if dep[0] == "root":
			deptree_root_token = tokens[dep[2]]
			break

	for dep in question_deptree:
		if dep[0] == "root":
			question_deptree_root_token = question_tokens[dep[2]]
			break

	root_match['similar_roots'] = (cosine_similarity(
		word_vectors[deptree_root_token], word_vectors[question_deptree_root_token]) >= cosine_similarity_threshold)

	# Find if the sentence contains the root of the dependency parse tree of the question
	root_match['question_in_dep'] = False
	for token in tokens:
		if (cosine_similarity(
				word_vectors[token], word_vectors[question_deptree_root_token]) >= cosine_similarity_threshold):
			root_match['question_in_dep'] = True
			break

	# Find if the question contains the root of the dependency parse tree of the sentence
	root_match['sentence_in_dep'] = False
	for token in question_tokens:
		if (cosine_similarity(
				word_vectors[token], word_vectors[deptree_root_token]) >= cosine_similarity_threshold):
			root_match['sentence_in_dep'] = True
			break

	return root_match

def sum_tf_idf(span):
	tf_idf_sum = 0
	for token in span:
		tf_idf_sum += tf_idf[token]

	return tf_idf_sum

def length_feature(span, tokens):
    # Calculate different length-related features
    features = {
        # Num words to the left
        'left': span['start'],
        'right': len(tokens) - span['end'],
        'inside': len(span['text'].split())
    }

    return features

def pos_feature(span, pos):
    # Calculate POS tags of the constituent
    pos_tags = pos[span['start'], span['']]
    return pos_tags

def lemmas_feature(span, lemmas, question_lemmas):
	lemma_similarity = [0, 0, 0, 0]
	lemma_indices = [span['start'] - 2, span['start'] - 1, span['end'] + 1, span['end'] + 2]
	# Compute similarity of lemmas of required words with all the question words, keep the max value
	for i in lemma_indices:
		if i in range(len(lemmas)):
			for q_lemma in question_lemmas:
				lemma_similarity[i] = max(
					lemma_similarity[i], cosine_similarity(word_vectors[lemmas[lemma_indices]], word_vectors[q_lemma]))

	return lemma_similarity

def deptree_path_feature(span, tokens, deptree, question_tokens, question_deptree):
	deptree_paths = []

	# Find similar words in sentence and question
	for token in tokens:
		for q_token in question_tokens:
			if cosine_similarity(word_vectors[token], word_vectors[q_token]) > cosine_similarity_threshold:
				# Calculate path from the token to the span
				deptree_path = []
				token_index = tokens.index(token)
				# TODO: Complete this implementation!

	return deptree_paths

def parse_data(path):
	# i =0
	print "hey"

	# Read in word vectors
	with open(word_vectors_filename) as f:
		for line in f:
			line_list = line.split()
			word = line_list.pop(0)
			word_vectors[word] = line_list

			# Temporarily calculating here
			tf_idf[word] = 1

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

			# Create features for each constituent in ans_features, related to each in q_features
			for question in q_features:
				curr_question_tokens = question['tokens']
				curr_question_deptree = question['deps_basic']
				curr_question_lemmas = question['lemmas']
				for ans_feature in ans_features:
					curr_tokens = ans_feature['tokens']
					curr_lemmas = ans_feature['lemmas']
					curr_pos = ans_feature['pos']
					curr_deptree = ans_feature['deps_basic']

					# Features that are sentence-dependent
					matching_word_freqs = matching_word_frequencies_feature(curr_tokens, curr_question_tokens, 1)

					matching_bigram_freqs = matching_word_frequencies_feature(curr_tokens, curr_question_tokens, 2)

					root_match = root_match_feature(
						curr_deptree, curr_tokens, curr_question_deptree, curr_question_tokens)

					# Features that are constituent-dependent
					for constituent in ans_feature['constituents']:
						constituent_length_features = length_feature(constituent, curr_tokens)
						constituent_length_features['sentence'] = len(curr_tokens)

						constituent_word_freqs = sum_tf_idf(constituent)

						constituent_label_feature = constituent['label'] if 'label' in constituent else 0

						constituent_pos_tag_feature = pos_feature(constituent, curr_pos)

						constituent_lemmas_feature = lemmas_feature(constituent, curr_lemmas, curr_question_lemmas)

						constituent_deptree_path = deptree_path_feature(
							constituent, curr_tokens, curr_deptree, curr_question_tokens, curr_question_deptree)

"""
Read in processed data from JSON, create features, save to CSV
"""
if __name__ == '__main__':
	parse_data("../data/processed")
