import os
import json
import math
import operator
from feature_extractor import *
from collections import Counter
import pandas
import re
import string
import networkx
import traceback

#word_vectors_filename = "../data/glove/glove.6B.50d.txt"
#word_vectors = {}
df = {}
N = 0
cosine_similarity_threshold = 0.75

#stanford_deps_hierarchy = networkx.DiGraph()

output_file_path = '../data/featuredata_wo_vectors_classifier'
# Number of files' data that is written into a single output file
chunk_size = 10
curr_file = 1
num_files_written = 1

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
	ylabel = 'N'
	prediction_tokens = normalize_answer(prediction).split()
	ground_truth_tokens = normalize_answer(ground_truth).split()
	common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
	num_same = sum(common.values())
	if num_same == len(ground_truth_tokens):
		if len(prediction_tokens)==len(ground_truth_tokens) :
			ylabel = 'Y'
		else :
			ylabel = 'M'
	return ylabel


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def get_tf_idf_for_word(word, tf):
	idf = 0
	normalized_word = word.lower()
	df_cur = df.get(normalized_word,0) 
	if df_cur:
		idf = N/df_cur
	tf_idf = tf * math.log(1 + idf)
	return tf_idf

def vector_magnitude(vec):
	magnitude = 0
	vec = map(float,vec)
	magnitude = sum(map(operator.mul,vec,vec))
	magnitude = math.sqrt(magnitude)
	return magnitude

def cosine_similarity(vec1, vec2):
	if vec1 is None or vec2 is None:
		return 0

	if(len(vec1) != len(vec2)):
		print("Vectors not of same length!")
		return 0

	product = 0
	vec1 = map(float,vec1)
	vec2 = map(float,vec2)
	vec1_magnitude = vector_magnitude(vec1)
	vec2_magnitude = vector_magnitude(vec2)
	product = sum(map(operator.mul,vec1,vec2))
	return max(-1, min(1, product / (vec1_magnitude * vec2_magnitude)))

def root_match_feature(deptree, tokens, question_deptree, question_tokens):
	root_match = []
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

	root_match.append(deptree_root_token == question_deptree_root_token)

	# Find if the sentence contains the root of the dependency parse tree of the question
	root_match.append(False)
	if question_deptree_root_token in tokens:
		root_match[1] = True

	# Find if the question contains the root of the dependency parse tree of the sentence
	root_match.append(False)
	if deptree_root_token in question_tokens:
		root_match[2] = True

	return root_match

def sum_tf_idf(span, sent_tokens, tf, q_tokens, n=1):
	tf_idf_sum = 0
	tf_idf_sum_left = 0
	tf_idf_sum_right = 0
	tf_idf_sum_in = 0
	sim = 0
	span_wrd_freq = 0
	left = span['start']
	right = span['end']

	for i in range(len(sent_tokens) - n + 1):
		token_tf_idf = 0
		token = sent_tokens[i:i+n]
		for t in token:
			t = t.lower()
			token_tf_idf += get_tf_idf_for_word(t, tf.get(t,0))
		if i >= left and i <= right:
			span_wrd_freq += token_tf_idf
		if t in q_tokens:
			if i < left:
				tf_idf_sum_left += token_tf_idf
			elif i > right:
				tf_idf_sum_right += token_tf_idf
			else:
				tf_idf_sum_in += token_tf_idf
			tf_idf_sum += token_tf_idf

	tf_idf_list= [tf_idf_sum, tf_idf_sum_in, tf_idf_sum_left, tf_idf_sum_right, span_wrd_freq]
	return tf_idf_list

def length_feature(span, tokens):
    # Calculate different length-related features
    features = [
        # Num words to the left
        span['start'],
		# Num words to the right
        len(tokens) - span['end'] - 1,
		# Num words in the span
        len(span['text'].split())
    ]

    return features

def constituent_feature(constituent_label, q_pos, qid):
	wh_tag = ''
	pos_dict = {'WDT': ['NP'],
				'WP': ['NP', 'PNP'],
				'WP$': ['NP', 'PNP'],
				'WRB': ['ADJP', 'ADVP', 'VP', 'PRT']}

	for pos_qs in q_pos:
		if re.match('W', pos_qs):
			wh_tag = pos_qs  # re.split('W',)
			break
	if constituent_label in pos_dict.get(wh_tag, []):
		return 1
	else:
		return 0


def pos_feature(span, pos, q_pos):
	# Calculate POS tags of the constituent
	length = int(span['end']) - int(span['start']) + 1
	penalty = 1.0/length
	score = 0.0
	wh_tag =''
	pos_dict = {'WDT' : ['DT', 'NN','NNP', 'NNPS', 'NNS'],
				'WP' : ['NN','NNP', 'NNPS', 'NNS'],
				'WP$': ['PRP$', 'NN','NNP', 'NNPS', 'NNS'],
				'WRB': ['RB','RBR','RBS','CD']}

	for pos_qs in q_pos:
                if re.match('W', pos_qs):
                        wh_tag = pos_qs#re.split('W',)
			break
	#if wh_tag == '':
		#print span['text'], q_pos
	pos_tags = pos[int(span['start']):int(span['end']) + 1]
	for tag in pos_tags: 
		if tag in pos_dict.get(wh_tag,[]):
			wrong_tags = pos_tags.index(tag)
			score += 1 - float(penalty * wrong_tags)
		else:
			score -= penalty
		score /=length
	return score

def ner_feature(span, ners, q_tokens):
	# Calculate POS tags of the constituent
	length = int(span['end']) - int(span['start']) + 1
	penalty = 1.0 / length
	score = 0.0
	wh_tag = ''
	pos_dict = {'what': ['TIME', 'DATE', 'ORGANIZATION', 'LOCATION', 'PERSON'],
				'which': ['TIME', 'DATE', 'ORGANIZATION', 'LOCATION', 'PERSON'],
				'who': ['PERSON', 'ORGANIZATION'],
				'whom': ['PERSON', 'ORGANIZATION'],
				'whose': ['PERSON', 'ORGANIZATION'],
				'where': ['LOCATION'],
				'when': ['DATE','TIME'],
				'how': ['MONEY', 'PERCENT', 'NUMBER']}

	wh_tag = ''
	for token in q_tokens:
		if normalize_answer(token) in pos_dict.keys():
			wh_tag = normalize_answer(token)
			break
	ner_tags = ners[int(span['start']):int(span['end']) + 1]
	for tag in ner_tags:
		if tag in pos_dict.get(wh_tag, []):
			wrong_tags = ner_tags.index(tag)
			score += 1 - float(penalty * wrong_tags)
		score /= length
	return score

def find_parent_index_in_deptree(token_index, deptree):
	# Find the place where token_index is the third value of a deptree element (this should only happen once!)
	for dep in deptree:
		if int(dep[2]) == token_index:
			# Return the index of the token that is one level above this current dependency in the deptree
			return int(dep[1])

	return -1

def lemmas_feature(deptree, span, lemmas, question_lemmas):
	lemma_similarity = -1
	ancestor_lemma_tokens = set()
	# Scan up through the dependency tree, add all token lemmas into ancestor_lemma_tokens
	for i in range(len(span)):
		parent_word_index = find_parent_index_in_deptree(i, deptree)
		if parent_word_index is not -1:
			ancestor_lemma_tokens.add(lemmas[parent_word_index])
		# Repeat for one more level
		grandparent_word_index = find_parent_index_in_deptree(parent_word_index, deptree)
		if grandparent_word_index is not -1:
			ancestor_lemma_tokens.add(lemmas[grandparent_word_index])

	# Compute similarity of all the above lemmas with all the question words, keep the max value
	for lemma in ancestor_lemma_tokens:
		if lemma in question_lemmas:
			lemma_similarity += 1
	return lemma_similarity

def parse_data(path):
	i =1
	global df
	global N
	global curr_file
	global num_files_written


	# Empty list to store feature values
	combined_features = []
	written_files_path = output_file_path+"/written_files.txt"
	unwritten_files_path = output_file_path+"/unwritten_files.txt"
	written_files = []
	already_written_files = []
	if os.path.isfile(written_files_path): 
		with open(written_files_path) as f:
			already_written_files = f.read().splitlines()
	#print already_written_files
	for (root, files, filenames) in os.walk(path):
		for file in filenames:
			file = os.path.splitext(file)[0]	
			if file in already_written_files:
				continue
			
			if file.find('_q') >= 0:
				continue
			# if file.find('.json')<0:
			# 	continue	
			print "Processing {0}".format(file)
			try :
				ans_features, q_features = parse_json(os.path.join(root, file))
			except Exception, e:
				traceback.print_exc()
				#append to unwritten files and reset
				with open(unwritten_files_path,'a') as unwriting_file:
					unwriting_file.write(file)
					unwriting_file.write("\n")
				continue
			i += 1

			all_tokens = []
			tf_list = []
			N = 0
			for j in ans_features[0].get('tokens'):
				j = map(unicode.lower,j)
				tf_list.append(Counter(j))
				all_tokens.extend(set(j))
				N += 1
			df = Counter(all_tokens)
			written_files.append(file)
			# Create features for each constituent in ans_features, related to each in q_features
			for i in range(len(q_features[0].get('tokens',[]))):
				curr_question_g_truth = []
				#curr_question_tokens = question['questions']['tokens']
				curr_question_tokens = q_features[0].get('tokens',[])[i]
				curr_question_deptree = q_features[0].get('deps_basic',[])[i]
				curr_question_lemmas = q_features[0].get('lemmas',[])[i]
				curr_question_pos	 = q_features[0].get('pos',[])[i]
				curr_question_g_truth = q_features[0].get('ground_truth',[])[i]
				qid = q_features[0].get('id','')[i]
				#print('Question ID: ' + str(qid))

				# Construct networkx graph from the deptree for the question
				#curr_question_graph = networkx.Graph()
				# Add edges for each dep in deptree
				#for dep in curr_question_deptree:
				#	curr_question_graph.add_edge(dep[1], dep[2], {'dep': dep[0]})

				curr_question_wh_word_loc = -1
				# Find wh-word for the question
				for wh_index in range(len(curr_question_pos)):
					q_word = curr_question_pos[wh_index]
					if re.match('W', q_word):
						curr_question_wh_word_loc = wh_index
						break

				max_sentence_length = len(max(ans_features[0].get('tokens'), key=len))
				#deptree_path_scaling = 1.0/(2 * max_sentence_length - 3)

				for j in range(len(ans_features[0].get('tokens',[]))):
					curr_tokens = ans_features[0].get('tokens',[])[j]
					curr_lemmas = ans_features[0].get('lemmas',[])[j]
					curr_pos = ans_features[0].get('pos',[])[j]
					curr_deptree = ans_features[0].get('deps_basic',[])[j]
					curr_constituents = ans_features[0].get('constituents',[])[j]
					curr_ners = ans_features[0].get('ners',[])[j]

					curr_tf = tf_list[j]

					# Construct networkx graph from the deptree for the sentence
					#curr_graph = networkx.Graph()
					# Add edges for each dep in deptree
					#for dep in curr_deptree:
					#	curr_graph.add_edge(dep[1], dep[2], {'dep': dep[0]})

					# List for current features
					curr_features = []

					# Features that are sentence-dependent

					root_match = root_match_feature(
						curr_deptree, curr_tokens, curr_question_deptree, curr_question_tokens)
					curr_features.extend(root_match)

					# Features that are constituent-dependent
					for constituent  in curr_constituents:
						
						span_words = normalize_answer(constituent['text'])
						
						constituent_length_features = length_feature(constituent, curr_tokens)
						constituent_length_features.append(len(curr_tokens))
						curr_features.extend(constituent_length_features)

						constituent_word_freqs = sum_tf_idf(constituent, curr_tokens, curr_tf, curr_question_tokens)
						curr_features.extend(constituent_word_freqs)

						constituent_bigram_freqs = sum_tf_idf(constituent, curr_tokens, curr_tf, curr_question_tokens, 2)
						curr_features.extend(constituent_bigram_freqs[:-1])											

						constituent_label_feature = constituent_feature(constituent['label'], curr_question_pos, qid)
						curr_features.append(constituent_label_feature)

						constituent_pos_tag_feature = pos_feature(constituent, curr_pos, curr_question_pos)
						curr_features.append(constituent_pos_tag_feature)

						constituent_ner_tag_feature = ner_feature(constituent, curr_ners, curr_question_tokens)
						curr_features.append(constituent_ner_tag_feature)

						constituent_lemmas_feature = lemmas_feature(
							curr_deptree, curr_tokens, curr_lemmas, curr_question_lemmas)
						curr_features.append(constituent_lemmas_feature)

						# curr_question_g_truth = map(normalize_answer,curr_question_g_truth)
						# if span_words in curr_question_g_truth:
						# 	constituent_answer = 'Y'
						
						F1_const_score = metric_max_over_ground_truths(f1_score, span_words, curr_question_g_truth)

						curr_features.append(F1_const_score)

						#span words
						curr_features.append(span_words.encode('ascii','ignore'))
						#qs
						#curr_features.append(' '.join(curr_question_tokens))
						curr_features.append(qid)
						#ground truth
						curr_features.append(curr_question_g_truth)
						combined_features.append(curr_features[:])
						del curr_features[3:]
				# print "Done with qs"
			if curr_file == chunk_size:
				# Write to file
				print('Writing!')
				features = ['root match 1', 'sent_root_qs', 'qs_root_sent',  'n_wrds_l', 'n_wrds_r', 
				'n_wrds_in', 'n_wrds_sent', 'm_u_sent', 'm_u_span', 'm_u_l', 'm_u_r', 'span_wf', 
				'm_b_sent', 'm_b_span', 'm_b_l', 'm_b_r', 'constituent_label', 'pos', 'ner', 'lemma', 'F1_score',
				'span_words', 'q_words', 'ground_truth' ]

				df = pandas.DataFrame.from_records(combined_features, columns = features)
				if not os.path.exists(output_file_path):
					os.makedirs(output_file_path)
				df.to_csv(os.path.join(output_file_path, str(num_files_written) + '.csv'))

				# Reset
				combined_features = []
				curr_file = 0
				num_files_written += 1
				#append to written files and reset
				with open(written_files_path,'a') as writing_file:
					for w_file in written_files:
						writing_file.write(w_file)
						writing_file.write("\n")
				written_files = []
				
			curr_file += 1

	print "All done"
"""
Read in processed data from JSON, create features, save to CSV
"""
if __name__ == '__main__':
	parse_data("../data/processed/processed_train")
