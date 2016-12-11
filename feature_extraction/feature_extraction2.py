import os
import json
import math
import operator
from feature_extractor import *
from collections import Counter
import pandas
import re
import networkx

word_vectors_filename = "../data/glove/glove.6B.50d.txt"
word_vectors = {}
df = {}
N = 1
cosine_similarity_threshold = 0.75

stanford_deps_hierarchy = networkx.DiGraph()

output_file_path = '../data/featuredata'
# Number of files' data that is written into a single output file
chunk_size = 1
curr_file = 0
num_files_written = 1

def init_deps_hierarchy():
	global stanford_deps_hierarchy
	stanford_deps_hierarchy.add_edge('treeroot', 'root')
	stanford_deps_hierarchy.add_edge('treeroot', 'dep')

	stanford_deps_hierarchy.add_edge('dep', 'aux')
	stanford_deps_hierarchy.add_edge('dep', 'arg')
	stanford_deps_hierarchy.add_edge('dep', 'case')
	stanford_deps_hierarchy.add_edge('dep', 'cc')
	stanford_deps_hierarchy.add_edge('dep', 'clf')
	stanford_deps_hierarchy.add_edge('dep', 'conj')
	stanford_deps_hierarchy.add_edge('dep', 'dislocated')
	stanford_deps_hierarchy.add_edge('dep', 'expl')
	stanford_deps_hierarchy.add_edge('dep', 'foreign')
	stanford_deps_hierarchy.add_edge('dep', 'list')
	stanford_deps_hierarchy.add_edge('dep', 'mod')
	stanford_deps_hierarchy.add_edge('dep', 'name')
	stanford_deps_hierarchy.add_edge('dep', 'obl')
	stanford_deps_hierarchy.add_edge('dep', 'orphan')
	stanford_deps_hierarchy.add_edge('dep', 'parataxis')
	stanford_deps_hierarchy.add_edge('dep', 'goeswith')
	stanford_deps_hierarchy.add_edge('dep', 'punct')
	stanford_deps_hierarchy.add_edge('dep', 'ref')
	stanford_deps_hierarchy.add_edge('dep', 'remnant')
	stanford_deps_hierarchy.add_edge('dep', 'reparandum')
	stanford_deps_hierarchy.add_edge('dep', 'sdep')
	stanford_deps_hierarchy.add_edge('dep', 'vocative')

	stanford_deps_hierarchy.add_edge('aux', 'auxpass')
	stanford_deps_hierarchy.add_edge('aux', 'cop')

	stanford_deps_hierarchy.add_edge('arg', 'agent')
	stanford_deps_hierarchy.add_edge('arg', 'comp')
	stanford_deps_hierarchy.add_edge('arg', 'subj')

	stanford_deps_hierarchy.add_edge('cc', 'cc:preconj')

	stanford_deps_hierarchy.add_edge('comp', 'acomp')
	stanford_deps_hierarchy.add_edge('comp', 'ccomp')
	stanford_deps_hierarchy.add_edge('comp', 'xcomp')
	stanford_deps_hierarchy.add_edge('comp', 'obj')

	stanford_deps_hierarchy.add_edge('obj', 'dobj')
	stanford_deps_hierarchy.add_edge('obj', 'iobj')
	stanford_deps_hierarchy.add_edge('obj', 'pobj')

	stanford_deps_hierarchy.add_edge('subj', 'nsubj')
	stanford_deps_hierarchy.add_edge('subj', 'csubj')

	stanford_deps_hierarchy.add_edge('nsubj', 'nsubjpass')

	stanford_deps_hierarchy.add_edge('csubj', 'csubjpass')

	stanford_deps_hierarchy.add_edge('mod', 'acl')
	stanford_deps_hierarchy.add_edge('mod', 'amod')
	stanford_deps_hierarchy.add_edge('mod', 'appos')
	stanford_deps_hierarchy.add_edge('mod', 'advcl')
	stanford_deps_hierarchy.add_edge('mod', 'det')
	stanford_deps_hierarchy.add_edge('mod', 'discourse')
	stanford_deps_hierarchy.add_edge('mod', 'preconj')
	stanford_deps_hierarchy.add_edge('mod', 'vmod')
	stanford_deps_hierarchy.add_edge('mod', 'mwe')
	stanford_deps_hierarchy.add_edge('mod', 'advmod')
	stanford_deps_hierarchy.add_edge('mod', 'rcmod')
	stanford_deps_hierarchy.add_edge('mod', 'quantmod')
	stanford_deps_hierarchy.add_edge('mod', 'nn')
	stanford_deps_hierarchy.add_edge('mod', 'npadvmod')
	stanford_deps_hierarchy.add_edge('mod', 'num')
	stanford_deps_hierarchy.add_edge('mod', 'nmod')
	stanford_deps_hierarchy.add_edge('mod', 'number')
	stanford_deps_hierarchy.add_edge('mod', 'nummod')
	stanford_deps_hierarchy.add_edge('mod', 'prep')
	stanford_deps_hierarchy.add_edge('mod', 'possessive')
	stanford_deps_hierarchy.add_edge('mod', 'prt')

	stanford_deps_hierarchy.add_edge('acl', 'acl:relcl')

	stanford_deps_hierarchy.add_edge('det', 'det:predet')

	stanford_deps_hierarchy.add_edge('mwe', 'compound')
	stanford_deps_hierarchy.add_edge('mwe', 'fixed')
	stanford_deps_hierarchy.add_edge('mwe', 'flat')
	stanford_deps_hierarchy.add_edge('mwe', 'mark')

	stanford_deps_hierarchy.add_edge('compound', 'compound:prt')

	stanford_deps_hierarchy.add_edge('nmod', 'nmod:npmod')
	stanford_deps_hierarchy.add_edge('nmod', 'nmod:tmod')
	stanford_deps_hierarchy.add_edge('nmod', 'nmod:poss')

	stanford_deps_hierarchy.add_edge('advmod', 'neg')

	stanford_deps_hierarchy.add_edge('npadvmod', 'tmod')

	stanford_deps_hierarchy.add_edge('sdep', 'xsubj')

def get_vector_for_word(word):
	normalized_word = word.lower()
	return word_vectors.get(normalized_word)

def get_tf_idf_for_word(word, tf):
	normalized_word = word.lower()
	idf = N/df.get(normalized_word,1)
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

	# print vec1,vec2
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

# def matching_word_frequencies_feature(tokens, q_tokens, n):
# 	tf_idf_sum = 0
# 	vec_not_found = 0

# 	# Find if the token is similar enough to a token in the question, add its tf_idf
# 	for i in range(len(tokens) - n ):
# 		curr_ngram = tokens[i : i + n]
# 		for j in range(len(q_tokens) - n):
# 			curr_q_ngram = q_tokens[j : j + n]
# 			# Check similarity by seeing if the normalised inner product of the two associated word vectors is close to 1
# 			token_vec = [get_vector_for_word(token) for token in curr_ngram]
# 			# print token_vec
# 			q_token_vec = [get_vector_for_word(token) for token in curr_q_ngram]
# 			# print q_token_vec
# 			if None in token_vec or None in q_token_vec:
# 				vec_not_found += 1
# 			elif not token_vec or not q_token_vec:
# 				vec_not_found += 1
# 			else:
# 				similarity = [cosine_similarity(vec1, vec2) for vec1, vec2 in zip(token_vec, q_token_vec)]
# 				# similarity = cosine_similarity(token_vec, q_token_vec)# for vec1, vec2 in zip(token_vec, q_token_vec)]
# 				if all(s >= cosine_similarity_threshold for s in similarity):
# 					# tf_idf_sum = map(sum,tf_idf_sum,tf_idf[])
# 					for word in curr_ngram:
# 						tf_idf_sum += get_tf_idf_for_word(word)

# 	return tf_idf_sum

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

	root_match.append(cosine_similarity(
		get_vector_for_word(deptree_root_token), get_vector_for_word(question_deptree_root_token)) >=
								   cosine_similarity_threshold)

	# Find if the sentence contains the root of the dependency parse tree of the question
	root_match.append(False)
	for token in tokens:
		if (cosine_similarity(
				get_vector_for_word(token), get_vector_for_word(question_deptree_root_token)) >=
				cosine_similarity_threshold):
			root_match[1] = True
			break

	# Find if the question contains the root of the dependency parse tree of the sentence
	root_match.append(False)
	for token in question_tokens:
		if (cosine_similarity(
				get_vector_for_word(token), get_vector_for_word(deptree_root_token)) >= cosine_similarity_threshold):
			root_match[2] = True
			break

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
			token_tf_idf += get_tf_idf_for_word(t, tf.get(t,0))
		span_wrd_freq += token_tf_idf
		for j in range(len(q_tokens) - n +1):
			vec_not_found = 0
			q = q_tokens[j:j+n]
			if q == token:
				sim = 1
			else:
				sim = 0
				# Check similarity by seeing if the normalised inner product of the two associated word vectors is close to 1
				token_vec = [get_vector_for_word(t) for t in token]
				# print token_vec
				q_token_vec = [get_vector_for_word(q_t) for q_t in q]
				# print q_token_vec
				if None in token_vec or None in q_token_vec:
					vec_not_found += 1
				elif not token_vec or not q_token_vec:
					vec_not_found += 1
				else:
					sim = [cosine_similarity(vec1, vec2) > cosine_similarity_threshold for vec1, vec2 in zip(token_vec, q_token_vec)]
					sim = all(sim)
			if sim:
				tf_idf_sum_left = 0
				tf_idf_sum_right = 0
				tf_idf_sum_in = 0
				# for t in token:
				# 	token_tf_idf += get_tf_idf_for_word(token, tf.get(t,0))
				if i < left:
					tf_idf_sum_left += token_tf_idf
				elif i > right:
					tf_idf_sum_right += token_tf_idf
				else:
					tf_idf_sum_in +=  token_tf_idf	
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

def pos_feature(span, pos, q_pos):
	# Calculate POS tags of the constituent
	length = int(span['end']) - int(span['start']) + 1
	penalty = 1/length
	score = -1
	
	pos_dict = {'WDT' : ['D'],
				'WP' : ['N','P'],
				'WP$': ['PRP$'],
				'WRB': ['R','C']}

	for pos_qs in q_pos:
		if re.match('W', pos_qs):
			wh_tag = pos_qs#re.split('W',)
			break
	
	pos_tags = pos[int(span['start']):int(span['end']) + 1]
	for tag in pos_tags: 
		if tag[0] in pos_dict.get(wh_tag,[]) or tag in pos_dict.get(wh_tag,[]):
			wrong_tags = pos_tags.index(tag)
			score = 1 - penalty * wrong_tags
	return score

def find_parent_index_in_deptree(token_index, deptree):
	# Find the place where token_index is the third value of a deptree element (this should only happen once!)
	for dep in deptree:
		if int(dep[2]) == token_index:
			# Return the index of the token that is one level above this current dependency in the deptree
			return int(dep[1])

	return -1


def lemmas_feature(span, deptree, tokens, lemmas, question_lemmas):
	lemma_similarity = -1
	ancestor_lemma_tokens = set()
	# Scan up through the dependency tree, add all token lemmas into ancestor_lemma_tokens
	for i in range(span.get('start'), span.get('end') + 1):
		parent_word_index = find_parent_index_in_deptree(i, deptree)
		if parent_word_index is not -1:
			ancestor_lemma_tokens.add(lemmas[parent_word_index])
		# Repeat for one more level
		grandparent_word_index = find_parent_index_in_deptree(parent_word_index, deptree)
		if grandparent_word_index is not -1:
			ancestor_lemma_tokens.add(lemmas[grandparent_word_index])

	# Compute similarity of all the above lemmas with all the question words, keep the max value
	for lemma in ancestor_lemma_tokens:
		for q_lemma in question_lemmas:
			lemma_similarity = max(lemma_similarity,
								   cosine_similarity(get_vector_for_word(lemma), get_vector_for_word(q_lemma)))

	return lemma_similarity

def compare_paths(path1, path2, pos1, pos2, graph1, graph2):
	similarity = 0

	# Compare both nodes as well as graph connections
	for index in min(range(len(path1)), range(len(path2))):
		curr_node1 = path1[index]
		curr_node2 = path2[index]
		next_node1 = path1[index + 1] if index in range(len(path1) - 1) else None
		next_node2 = path2[index + 1] if index in range(len(path2) - 1) else None
		# Compare node (POS tags) except for the first and last index
		if index > 0 and index < min((len(path1)), (len(path2))) - 1:
			pos_tag1 = pos1[curr_node1]
			pos_tag2 = pos2[curr_node2]
			if pos_tag1 == pos_tag2 or pos_tag1 in pos_tag2 or pos_tag2 in pos_tag1:
				similarity += 1
		# Compare graph connection
		if next_node1 is not None and next_node2 is not None:
			conn1 = (graph1[curr_node1][next_node1]).get('dep')
			conn2 = (graph2[curr_node2][next_node2]).get('dep')
			if conn1 == conn2 or conn1 in stanford_deps_hierarchy.predecessors(conn2) or \
				conn2 in stanford_deps_hierarchy.predecessors(conn1) or \
							stanford_deps_hierarchy.predecessors(conn1) == stanford_deps_hierarchy.predecessors(conn2):
				similarity += 1

	return similarity

def deptree_path_feature(span, tokens, graph, pos, question_tokens, question_graph, question_pos, wh_word_loc):
	max_path_similarity = 0

	# Find similar words in the sentence and question
	for i in range(len(tokens)):
		token = tokens[i]
		for j in range(len(question_tokens)):
			q_token = question_tokens[j]
			if cosine_similarity(get_vector_for_word(token),
								 get_vector_for_word(q_token)) > cosine_similarity_threshold:
				# Calculate path from wh-word in the question to the question word
				wh_word_to_word = networkx.shortest_path(question_graph, wh_word_loc, j)

				# Calculate path from word to each of the span words
				for k in range(span.get('start'), span.get('end') + 1):
					span_word_to_word = networkx.shortest_path(graph, k, i)

					# Compare this path with the question graph path
					curr_similarity = compare_paths(wh_word_to_word, span_word_to_word, question_pos, pos, question_graph, graph)
					if max_path_similarity < curr_similarity:
						max_path_similarity = curr_similarity

	return max_path_similarity

def parse_data(path):
	i =0
	print "hey"
	global idf
	global N
	global curr_file
	global num_files_written

	# Read in word vectors
	with open(word_vectors_filename) as f:
		for line in f:
			line_list = line.split()
			word = line_list.pop(0)
			word_vectors[word] = line_list

	# Initialize the dependencies hierarchy, as given in the Stanford Dependencies Manual
	init_deps_hierarchy()

	for (root, files, filenames) in os.walk(path):
		for file in filenames:
			if (i == 2):
				break
			file = os.path.splitext(file)[0]
			if file.find('_q') >= 0:
				print file
				continue
			ans_features, q_features = parse_json(os.path.join(root, file))
			# print 'Answer_features :', ans_features, "Question_features", q_features
			i += 1

			all_tokens = []
			tf_list = []
			for j in ans_features[0].get('tokens'):
				j = map(unicode.lower,j)
				tf_list.append(Counter(j))
				all_tokens.extend(j)
			df = Counter(all_tokens)
			N = sum(df.values())

			# Empty list to store feature values
			combined_features = []

			# Create features for each constituent in ans_features, related to each in q_features
			for i in range(len(q_features[0].get('tokens'))):
				#curr_question_tokens = question['questions']['tokens']
				curr_question_tokens = q_features[0].get('tokens')[i]
				curr_question_deptree = q_features[0].get('deps_basic')[i]
				curr_question_lemmas = q_features[0].get('lemmas')[i]
				curr_question_pos	 = q_features[0].get('pos')[i]

				# Construct networkx graph from the deptree for the question
				curr_question_graph = networkx.Graph()
				# Add edges for each dep in deptree
				for dep in curr_question_deptree:
					curr_question_graph.add_edge(dep[1], dep[2], {'dep': dep[0]})

				curr_question_wh_word_loc = -1
				# Find wh-word for the question
				for wh_index in range(len(curr_question_pos)):
					q_word = curr_question_pos[wh_index]
					if re.match('W', q_word):
						curr_question_wh_word_loc = wh_index
						break

				max_sentence_length = len(max(ans_features[0].get('tokens'), key=len))
				deptree_path_scaling = 1.0/(2 * max_sentence_length - 3)

				for j in range(len(ans_features[0].get('tokens'))):
					curr_tokens = ans_features[0].get('tokens')[j]
					curr_lemmas = ans_features[0].get('lemmas')[j]
					curr_pos = ans_features[0].get('pos')[j]
					curr_deptree = ans_features[0].get('deps_basic')[j]
					curr_constituents = ans_features[0].get('constituents')[j]
					curr_tf = tf_list[j]

					# Construct networkx graph from the deptree for the sentence
					curr_graph = networkx.Graph()
					# Add edges for each dep in deptree
					for dep in curr_deptree:
						curr_graph.add_edge(dep[1], dep[2], {'dep': dep[0]})

					# List for current features
					curr_features = []

					# Features that are sentence-dependent
					# matching_word_freqs = matching_word_frequencies_feature(curr_tokens, curr_question_tokens, 1)
					# curr_features.append(matching_word_freqs)

					# matching_bigram_freqs = matching_word_frequencies_feature(curr_tokens, curr_question_tokens, 2)
					# curr_features.append(matching_bigram_freqs)

					root_match = root_match_feature(
						curr_deptree, curr_tokens, curr_question_deptree, curr_question_tokens)
					curr_features.extend(root_match)

					# Features that are constituent-dependent
					#print ans_feature['constituents']
					#constituent_word_freqs = sum_tf_idf(constituents)
					for constituent  in curr_constituents:
						#constituent = constituents[i]
						constituent_length_features = length_feature(constituent, curr_tokens)
						constituent_length_features.append(len(curr_tokens))
						curr_features.extend(constituent_length_features)

						constituent_word_freqs = sum_tf_idf(constituent, curr_tokens, curr_tf, curr_question_tokens)
						curr_features.extend(constituent_word_freqs)

						constituent_bigram_freqs = sum_tf_idf(constituent, curr_tokens, curr_tf, curr_question_tokens, 2)
						curr_features.extend(constituent_bigram_freqs[:-1])											

						constituent_label_feature = constituent['label'] if 'label' in constituent else 0
						curr_features.append(constituent_label_feature)

						constituent_pos_tag_feature = pos_feature(constituent, curr_pos, curr_question_pos)
						curr_features.append(constituent_pos_tag_feature)

						constituent_lemmas_feature = lemmas_feature(
							constituent, curr_deptree, curr_tokens, curr_lemmas, curr_question_lemmas)
						curr_features.append(constituent_lemmas_feature)

						constituent_deptree_path = deptree_path_scaling * deptree_path_feature(
							constituent, curr_tokens, curr_graph, curr_pos, curr_question_tokens, curr_question_graph,
							curr_question_pos, curr_question_wh_word_loc)
						curr_features.append(constituent_deptree_path)

						combined_features.append(curr_features[:])
						del curr_features[3:]
			if curr_file == chunk_size:
				# Write to file
				print('Writing!')

				df = pandas.DataFrame.from_records(combined_features)
				if not os.path.exists(output_file_path):
					os.makedirs(output_file_path)
				df.to_csv(os.path.join(output_file_path, str(num_files_written) + '.csv'))

				# Reset
				combined_features = []
				curr_file = 0
				num_files_written += 1
			curr_file += 1

"""
Read in processed data from JSON, create features, save to CSV
"""
if __name__ == '__main__':
	parse_data("../data/processed/processed_dev")
