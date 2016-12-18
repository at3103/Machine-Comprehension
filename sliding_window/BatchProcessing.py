import os
import argparse

def cli():
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter)

    parser.add_argument(
        'parsing_type',
        type=str,
        help='Provide type of processing required. "context" for context parsing and "q" for question parsing')

    parser.add_argument(
        'datasets',
        type=str,
        help='Provide dataset name to be processed')

    parser.add_argument(
        '--parse',
        action='store_true',
        help='Switch on parsing along with processing')

    args = parser.parse_args()
    return args.datasets,args.parsing_type,args.parse

datasets,parsing_type,parse = cli()

def main():
	if parsing_type == "q":
		for dataset in datasets.split("&"):
			print "Working on {0} questions".format(dataset)
			if parse:
				os.system("python QParsing.py {0}".format(dataset))
			os.system("java -cp .:stanford-corenlp-3.7.0.jar:json-simple-1.1.jar corenlp/ GetConstituentsQ {0}".format(dataset))
	elif parsing_type == "context":
		for dataset in datasets.split("&"):
			print "Working on {0} contexts".format(dataset)
			if parse:
				os.system("python ContextParsing.py {0}".format(dataset))
			os.system("java -cp .:stanford-corenlp-3.7.0.jar:json-simple-1.1.jar corenlp GetConstituents {0}".format(dataset))


if __name__ == '__main__':
	main()

	