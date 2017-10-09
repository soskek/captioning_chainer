import argparse
import sys

import datasets

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', default='mscoco')
parser.add_argument('--threshold', '-t', type=int, default=5)
args = parser.parse_args()

directory = datasets.get_default_dataset_path(args.dataset)
vocab = datasets.construct_vocab(
    directory, max_vocab_size=1e8, min_freq=5)

for w, i in sorted(vocab.items(), key=lambda x: (x[1], x[0])):
    print('\t'.join([w, str(i)]))

sys.stderr.write('# of words: {}\n'.format(len(vocab)))
