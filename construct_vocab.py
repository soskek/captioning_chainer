import argparse
import json
import sys

import datasets

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', default='mscoco')
parser.add_argument('--threshold', '-t', type=int, default=5)
parser.add_argument('--out', '-o', default='vocab.txt')
args = parser.parse_args()

directory = datasets.get_default_dataset_path(args.dataset)
vocab, count = datasets.construct_vocab(
    directory, max_vocab_size=1e8, min_freq=5, with_count=True)

json.dump(vocab, open(args.out, 'w'))
json.dump(count, open(args.out + '.count', 'w'))

sys.stderr.write('# of words: {}\n'.format(len(vocab)))
