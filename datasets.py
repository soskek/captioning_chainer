from __future__ import print_function
import collections
import json
import os

import numpy as np
import scipy.io

import utils


def get_default_dataset_path(dataset):
    if 'coco' in dataset:
        return 'data/coco'
    elif 'flickr8k' in dataset:
        return 'data/flickr8k'
    elif 'flickr30k' in dataset:
        return 'data/flickr30k'
    else:
        raise NotImplementedError()


def construct_vocab(directory, max_vocab_size=50000, min_freq=5,
                    with_count=False):
    counts = collections.defaultdict(int)
    caption_path = os.path.join(directory, 'dataset.json')
    caption_dataset = json.load(open(caption_path))['images']
    for cap_set in caption_dataset:
        this_split = cap_set['split']
        sentences = cap_set['sentences']
        if this_split == 'train':
            for sent in sentences:
                tokens = sent['tokens']
                for token in tokens:
                    counts[token] += 1

    vocab = {'<eos>': 0, '<unk>': 1}
    for w, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if len(vocab) >= max_vocab_size or c < min_freq:
            break
        vocab[w] = len(vocab)
    if with_count:
        return vocab, dict(counts)
    else:
        return vocab


def load_caption_dataset(vocab, directory, split):
    dataset = []
    vec_path = os.path.join(directory, 'vgg_feats.mat')
    vecs = scipy.io.loadmat(vec_path)['feats'].T
    vecs = [v[0] for v in np.split(vecs, vecs.shape[0], axis=0)]
    caption_path = os.path.join(directory, 'dataset.json')
    caption_dataset = json.load(open(caption_path))['images']
    for cap_set in caption_dataset:
        this_split = cap_set['split']
        img_id = cap_set['imgid']
        sentences = cap_set['sentences']
        other = {'img_id': img_id}
        if this_split == split:
            img_vec = vecs[img_id]
            for sent in sentences:
                tokens = sent['tokens']
                sent_array = utils.make_array(
                    tokens, vocab, add_eos=True, add_bos=True)
                dataset.append((img_vec, sent_array, other))
    return dataset


def get_caption_dataset(vocab, directory, split):
    if isinstance(split, (list, tuple)):
        datas = [load_caption_dataset(vocab, directory, sp)
                 for sp in split]
        return datas
    else:
        return load_caption_dataset(vocab, directory, split)
