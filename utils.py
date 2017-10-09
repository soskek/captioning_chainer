import collections
import io

import numpy as np

import chainer
from chainer import cuda
from chainer import serializer


def make_vocab(dataset, max_vocab_size=20000, min_freq=2):
    counts = collections.defaultdict(int)
    for tokens, _ in dataset:
        for token in tokens:
            counts[token] += 1

    vocab = {'<eos>': 0, '<unk>': 1}
    for w, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if len(vocab) >= max_vocab_size or c < min_freq:
            break
        vocab[w] = len(vocab)
    return vocab


def make_array(tokens, vocab, add_eos=True, add_bos=True):
    ids = [vocab.get(token, vocab['<unk>']) for token in tokens]
    if add_eos:
        ids.append(vocab['<eos>'])
    if add_bos:
        ids.insert(0, vocab['<eos>'])
    return np.array(ids, 'i')


def read_vocab(file_name):
    vocab = {}
    for l in io.open(file_name, encoding='utf-8', errors='ignore'):
        sp = l.rstrip().split('\t')
        if len(sp) == 2:
            vocab[sp[0]] = int(sp[1])
    necessary = ['<eos>', '<unk>', ' ']
    for token in necessary:
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def convert(batch, device):
    def to_device_batch(batch):
        if device is None:
            return batch
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = np.cumsum([len(x) for x in batch[:-1]], dtype='i')
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = xp.split(concat_dev, sections)
            return batch_dev

    vecs, sentences = zip(*batch)
    return {'xs': to_device_batch(list(vecs)),
            'ys': to_device_batch(list(sentences))}


class PartialNpzDeserializer(serializer.Deserializer):

    """Partial Deserializer for NPZ format.

    This is the standard deserializer in Chainer. This deserializer can be used
    to read an object serialized by :func:`save_npz`.
    Only params with selected names "targets" are copied.

    Args:
        npz: `npz` file object.
        path: The base path that the deserialization starts from.
        strict (bool): If ``True``, the deserializer raises an error when an
            expected value is not found in the given NPZ file. Otherwise,
            it ignores the value and skip deserialization.

    """

    def __init__(self, npz, path='', strict=True, target_words=[]):
        assert(len(target_words) >= 1)
        self.target_words = target_words
        self.npz = npz
        self.path = path
        self.strict = strict

    def __getitem__(self, key):
        key = key.strip('/')
        return PartialNpzDeserializer(
            self.npz, self.path + key + '/', strict=self.strict,
            target_words=self.target_words)

    def __call__(self, key, value):
        key = self.path + key.lstrip('/')
        if not any(target in key for target in self.target_words):
            print('\t{}\tis NOT in targets'.format(key))
            return None
        else:
            print('{}\tis in targets'.format(key))

        if not self.strict and key not in self.npz:
            return value

        dataset = self.npz[key]
        if dataset[()] is None:
            return None

        if value is None:
            return dataset
        elif isinstance(value, np.ndarray):
            np.copyto(value, dataset)
        elif isinstance(value, cuda.ndarray):
            value.set(np.asarray(dataset))
        else:
            value = type(value)(np.asarray(dataset))
        return value


def load_npz_partially(filename, obj, strict=True, target_words=[]):
    """Loads an object from the file in NPZ format.
    This is a short-cut function to load from an `.npz` file that contains only
    one object.
    Args:
        filename (str): Name of the file to be loaded.
        obj: Object to be deserialized. It must support serialization protocol.
    """
    with np.load(filename) as f:
        d = PartialNpzDeserializer(f, strict=strict, target_words=target_words)
        d.load(obj)
