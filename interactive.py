from __future__ import print_function
import argparse
import json

import chainer
import numpy as np
from PIL import Image

import nets

try:
    # p2
    input_method = raw_input
except Exception:
    # p3
    input_method = input

print('import finished')


def main():
    parser = argparse.ArgumentParser(description='Image Comprehension')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--vocab', '-v', default='vocab.json')
    parser.add_argument('--resume', '-r', required=True,
                        help='Resume the training from snapshot')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    print('read vocab')
    vocab = json.load(open(args.vocab))
    rev_vocab = {i: w for w, i in vocab.items()}

    print('setup model')
    tmp = np.load(args.resume)
    n_vocab, n_units = tmp['embed/W'].shape
    n_layer = max(int(key.split('/')[1])
                  for key in tmp.keys() if 'rnn' in key) + 1
    model = nets.RNNDecoder(
        n_layer, n_vocab, n_units,
        dropout=0.,
        eos_id=vocab['<eos>'])
    cnn = chainer.links.VGG16Layers()
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
        cnn.to_gpu()

    if args.resume:
        print('load model', args.resume)
        chainer.serializers.load_npz(args.resume, model)

    while 1:
        inp = input_method('image>>').strip()
        path = inp.strip().lower()
        if not path:
            continue

        vecs = cnn.extract([Image.open(path)]).popitem()[1].data
        result = model.decode(vecs)[0]
        outs = result['outs']
        score = result['score']
        print(' '.join(rev_vocab[wi] for wi in outs))
        print(score)


if __name__ == '__main__':
    main()
