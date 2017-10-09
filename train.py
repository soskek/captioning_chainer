from __future__ import print_function
import argparse
import json

import numpy as np

import chainer
from chainer import training
from chainer.training import extensions

import datasets
import nets
import utils


def main():
    parser = argparse.ArgumentParser(description='Image Comprehension')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--learnrate', '-l', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--unit', '-u', type=int, default=512,
                        help='Number of units')
    parser.add_argument('--layer', type=int, default=1)
    parser.add_argument('--dropout', '-d', type=float, default=0.2,
                        help='Dropout rate for MLP')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--vocab', '-v', required=True,
                        help='Text file of vocab')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--dataset', default='mscoco',
                        choices=['mscoco', 'flickr8k', 'flickr30k'])
    parser.add_argument('--resume')
    parser.add_argument('--resume-rnn')
    parser.add_argument('--resume-wordemb')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    print('read vocab')
    vocab = utils.read_vocab(args.vocab)

    print('read dataset')
    directory = datasets.get_default_dataset_path(args.dataset)
    train, valid = datasets.get_caption_dataset(
        vocab, directory, ['train', 'val'])
    valid = valid[:20]

    print('# train', len(train))
    print('# valid', len(valid))

    print('setup model')
    np.random.seed(777)
    model = nets.RNNDecoder(
        args.layer, len(vocab), args.unit,
        dropout=args.dropout,
        eos_id=vocab['<eos>'])
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    print('setup trainer')
    print('       optimizer')
    optimizer = chainer.optimizers.Adam(args.learnrate)
    optimizer.setup(model)

    if args.resume:
        print('load model', args.resume)
        chainer.serializers.load_npz(args.resume, model)
    if args.resume_rnn:
        print('load RNN model', args.resume_rnn)
        utils.load_npz_partially(args.resume_rnn, model,
                                 target_words=['rnn/'])
    if args.resume_wordemb:
        print('load Word Embedding model', args.resume_wordemb)
        utils.load_npz_partially(args.resume_wordemb, model,
                                 target_words=['embed/'])

    print('       iterator')
    model.xp.random.seed(777)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    valid_iter = chainer.iterators.SerialIterator(valid, args.batchsize,
                                                  repeat=False, shuffle=False)

    print('       updater')
    updater = training.StandardUpdater(
        train_iter, optimizer,
        converter=utils.convert, device=args.gpu,
        loss_func=model.calculate_loss)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    print('       extensions')
    iter_per_epoch = len(train) // args.batchsize
    log_trigger = (iter_per_epoch // 2, 'iteration')
    eval_trigger = (log_trigger[0] * 2, 'iteration')  # every 1 epoch

    trainer.extend(extensions.Evaluator(
        valid_iter, model,
        converter=utils.convert, device=args.gpu,
        eval_func=model.evaluate),
        trigger=eval_trigger)

    trainer.extend(utils.SentenceEvaluater(
        model, valid, vocab,
        'val/', device=args.gpu),
        trigger=eval_trigger)

    record_trigger = training.triggers.MaxValueTrigger(
        'validation/main/acc',
        trigger=eval_trigger)
    trainer.extend(extensions.snapshot_object(
        model, 'best_model.npz'),
        trigger=record_trigger)

    trainer.extend(extensions.LogReport(trigger=log_trigger),
                   trigger=log_trigger)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration',
         'main/perp', 'validation/main/perp',
         'main/acc', 'validation/main/acc',
         'val/bleu',
         'val/rouge',
         'val/cider',
         'val/meteor',
         'elapsed_time']),
        trigger=log_trigger)

    if eval_trigger[0] % log_trigger[0] != 0:
        print('eval_trigger % log_trigger != 0.\n'
              'So, some evaluation results can not be logged and shown.')

    trainer.extend(extensions.ProgressBar())

    if args.resume:
        print(extensions.Evaluator(
            valid_iter, model,
            converter=utils.convert, device=args.gpu,
            eval_func=model.evaluate)())

    print('log_trigger ', log_trigger)
    print('eval_trigger', eval_trigger)
    print('START training. # iter/epoch=', iter_per_epoch)
    trainer.run()


if __name__ == '__main__':
    main()
