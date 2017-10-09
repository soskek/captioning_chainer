import collections

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import reporter


def sequence_embed(embed, xs, dropout=0.):
    """Efficient embedding function for variable-length sequences

    This output is equally to
    "return [F.dropout(embed(x), ratio=dropout) for x in xs]".
    However, calling the functions is one-shot and faster.

    Args:
        embed (callable): A :func:`~chainer.functions.embed_id` function
            or :class:`~chainer.links.EmbedID` link.
        xs (list of :class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): i-th element in the list is an input variable,
            which is a :math:`(L_i, )`-shaped int array.
        dropout (float): Dropout ratio.

    Returns:
        list of ~chainer.Variable: Output variables. i-th element in the
        list is an output variable, which is a :math:`(L_i, N)`-shaped
        float array. :math:`(N)` is the number of dimensions of word embedding.

    """
    x_len = [len(x) for x in xs]
    x_section = numpy.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    ex = F.dropout(ex, ratio=dropout)
    exs = F.split_axis(ex, x_section, 0)
    return exs


def get_topk(output, k=20):
    batchsize, n_out = output.shape
    xp = cuda.get_array_module(output)
    argsort = xp.argsort(output, axis=1)
    argtopk = argsort[:, ::-1][:k]
    assert(argtopk.shape == (batchsize, k))
    topk_score = output.take(
        argtopk + xp.arange(batchsize)[:, None] * n_out)
    return argtopk, topk_score


def update_beam_state(state, topk, topk_score, h, c, eos_id):
    # {beam_i: {'instance_i': instance_i, 'outs': [],
    #           'score': 0., 'end': False}, ...}
    xp = cuda.get_array_module(h)
    k = topk.shape[1]
    middle = collections.defaultdict(list)
    next_state = {}
    next_hs = []
    next_cs = []
    input_ys = []
    for beam_i, st in state.items():
        if st['end']:
            middle[st['instance_i']].append(
                [beam_i, st['outs'][-1], st['score']])
        else:
            scores = st['score'] + topk_score[beam_i]
            for y, score in zip(topk[beam_i], scores):
                middle[st['instance_i']].append([beam_i, y, score])

    for instance_i, cands in middle.items():
        topk_cands = sorted(cands, key=lambda x: -x[-1])[:k]
        for beam_i, y, score in topk_cands:
            end = state[beam_i]['end']
            outs = state[beam_i]['outs'] + ([y] if not end else [])
            next_state[len(next_state)] = {
                'instane_i': instance_i,
                'outs': outs,
                'score': score,
                'end': y == eos_id or end
            }
            # first axis = i-th layer
            next_hs.append(h[:, beam_i])
            next_cs.append(c[:, beam_i])
            input_ys.append(xp.array([y], 'i'))
    next_h = F.stack(next_hs, axis=1)
    next_c = F.stack(next_cs, axis=1)
    return next_state, input_ys, next_h, next_c


def finish_beam(state):
    result_batch = collections.defaultdict(
        lambda: {'outs': [], 'score': -1e8})
    for beam_i, st in state.items():
        instance_i = st['instance_i']
        if result_batch[instance_i]['score'] < st['score']:
            result_batch[instance_i] = {
                'outs': st['outs'], 'score': st['score']}
    result_batch = [
        result for i, result in
        sorted(result_batch.items(), key=lambda x:x[0])]
    return result_batch


class RNNDecoder(chainer.Chain):

    """A LSTM-RNN Decoder with Word Embedding.

    This model encodes a sentence sequentially using LSTM.

    Args:
        n_layers (int): The number of LSTM layers.
        n_vocab (int): The size of vocabulary.
        n_units (int): The number of units of a LSTM layer and word embedding.
        dropout (float): The dropout ratio.

    """

    def __init__(self, n_layers, n_vocab, n_units, dropout=0.5, eos_id=0):
        super(RNNDecoder, self).__init__(
            transform=L.Linear(None, n_units),
            embed=L.EmbedID(n_vocab, n_units),
            rnn=L.NStepLSTM(n_layers, n_units, n_units, dropout),
            output=L.Linear(n_units, n_vocab),
        )
        self.n_layers = n_layers
        self.n_units = n_units
        self.dropout = dropout
        self.eos_id = eos_id
        self.max_decode_length = 40

    def __call__(self, xs, ys):
        return self.calculate_loss(xs, ys)

    def calculate_loss(self, xs, ys):
        h, c = self.prepare(xs)
        input_ys = [y[:-1] for y in ys]
        target_ys = [y[1:] for y in ys]
        es = sequence_embed(self.embed, input_ys, self.dropout)
        h, c, hs = self.rnn(h, c, es)
        concat_h = F.dropout(F.concat(hs, axis=0), self.dropout)
        concat_output = self.output(concat_h)
        concat_target = F.concat(target_ys, axis=0)
        loss = F.softmax_cross_entropy(concat_output, concat_target)
        accuracy = F.accuracy(concat_output, concat_target)
        reporter.report({'loss': loss.data}, self)
        reporter.report({'perp': self.xp.exp(loss.data)}, self)
        reporter.report({'acc': accuracy.data}, self)
        return loss

    def evaluate(self, *args, **kargs):
        return self.calculate_loss(*args, **kargs)

    def decode(self, xs):
        batchsize = len(xs)
        state = {i: {'instance_i': i, 'outs': [], 'score': 0., 'end': False}
                 for i in range(batchsize)}

        # TODO: "encode image" option, using VGG
        h, c = self.prepare(xs)
        input_ys = [self.xp.array([self.eos_id], 'i')] * batchsize
        for i in range(self.max_decode_length):
            es = sequence_embed(self.embed, input_ys, 0)
            h, c, hs = self.rnn(h, c, es)

            concat_h = F.concat(hs, axis=0)
            concat_output = self.output(concat_h)
            topk, topk_score = get_topk(F.log_softmax(concat_output).data)

            state, input_ys, h, c = update_beam_state(
                state, topk, topk_score, h, c, self.eos_id)
        result = finish_beam(state)
        return result

    def prepare(self, xs):
        xs = F.split_axis(F.dropout(self.transform(
            self.xp.stack(xs, axis=0)), self.dropout),
            len(xs), axis=0)
        h, c, _ = self.rnn(None, None, xs)
        return h, c
