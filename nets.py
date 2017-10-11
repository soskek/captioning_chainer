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
    argtopk = argsort[:, ::-1][:, :k]
    assert(argtopk.shape == (batchsize, k)), (argtopk.shape, (batchsize, k))
    topk_score = output.take(
        argtopk + xp.arange(batchsize)[:, None] * n_out)
    return argtopk, topk_score


def update_beam_state(outs, total_score, topk, topk_score, h, c, eos_id):
    xp = cuda.get_array_module(h)
    full = outs.shape[0]
    prev_full, k = topk.shape
    batch = full // k
    prev_k = prev_full // batch
    assert(prev_k in [1, k])

    if total_score is None:
        total_score = topk_score
    else:
        is_end = xp.max(outs == eos_id, axis=1)
        is_end = xp.broadcast_to(is_end[:, None], topk_score.shape)
        bias = xp.zeros_like(topk_score, numpy.float32)
        bias[:, 1:] = -10000.  # remove ended cands except for a consequence
        total_score = xp.where(
            is_end,
            total_score[:, None] + bias,
            total_score[:, None] + topk_score)
        assert(xp.all(total_score < 0.))
        topk = xp.where(is_end, eos_id, topk)  # this is not required
    total_score = total_score.reshape((prev_full // prev_k, prev_k * k))
    argtopk, total_topk_score = get_topk(total_score, k=k)
    assert(argtopk.shape == (prev_full // prev_k, k))
    assert(total_topk_score.shape == (prev_full // prev_k, k))
    total_topk = topk.take(
        argtopk + xp.arange(prev_full // prev_k)[:, None] * prev_k * k)
    total_topk = total_topk.reshape((full, ))
    total_topk_score = total_topk_score.reshape((full, ))

    hs = F.separate(h, axis=1)
    cs = F.separate(c, axis=1)

    argtopk = argtopk // k + \
        xp.arange(prev_full // prev_k)[:, None] * prev_k

    argtopk = argtopk.reshape((full, )).tolist()

    next_h = F.stack([hs[i] for i in argtopk], axis=1)
    next_c = F.stack([cs[i] for i in argtopk], axis=1)
    outs = xp.stack([outs[i] for i in argtopk], axis=0)

    outs = xp.concatenate([outs, total_topk[:, None]],
                          axis=1).astype(numpy.int32)

    return outs, total_topk_score, next_h, next_c


def finish_beam(outs, total_score, batchsize, eos_id):
    k = outs.shape[0] // batchsize
    result_batch = collections.defaultdict(
        lambda: {'outs': [], 'score': -1e8})
    for i in range(batchsize):
        for j in range(k):
            score = total_score[i * k + j]
            if result_batch[i]['score'] < score:
                out = outs[i * k + j].tolist()
                if eos_id in out:
                    out = out[:out.index(eos_id)]
                result_batch[i] = {'outs': out, 'score': score}

    result_batch = [
        result for i, result in
        sorted(result_batch.items(), key=lambda x: x[0])]
    return result_batch


class RNNDecoder(chainer.Chain):

    """A LSTM-RNN Decoder with Word Embedding.

    This model decodes a sentence sequentially using LSTM.

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

    def __call__(self, xs, ys, others):
        return self.calculate_loss(xs, ys, others)

    def calculate_loss(self, xs, ys, others):
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

    def decode(self, xs, k=20):
        batchsize = len(xs)
        # TODO: "encode image" option, using VGG
        h, c = self.prepare(xs)
        input_ys = [self.xp.array([self.eos_id], 'i')] * batchsize
        outs = self.xp.array([[]] * batchsize * k, 'i')
        total_score = None
        for i in range(self.max_decode_length):
            es = sequence_embed(self.embed, input_ys, 0)
            h, c, hs = self.rnn(h, c, es)

            concat_h = F.concat(hs, axis=0)
            concat_output = self.output(concat_h)
            topk, topk_score = get_topk(
                F.log_softmax(concat_output).data, k=k)
            assert(self.xp.all(topk_score < 0.))

            outs, total_score, h, c = update_beam_state(
                outs, total_score, topk, topk_score, h, c, self.eos_id)
            assert(self.xp.all(total_score < 0.)), i
            input_ys = self.xp.split(outs[:, -1], outs.shape[0], axis=0)
            if self.xp.max(outs == self.eos_id, axis=1).sum() == outs.shape[0]:
                # all cands meet eos, end
                break
        result = finish_beam(outs, total_score, batchsize, self.eos_id)
        return result

    def prepare(self, xs):
        xs = F.split_axis(F.dropout(self.transform(
            self.xp.stack(xs, axis=0)), self.dropout),
            len(xs), axis=0)
        h, c, _ = self.rnn(None, None, xs)
        return h, c
