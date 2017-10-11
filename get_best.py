from __future__ import print_function
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot


def plot_result(key, xy, out_path):
    f = plot.figure()
    a = f.add_subplot(111)
    a.set_xlabel(key)
    a.grid()

    xy = np.array(xy)
    a.plot(xy[:, 0], xy[:, 1], marker='.', label=key)

    l = a.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    f.savefig(out_path, bbox_extra_artists=(l,), bbox_inches='tight')

    plot.close()


def main():
    parser = argparse.ArgumentParser(description='Image Comprehension')
    parser.add_argument('--key', '-n', default='val/bleu')
    parser.add_argument('--log', '-l', required=True)
    parser.add_argument('--min', action='store_true')
    parser.add_argument('--plot-path', '-p', default='plot.png')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    log = json.load(open(args.log))
    best = None
    best_iter = -1
    xy = []
    for out in log:
        i = out['iteration']
        if args.key in out:
            v = out[args.key]
            xy.append((i, v))
        else:
            continue
        update = False
        if best is None:
            update = True
        if args.min:
            if v < best:
                update = True
        else:
            if v > best:
                update = True
        if update:
            best = v
            best_iter = i
    print('iter', best_iter)
    print(args.key, best)

    plot_result(args.key, xy, args.plot_path)


if __name__ == '__main__':
    main()
