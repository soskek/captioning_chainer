# Image Captioning by Chainer

A Chainer implementation of [Neural Image Caption](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf), which generates captions given images.

This implementation is fast, because it uses cudnn-based LSTM (NStepLSTM) and beam search can deal with batch processing.

This code uses the [coco-caption](https://github.com/tylin/coco-caption) as a submodule.
So, please clone this repository as follows:
```
git clone --recursive https://github.com/soskek/captioning_chainer.git
```

Furthermore, the [coco-caption](https://github.com/tylin/coco-caption) works on python 2.7 only. Thus, this repository also follows it.


## Train an Image Caption Generator

```
sh prepare_scripts/prepare_dataset.sh
```

```
# flickr8k, flickr30k, mscoco
python -u train.py -g 0 --vocab data/flickr8k/vocab.txt --dataset flickr8k -b 64
python -u train.py -g 0 --vocab data/flickr30k/vocab.txt --dataset flickr30k -b 64
python -u train.py -g 0 --vocab data/coco/vocab.txt --dataset mscoco -b 64
```

On the mscoco dataset, with beam size of 20, a trained model reached BELU 25.9.
The paper uses ensemble and (unwritten) hyperparameters, which can cause the gap between this and the value reported in the paper.

## Use the model

```
python interactive.py --resume result/best_model.npz --vocab data/flickr8k/vocab.txt
```

After launched, enter the path of an image file.


## See Best Result and Plot Curve

```
python get_best.py --log result/log
```


## Citation

```
@article{Vinyals2015ShowAT,
  title={Show and tell: A neural image caption generator},
  author={Oriol Vinyals and Alexander Toshev and Samy Bengio and Dumitru Erhan},
  journal={2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2015},
  pages={3156-3164}
}
```
