# Image Captioning by Chainer

A Chainer implementation of [Neural Image Caption](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf), which generates captions given images.

This implementation is fast, because it uses cudnn-based LSTM (NStepLSTM) and beam search can deal with batch processing.

## Train an Image Caption Generator

```
sh prepare_scripts/prepare_dataset.sh
```

```
python -u train.py -g 0 --vocab data/flickr8k/vocab.txt --dataset flickr8k -b 64
```

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
