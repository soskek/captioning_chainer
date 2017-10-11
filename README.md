# vec2text
Sentence Generation from Vector (image, video, feature, ...)

# Train an Image Caption Generator

```
sh prepare_scripts/prepare_dataset.sh
```

```
python -u train.py -g 0 --vocab data/flickr8k/vocab.txt --dataset flickr8k -b 64
```

# Use the model

```
python interactive.py --resume result/best_model.npz --vocab data/flickr8k/vocab.txt
```

After launched, enter the path of an image file.


## See Best Result and Plot Curve

```
python get_best.py --log result/log
```
