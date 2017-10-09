# vec2text
Sentence Generation from Vector (image, video, feature, ...)

# Train an Image Caption Generator

```
sh prepare_dataset.sh
```

```
python -u train.py -g -1 --vocab data/flickr8k/vocab.txt --dataset flickr8k -b 32
```
