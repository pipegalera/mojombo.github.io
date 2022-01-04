---
layout: course
title: Fastbook - Chapter 7 - Training a State-of-the-Art Model
---

{{ page.title }}
================



## Imagenette Dataset


Imagenette is a lighter version of the dataset ImageNet.

Trayining models using ImageNet took several hours so fastai created this lighter version. The philosophy behind is that you should aim to have an iteration speed of no more than a couple of minutes - that is, when you come up with a new idea you want to try out, you should be able to train a model and see how it goes within a couple of minutes. 

- **ImageNet**: 1.3 million images of various sizes, around 500 pixels across, in 1,000 categories.

- **Imagenette**: Smaller version of ImageNet that takes only 10 classes that looks very different from one another.



```python
# Imagenette
path = untar_data(URLs.IMAGENETTE)
```


```python
dblock = DataBlock(blocks=(ImageBlock(), CategoryBlock()),
                   get_items=get_image_files,
                   get_y=parent_label,
                   item_tfms=Resize(460),
                   batch_tfms=aug_transforms(size=224, min_scale=0.75))
# bs indicates how many samples per batch to load
dls = dblock.dataloaders(path, bs=64)
```


## Normalization

When training a model, it helps if your input data is normalized — that is, has a mean
of 0 and a standard deviation of 1. But most images and computer vision libraries use
values between 0 and 255 for pixels, or between 0 and 1; in either case, your data is
not going to have a mean of 0 and a standard deviation of 1.

To normalize the dat, you can add `batch_tfms` to the datablock to transform the mean andstandard deviation that you want to use. 


```python
dblock_norm = DataBlock(blocks=(ImageBlock(), CategoryBlock()),
                   get_items=get_image_files,
                   get_y=parent_label,
                   item_tfms=Resize(460),
                   batch_tfms= [*aug_transforms(size=224, min_scale=0.75), 
                                # Normalization
                                Normalize.from_stats(*imagenet_stats)])

dls_norm = dblock_norm.dataloaders(path, bs=64)
```


Let's compare two models, one with normalized data and one without normalization. The baseline model is `xResNet50`. To keep it short, `xResNet50` is a twist of `ResNet50` that have shown favourable results when compared to other RestNets **when training from scratch**. For testing use `fit_one_cycle()` and not`fine_tune()`, as it faster.

### Non-normalzied xRestNet50

```python
model = xresnet50()
learn = Learner(dls, model, loss_func = CrossEntropyLossFlat(), metrics=accuracy)
learn.fit_one_cycle(5, 3e-3)
```


<table  class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.639044</td>
      <td>7.565507</td>
      <td>0.211725</td>
      <td>02:20</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.264875</td>
      <td>1.688994</td>
      <td>0.523152</td>
      <td>02:16</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.961111</td>
      <td>1.115392</td>
      <td>0.664302</td>
      <td>02:17</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.717251</td>
      <td>0.651410</td>
      <td>0.789768</td>
      <td>02:22</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.589625</td>
      <td>0.550697</td>
      <td>0.825243</td>
      <td>02:16</td>
    </tr>
  </tbody>
</table>


### Normalized xRestNet50


```python
# Normalized data
learn_norm = Learner(dls_norm, model, loss_func = CrossEntropyLossFlat(), metrics=accuracy)
learn_norm.fit_one_cycle(5, 3e-3)
```


<table class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.817426</td>
      <td>1.625511</td>
      <td>0.572069</td>
      <td>02:17</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.790636</td>
      <td>1.329097</td>
      <td>0.592233</td>
      <td>02:15</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.671544</td>
      <td>0.681273</td>
      <td>0.781553</td>
      <td>02:17</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.501642</td>
      <td>0.431404</td>
      <td>0.864078</td>
      <td>02:15</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.395240</td>
      <td>0.387665</td>
      <td>0.875280</td>
      <td>02:17</td>
    </tr>
  </tbody>
</table>


**Normalizing the data helped achive 4% to 5% more accuracy!**

Normalization is specially important in pre-trained models. If the model was trained with normalized data (pixels with mean 1 and standard deviation 1), then it will perform better if your data is also normalized. Matching the statistics is very important for transfer learning to work well.

The default behaviour in fastai `cnn_learner` is adding the proper `Normalize` function automatically, but you will have to add it manually when training models from scratch.

## Progressive Resizing

Progressive resizing is gradually using larger and larger images as you train the model. 

Benefits:

- Training complete much faster, as most of the epochs are used training small images.

- You will have better generalization of your models, as progressive resizing is just a method of data augmentation and therefore tend to improve external validity.

How it works? 

First, we create a `get_dls` function that calls the exactly same datablock that we made before, **but with arguments for the size of the images and the size of the batch** - so we can test different batch sizes.



```python
def get_dls(batch_size, image_size):
  dblock_norm = DataBlock(blocks=(ImageBlock(), CategoryBlock()),
                    get_items=get_image_files,
                    get_y=parent_label,
                    item_tfms=Resize(460),
                    batch_tfms= [*aug_transforms(size=image_size, min_scale=0.75), 
                                  Normalize.from_stats(*imagenet_stats)])
  
  return dblock_norm.dataloaders(path, bs=batch_size)
```

Let's start with 128 batch of images of 128 pixels each:


```python
dls = get_dls(128, 128)
learn = Learner(dls, xresnet50(), loss_func=CrossEntropyLossFlat(), metrics=accuracy)
learn.fit_one_cycle(4, 3e-3)
```


<table class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.859451</td>
      <td>2.136631</td>
      <td>0.392084</td>
      <td>01:14</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.297873</td>
      <td>1.321736</td>
      <td>0.585138</td>
      <td>01:12</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.979822</td>
      <td>0.863942</td>
      <td>0.723674</td>
      <td>01:12</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.761521</td>
      <td>0.687464</td>
      <td>0.781927</td>
      <td>01:11</td>
    </tr>
  </tbody>
</table>


As with transfered learning, we take the model and we train it 5 more batches with 64 more images but this time with a larger size of 224 pixels: 


```python
learn.dls = get_dls(64, 224)
learn.fine_tune(5, 1e-3)
```


<table class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.863330</td>
      <td>1.115129</td>
      <td>0.645631</td>
      <td>02:16</td>
    </tr>
  </tbody>
</table>



<table class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.677025</td>
      <td>0.756777</td>
      <td>0.762136</td>
      <td>02:15</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.659812</td>
      <td>0.931320</td>
      <td>0.712099</td>
      <td>02:15</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.592581</td>
      <td>0.682786</td>
      <td>0.775579</td>
      <td>02:15</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.481050</td>
      <td>0.454066</td>
      <td>0.855863</td>
      <td>02:17</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.427033</td>
      <td>0.425391</td>
      <td>0.868185</td>
      <td>02:23</td>
    </tr>
  </tbody>
</table>


Pregressive resizing can be done at more epochs and for as big an image as you wish, but notice that you will not get any benefit by using an image size larger that the size of the images.


## Test Time augmentation

We have been using random cropping as a way to get some useful data augmentation,
which leads to better generalization, and results in a need for less training data. When
we use random cropping, fastai will automatically use center-cropping for the validation
set — that is, it will select the largest square area it can in the center of the image,
without going past the image’s edges.

This can often be problematic. For instance, in a multi-label dataset, sometimes there
are small objects toward the edges of an image; these could be entirely cropped out by
center cropping.

*Squishing* could be a solution but also can make the image recognition
more difficult for our model. It has to learn how to recognize squished and
squeezed images, rather than just correctly proportioned images.

**Test Time Augmentation (TTA)** is a method that instead of centering or squishing, takes a number of
areas to crop from the original rectangular image, pass each of them through our
model, and take the maximum or average of the predictions.

It does not change the time required to train at all, but will
increase the amount of time required for validation or inference by the number of
test-time-augmented images requested. By default, fastai will use the unaugmented
center crop image plus four randomly augmented images

To use it, pass the DataLoader to fastai’s `tta` method; by default, it will crop your validation set - you just have to store the "new validation set" in a variable. 

Run it to observe the output shape:


```python
learn.tta()

    (TensorBase([[1.3654e-03, 1.1131e-04, 4.8078e-05,  ..., 8.0065e-09, 1.8123e-08,
              2.7091e-08],
             [1.8131e-04, 3.0205e-04, 4.8520e-03,  ..., 1.0132e-11, 8.4396e-12,
              1.2754e-11],
             [7.4551e-05, 4.6013e-03, 9.6602e-03,  ..., 3.2817e-09, 2.7115e-09,
              6.0039e-09],
             ...,
             [6.5209e-05, 9.8668e-01, 7.5150e-07,  ..., 1.3289e-11, 1.2414e-11,
              9.5075e-12],
             [9.9031e-01, 1.3725e-04, 3.4502e-04,  ..., 3.1489e-11, 2.6372e-11,
              2.8058e-11],
             [1.1344e-05, 6.2957e-05, 9.8214e-01,  ..., 1.0300e-11, 1.2358e-11,
              2.7416e-11]]),
     TensorCategory([4, 6, 4,  ..., 1, 0, 2]))
```


The outputs are:

- The validation set (after this "random average cropping" technique), and
- The real labels

Notice that the model do not have to be retrained because **we don't use the validation set in the training phase**. We only take cropping averages of the images in the validation set, so the model doesn't change.  


```python
preds, targs = learn.tta()
accuracy(preds, targs).item()

    0.869305431842804
```


**TTA gives a little boost in performance (~1%) - taking into account that it doesn't require additional model training.**

However, it does make inference slower. For example, if you’re averaging five images for TTA inference will be five times slower.

## Mixup

Mixup is a powerful data augmentation technique that **can provide
dramatically higher accuracy, especially when you don’t have much data** and don’t
have a pretrained model that was trained on data similar to your dataset


Mixup is a technique that uses the weighted average of random images to improve the accuracy of the model. It iterates through the images in the dataset to combine:

1. The pixel and label values of each image with;
2. The pixel and label values of a random image.

For example, the following image is a mixup of a church with a gas station image:

![Mixing a church and a gas station](https://i.postimg.cc/kG6kLbJ5/fig1.png)


The constructed image is a **linear combination of the first and the second images** - like a linear regresion in which the dependent variable is the mixup image and the dependent variables the 2 images. It is built by adding 0.3 times the first one and 0.7 times the second. 

In this example, should the model predict “church” or “gas station”? 

The right answer is 30% church and 70% gas station, since that’s what we’ll get if we take the linear combination
of the one-hot-encoded targets. 

For instance, suppose we have 10 classes,
and “church” is represented by the index 2 and “gas station” by the index 7. The onehot-
encoded representations are as follows:

```
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0] and [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
```

So here is our final target:

```
[0, 0, 0.3, 0, 0, 0, 0, 0.7, 0, 0]
```


<div class="alert alert-block alert-info"> Notice that for Mixup to work, our targets need to be one-hot encoded. </div>

Here is how we train a model with Mixup:


```python
model = xresnet50()
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(),
                metrics=accuracy, 
                # Mixup!
                cbs= MixUp(0.5))
learn.fit_one_cycle(46, 3e-3)
```



<table class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2.328936</td>
      <td>1.526767</td>
      <td>0.511576</td>
      <td>01:11</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.774001</td>
      <td>1.380210</td>
      <td>0.552651</td>
      <td>01:11</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.623476</td>
      <td>1.196524</td>
      <td>0.612397</td>
      <td>01:11</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.564727</td>
      <td>1.234234</td>
      <td>0.609783</td>
      <td>01:11</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.564727</td>
      <td>1.234234</td>
      <td>0.609783</td>
      <td>01:11</td>
    </tr>
    <tr>
      <td>[...]</td>
      <td>[...]</td>
      <td>[...]</td>
      <td>[...]</td>
      <td>[...]</td>
    </tr>
    <tr>
      <td>29</td>
      <td>0.862966</td>
      <td>0.427176</td>
      <td>0.874160</td>
      <td>01:09</td>
    </tr>
    <tr>
      <td>30</td>
      <td>0.856436</td>
      <td>0.375472</td>
      <td>0.889096</td>
      <td>01:09</td>
    </tr>
    <tr>
    <tr>
      <td>[...]</td>
      <td>[...]</td>
      <td>[...]</td>
      <td>[...]</td>
      <td>[...]</td>
    </tr>
      <td>46</td>
      <td>0.714792</td>
      <td>0.288479</td>
      <td>0.922704</td>
      <td>01:08</td>
    </tr>
  </tbody>
</table>





**Mixup requires far more epochs to train to get better accuracy**, compared with other models.

With normalization, we reached 87% accuracy after 5 epochs, while by using mixup we needed 29. 

The model is harder to train, because it’s harder to see what’s in each image. And the
model has to predict two labels per image, rather than just one, as well as figuring out
how much each one is weighted. 

Overfitting seems less likely to be a problem, however,
because we’re not showing the same image in each epoch, but are instead showing
a random combination of two images.

# Label Smoothing

ML models optimize for the metric that you select. If the metric is accuracy, the model search for the maximum accuracy - minimazing the loss function by SGD. 

The optimization process, in practice, tells the model to return 0 for all categories but one, for which it is trained to return 1. Even 0.999 is not “good enough”; the model will get gradients and learn to predict activations with even higher confidence. This can become very harmful if your data is not perfectly labeled, and it never is in real life scenarios.

**Label smoothing** replace all the 1 with a number a bit less than 1, and the 0s with a number a bit more than 0. When you train the model, the model doesn't have to be 100% sure that it found the correct label - with 99% is good enough.

For example, for a 10 class classification problem (Imagenette) with the correct label in the index 3:

```
[0.01, 0.01, 0.01, 0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
```

Label smoothing can be incorporated in the `loss_func` argument: `loss_func=LabelSmoothingCrossEntropy()`



```python
model = xresnet50()
learn = Learner(dls, model, loss_func=LabelSmoothingCrossEntropy(),
                metrics=accuracy)
learn.fit_one_cycle(5, 3e-3)
```


<table class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2.512356</td>
      <td>2.483313</td>
      <td>0.449216</td>
      <td>02:24</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2.120067</td>
      <td>2.909898</td>
      <td>0.462659</td>
      <td>02:24</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.868167</td>
      <td>1.840382</td>
      <td>0.730769</td>
      <td>02:28</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.704343</td>
      <td>1.646435</td>
      <td>0.801344</td>
      <td>02:28</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.598507</td>
      <td>1.552380</td>
      <td>0.827110</td>
      <td>02:28</td>
    </tr>
  </tbody>
</table>


As with Mixup, you won’t generally see significant improvements from label smoothing
until you train more epochs.



