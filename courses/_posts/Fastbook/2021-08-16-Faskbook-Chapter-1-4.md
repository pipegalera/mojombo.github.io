---
layout: course
title: Fastbook - Chapters 1 to 4
---

{{ page.title }}
================

## About Fastbook

Fastbook is a book is focused on the practical side of deep learning. It starts with the big picture, such as definitions and general applications of deep learning, and progressively digs beneath the surface into concrete examples.

The book is based on *fastai* API, an API on top of Pytorch that makes it easier to use state-of-the-art methods in deep learning. It doesn't need you to understand models such as Convolutional Neural Networks and how they work, but it definitely have helped me following the book.

The fastbook package includes fastai and several easy-access datasets to test the models.

<strike>I have used Google Colab notebooks as it provides free GPU. The downside is that it doesn't have any memory available so you will have to install fastai every time you run a notebook.</strike>

*Amazon Segamaker Studio Lab* ([link](https://aws.amazon.com/sagemaker/studio-lab/)) provides free GPU (Tesla T4) and free storage (up to 15GB) for ML projects - You can use it for the course and reproduce the book results in notebooks.

## Installing fastai in Segamaker Studio Lab

1. Iniciate a Terminal and create a conda enviroment:

    ```python
    conda create -n fastai python=3.8
    ```
In the terminal activate the virtual enviroment: `conda activate fastai`

2. Install Pytorch and Fastai

    ```python
    # Pytorch
    conda install pytorch torchvision torchaudio
    ```

    ```python
    # Fastai
    conda install -c fastchan fastai
    ```

3. Import fastai

    ```python
    import fastbook
    fastbook.setup_book()
    from fastai.vision.all import *
    from fastai.vision import *
    ```

For a local instalation you should install CUDA Toolkit 11.3 before anything, and add `cudatoolkit=11.3 -c pytorch` at the end when you install Pytorch. Please notice that unless you have a really powerful GPU (Nvidia 3080+) you won't get the same times training the models.

## Machine Learning Intro

Machine Learning: The training of programs developed by allowing a computer to learn from its experience, rather than through manually coding the individual steps.

Deep Learning is a branch of Machine Learning focus in Neural Networks. Visually, this is how they work:

![png](/images/Fastbook/Chapter_1-4/2.png)


Neural Networks, in theory, can solve any problem to any level of accuracy based on the parametrization of the weights - *Universal approximation theorem*.


### Weights

The key for the parametrization to be correct is updating the weight. The weights are "responsible" of finding the right solution to the problem at hand. For example, weighting correctly the pixels in a picture to solve the question "Is a Dog or a Cat picture?".

The weight updating is made by Stochastic gradient descent (SGD).

### Terminology

The terminology has changed. Here is the modern deep learning terminology for all the pieces we have discussed:


- The functional form of the *model* is called its *architecture* (but be careful—sometimes people use *model* as a synonym of *architecture*, so this can get confusing).
- The *weights* are called *parameters*.
- The *predictions* are calculated from the *independent variable*, which is the *data* not including the *labels*.
- The *results* of the model are called *predictions*.
- The measure of *performance* is called the *loss*.
- The loss depends not only on the predictions, but also the correct *labels* (also known as *targets* or the *dependent variable*); e.g., "dog" or "cat."

Clarification: In the course they use "regression" not as a linear regression but as any prediction model in which the result is a continuous variable.

After making these changes, our diagram looks like:

![png](/images/Fastbook/Chapter_1-4/3.png)

## Ethic considerations and bias

### Positive feedback loop


Positive feedback loop is the effect of a small increase in the values of one part of a system that increases other values in the remaining system. Given that the definition is kinda technical, let's use the case of a predictive policing model.

Let's say that a predictive policing model is created based on where arrests have been made in the past. In practice, this is not actually predicting crime, but rather predicting arrests, and is therefore partially simply reflecting biases in existing policing processes. Law enforcement officers then might use that model to decide where to focus their police activity, resulting in creased arrests in those areas. These additional arrests would then feed back to re-trainning future versions of the model. The more the model is used, the more biased the data becomes, making the model even more biased, and so forth.

This is an example of a Positive feedback loop, where the system is this predictive policing model and the values are arrests.

**You cannot avoid positive feedback loop, use human interaction to notice the weird stuff that your algorithm might create.**

### Proxy bias

Taking the previous example - If the proxy for the variable that you are interested (arrests as proxied for crime) is bias, the variable that you are predicting too.

## Metric and Loss difference

Loss function: measure of performance **for the computer** to see if the model is doing better or worse in order **to update the parameters**. Accuracy is an example of loss function.

Metric: a function that measures quality of the model prediction **for you**. For example, the % of true labels predicted accurately. It can be the case that the loss change but identify the same number of true labels.

## Transfer learning

Using a pretrained model for a task different to what it was originally trained for. It is key to use models with less data. Basically, instead of the model starting with random weights, it is already trained by someone else and parametrized.

## Fine tuning

A transfer learning technique where the parameters of a pretrained model are updated by training for additional epochs using a different task to that used for pretaining.

An epoch is how many times the model looks at the data.

##  Dictionary

![png](/images/Fastbook/Chapter_1-4/4.png)

## P-values principles

The practical importance of a model is not given by the p-values but by the results and implications. **It only says that the confidence of the event happening by chance.**

**Principle 1**: P-values can indicate how incompatible the data are with a specified statistical model.

**Principle 2**: P-values do not measure the probability that the studied hypothesis is true, or the probability that the data were produced by random chance alone.

**Principle 3**: Scientific conclusions and business or policy decisions should not be based only on whether a P-value passes a specific threshold.

**Principle 4**: Proper inference requires full reporting and transparency.

**Principle 5**: A P-value, or statistical significance, does not measure the size of an effect or the importance of a result.

The threshold of statistical significance that is commonly used is a P-value of 0.05. This is conventional and arbitrary.

**Principle 6**: By itself, a P-value does not provide a good measure of evidence regarding a model or hypothesis.


## Starting a Machine Learning Project: Defining the problem

First, define the problem that you want to solve and the levers or variables that you can pull to change the outcome. **What its the point of predicting an outcome if you cannot do anything about it?**
