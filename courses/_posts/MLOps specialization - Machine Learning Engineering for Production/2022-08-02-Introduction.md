---
layout: course
title: MLOps Specialization - Introduction to Machine Learning in Production
---

{{ page.title }}
================


MLOps stands for Machine Learning Operations. MLOps is a core function of Machine Learning engineering, focused on streamlining the process of taking machine learning models to production, and then maintaining and monitoring them. 


# Overview of the ML Life cycle and Deployment

## The Machine Learning Project Life cycle 

Beyond building the model, there are several aspects that determine the success of a machine learning in production.

![](/images/MLOps_specialization/pic1.png)

Building the deep learning model is only 5% of the ML Project. What its the 95% missing? 

- Configuration
- Data collection
- Feature Extraction
- Data verification
- Machine Resource Management
- Analysis Tools
- Process Management Tools
- Serving Infrastructure 
- Monitoring


![](/images/MLOps_specialization/pic0.png)


The workflow of a machine learning project is to systematically plan out the life cycle of a machine learning project. The framework contains 4 steps:

- (i) Scoping

Decide what to work on. What exactly do you want to apply Machine Learning to, and what is your features and what is your target. Scoping also involve define key metrics such as loss metric, latency, how long the system (queries per second), timeline, budget...

- (ii) Data

Defining the data and establishing a baseline, and then also labelling and organizing the data. Is the data labelled consistently? 

- (iii) Modelling

Model selection, model training, and perform error analysis. What architecture you use? What hyperparameters (e.g. learning rate)?

In Research/Academia, the modelling focus on the model experimentaion and hyperparameters optimization. However, in industry production, usually is more important optimizing the data, keeping the model fixed. Error analysis is used to systematically improve the data (e.g. targeting what data to collect).

- (iv) Deployment

Deployment of the model in production, writing the software needed to put into production, monitor the system, track the data that continues to come in, and maintain the system to avoid performance degradation. One of the key concept maintaining the systems is ***Concept drift or Data drift***: How the model performs when the input data comes from a specific segment that it is not usual for the model.

For example, because of little exposure of this segment in the training phase, or even this data segment is out of the training distribution (More on these problems in the next section). Sample bias, deployment in a different environment can be the causes among many others.

The framework of Scoping, Data, Modelling, and Deployment is not a linear process. When modelling, you might discover new features that can be created, or that the labelling is wrong, or even that the proxy for the target variable is flawed. Working on one step can create improvement in other steps, and this iteration process is important to make the model on production works.

![](/images/MLOps_specialization/pic2.png)

When the model is deployed, the iteration between steps is even more relevant. 

For example, the data distribution of the input data can be different from the training and you would need to update the model with new data. Or the model is too slow, and you need to change the model architecture. In summary, hundreds of different things can happen when the model is put into production that affect the performance. **Monitoring the model behaviour once deployed** is essential to ensure that it is working propertly.


## Deployment challenges

There are 3 main challenges when deploying a model: Concept Drift, Data Drift, and Software system issues. 

- **Concept Drift** (changes in Y)

Concept drift means that the statistical properties of the target variable, which the model is trying to predict, change over time in unforeseen ways. Given that the statistical properties of Y change, the model begin to explain the variation of Y less accurately. 

Online shopping is a good example of concept drift. Let's say that a sales prediction model could estimate the sales amount (Y) given consumer profile features (X). Then Covid pandemic happened and the online retail sales raised rapidly, independently of the X factors modelled. In consequence, the sales prediction model would fail to estimate the sales amount systematically because of the concept drift in sales (Y) due to Covid-19 pandemic. 

- **Data or Covariate Drift** (changes in X distribution)

Data Drift is the change in the distribution of one or more of the independent variables or input variables of the dataset (X). The function or relation between X and Y could be unaffected, but if the distribution of the input data in production changes with respect to the training set the model will be inaccurate. The model normally do not behave well on out of sample training data.

- **Software system issues**

Checklist of question before launching a model in deployment:

1. Does the model can run estimations offline? Batching real-time data is necessary?
2. Does the model run in the cloud, or "in the edge" (in the device)? Internet connection is necessary for the product?
3. Compute resources: does it run in the CPU or GPU?
4. Latency is important? Users can wait for the results of the model?
5. Do you have a logging system that analyse and monitor the model behaviour?
6. Does your application have the right information security and privacy?

All of the questions must have a clear answer before deployment to avoid costly surprises. 

## Deployment implementation

**Monitoring, Shadow mode, and scaled deployment**.

Having done the checklist and building a software to host the model doesn't mean that the deployment is ready. The correct implementation of an ML system start with a **gradual ramp up with monitoring**. 

The new ML system starts its implementation in **shadow mode**. The model is implemented, but shadowing the process that it would replace, so both run in parallel. It allows for comparison between the model and the current system. It allows to **rollback** in case the system doesn't behave well, or affect the business continuity, and also helps to incorporate improvements into the model before full deployment. 

Once the model looks like in does a good job in the shadow mode, the ML system can be deployed partially, let's say to 5% of capacity. This mode is also called **canary deployment**. The new system that tries to replace the previous process is implemented in a part of the process, and its monitored to see their behaviour. The goal, as in shadow mode, is **minimizing risks**. If the model does mistakes, it would only affect a minimal part of the production, and once it proves its accuracy the system can ramp up gradually.

An alternative to canary deployment is **Blue-Green Deployment**. The blue-green deployment means that you have 2 systems: your current one (Blue), and the new ML system to be implemented (Green). As you prepare a new release of your ML system, you do your final stage of testing in the green environment. Once the ML model is working in the green environment, you switch the router so that all incoming requests go to the green environment - and the blue old environment is now idle.

![](/images/MLOps_specialization/pic3.png)

The router changes from the old to the new environment when the release is ready. Blue-green deployment also gives you a rapid way to rollback - if anything goes wrong you switch the router back to your blue environment. 


## Degrees of automation

A ML model deployment is not a binary choice, implemented or not. It can be partially implemented depending on the problem it is trying to solve. Instead, think about the appropriate degree of automation of the ML model. 

![](/images/MLOps_specialization/pic4.png)

You can chose to stop in any step, from fully human to fully AI making decisions.

Even if the ML model is well designed, it might be the case that it simply cannot predict accurate enough the output. Even when it can predict very well, let's say 95% of accuracy, the resting 5% can be critical. 

Think about a computer vision model predicting cancer - false negatives are critical and a 95% accuracy is not enough. However, a ML system doesn't have to be implemented to full automation, it can also simply help humans to make a better decision (**AI assistance**). From the previous example, the computer vision model can predict the cases when the model is 100% sure and forward the cases to a doctor when is below 100%.


## Monitoring

**Key metrics to track around the ML system**.

- Software metrics: memory, compute load, latency, throughput (time to user), server load.
- Input metrics: average lenght (text/audio), missing values, enviromental variables (e.g. light)
- Output metrics: null returns, user retry, user quit app, user time in the system.

It is important to decide beforehand the software, input, and output metrics that you will monitor. However, they are not set in stone. As with the iterative process of The Machine Learning Project Life cycle, the deployment step has as well its own iterative process of seeing the new input data that the ML system receives and deciding the correct measures and key metrics. 


![](/images/MLOps_specialization/pic5.png)

When monitoring, set alarms when certain parameters goes below or above a threshold. For example, the GPU usage in cloud computing going above a threshold can incur in extra cost for the company.  

![](/images/MLOps_specialization/pic6.png)

# Select and Train a Model

## Key challenges in model development

The modelling step is an iteration between modelling, monitoring, and continuous analysis to improve the life performance in production. This means choosing the right model, the right hyper parameters, the right datasets, training the model, and evaluating the errors to check improvements. 

**The focus should be on improving the data, not the model tunning**. Usually, the biggest leap in performance comes when the data is better - not when the hyper parameters are tweaked. 

![](/images/MLOps_specialization/pic7.png)

A final audit performance is necessary before phasing to production, to make sure everything works as expected. 

Usually, data science teams focus on doing a high accuracy model but test accuracy is not the end goal. The end goal is always solving the business problem. Test accuracy is not itself helping the business, neither stands by its own to justify the ML application.

![](/images/MLOps_specialization/pic8.png)

They are 3 main key metrics that data teams need to address to determine the success of the ML project: 

1. Doing well on training set.
2. Doing well on dev/test sets.
3. **Doing well on business metrics/project goals.**

## Why low average error isn't good enough

Simply put, it is not good enough because **even good models can have spill-over effects or unintended consequences**. Let's see 3 examples of good models in terms of pure performance that can work terribly used without any ethical or spill-over effect consideration. 

- *Using ML for loan approval*. Some unintended consequences might include discrimination by ethnicity, gender, location, language or other attributes that the model is using as predictors. Not only is it unethical, also it is against many country regulations that can get your company in risk compliance and legal troubles.

- *Using ML based product recommendations in retail industry*. The model can promote only mayor retailers, as take most of the training sample, and recommend the same products over an over. This might upset other retailers in the platform and do not catch up well new product that could be more relevant to the user.  

- *Using ML for detecting cancer cells for medical evaluation*. Rare cancers have really small sample dataset. Skew distributions or rare classes make the target very small, making it easy for the model to just "skip them" while achieving good performance overall. Remember that always detecting non-cancer is accurate 90% of the time, so use a relevant baseline.

## Establish a baseline

**A baseline is a benchmark to measure the model against**. 

When thinking about the baseline logic, think about the classic New Year Eve's resolutions. On average, we only follow up with 8% of the resolutions that we make at the end of the December. 8 out of a 100 resolutions made looks like a failure. In really, it is an amazing number compared with any other month or event. Consider the zero improvements plans or resolutions that we do at the end of February, 8% over 0 now feels like a better figure, doesn't it? 

The baseline to measure the success of New Year Eve's resolutions is not 100% resolutions made, it's the average of close to zero resolutions that we make every other month of the year. New Year Eve's resolutions are a success, not an apparent failure, made it clearer by a simple baseline comparison. 

The same happens with ML modelling. No model it's good on itself, it is good compared against a baseline. 

**What are good model benchmarks or baselines?**

- **Human level performance (HLP)**.

*Human level performance* (HLP) can be useful baseline that helps to focus you attention to the most important parts of the model to develop. For example, a test set of 70% accuracy can be good enough if the human level performance is 50%.

Remember that **humans are better with unstructured data than structure data**. We are better getting conclusions and decoding images, audio or text. We are usually worse than machines looking at structured tables or spreadsheet data.

Other baselines to consider:

- State of the art accuracy (last literature).
- Basic statistic (predicting the most frequent class, or linear regression).
- Performance of the previous system prediction.

## Tips for getting started a ML project

- Don't try to implement the state-of-the-art approach, at least when you start. Find a reasonable method or model that is well-explained in the web (blogs) and you can apply.

- Find open-source implementation if available. 

- A reasonable algorithm with good data will often outperform a great algorithm with no so good data. 

- Take into account computational constraints if the goal is to deploy the model. Not so important if the goal is set the boundary of its possible with a model to achieve.  

- Try to over-fit a small training dataset before training on a large one. Make it sure it works in a very small batch of data before scaling up. 

## Error analysis

Error analysis can tell you what's the most efficient use of your time in terms of **what you should do to improve** your learning algorithm's performance.

It is basically **a human inspection**. Take the predictions that were not accurate or misclassified, one by one, and try to investigate what the model is not picking. You can tag the variables that are not captured in the model but they seem to influence the prediction accuracy. While it might seem complex, a simple spreadsheet can be used to error analysis, adding columns for every tag that you find. 

Take the example of *using ML for speach detection*. Listening to a couple of wrong predictions from the model you realize that the noise in the background might affect the accuracy. Most of the wrong predictions listened had some sort of noise or inference. From this human inspection, you could create a 

"car noise", "people noise" tag, and "low bandwidth" tag that the speech recognition algorithm is not capturing but seems to influence the prediction accuracy. 

![](/images/MLOps_specialization/pic10.png)

This error analysis can be used to decide what to focus first in improving the data.


## Prioritizing data improvements

The marginal effect on accuracy can be the reference to what to work on improving the data.

| Type          | Accuracy | Human level Performance | GAP to HLP | % of data | Marginal effect on accuracy |
| Clean Speech  | 94%      | 95%                     | 1%         | 60%       | **0.6%**                    |
| Car Noise     | 89%      | 93%                     | 4%         | 4%        | 0.16%                       |
| People Noise  | 87%      | 89%                     | 2%         | 30%       | **0.6%**                    |
| Low Bandwidth | 70%      | 70%                     | 0%         | 60%       | 0%                          |


**The marginal effect on accuracy is just the gap to HLP multiplied by the % of the data in your dataset that has this problem**. Please don't jump on fixing the tag guided only by this figure. How easy is to fix this tag also plays a role - maybe getting more clean speech samples is easier to fix than removing people noise, for example. 

**How to fix the tags identified by error analysis?**

- Collect more data with that particular tag
- Use data augmentation to get more data
- Improve label accuracy 

**By improving one tag/category, it usually happens that the model overall improvements on other tags improves**.

Prioritization process is part of the **error analysis**, and it serves to guide where to drive the efforts to improve the model data. 

## Skewed datasets

Skewed datasets are data where the ratio of positive to negative examples is very far from 50-50. For those datasets, accuracy is not a relevant metric. 

You are more interested in this cases in 2 main metrics:

1. **Recall/Sensitivity**. When the actual label is 1, how often does it predict 1?

This metric is better than accuracy in medical evaluation, for example. For a ML based cancer detector the crucial metric is not average accuracy, it is getting right all the people that have cancer. It can be okay to have some false positives, as long as when the actual label is "cancer", predict "cancer". 

For the cases that detecting true positives is crucial, recall tend to be a better metric than accuracy. 

2. **Precision**. When the model predicts 1, how often is it correct in reality?

For spam detectors ML models, this metric is important. Most of the emails we send and receive are not spam, so the dataset is very skewed and simply detecting "not spam" would have 99.99% accuracy. But it would be a terrible spam detector model. 

We much better evaluate the model using the precision metric: when the model predicts spam, how often is spam? 

Why not recall? Because you rather prefer to have a spam mail on my inbox than an actual mail on the spam folder. 

![](/images/MLOps_specialization/pic9.png)

Normally, you have to balance between recall and precision in order to select the best model. Most experiments will end up with models with more recall but less precision than other models, and vice versa. 

**How to choose the right metric for screwed datasets?** 

1. You can **weight each according to your needs**. For cancer prediction, you want to avoid False negatives at all cost so you weight recall way more than precision.

2. Use **F1-score**. The F1-score combines the precision and recall of a classifier into a single metric by taking their harmonic mean. It is primarily used to compare the performance of two classifiers.

![](/images/MLOps_specialization/pic11.png)

More metrics beyond accuracy implemented in sklearn [here](https://scikit-learn.org/stable/modules/classes.html#classification-metrics)


## Performance auditing 

Performance auditing is a framework or set of **steps for how you can double check your system for accuracy, fairness/bias and other possible problems BEFORE deployment**.

 1. Brainstorm the ways the system go wrong.

 - Check performance on subsets of data that the model performs worse (e.g. prediction error spikes for a certain class or tag).
 - Check the False Negatives are not grouped in the same class.
 - Check performance on rare classes with very few training samples.

 2. Stablish metrics to assess performance against these issues on the appropriate **slices of data** (e.g. train the model only on certain ethnicity that the model predict worse than average)

 3. Take input with stakeholders with knowledge of the matter. If the model predict badly certain class, spar with them about why they think is difficult to predict certain class.


## Data augmentation and feature engineering

**Data augmentation is basically create more data out of the already available**. Data augmentation can be a very efficient way to get more data, especially for unstructured data problems such as images, audio, and text.

- For image data, new samples can be created by flipping or cropping parts of the images (Pytorch and Tensorflow have a whole set of techniques already implemented in their APIs. 

- For audio data, you could mix 2 clips of audio, one with background noise and one with the only the target audio.

**The rule of thumb with data augmentation is that the main target is still has to be recognizable by a human**. So, after the data transformation, you should still recognizing the object in an image or the sound meaning. Create realistic examples that the algorithm does poorly on, but humans do well on.

**Feature engineering is creating new variables or features in the data by discovery or domain knowledge**. For example, with the variable "Date" you could extract the day of the week, if that day was holiday or weekend, that can improve the quality of results from a machine learning model.

The larger data set, the more likely it is that a pure end-to-end deep learning algorithm can work without feature engineering. However, it is usually not the case that the data is that rich and especially for structured data problems can still be a very important driver of performance improvements.  

## Experiment tracking

When you're running dozens or hundreds or maybe even more experiments, it's easy to forget what experiments you have already run. Having a system for tracking your experiments can help you be more efficient in making the decisions on the data or the model or hyper parameters to systematically improve your algorithm's performance. 

**Keep track of**:

1. What algorithm you're using and what version of code. 
2. The data set you use. 
3. Hyper parameters tuning 
4. The metric results (accuracy)
5. CPU results (resource monitoring)

Spreadsheets for a small data team is enough. For a formal approach consider moving to a experiment tracking system such as [Weights & Biases](www.wandb.ai).


# Best practices for the data preparion stage

Week 3 dives into best practices for the data stage of the full cycle of a machine learning project. Specifically, how to define what is the data, what should be x and what should be y and establish a baseline. This process would give you a good data set for when you move into the modelling phase.


## Scoping

**Scoping is picking what project to work on and also planning out the scope of the project**. 

Imaging that you are working to increase sales using ML for a E-commerce. You could start by improve the recommender system, or the search algorithm, or the data catalogue, or data quality, or the price optimization...All of those aspects affect sales.

What project should you focus on? What are the metrics for success? What are the resources needed?

**Given the amount of opportunities to use AI, it is key to identify the business problem, not the AI problem**. The main challenge is not improving the recommender system, is increasing the conversion rate for the E-commerce. Sometimes there is not a AI solution. Even when it has, multiple AI solutions can be considered.

Different business problems require different AI solutions:

| Business Problem         | AI Solution                             |
| ------------------------ | --------------------------------------- |
| Increase conversion rate | Search, recommendation system           |
| Reduce inventory         | Demand prediction, marketing promotion  |
| Increase margins         | Product optimisation, recommend bundles |


## Diligence on feasibility and value

To start considering about using a ML model to solve a business problem, it is essential to think about feasibility: it is this problem technically possible?

HLP is very useful and give you **an initial sense of whether a project is doable**. When evaluating HLP, You could give a human to same data that would be fed to a learning algorithm and just ask:

*Can a human, given the same data, perform the task reliably?*

**If a human can do it, then it significantly increases the hope they can also get the learning algorithm to do it**. If its humanly impossible, then it probably very hard for the machine to achieve "super-human performance".

Alternatives to HLP are external benchmark (competitors, literature, field leaders) or the history of the project (what was the last performance score?)





