---
layout: course
title: Fastbook - Chapter 9 - Tabular Modeling Deep Dive
---

{{ page.title }}
================



<div class="alert alert-block alert-info"> This Notebook is completelly reproductible in Amazon SageMaker StudioLab <a href="https://aws.amazon.com/sagemaker/studio-lab/">(more info here)</a> </div>

For this Chapter we will use more than the `fastai` package so I let below the necessary imports:

```python
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import IPython
import graphviz
from dtreeviz.trees import *
from scipy.cluster import hierarchy as hc
from sklearn.model_selection import train_test_split, 
                                    cross_val_score
from sklearn.tree import DecisionTreeRegressor, 
                         DecisionTreeClassifier, 
                         export_graphviz
from sklearn.ensemble import BaggingClassifier, 
                             RandomForestClassifier, 
                             BaggingRegressor, 
                             RandomForestRegressor, 
                             GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,
                            confusion_matrix, 
                            classification_report
from fastai.tabular.all import *

plt.style.use('seaborn-white')

import warnings
warnings.filterwarnings('ignore')
```

```python
x = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
print(x)

      Tesla T4
```
Tabular modeling takes data in the form of a table (like a spreadsheet or CSV). The
objective is to predict the value in one column based on the values in the other columns.


## Beyond Deep Learning

So far, the solution to all of our modeling problems has
been to train a deep learning model. And indeed, that is a pretty good rule of thumb
for complex unstructured data like images, sounds, natural language text, and so
forth. Deep learning also works very well for collaborative filtering. But it is not
always the best starting point for analyzing tabular data.

Although deep learning is nearly always clearly superior for unstructured data, Ensembles of decision trees tend to give **quite similar results for many kinds of structured data**. Also, they train faster, are often easier to interpret, do not require special GPU hardware, and require less hyperparameter tuning.

## The Dataset


The dataset we use in this chapter is from the Blue Book for Bulldozers Kaggle competition, which has the following description: *"The goal of the contest is to predict
the sale price of a particular piece of heavy equipment at auction based on its usage, equipment type, and configuration."*


```python
df = pd.read_csv('/home/studio-lab-user/sagemaker-studiolab-notebooks/TrainAndValid.csv', low_memory=False)
df.head()
```

<table class="dataframe">
  <thead>
    <tr >
      <th></th>
      <th>SalesID</th>
      <th>SalePrice</th>
      <th>MachineID</th>
      <th>ModelID</th>
      <th>datasource</th>
      <th>auctioneerID</th>
      <th>YearMade</th>
      <th>MachineHoursCurrentMeter</th>
      <th>UsageBand</th>
      <th>saledate</th>
      <th>...</th>
      <th>Undercarriage_Pad_Width</th>
      <th>Stick_Length</th>
      <th>Thumb</th>
      <th>Pattern_Changer</th>
      <th>Grouser_Type</th>
      <th>Backhoe_Mounting</th>
      <th>Blade_Type</th>
      <th>Travel_Controls</th>
      <th>Differential_Type</th>
      <th>Steering_Controls</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1139246</td>
      <td>66000.0</td>
      <td>999089</td>
      <td>3157</td>
      <td>121</td>
      <td>3.0</td>
      <td>2004</td>
      <td>68.0</td>
      <td>Low</td>
      <td>11/16/2006 0:00</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Standard</td>
      <td>Conventional</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1139248</td>
      <td>57000.0</td>
      <td>117657</td>
      <td>77</td>
      <td>121</td>
      <td>3.0</td>
      <td>1996</td>
      <td>4640.0</td>
      <td>Low</td>
      <td>3/26/2004 0:00</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Standard</td>
      <td>Conventional</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1139249</td>
      <td>10000.0</td>
      <td>434808</td>
      <td>7009</td>
      <td>121</td>
      <td>3.0</td>
      <td>2001</td>
      <td>2838.0</td>
      <td>High</td>
      <td>2/26/2004 0:00</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1139251</td>
      <td>38500.0</td>
      <td>1026470</td>
      <td>332</td>
      <td>121</td>
      <td>3.0</td>
      <td>2001</td>
      <td>3486.0</td>
      <td>High</td>
      <td>5/19/2011 0:00</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1139253</td>
      <td>11000.0</td>
      <td>1057373</td>
      <td>17311</td>
      <td>121</td>
      <td>3.0</td>
      <td>2007</td>
      <td>722.0</td>
      <td>Medium</td>
      <td>7/23/2009 0:00</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 53 columns</p>



The **metric** selected to evaluate the model is the **root mean squared log error (RMLSE)** between the actual and predicted auction prices. We are going to transform the sales price column into a logarithm, so when we apply the RMSE, it is already taking the logarithm into account.


```python
df['SalePrice'] = np.log(df['SalePrice'])
```
## Categorical Embeddings

Categorical embeddings transforms the categorical variables into inputs that are both continuous and meaningful. Clustering or ordening different categories is important because models are better at understanding continuous variables. This is unsurprising considering
models are built of many continuous parameter weights and continuous activation values, which are updated via gradient descent.

Categorical embedding also:

- Reduces memory usage and speeds up neural networks compared with one-hot encoding.
- Reveals the intrinsic properties of the categorical variables - increasing their predictive power.
- It can be used for visualizing categorical data and for data clustering. The model learns an embedding for these entities that defines a continuous
notion of distance between them.
- Avoid overfitting. It is especially useful for datasets with lots of high cardinality features, where other methods tend to overfit.

We will start by embedding the "Product Size" variable, giving it it's natural order:


```python
df['ProductSize'].unique()

    array([nan, 'Medium', 'Small', 'Large / Medium', 'Mini', 'Large',
           'Compact'], dtype=object)
```



```python
df['ProductSize'].dtype

    dtype('O')
```



```python
# Order
sizes = ['Large','Large / Medium','Medium','Small','Mini','Compact']

df['ProductSize'] = df['ProductSize'].astype('category')
df['ProductSize'] = df['ProductSize'].cat.set_categories(sizes, ordered=True)
```


```python
df['ProductSize'].dtype

    CategoricalDtype(categories=['Large', 'Large / Medium', 'Medium', 'Small', 'Mini',
                      'Compact'],
    , ordered=True)
```


**It is not needed to do hot-encoding**. For binary classification
and regression, it was shown that ordering the predictor categories in each
split leads to exactly the same splits as the standard approach. This reduces computational
complexity because only k − 1 splits have to be considered for a nominal predictor
with k categories

## Feature Engineering: Dates

The fundamental basis of the decision tree is **bisection** — dividing a group into two. 

We look at the ordinal variables and divide the dataset
based on whether the variable’s value is greater (or lower) than a threshold, and we look at the categorical variables and divide the dataset based on whether the variable’s level is a particular level. So this algorithm has a way of dividing the dataset based on both ordinal and categorical data.

**But how does this apply to a common data type, the date?**

We might want our model to make decisions based on that date’s day of the week, on whether a day is a holiday, on what month it is in, and so forth. fastai comes with a function that will do this for us: `add_datepart` 


```python
df = add_datepart(df, 'saledate')
```


```python
# Last 15 columns, now we added more feature columns based on the day
df.sample(5).iloc[:,-15:]
```

<table class="dataframe">
  <thead>
    <tr >
      <th></th>
      <th>Differential_Type</th>
      <th>Steering_Controls</th>
      <th>saleYear</th>
      <th>saleMonth</th>
      <th>saleWeek</th>
      <th>saleDay</th>
      <th>saleDayofweek</th>
      <th>saleDayofyear</th>
      <th>saleIs_month_end</th>
      <th>saleIs_month_start</th>
      <th>saleIs_quarter_end</th>
      <th>saleIs_quarter_start</th>
      <th>saleIs_year_end</th>
      <th>saleIs_year_start</th>
      <th>saleElapsed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>295937</th>
      <td>Standard</td>
      <td>Conventional</td>
      <td>2007</td>
      <td>4</td>
      <td>16</td>
      <td>18</td>
      <td>2</td>
      <td>108</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1.176854e+09</td>
    </tr>
    <tr>
      <th>177280</th>
      <td>Standard</td>
      <td>Conventional</td>
      <td>2005</td>
      <td>3</td>
      <td>12</td>
      <td>21</td>
      <td>0</td>
      <td>80</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1.111363e+09</td>
    </tr>
    <tr>
      <th>198868</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2007</td>
      <td>3</td>
      <td>13</td>
      <td>27</td>
      <td>1</td>
      <td>86</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1.174954e+09</td>
    </tr>
    <tr>
      <th>55758</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1991</td>
      <td>5</td>
      <td>21</td>
      <td>21</td>
      <td>1</td>
      <td>141</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>6.747840e+08</td>
    </tr>
    <tr>
      <th>154301</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2006</td>
      <td>2</td>
      <td>8</td>
      <td>23</td>
      <td>3</td>
      <td>54</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1.140653e+09</td>
    </tr>
  </tbody>
</table>



## Using TabularPandas and TabularProc


A second piece of preparatory processing is to be sure we can handle strings and missing data. fastai includes `Categorify` for the fists and `FillMissing` for the second. 

- `Categorify` is a TabularProc that replaces a column with a numeric categorical transformation - levels chosen consecutively as they are seen in a column.
- `FillMissing` is a TabularProc that replaces missing values with the median of
the column, and creates a new Boolean column that is set to True for any row where the value was missing.


```python
procs = [Categorify, FillMissing]
```

The Kaggle training data ends in April 2012, so we will define a narrower
training dataset that consists only of the Kaggle training data from before November
2011, and we’ll define a validation set consisting of data from after November 2011.


```python
cond = (df.saleYear < 2011) | (df.saleMonth< 10)
train_idx = np.where(cond)[0]
valid_idx = np.where(~cond)[0]

splits = (list(train_idx), list(valid_idx))
```

**TabularPandas needs to be told which columns are continuous and which are categorical**.
We can handle that automatically using the helper function cont_cat_split:


```python
cont, cat = cont_cat_split(df, 1, dep_var='SalePrice')

to = TabularPandas(df, 
                   procs = procs, 
                   cat_names=cat, 
                   cont_names=cont, 
                   y_names='SalePrice', 
                   splits=splits)
```


```python
len(to.train), len(to.valid)


    (404710, 7988)
```


Fastai `TabularPandas` helps pre-processing the data. The following table is the first items of the orginal dataset:


```python
df.head(5)[['UsageBand', 'fiModelDesc','fiBaseModel', 'fiSecondaryDesc', 'fiModelSeries']]
```

<table class="dataframe">
  <thead>
    <tr >
      <th></th>
      <th>UsageBand</th>
      <th>fiModelDesc</th>
      <th>fiBaseModel</th>
      <th>fiSecondaryDesc</th>
      <th>fiModelSeries</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Low</td>
      <td>521D</td>
      <td>521</td>
      <td>D</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Low</td>
      <td>950FII</td>
      <td>950</td>
      <td>F</td>
      <td>II</td>
    </tr>
    <tr>
      <th>2</th>
      <td>High</td>
      <td>226</td>
      <td>226</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>High</td>
      <td>PC120-6E</td>
      <td>PC120</td>
      <td>NaN</td>
      <td>-6E</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Medium</td>
      <td>S175</td>
      <td>S175</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>


And this is how `to` looks afert the transformation:


```python
# Numerical version of the columns
to.items.head(5)[['UsageBand', 'fiModelDesc','fiBaseModel', 'fiSecondaryDesc', 'fiModelSeries']]
```


<table class="dataframe">
  <thead>
    <tr >
      <th></th>
      <th>UsageBand</th>
      <th>fiModelDesc</th>
      <th>fiBaseModel</th>
      <th>fiSecondaryDesc</th>
      <th>fiModelSeries</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>963</td>
      <td>298</td>
      <td>43</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1745</td>
      <td>529</td>
      <td>57</td>
      <td>98</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>336</td>
      <td>111</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>3716</td>
      <td>1381</td>
      <td>0</td>
      <td>45</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>4261</td>
      <td>1538</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>




The conversion of categorical columns to numbers is done by simply replacing each unique level with a number. **The numbers associated with the levels are chosen consecutively as they are seen in a column**, so there’s no particular meaning to the numbers in categorical columns after conversion.

The exception is if you first convert a
column to a Pandas ordered category (as we did for ProductSize earlier), in which
case the ordering you chose is used. We can see the mapping by looking at the
classes attribute:


```python
df['ProductSize'].unique()

    [NaN, 'Medium', 'Small', 'Large / Medium', 'Mini', 'Large', 'Compact']
    Categories (6, object): ['Large' < 'Large / Medium' < 'Medium' < 'Small' < 'Mini' < 'Compact']
```



```python
to['ProductSize'].unique()


    array([0, 3, 4, 2, 5, 1, 6], dtype=int8)
```



```python
to.classes['ProductSize']

    ['#na#', 'Large', 'Large / Medium', 'Medium', 'Small', 'Mini', 'Compact']
```



```python
# Save the progress
save_pickle('to.pkl',to)
# To load progress:
#to = load_pickle('to.pkl')
```

## Decision Trees: Avoiding Overfitting

To begin, we define our independent and dependent variables. The `TabularPandas` dataframe knows that the dependent variable is the sale price, because we specify it at `y_names='SalePrice'` inside the transformation. It is also stored which rows are from the test and which rows are from the validation dataset as we set it by the `splits=splits` in which we splitted the data based on the condition `cond = (df.saleYear < 2011) | (df.saleMonth< 10)`

The arguments `xs`, `y`, and `train`, `valid` can be used to split the data accordingly - and very fast! 


```python
# X train and y train
X, y = to.train.xs, to.train.y

# X valid and y valid
X_valid, y_valid = to.valid.xs, to.valid.y
```

Now that our data is all numeric, and there are no missing values, we can create a decision tree:


```python
tree_model = DecisionTreeRegressor(max_leaf_nodes=4)
tree_model.fit(X, y)
```

To keep it simple, we’ve told sklearn to create just four leaf nodes. To see what it’s
learned, we can display the tree:

{% raw %}
```python
def draw_tree(t, df, size=10, ratio=0.6, precision=0, **kwargs):
    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True, rounded=True,
                      special_characters=True, rotate=False, precision=precision, **kwargs)
    return graphviz.Source(re.sub('Tree {', f'Tree {{ size={size}; ratio={ratio}', s))
```
{% endraw %}

 
 

```python
draw_tree(tree_model, X, size=7, leaves_parallel=True, precision=2)
```

![png](/images/Fastbook/Chapter_9/output_47_0.svg)


We see the importance of bisection: only dividing the dataset based on the value of `Copler_System` predicts an average value of 9.21 versus 10.1. The deeper the model, the more questions it will be able to ask separating high-value from low-value auction results.

We will use the package `dtreeviz` to see the distribution of the tree leafs, and catch possible data quality issues.


```python
# Random sample of the data
samp_idx = np.random.permutation(len(y))[:500]

# Representation for decision tree visualization and model interpretation
dtreeviz(tree_model, 
         X.iloc[samp_idx], 
         y.iloc[samp_idx], 
         X.columns, 
         'SalePrice',
         fontname='DejaVu Sans', scale=1.6, label_fontsize=10, orientation='LR')
```




    
![svg](/images/Fastbook/Chapter_9/output_49_0.svg)
    



We can clearly see that there’s a problem with our YearMade data: there are bulldozers made in the year 1000. Let’s replace it with
1950:


```python
X.loc[X['YearMade']<1900, 'YearMade'] = 1950
X_valid.loc[X_valid['YearMade']<1900, 'YearMade'] = 1950
```

That change makes the split much clearer in the tree visualization, even although it
doesn’t change the result of the model in any significant way. This is a great example
of how resilient decision trees are to data issues.


```python
tree_model_2 = DecisionTreeRegressor(max_leaf_nodes=4)
tree_model_2.fit(X, y)

dtreeviz(tree_model_2, 
         X.iloc[samp_idx], 
         y.iloc[samp_idx], 
         X.columns, 
         'SalePrice',
         fontname='DejaVu Sans', scale=1.6, label_fontsize=10, orientation='LR')
```




    
![svg](/images/Fastbook/Chapter_9/output_53_0.svg)
    



We’ll create a little function to check the root mean squared error of our model
(m_rmse), since that’s how the competition was judged:


```python

def r_mse(a, b): 
    # Formula: Root mean squared error between 2 values: a and b
    return round(math.sqrt(((a-b)**2).mean()), 6)

def m_rmse(model, X, y): 
    # Model application: RMSE between the predictions of the model and the y
    return r_mse(model.predict(X), y)

def print_rmse(model):
    print("Training RMSE: {}".format(m_rmse(model, X, y)))
    print("Validation RMSE: {}".format(m_rmse(model, X_valid, y_valid)))
```

To ilustrate overfitting, let the model create a tree model without a limit of leafs:


```python
tree_model_3 = DecisionTreeRegressor()
tree_model_3.fit(X, y)

print_rmse(tree_model_3)
```

    Training RMSE: 0.0
    Validation RMSE: 0.332212


The model perfectly predicts the price of the auctions on the training, but checking it on the validation sets it seems to be overfitting indeed. 


```python
tree_model_3.get_n_leaves()

    324565
```


The model uses around 325k leafs for 400k datapoints - of course it is overfitting, we have nearly as many leaf nodes as data points.

Let's try a new model with at least 25 autions per leaf.


```python
tree_model_4 = DecisionTreeRegressor(min_samples_leaf=25)
tree_model_4.fit(X, y)

print_rmse(tree_model_4)


    Training RMSE: 0.211706
    Validation RMSE: 0.268875
```


```python
tree_model_4.get_n_leaves()


    12400
```


## Random Forests

Random Forests are based on a process called *bagging*:

1. Randomly choose a subset of the rows of your data.
2. Train a model using this subset.
3. Save that model, and then return to step 1 a few times.
4. This will give you multiple trained models. To make a prediction, predict using all of the models, and then take the average of each of those model’s predictions.


Although each of the models trained on a subset of data will make more errors than a
model trained on the full dataset, **those errors will not be correlated with each other**.
Different models will make different errors. The average of those errors, therefore, is
zero! So if we take the average of all of the models’ predictions, we should end up
with a prediction that gets closer and closer to the correct answer, the more models
we have. 

In the following function:

- `n_estimators` defines the number of trees we want.
- `max_samples` defines how many rows to sample for training each tree.
- `max_features` defines how many columns to sample at each split point (where 0.5 means “take half the total number of columns”). 
- `min_samples_leaf` specify when to stop splitting the tree nodes, effectively limiting the depth of the tree.
- `n_jobs=-1` tells sklearn to use all our CPUs to build the trees in parallel.


```python
def random_forest(X, y, n_estimators=100, 
              max_samples=200_000, 
              max_features=0.5,
              min_samples_leaf=5, **kwargs):
    
    return RandomForestRegressor(n_jobs=-1, 
                                 n_estimators=n_estimators, 
                                 max_samples=max_samples, 
                                 max_features=max_features,
                                 min_samples_leaf=min_samples_leaf, 
                                 oob_score=True).fit(X, y)
```


```python
rf_model = random_forest(X, y)
```

Our validation RMSE is now much improved over our last result produced by the
DecisionTreeRegressor, which made just one tree using all the available data:


```python
print_rmse(rf_model)


    Training RMSE: 0.169543
    Validation RMSE: 0.231052
```

<div class="alert alert-block alert-info"> You can set <b>n_estimators</b> to as high a number as you have time to train — the more trees you have, the more accurate the model will be. </div>



## Out-of-Bag Error and Prediction

The OOB error is a way of measuring prediction error in the training dataset by
including in the calculation of a row’s error trees only where that row was *not*
included in training. Imagining that every tree it has also has its own validation
set. That validation set is simply the rows that were not selected for
that tree’s training.

The OOB predictions are available in the `oob_prediction_`
attribute.


```python
rf_model.oob_prediction_

    array([10.96384715, 10.89122526,  9.39799785, ...,  9.30305792,
            9.46965767,  9.5851676 ])
```



```python
r_mse(rf_model.oob_prediction_, y)

    0.20854
```


`sklearn` also have a a `oob_score_` attribute that calculates the number of correctly predicted rows from the out of bag sample.


```python
rf_model.oob_score_


    0.909784962573175
```


We can include them in the definition above to having a full picture of the RMSE loss:


```python
def print_rmse(model, X, X_valid, y, y_valid):
    print("Training RMSE: {}".format(m_rmse(model, X, y)))
    print("Validation RMSE: {}".format(m_rmse(model, X_valid, y_valid)))
    print("Out-of-Bag RMSE: {}".format(r_mse(model.oob_prediction_, y)))
    print("Out-of-Bag Accuracy: {}".format(model.oob_score_.round(3)))
```

## Model Simplification and Improvements

For tabular data, model interpretation is particularly important. For a given model,
we are most likely to be interested in are the following:

- How confident are we in our predictions using a particular row of data?
- For predicting with a particular row of data, what were the most important factors, and how did they influence that prediction?
- Which columns are the strongest predictors, which can we ignore?
- Which columns are effectively redundant with each other, for purposes of prediction?
- How do predictions vary as we vary these columns?

This section covers the first questions above, not necessarery improving the accuracy of the model but simplyfing the variables to focus to and identifying the part of the data that the model have more problems with.


### 1. Tree Variance for Prediction Confidence

How can we know the conficdence of the estimate? One simple way is to use the standard deviation of predictions across the tree, instead of just the mean. This tells us the relative confidence of predictions.

Therefor the task is taking all the trees in the model, stack the different predictions and call the standard deviation between them - insteal of the mean.

- The different trees from a `RandomForestRegressor` model can be called as elements of a list: 


```python
rf_model[0]

    DecisionTreeRegressor(max_features=0.5, min_samples_leaf=5,
                          random_state=1600246232)
```


- `predict()` can be called on individual trees - for every one of the 7988 auction predictions.



```python
rf_model[0].predict(X_valid)


    array([ 9.97461469, 10.10483758,  9.32772859, ...,  9.38366519,
            9.37713079,  9.37713079])
```



```python
len(rf_model[0].predict(X_valid))


    7988
```


- All the trees are under the `m.estimators_`.


```python
len(rf_model.estimators_)


    100
```


Let's stack all the predictions. 7988 predictions, in each of 100 trees.


```python
preds_stacked = np.stack(i.predict(X_valid) for i in rf_model.estimators_)
preds_stacked

    array([[ 9.97461469, 10.10483758,  9.32772859, ...,  9.38366519,
             9.37713079,  9.37713079],
           [10.02496635,  9.99724274,  9.23241147, ...,  9.3199054 ,
             9.38743793,  9.38743793],
           [ 9.93373553,  9.96011698,  9.01997169, ...,  9.1301562 ,
             9.22006596,  9.22006596],
           ...,
           [ 9.84292495,  9.95399866,  9.43168683, ...,  9.41749875,
             9.11293326,  9.11293326],
           [ 9.91806875, 10.12426186,  9.37828723, ...,  9.47753124,
             9.22080501,  9.22080501],
           [10.29240811, 10.00102539,  9.40523815, ...,  9.34376642,
             9.50345051,  9.50345051]])
```



```python
preds_stacked.shape

    (100, 7988)
```



```python
preds_stacked[0]

    array([ 9.97461469, 10.10483758,  9.32772859, ...,  9.38366519,
            9.37713079,  9.37713079])
```


- Lastly, we use `std` to calculate the standard deviation for every auction. 

We are setting the axis to 0 calculate the standard deviation at a column level - it takes the 100 tree prediction of the 1st auction and compares the results, takes 100 predictions of the 2nd auction and compares the results, and so forth. Giving 7988 standard deviations - one for every auction. The ones with a high standard deviation means that the trees dissagree more. If every tree gives the same prediction, the standard deviation would be 0. 


Wrapping everything into a function:


```python
def tree_variance(model):
    # Stack the estimations for every tree
    preds_stacked = np.stack(i.predict(X_valid) for i in model.estimators_)
    
    # Calculate the standard deviation
    pres_std = preds_stacked.std(0)
    
    # Discrepancies
    max_std = pres_std.max().round(3)
    max_row = np.where(pres_std == pres_std.max())[0].astype(np.int)
    min_std = pres_std.min().round(3)
    min_row = np.where(pres_std == pres_std.min())[0].astype(np.int)
    
    # Checking differences
    print("The row {} have the MAX standard deviation between trees ({})".format(max_row, max_std))
    print("The row {} have the MIN standard deviation between trees ({})".format(min_row, min_std))
    
```


```python
tree_variance(rf_model)

    The row [7083 7084] have the MAX standard deviation between trees (0.625)
    The row [5364] have the MIN standard deviation between trees (0.058)
```

As you can see, the confidence in the predictions varies widely. For the auction in the index position 6722th, the trees disagree "a lot". For the the auction 5364th, the trees predictions varely differ. 



### 2. Feature Importance

We also want to know how the model its making predictions. The feature importances give us
this insight.

The attribute `feature_importances_` gives the list of importance of every feature the model is using to create the splits. 

The feature importance algorithm loops through each tree, and then recursively explores each
branch. At each branch, it looks to see what feature was used for that split, and how
much the model improves as a result of that split. The improvement (weighted by the
number of rows in that group) is added to the importance score for that feature. This
is summed across all branches of all trees, and finally the scores are normalized such
that they add to 1.

Sadly, `.feature_importances_` doesn't provide a visual way to present the data - so we will create a function to translate the array output into a visual representation.


```python
rf_model.feature_importances_


    array([5.60696687e-04, 3.31017616e-02, 2.25121316e-02, 4.93692577e-02,
           3.80074883e-03, 2.32673274e-02, 1.17510826e-01, 7.59427101e-02,
           4.71958629e-03, 1.34531419e-02, 1.60803859e-02, 8.05581866e-03,
           2.71543507e-02, 5.76569695e-04, 2.19521447e-03, 6.22032682e-04,
           2.19242811e-03, 1.59219664e-03, 3.07939210e-04, 5.75216860e-04,
           1.05861815e-03, 2.83936750e-04, 1.14647214e-03, 8.98867415e-03,
           7.09370102e-04, 2.32816635e-03, 1.42691323e-03, 9.85881769e-04,
           7.29902352e-03, 1.87666527e-03, 1.26387160e-01, 3.53398306e-02,
           3.44002655e-02, 1.95951710e-03, 9.28346841e-04, 1.29951677e-03,
           5.37011193e-04, 3.16194890e-04, 3.62936850e-04, 4.99996690e-04,
           2.58723627e-03, 2.19530794e-03, 2.28332319e-04, 2.65172973e-04,
           2.80335713e-05, 1.72524636e-05, 1.06478776e-05, 4.18794614e-06,
           0.00000000e+00, 0.00000000e+00, 1.16771593e-04, 6.35195462e-04,
           2.45309332e-02, 1.70504793e-02, 5.21241853e-02, 8.65017421e-04,
           2.74614566e-03, 1.76327989e-01, 1.73509851e-03, 1.98023956e-02,
           1.60375078e-03, 3.41878336e-03, 4.36842581e-03, 2.15207722e-03,
           4.90070273e-03, 5.05610397e-02])
```



```python
def plot_importance(model, features, n):
    df_importance = pd.DataFrame({'Feature': features.columns, 'Importance': model.feature_importances_})
    df_importance_storted = df_importance.sort_values('Importance', ascending = False).reset_index(drop = True)
    df_importance_top = df_importance_storted.head(n)
    
    fig, ax = plt.subplots(figsize=(12,n))
    sns.barplot(x = 'Importance', y = 'Feature', 
                data = df_importance_top,
                palette = 'Blues_r')
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.ylabel('')
    sns.despine(left=True);
    
```


```python
plot_importance(rf_model, X, 10)
```


    
![png](/images/Fastbook/Chapter_9/output_102_0.png)
    


### 3. Removing Low-Importance Variables

We have 66 features in the initial mode, let's try keeping just those with a feature importance greater than 0.005:


```python
columns_keep = X.columns[rf_model.feature_importances_ > 0.005]
columns_keep


    Index(['fiModelDesc', 'fiBaseModel', 'fiSecondaryDesc', 'fiModelDescriptor',
           'ProductSize', 'fiProductClassDesc', 'ProductGroup', 'ProductGroupDesc',
           'Drive_System', 'Enclosure', 'Hydraulics', 'Tire_Size',
           'Coupler_System', 'Grouser_Tracks', 'Hydraulics_Flow', 'SalesID',
           'MachineID', 'ModelID', 'YearMade', 'saleYear', 'saleElapsed'],
          dtype='object')
```

We should trim the column both in the train and in the validation set - and then we test the model again:


```python
# Only keep important features based on `feature_importances_` attribute
X_imp = X[columns_keep]
X_valid_imp = X_valid[columns_keep]

# Retrain the model with less features
rf_model_2 = random_forest(X_imp, y)
```


```python
# New model with less features
print_rmse(rf_model_2, X_imp, X_valid_imp)


    Training RMSE: 0.180246
    Validation RMSE: 0.229436
    Out-of-Bag RMSE: 0.212309
    Out-of-Bag Accuracy: 0.906
```


```python
# Previous model
print_rmse(rf_model, X, X_valid)


    Training RMSE: 0.169543
    Validation RMSE: 0.231052
    Out-of-Bag RMSE: 0.20854
    Out-of-Bag Accuracy: 0.91
```

Our validation accuracy is about the same than before (`rf_model`), even a little bit better, and we have 45(!!) fewer columns to study:


```python
len(X.columns) - len(columns_keep)


    45
```


### 4. Removing Redundant Features

We will create a function using Spearman or rank correlation between the variables. 

Intuitively, the **Spearman correlation** between variables will be high when observations have a similar rank, and low when observations have a dissimilar (or fully opposed for a correlation of −1) rank between the two variables. 

We use rank correlation because not all the variables follow the same normal distribution and range of values (a.k.a *distribution-free/nonparametric*). For example, the distribution and range of values of the `YearID` and the `Tire_Size` of the auctions are widely different.

The only requirement to Spearman correlation is that the variables follow a given order (a.k.a *monotonic*). 


```python
def cluster_columns(df, figsize=(10,6), font_size=12):
    corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
    corr_condensed = hc.distance.squareform(1-corr)
    z = hc.linkage(corr_condensed, method='average')
    fig = plt.figure(figsize=figsize)
    hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=font_size)
    plt.show()
```


```python
cluster_columns(X_imp)
```


    
![png](/images/Fastbook/Chapter_9/output_115_0.png)
    


The more correlated the features, the early the group at the right of the rank.

Out of the 21 variables `saleElapsed` and `saleYear` seems to be closelly correlated. Same goes for:

- `Hydraulics_Flow`, `Grouser_Tracks`, and `Coupler_System`.
- `ProductGroupDesc` and `ProductGroup`.
- `fiBaseModel` and `fiModelDesc`.


Let’s try removing some of these closely related features to see if the model can be simplified. We will use OOB acurracy to see the effect of removing the variables one by one.


```python
var_redundant = ['saleElapsed', 'saleYear', 
                 'Hydraulics_Flow', 'Grouser_Tracks', 'Coupler_System', 
                 'ProductGroupDesc', 'ProductGroup', 
                 'fiBaseModel', 'fiModelDesc']
```


```python
def random_forest_redundancy(X, redundant_variables):
    print("Baseline Model with the {} most important variables".format(len(X.columns)), random_forest(X, y).oob_score_.round(3))
    {print("Model Accuracy without", i, ":", random_forest(X.drop(i, axis = 1), y).oob_score_.round(3)) for i in redundant_variables}

```


```python
random_forest_redundancy(X_imp, var_redundant)

    Baseline Model with the 21 most important variables 0.906
    Model Accuracy without saleElapsed : 0.901
    Model Accuracy without saleYear : 0.906
    Model Accuracy without Hydraulics_Flow : 0.907
    Model Accuracy without Grouser_Tracks : 0.906
    Model Accuracy without Coupler_System : 0.907
    Model Accuracy without ProductGroupDesc : 0.906
    Model Accuracy without ProductGroup : 0.907
    Model Accuracy without fiBaseModel : 0.906
    Model Accuracy without fiModelDesc : 0.906
```

As we see, removing redundant variables doesn't seem to affect the accuracy. 

We can try to keep 4 and remove 5 that seems redundant, and see the accuracy impact - e.g. from `['Hydraulics_Flow', 'Grouser_Tracks', 'Coupler_System']` only keeping `Grouser_Tracks`.


```python
var_drop = ['saleElapsed', 'Hydraulics_Flow', 'Grouser_Tracks', 'ProductGroupDesc','fiBaseModel']
```


```python
# We remove the redundant variables
X_final = X_imp.drop(var_drop, axis=1)
X_valid_final = X_valid_imp.drop(var_drop, axis=1)

# Fit the model with the reduced features dataset
rf_model_3 = random_forest(X_final, y)
```


```python
print_rmse(rf_model_3, X_final, X_valid_final)


    Training RMSE: 0.188442
    Validation RMSE: 0.230612
    Out-of-Bag RMSE: 0.219032
    Out-of-Bag Accuracy: 0.9
```

The validation RMSE and Out-of-Bag RMSE is the metrics that we most care about, as they rely on data that the model hasn't seen before. And they are looking good! 

We made a model with 17 features that achieve almost the same loss as the model using 53 features.


```python
X_final.to_csv('data/X_final.csv', index=False)
X_valid_final.to_csv('data/X_valid_final.csv', index=False)
```

### 5. Partial Dependence

Partial dependence plots try to answer the question: if a row varied on nothing other than the feature in question, how would it impact the dependent variable?

As we’ve seen, the two most important predictors are `ProductSize` and `YearMade`.

What we do is replace every single value in the `YearMade` column with 1950,
and then calculate the predicted sale price for every auction, and take the average
over all auctions. Then we do the same for 1951, 1952, and so forth until our final
year of 2011. This isolates the effect of only YearMade


```python
from sklearn.inspection import plot_partial_dependence
```


```python
fig,ax = plt.subplots(figsize=(12, 4))
plot_partial_dependence(rf_model_3, X_valid_final, ['YearMade','ProductSize'], ax=ax);
```


    
![png](/images/Fastbook/Chapter_9/output_129_0.png)
    


- The `YearMade` partial plot show a nearly linear relationship between `YearMade` and `Salesprice` after year 1970

- The `ProductSize` partial shows that for 5 and 6 classes the auctions have the lowest `Salesprice`.

This kind of insights can give an extra advantage to squish a bit of accuracy in, for example, Kaggle competitions.

### 6. Data Leakage

Data leakage is another way to get an advantage in programming competitions.

Tips to identify data leakages:

- Check whether the accuracy of the model is **too good to be true**.
- Look for **important predictors that don’t make sense in practice**.
- Look for **partial dependence plot** results that don’t make sense in practice.

The only question that remains is:

*For predicting with a particular row of data, what were the most important factors, and how did they influence that prediction?*

## Tree-models and Extrapolation

A problem with random forests, like all machine learning or deep learning algorithms, is that they don’t always generalize well to new data.

Let’s consider the simple task of creating a `RandomForestRegressor()` that learns from the first 30 points and try to predict the next 10. The "features" is a one-dimensional tensor and the "target" is the same one-dimensional tensor plus some noise mady by adding random numbers from a normal distribution.

Therefore, the relation between the features and the target is almost linear plus noise - they are almost the same number. Any human could see the corralation between the `X` numbers and the `y` numbers in the tensors below:


```python
X = torch.linspace(0,20, steps=40)
target = X + torch.randn_like(X)
```


```python
X


    tensor([ 0.0000,  0.5128,  1.0256,  1.5385,  2.0513,  2.5641,  3.0769,  3.5897,
             4.1026,  4.6154,  5.1282,  5.6410,  6.1538,  6.6667,  7.1795,  7.6923,
             8.2051,  8.7179,  9.2308,  9.7436, 10.2564, 10.7692, 11.2821, 11.7949,
            12.3077, 12.8205, 13.3333, 13.8462, 14.3590, 14.8718, 15.3846, 15.8974,
            16.4103, 16.9231, 17.4359, 17.9487, 18.4615, 18.9744, 19.4872, 20.0000])
```



```python
target

    tensor([-4.8952e-01,  1.1971e-02,  2.6504e+00,  1.0710e+00,  3.3717e+00,
             7.5045e-01,  1.3698e+00,  2.2385e+00,  5.2067e+00,  4.5659e+00,
             5.5455e+00,  4.8772e+00,  7.8788e+00,  5.7786e+00,  6.2888e+00,
             6.7935e+00,  8.7160e+00,  9.1112e+00,  8.8788e+00,  1.0618e+01,
             1.0592e+01,  1.2324e+01,  1.1950e+01,  1.1621e+01,  1.1374e+01,
             1.1379e+01,  1.4004e+01,  1.4633e+01,  1.4821e+01,  1.5068e+01,
             1.4898e+01,  1.4943e+01,  1.6194e+01,  1.6307e+01,  1.8478e+01,
             1.7215e+01,  1.9295e+01,  1.9452e+01,  2.0081e+01,  1.9914e+01])
```



```python
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
ax1.title.set_text('Generated "features"')
ax2.title.set_text('Generated target')
ax3.title.set_text('Relationship')
ax1.plot(X)
ax2.plot(target)
ax3.scatter(X, target)
```




    
![png](/images/Fastbook/Chapter_9/output_139_1.png)
    


The linear relationship is really straightforward, as we can see in the "Relationship" plot - X and y values are very correlated. 

It should be easy for the model to take the relationship between the first 30 points and extrapolate it to predict the next 10 right? Let's try!


```python
X = X.unsqueeze(1)
# Fitting the first 30 datapoints
tree_model = RandomForestRegressor().fit(X[:30], target[:30])
# Predictions
y_preds = tree_model.predict(X)
```


```python
# Real values in blue
plt.scatter(X, target)
# Predictions in red
plt.scatter(X, y_preds, color='red', alpha=0.5);
```


    
![png](/images/Fastbook/Chapter_9/output_142_0.png)
    


**The random forest is not able to see the "clear" linear relationship between our linear points!**

Remember that a random forest just averages the predictions of a number of trees. And a tree simply predicts the average value of the rows in a leaf. Therefore, **a tree and a
random forest can never predict values outside the range of the training data**. 

**This is particularly problematic for data indicating a trend over time, such as inflation, and you wish to make predictions for a future time. Your predictions will be systematically too low.**

Random forests are not able to extrapolate outside the types of data they have seen, in a more general sense. That’s why we need to make sure our validation set does not contain out-of-domain data.


## Finding Out-of-Domain Data

The main problem above is that test set is distributed in a different way than the training data. **If the tree model hasn't seen a value more than 16, it will never predict more than 16**. 

Sometimes it is hard to know whether your test set is distributed in the same way as your training data, or, if it is different, which columns reflect that difference. There’s
an easy way to figure this out, which is ironically using a random forest!

But in this case, we don’t use the random forest to predict our actual dependent variable. Instead, **we try to predict whether a row is in the validation set or the training
set**. To see this in action, let’s combine our training and validation sets, create a dependent variable that represents which dataset each row comes from, build a random forest using that data, and get its feature importance


```python
# Create a column with the target
X_final['is_valid'] = 0
X_valid_final['is_valid'] = 1

# Concat the dfs and create variables
X = pd.concat([X_final, X_valid_final])
is_valid = X['is_valid'].copy()

# Drop the new variable from the features dataset
X = X.drop('is_valid', axis=1)
X_final = X_final.drop('is_valid', axis=1)
X_valid_final = X_valid_final.drop('is_valid', axis=1)

# Create a model with the target being `is_valid`
rf_model_ODD = random_forest(X, is_valid)
```


```python
plot_importance(rf_model_ODD, X, 10)
```


    
![png](/images/Fastbook/Chapter_9/output_147_0.png)
    


- The difference in `SalesID` suggests that identifiers for auction sales might increment over time, we'll find bigger `SalesID` values in the validation set. 
- `saleYear` suggest that the latest auctions are in the validation set.
- `MachineID` suggests something similar might be happening for individual items sold in those auctions, we'll find bigger `MachineID` values in the validation set. 
- `YearMade`, same same.

All these features that are different in the training and validation set have something in common: **they encode the date of the auction**. This is an issue because we are training the past datapoints to predict future datapoints, and as we have seen in the *Tree-models and Extrapolation* section this works badly.

Let's try to remove the "date variables" to see if we lose accuracy removing the variables 1 by 1:


```python
random_forest_redundancy(X_final, ['SalesID', 'saleYear', 'MachineID'])


    Baseline Model with the 16 most important variables 0.9
    Model Accuracy without SalesID : 0.899
    Model Accuracy without saleYear : 0.842
    Model Accuracy without MachineID : 0.901
```

We should not remove `saleYear`, as we see a drop in the accuracy.But we can remove `SalesID` and `MachineID`.

We should look as well at the RMSE loss, not only the accuracy:


```python
# Reduced datasets
X_final_2 = X_final.drop(['SalesID', 'MachineID'], axis = 1)
X_valid_final_2 = X_valid_final.drop(['SalesID', 'MachineID'], axis = 1)

# Re-train the model
rf_model_4 = random_forest(X_final_2, y)
```


```python
# New model
print_rmse(rf_model_4, X_final_2, X_valid_final_2)


    Training RMSE: 0.200645
    Validation RMSE: 0.227668
    Out-of-Bag RMSE: 0.219294
    Out-of-Bag Accuracy: 0.9
```

We have improved a bit the model wrt the previous one:

- Training RMSE: 0.188482
- Validation RMSE: 0.230356
- Out-of-Bag RMSE: 0.219128
- Out-of-Bag Accuracy: 0.9

What we do with `salesYear`. The distrubtion of this variable is different in the training and in the validation set, but remocing it reduces the accuracy. However... can we trim it?

One thing that might help in our case is to simply avoid using old data. Often, old
data shows relationships that just aren’t valid anymore. Let’s try just using the most
recent few years of the data:


```python
X['saleYear'].hist();
```

    
![png](/images/Fastbook/Chapter_9/output_154_0.png)
    

```python
X_final_2[X_final_2['saleYear'] > 2004]
```


<table class="dataframe">
  <thead>
    <tr >
      <th></th>
      <th>fiModelDesc</th>
      <th>fiSecondaryDesc</th>
      <th>fiModelDescriptor</th>
      <th>ProductSize</th>
      <th>fiProductClassDesc</th>
      <th>ProductGroup</th>
      <th>Drive_System</th>
      <th>Enclosure</th>
      <th>Hydraulics</th>
      <th>Tire_Size</th>
      <th>Coupler_System</th>
      <th>ModelID</th>
      <th>YearMade</th>
      <th>saleYear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>963</td>
      <td>43</td>
      <td>0</td>
      <td>0</td>
      <td>59</td>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>17</td>
      <td>0</td>
      <td>3157</td>
      <td>2004</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3716</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>8</td>
      <td>4</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>332</td>
      <td>2001</td>
      <td>2011</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4261</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>17311</td>
      <td>2007</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>5</th>
      <td>500</td>
      <td>59</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4605</td>
      <td>2004</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>7</th>
      <td>749</td>
      <td>43</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3539</td>
      <td>2001</td>
      <td>2005</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>412693</th>
      <td>490</td>
      <td>108</td>
      <td>0</td>
      <td>5</td>
      <td>13</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>21435</td>
      <td>2005</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>412694</th>
      <td>491</td>
      <td>108</td>
      <td>0</td>
      <td>5</td>
      <td>17</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>21436</td>
      <td>2005</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>412695</th>
      <td>490</td>
      <td>108</td>
      <td>0</td>
      <td>5</td>
      <td>13</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>21435</td>
      <td>2005</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>412696</th>
      <td>490</td>
      <td>108</td>
      <td>0</td>
      <td>5</td>
      <td>13</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>21435</td>
      <td>2006</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>412697</th>
      <td>491</td>
      <td>108</td>
      <td>0</td>
      <td>5</td>
      <td>17</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>21436</td>
      <td>2006</td>
      <td>2012</td>
    </tr>
  </tbody>
</table>
<p>230144 rows × 14 columns</p>




```python
y[X_final_2['saleYear']>2004]


    0         11.097410
    3         10.558414
    4          9.305651
    5         10.184900
    7         10.203592
                ...    
    412693     9.210340
    412694     9.259130
    412695     9.433484
    412696     9.210340
    412697     9.472705
    Name: SalePrice, Length: 230144, dtype: float32
```



```python
X_trimmed =       X_final_2[X_final_2['saleYear'] > 2004]
X_valid_trimmed = X_valid_final_2[X_valid_final_2['saleYear'] > 2004]
y_trimmed =       y[X_final_2['saleYear']>2004]

rf_model_5 = random_forest(X_trimmed, y_trimmed)
```


```python
# Previous RMSE
print_rmse(rf_model_4, X_final_2, X_valid_final_2, y, y_valid)


    Training RMSE: 0.200645
    Validation RMSE: 0.227668
    Out-of-Bag RMSE: 0.219294
    Out-of-Bag Accuracy: 0.9
```


```python
# New RMSE
print_rmse(rf_model_5, X_trimmed, X_valid_trimmed, y_trimmed, y_valid)


    Training RMSE: 0.19193
    Validation RMSE: 0.227894
    Out-of-Bag RMSE: 0.218182
    Out-of-Bag Accuracy: 0.904
```

It’s a tiny bit better, which shows that you shouldn’t always use your entire dataset;
sometimes a subset can be better.
Let’s see if using a neural network helps.

## Neural Networks for tabular data

We can use the same approach to build a neural network model. Let’s first replicate
the steps we took to set up the TabularPandas object:


```python
X_final_2
```

<table class="dataframe">
  <thead>
    <tr >
      <th></th>
      <th>fiModelDesc</th>
      <th>fiSecondaryDesc</th>
      <th>fiModelDescriptor</th>
      <th>ProductSize</th>
      <th>fiProductClassDesc</th>
      <th>ProductGroup</th>
      <th>Drive_System</th>
      <th>Enclosure</th>
      <th>Hydraulics</th>
      <th>Tire_Size</th>
      <th>Coupler_System</th>
      <th>ModelID</th>
      <th>YearMade</th>
      <th>saleYear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>963</td>
      <td>43</td>
      <td>0</td>
      <td>0</td>
      <td>59</td>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>17</td>
      <td>0</td>
      <td>3157</td>
      <td>2004</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1745</td>
      <td>57</td>
      <td>0</td>
      <td>3</td>
      <td>62</td>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>12</td>
      <td>0</td>
      <td>77</td>
      <td>1996</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>2</th>
      <td>336</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>39</td>
      <td>3</td>
      <td>0</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>7009</td>
      <td>2001</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3716</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>8</td>
      <td>4</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>332</td>
      <td>2001</td>
      <td>2011</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4261</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>17311</td>
      <td>2007</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>412693</th>
      <td>490</td>
      <td>108</td>
      <td>0</td>
      <td>5</td>
      <td>13</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>21435</td>
      <td>2005</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>412694</th>
      <td>491</td>
      <td>108</td>
      <td>0</td>
      <td>5</td>
      <td>17</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>21436</td>
      <td>2005</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>412695</th>
      <td>490</td>
      <td>108</td>
      <td>0</td>
      <td>5</td>
      <td>13</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>21435</td>
      <td>2005</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>412696</th>
      <td>490</td>
      <td>108</td>
      <td>0</td>
      <td>5</td>
      <td>13</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>21435</td>
      <td>2006</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>412697</th>
      <td>491</td>
      <td>108</td>
      <td>0</td>
      <td>5</td>
      <td>17</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>21436</td>
      <td>2006</td>
      <td>2012</td>
    </tr>
  </tbody>
</table>
<p>404710 rows × 14 columns</p>



```python
df_nn = pd.read_csv('TrainAndValid.csv', low_memory=False)
df_nn['ProductSize'] = df_nn['ProductSize'].astype('category')
df_nn['ProductSize'].cat.set_categories(sizes, ordered=True, inplace=True)
df_nn['SalePrice'] = np.log(df_nn['SalePrice'])
df_nn = add_datepart(df_nn, 'saledate')
df_nn_final = df_nn[list(X_final_2.columns) + ['SalePrice']]
```


Categorical columns are handled very differently in neural networks, compared to decision tree approaches - **For Neural Networks we will use embeddings**.

To create embeddings, fastai needs to determine which columns should be treated as categorical variables. It does this by
comparing the number of distinct levels in the variable to the value of the `max_card` parameter. If it’s lower, fastai will treat the variable as categorical. Embedding sizes
larger than 10,000 should generally be used only after you’ve tested whether there are better ways to group the variable, so we’ll use 9,000 as our max_card value:


```python
cont_nn, cat_nn = cont_cat_split(df_nn_final, max_card=9000, dep_var='SalePrice')
```


```python
cat_nn

    ['fiModelDesc',
     'fiSecondaryDesc',
     'fiModelDescriptor',
     'ProductSize',
     'fiProductClassDesc',
     'ProductGroup',
     'Drive_System',
     'Enclosure',
     'Hydraulics',
     'Tire_Size',
     'Coupler_System',
     'ModelID',
     'YearMade',
     'saleYear']
```



In this case, however, there’s one variable that we absolutely do not want to treat as
categorical: `saleYear`. A categorical variable cannot, by definition, extrapolate
outside the range of values that it has seen, but we want to be able to predict auction
sale prices in the future. Therefore, we need to make this a continuous variable:


```python
cont_nn.append('saleYear')
cat_nn.remove('saleYear')
```


```python
df_nn_final[cat_nn].nunique()


    fiModelDesc           5059
    fiSecondaryDesc        177
    fiModelDescriptor      140
    ProductSize              6
    fiProductClassDesc      74
    ProductGroup             6
    Drive_System             4
    Enclosure                6
    Hydraulics              12
    Tire_Size               17
    Coupler_System           2
    ModelID               5281
    YearMade                73
    dtype: int64
```



We can create our TabularPandas object in the same way as when we created our
random forest, with one very important addition: **normalization**. A random forest
does not need any normalization—the tree building procedure cares only about the
order of values in a variable, not at all about how they are scaled. But as we have seen,
a neural network definitely does care about this. Therefore, we add the Normalize
processor when we build our TabularPandas object:


```python
procs_nn = [Categorify, FillMissing, Normalize]
to_nn = TabularPandas(df_nn_final, 
                      procs_nn, 
                      cat_nn, 
                      cont_nn,
                      splits=splits, 
                      y_names='SalePrice')
```

We load the data into a `DataLoader`, and set a range for the target. It’s a good idea to set `y_range` for regression models, so let’s find
the min and max of our dependent variable:


```python
# Features into the dataloader
dls = to_nn.dataloaders(1024)
# Target
y = to_nn.train.y

# Range 
y.min(),y.max()




    (8.465899, 11.863583)
```


Lastly, we build the model:


```python
learn = tabular_learner(dls, y_range=(8,12), layers=[500,250], n_out=1, loss_func=F.mse_loss)
```

There’s no need to use fine_tune, so we’ll train with fit_one_cycle for a few epochs
and see how it looks:


```python
learn.fit_one_cycle(10, learn.lr_find()[0])
```



<table class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.040625</td>
      <td>0.051301</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.043241</td>
      <td>0.052923</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.043518</td>
      <td>0.053630</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.042394</td>
      <td>0.054047</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.040913</td>
      <td>0.052986</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.040410</td>
      <td>0.052649</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.038336</td>
      <td>0.051216</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.037320</td>
      <td>0.052022</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.036384</td>
      <td>0.051955</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.036191</td>
      <td>0.051794</td>
      <td>00:03</td>
    </tr>
  </tbody>
</table>


    
![png](/images/Fastbook/Chapter_9/output_179_2.png)
    


We can use our r_mse function to compare the result to the random forest result we
got earlier:

```python
preds,targs = learn.get_preds()
r_mse(preds,targs)


    0.227582
```


This gives us a similiar result than the best random forest achieved previously. Before we move on, let’s save our model in case we want to come back to it again
later:


```python
learn.save('nn')

    Path('models/nn.pth')
```


## Ensembling

We have two very different models, trained using very different algorithms:
a random forest and a neural network. It would be reasonable to expect that
the kinds of errors that each one makes would be quite different. Therefore, we might
**expect that the average of their predictions would be better than either one’s individual
predictions**.

When ensembling the results together, one minor issue we have to be aware of is that our PyTorch model and our sklearn
model create data of different types.

- PyTorch gives us a rank-2 tensor (a column matrix)
- NumPy gives us a rank-1 array (a vector). 


```python
learn.get_preds()


    (tensor([[10.2192],
             [10.0230],
             [ 9.3750],
             ...,
             [ 9.3017],
             [ 9.2062],
             [ 9.2062]]),
     tensor([[10.0432],
             [10.0858],
             [ 9.3927],
             ...,
             [ 9.3501],
             [ 9.1050],
             [ 8.9554]]))
```



```python
rf_model_5.predict(X_valid_final_2)

    array([10.07742579, 10.03322471,  9.35772406, ...,  9.34768389,
            9.24583077,  9.24583077])
```


`squeeze` removes any unit axes from a tensor, and to_np converts it into a NumPy array:


```python
to_np(preds.squeeze())

    array([10.219167, 10.023037,  9.375016, ...,  9.301746,  9.206213,
            9.206213], dtype=float32)
```


```python
ensemble_preds = (to_np(preds.squeeze()) + rf_model_5.predict(X_valid_final_2))/2
```


```python
r_mse(ensemble_preds, y_valid)


    0.22322
```


Notice that an RMSE of 0.223 is the best result so far - better than the most tunned random forest and the neural network!

## Boosting

In another important approach to ensembling, called boosting, where we add models
instead of averaging them. Here is how boosting works:
    
1. Train a small model that underfits your dataset.
2. Calculate the predictions in the training set for this model.
3. Subtract the predictions from the targets; these are called the residuals and represent
the error for each point in the training set.
4. Go back to step 1, but **instead of using the original targets, use the residuals as
the targets for the training**.
5. Continue doing this until you reach a stopping criterion, such as a maximum
number of trees, or you observe your validation set error getting worse.

Using this approach, each new tree will be attempting to fit the error of all of the previous
trees combined.

Note that, unlike with random forests, with this approach, **there is nothing to stop us
from overfitting**. Using more trees in a random forest does not lead to overfitting,
because each tree is independent of the others. But **in a boosted ensemble, the more
trees you have, the better the training error becomes**, and eventually you will see
overfitting on the validation set.


## Conclusion


We have discussed two approaches to tabular modeling: decision tree ensembles and
neural networks. We’ve also mentioned two decision tree ensembles: random forests
and gradient boosting machines. Each is effective but also requires compromises:

- **Random forests** are the easiest to train, because they are extremely resilient to
hyperparameter choices and require little preprocessing. They are fast to train,
and should not overfit if you have enough trees. But they can be a little less accurate,
especially if extrapolation is required, such as predicting future time periods.

- **Gradient boosting** machines in theory are just as fast to train as random forests,
but in practice you will have to try lots of hyperparameters. They can overfit, but
they are often a little more accurate than random forests.

- **Neural networks** take the longest time to train and require extra preprocessing,
such as normalization; this normalization needs to be used at inference time as
well. They can provide great results and extrapolate well, but only if you are careful
with your hyperparameters and take care to avoid overfitting.


**We suggest starting your analysis with a random forest**. This will give you a strong
baseline, and you can be confident that it’s a reasonable starting point. You can then
use that model for feature selection and partial dependence analysis, to get a better
understanding of your data.
