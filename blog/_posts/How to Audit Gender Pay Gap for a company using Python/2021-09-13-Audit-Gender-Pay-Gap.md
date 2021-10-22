---
layout: post
title: 'How to Audit Gender Pay Gap for a company using Python'
---

 {{ page.title }}
================

# Introduction

In this post I will go though a technical step-by-step guide for how to analyze your company’s gender pay gap (including example data and code).

Gender Pay Gap remains a recurrent problem at many companies. In the United States, Pew Research Center evaluated the trend in gender pay gap from the 1980 until 2020 and discovered that women earned 84% of what men earned. In other words, it would take an extra 42 days of work for women to earn what men did in 2020.

While this gap is closing, [specially for young workers](https://www.pewresearch.org/fact-tank/2021/05/25/gender-pay-gap-facts/), is still an issue that can affect the morale of the company as whole, the decision of prospect candidates to apply to your company, or corrupt the meritocracy system in promotions just to mention a few problems that can derive from not taking into account the Gender Pay Gap.

For this post I will go though a technical step-by-step guide for how to analyze your company’s gender pay gap (including example data and code) showing how to apply statistical methods to evauate Gender Pay Gap and move forward a better gender pay fairness. 

We will analyze two types:

1. **Unadjusted Gender Pay Gap**. Is there a difference between average pay for men and women in the company?

2. **Adjusted Gender Pay Gap**. Are there differences between average pay for men and women **after we’ve accounted for differences among workers** in education, experience, job roles, employee performance and other factors aside from gender
that affect pay?

The sections include several **insights** that explains the results marked with a &#128204; sign.

# 1. Unadjusted Gender Pay Gap

**Is there a difference between average pay for men and women in the company?**

We will start defining Unadjusted Gender Pay Gap: the **percentage of salary paid to male with respect to female workers**.

$$
\begin{aligned}
Unadjusted \ Gender \ Pay \ Gap = \frac{Average \ Male \ Pay - Average \ Female \ Pay}{Average \ Male \ Pay}
\end{aligned}
$$

This is the simplest definition for Gender Pay Gap. This Pay Gap is "unadjusted" in the sense that it doesn't take into account other factors but gender. Once we have this Unadjusted Gender Pay Gap we will include more factors that pay a role in pay differences.

For the analysis, we will use a **sample of data** from the job seekers website Glassdoor that contains 1000 people with the following information:

- Job title
- Gender
- Age
- Score on most recent performance evaluation (1 to 5)
- Education
- Company department
- Seniority (1 to 5)
- Annual base salary
- Annual bonuses, comimissions, stock awards or other compensations

We will load the data using `pandas` and we will use `numpy` to create numerical variables.

```python
import pandas as pd 
import numpy as np
```
```python
df = pd.read_csv("https://glassdoor.box.com/shared/static/beukjzgrsu35fqe59f7502hruribd5tt.csv")
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 9 columns):
     #   Column     Non-Null Count  Dtype 
    ---  ------     --------------  ----- 
     0   jobTitle   1000 non-null   object
     1   gender     1000 non-null   object
     2   age        1000 non-null   int64 
     3   perfEval   1000 non-null   int64 
     4   edu        1000 non-null   object
     5   dept       1000 non-null   object
     6   seniority  1000 non-null   int64 
     7   basePay    1000 non-null   int64 
     8   bonus      1000 non-null   int64 
    dtypes: int64(5), object(4)
    memory usage: 70.4+ KB
    


```python
df.head()
```

<table>
  <thead>
    <tr>
      <th>jobTitle</th>
      <th>gender</th>
      <th>age</th>
      <th>perfEval</th>
      <th>edu</th>
      <th>dept</th>
      <th>seniority</th>
      <th>basePay</th>
      <th>bonus</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Graphic Designer</td>
      <td>Female</td>
      <td>18</td>
      <td>5</td>
      <td>College</td>
      <td>Operations</td>
      <td>2</td>
      <td>42363</td>
      <td>9938</td>
    </tr>
    <tr>
      <td>Software Engineer</td>
      <td>Male</td>
      <td>21</td>
      <td>5</td>
      <td>College</td>
      <td>Management</td>
      <td>5</td>
      <td>108476</td>
      <td>11128</td>
    </tr>
    <tr>
      <td>Warehouse Associate</td>
      <td>Female</td>
      <td>19</td>
      <td>4</td>
      <td>PhD</td>
      <td>Administration</td>
      <td>5</td>
      <td>90208</td>
      <td>9268</td>
    </tr>
    <tr>
      <td>Software Engineer</td>
      <td>Male</td>
      <td>20</td>
      <td>5</td>
      <td>Masters</td>
      <td>Sales</td>
      <td>4</td>
      <td>108080</td>
      <td>10154</td>
    </tr>
    <tr>
      <td>Graphic Designer</td>
      <td>Male</td>
      <td>26</td>
      <td>5</td>
      <td>Masters</td>
      <td>Engineering</td>
      <td>5</td>
      <td>99464</td>
      <td>9319</td>
    </tr>
  </tbody>
</table>

<p class="alert-info" >Using your own data, please be sure to keep all personally identifying information out of the data file. Personal names or employee numbers should not be in the file. It’s important to protect employee privacy and anonymity at all times while conducting a gender pay audit. </p>


We can start the analysis by creating a pandas' `pivot_table()` that groups the workers by gender and prints the mean and median salaries:


```python
male_female_table = pd.pivot_table(data = df, 
                                   index = 'gender', 
                                   values = 'basePay', 
                                   aggfunc= [np.mean, np.median, 'count'])
male_female_table
```

<table>
  <thead>
    <tr>
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>count</th>
    </tr>
    <tr>
      <th></th>
      <th>basePay</th>
      <th>basePay</th>
      <th>basePay</th>
    </tr>
    <tr>
      <th>gender</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Female</th>
      <td>89942.818376</td>
      <td>89913.5</td>
      <td>468</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>98457.545113</td>
      <td>98223.0</td>
      <td>532</td>
    </tr>
  </tbody>
</table>




The company has **more male workers and they earn more by average**. 

<p class="mark">Why we have calculated the median? It is useful?</p>

Calculating the median pay along the average is important as **the average can be affected by outliers** and therefore could picture a misleading general image of the pay scheme. 

For example, a highly paid female CEO can raise the average pay of women, but it is only an outliear and would not represent the general pay scheme of women within the company. That is why is important to add the median and observe if there is a difference between the mean and the median that could signal outliers. 

For this specific dataset, the median and mean are similar, and both can be used to depict the general payroll of the company. 

We can code some cosmetic changes to get the right format, but they are entirely optional:


```python
# Cosmetic changes of the table
male_female_table = (male_female_table.stack(level=1)
                                      .reset_index()
                                      .drop(columns= 'level_1')
                                      .set_index("gender")
                                      .rename_axis(None)
                                      .astype(np.int64)               
                    )

male_female_table
```



<table >
  <thead>
    <tr>
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Female</th>
      <td>89942</td>
      <td>89913</td>
      <td>468</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>98457</td>
      <td>98223</td>
      <td>532</td>
    </tr>
  </tbody>
</table>



To calculate the exact difference between the base salary of male and female workers we can use `diff()` and `pct_change()`. We can use`diff()` to calculate the difference in absolute terms and `pct_change()` for the percentage difference between both genders. The table can be transposed using the `T` method to display the data more clearly.


```python
male_female_table.diff()[1:].T
```

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean</th>
      <td>8515.0</td>
    </tr>
    <tr>
      <th>median</th>
      <td>8310.0</td>
    </tr>
    <tr>
      <th>count</th>
      <td>64.0</td>
    </tr>
  </tbody>
</table>



```python
male_female_table.pct_change()[1:].T
```

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean</th>
      <td>0.094672</td>
    </tr>
    <tr>
      <th>median</th>
      <td>0.092423</td>
    </tr>
    <tr>
      <th>count</th>
      <td>0.136752</td>
    </tr>
  </tbody>
</table>


From this first analysis we can get a first round of insights.

## &#128204; First round of Insights &#128204;

- **Unadjusted** Gender Pay Gap is present in the company.
- **Male workers get paid 9.5% more base salary** than female workers on average, which represents roughly 8.5k more annually.
- There are roughly **14% more male workers** in the company.
- There is a difference between average pay for men and women in the company, but **we don't if the cause is gender or other characteristics**.


To get a little more deep into the gender differences within this hyphotetical company, let's group the workers into their respective roles using `pivot_table()` again.



```python
salary_by_gender_jobtitle = pd.pivot_table(data = df,
                                           values= 'basePay',
                                           index = 'jobTitle',
                                           columns= 'gender',
                                           aggfunc= [np.mean, 'count'])


salary_by_gender_jobtitle
```


<table>
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">mean</th>
      <th colspan="2" halign="left">count</th>
    </tr>
    <tr>
      <th>gender</th>
      <th>Female</th>
      <th>Male</th>
      <th>Female</th>
      <th>Male</th>
    </tr>
    <tr>
      <th>jobTitle</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Data Scientist</th>
      <td>95704.792453</td>
      <td>89222.629630</td>
      <td>53</td>
      <td>54</td>
    </tr>
    <tr>
      <th>Driver</th>
      <td>86867.630435</td>
      <td>91952.666667</td>
      <td>46</td>
      <td>45</td>
    </tr>
    <tr>
      <th>Financial Analyst</th>
      <td>95458.326531</td>
      <td>94607.034483</td>
      <td>49</td>
      <td>58</td>
    </tr>
    <tr>
      <th>Graphic Designer</th>
      <td>92243.291667</td>
      <td>89595.800000</td>
      <td>48</td>
      <td>50</td>
    </tr>
    <tr>
      <th>IT</th>
      <td>90475.720000</td>
      <td>91021.978261</td>
      <td>50</td>
      <td>46</td>
    </tr>
    <tr>
      <th>Manager</th>
      <td>127252.277778</td>
      <td>124848.930556</td>
      <td>18</td>
      <td>72</td>
    </tr>
    <tr>
      <th>Marketing Associate</th>
      <td>76119.177570</td>
      <td>81881.818182</td>
      <td>107</td>
      <td>11</td>
    </tr>
    <tr>
      <th>Sales Associate</th>
      <td>91894.209302</td>
      <td>94663.117647</td>
      <td>43</td>
      <td>51</td>
    </tr>
    <tr>
      <th>Software Engineer</th>
      <td>94701.000000</td>
      <td>106371.485149</td>
      <td>8</td>
      <td>101</td>
    </tr>
    <tr>
      <th>Warehouse Associate</th>
      <td>92428.260870</td>
      <td>86553.431818</td>
      <td>46</td>
      <td>44</td>
    </tr>
  </tbody>
</table>

Again, we apply some formating changes that will affect how the table's looks:


```python
# Cosmetic changes to change multilevel index and making an extra column
salary_by_gender_jobtitle.columns = salary_by_gender_jobtitle.columns.to_flat_index()
salary_by_gender_jobtitle.index.name = None
salary_by_gender_jobtitle.rename(columns = {

  salary_by_gender_jobtitle.columns[0]: 'Average Female Base Pay',
  salary_by_gender_jobtitle.columns[1]: 'Average Male Base Pay',
  salary_by_gender_jobtitle.columns[2]: 'Female headcount',
  salary_by_gender_jobtitle.columns[3]: 'Male headcount'}, 
                                inplace = True) 

salary_by_gender_jobtitle['Base Pay Difference'] = salary_by_gender_jobtitle['Average Female Base Pay'] - salary_by_gender_jobtitle['Average Male Base Pay']

salary_by_gender_jobtitle.sort_values(by = 'Base Pay Difference', inplace = True)

salary_by_gender_jobtitle.astype(np.int64)
```
<table>
  <thead>
    <tr>
      <th></th>
      <th>Average Female Base Pay</th>
      <th>Average Male Base Pay</th>
      <th>Female headcount</th>
      <th>Male headcount</th>
      <th>Base Pay Difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Software Engineer</th>
      <td>94701</td>
      <td>106371</td>
      <td>8</td>
      <td>101</td>
      <td>-11670</td>
    </tr>
    <tr>
      <th>Marketing Associate</th>
      <td>76119</td>
      <td>81881</td>
      <td>107</td>
      <td>11</td>
      <td>-5762</td>
    </tr>
    <tr>
      <th>Driver</th>
      <td>86867</td>
      <td>91952</td>
      <td>46</td>
      <td>45</td>
      <td>-5085</td>
    </tr>
    <tr>
      <th>Sales Associate</th>
      <td>91894</td>
      <td>94663</td>
      <td>43</td>
      <td>51</td>
      <td>-2768</td>
    </tr>
    <tr>
      <th>IT</th>
      <td>90475</td>
      <td>91021</td>
      <td>50</td>
      <td>46</td>
      <td>-546</td>
    </tr>
    <tr>
      <th>Financial Analyst</th>
      <td>95458</td>
      <td>94607</td>
      <td>49</td>
      <td>58</td>
      <td>851</td>
    </tr>
    <tr>
      <th>Manager</th>
      <td>127252</td>
      <td>124848</td>
      <td>18</td>
      <td>72</td>
      <td>2403</td>
    </tr>
    <tr>
      <th>Graphic Designer</th>
      <td>92243</td>
      <td>89595</td>
      <td>48</td>
      <td>50</td>
      <td>2647</td>
    </tr>
    <tr>
      <th>Warehouse Associate</th>
      <td>92428</td>
      <td>86553</td>
      <td>46</td>
      <td>44</td>
      <td>5874</td>
    </tr>
    <tr>
      <th>Data Scientist</th>
      <td>95704</td>
      <td>89222</td>
      <td>53</td>
      <td>54</td>
      <td>6482</td>
    </tr>
  </tbody>
</table>

Taking a look at the table, it is easy to see the Gender Pay Gap is affected by the kind of job that they have.

## &#128204; Second round of Insights &#128204;

- There is **no evident male gender gap for all the departments**. For example, Female Data Scientist make 6.5k more than males in the same position. At a company level, female still earn less on average.
- The company show sharp differences in gender compositions in Sofware Engineering. **Female Software Engineer are both under-represented (7%) and under-paid (earn 11.7k less)** with respect to males.
- There are **only 11 Male Marketing Associates but they make on average 5.7k more** annually than females with the same title. 

It is clear that the kind of job affects the gap. Going though how all the possible factors (age, performance evaluation, education...) that also could have an effect would be tedious and innacurate. We will use a linear regression to explain the relationship between the salary and all the factors.

# Adjusted Gender Pay Gap

**Are there differences between average pay for men and women after we’ve accounted for differences among workers in education, experience, job roles, employee performance and other factors aside from gender that affect pay?**

To estimate your company’s gender pay gap, you’ll need to estimate a linear regression model. This regression analysis is used to describe the relationships between the characteristics of the workers (gender, performance evaluation, job title...) and the salary (base pay). In the regression equation, the **coefficients represent the relationship or effect** between each of the variables and the salary.

The regression model allows to disentangle the effect on the salary of each variable, we will call them "controls", from the effect that we are looking for: gender.

##  Regression Model

$$
\begin{aligned}
Log \ (Base \ Pay_i) = \beta_0 + \beta_1 gender_i + B_2 controls_i + \epsilon_i
\end{aligned}
$$


Explanation of the equation:

- The variable of interest is the logarithm of the base pay, $$Log \ (Base \ Pay)$$, for every worker $$i$$.

- $$\beta_1$$ is the coefficient of interest, the effect of being a male worker on base pay. 

- $$\beta_2$$ represents the effect of all the other variables or characteristics, like education, job title and others.

**The reason of using the natural logarithm** of the base salary is that it will make that the coefficients of the regression represent percentages, instead of absolute numbers. For example, a $$\beta_1$$ of 0.1 can be read as a 10% increase in salary because being male. It doesn't change the magnitude of the effect or the significance of the coefficients, only it's interpretation.

**Including more variables will isolate the effect of gender**. The more relevant variables we add to the analysis, the more isolated the effect of gender on salary. To clarify how including more controls change the estimation, we are going to do three linear regressions, adding more variables every time:

- First Model. "Unadjusted", without controls:

$$
\begin{aligned}
Log \ (Base \ Pay_i) = \beta_0 + \beta_1 gender_i + \epsilon_i
\end{aligned}
$$

This equation is equivalent to the average difference in salaries by gender.

- Second model. Controling for human capital (*Performing Evaluation, Education, Age*):

$$
\begin{aligned}
Log \ (Base \ Pay_i) = \beta_0 + \beta_1 gender_i + \beta_2 perfEval + \beta_3 edu + \beta_4 age + \epsilon_i
\end{aligned}
$$


- Third model. Controling for human capital and job characteristics (*Seniority, Department and Job Title*):

$$
\begin{eqnarray}
Log \ (Base \ Pay_i) = \beta_0 + \beta_1 gender_i + \beta_2 perfEval + \beta_3 edu + \beta_4 age \\\ + \beta_5 seniority + \beta_6 dept + \beta_7 jobTitle+ \epsilon_i
\end{eqnarray}
$$


To build the models, we will use `statsmodels`. We will write the above formulas in the `formula` parameter, specify the dataset in `data` and call `fit()` to run the regressions. The package use is quite straightforward. 


```python
import statsmodels.api as sm
```


```python
# Model 1 
model_1 = sm.ols(formula = "np.log(basePay) ~ gender", data = df)
results_1 = model_1.fit()

# Model 2
model_2 = sm.ols(formula = "np.log(basePay) ~ gender + perfEval + edu + age", data = df)
results_2 = model_2.fit()

# Model 3
model_3 = sm.ols(formula = "np.log(basePay) ~ gender + perfEval + edu + age + seniority + dept + jobTitle", data = df)
results_3 = model_3.fit()
```

*But where are the results?*

I'm going to use the package `Stargazer` to print the results nicely formatted.

```python
from stargazer.stargazer import Stargazer
```

I applied some cosmetic changes to display the data in a more compact way. However, if you want to display all the effects of each variable you simply run `Stargazer([results_1, results_2, results_3])` and it will provide a full table.


```python
results_table = Stargazer([results_1, results_2, results_3])

# Cosmetic changes
results_table.title('Gender Pay Gap Regression Results')
results_table.custom_columns(['Model 1', 'Model 2', 'Model 3'], [1, 1, 1])
results_table.show_model_numbers(False)
results_table.covariate_order(['gender[T.Male]', 'age', 'perfEval', 'seniority'])
results_table.rename_covariates({'age': 'Age', 
                                 'gender[T.Male]': 'Male',
                                 'perfEval': 'Performance Evaluation',
                                 'seniority': 'Seniority'})
results_table.add_line('Controls included:', [' ', ' ', ' '])
results_table.add_line('Department', ['X', 'X',  u'\u2713'])
results_table.add_line('Job Title', ['X', 'X',  u'\u2713'])
results_table.add_line('Education', ['X', u'\u2713',  u'\u2713'])
results_table.show_degrees_of_freedom(False)
results_table
```




<p class="mark">Gender Pay Gap Regression Results</p>

<table>
<tr>
<td colspan="4" style="border-bottom: 1px solid black">
</td>
</tr>
<tr>
<td style="text-align:left">
</td>
<td colspan="3"><em>Dependent variable:np.log(basePay)</em>
</td>
<td style="text-align:left">
</td>
<tr>
<td>
</td>
<td colspan="1">Model 1
</td>
<td colspan="1">Model 2
</td>
<td colspan="1">Model 3
</td>
</tr>
<tr>
<td colspan="4" style="border-bottom: 1px solid black">
</td>
</tr>
<tr>
<td style="text-align:left">Male
</td>
<td>0.095<sup>***</sup>
</td>
<td>0.100<sup>***</sup>
</td>
<td>0.011<sup></sup>
</td>
</tr>
<tr>
<td style="text-align:left">
</td>
<td>(0.018)
</td>
<td>(0.015)
</td>
<td>(0.009)
</td>
</tr>
<tr>
<td style="text-align:left">Age
</td>
<td>
</td>
<td>0.011<sup>***</sup>
</td>
<td>0.011<sup>***</sup>
</td>
</tr>
<tr>
<td style="text-align:left">
</td>
<td>
</td>
<td>(0.001)
</td>
<td>(0.000)
</td>
</tr>
<tr>
<td style="text-align:left">Performance Evaluation
</td>
<td>
</td>
<td>-0.007<sup></sup>
</td>
<td>0.000<sup></sup>
</td>
</tr>
<tr>
<td style="text-align:left">
</td>
<td>
</td>
<td>(0.005)
</td>
<td>(0.003)
</td>
</tr>
<tr>
<td style="text-align:left">Seniority
</td>
<td>
</td>
<td>
</td>
<td>0.109<sup>***</sup>
</td>
</tr>
<tr>
<td style="text-align:left">
</td>
<td>
</td>
<td>
</td>
<td>(0.003)
</td>
</tr>
<tr>
<td style="text-align: left">Controls included:
</td>
<td> 
</td>
<td> 
</td>
<td> 
</td>
</tr>
<tr>
<td style="text-align: left">Department
</td>
<td>X
</td>
<td>X
</td>
<td>✓
</td>
</tr>
<tr>
<td style="text-align: left">Job Title
</td>
<td>X
</td>
<td>X
</td>
<td>✓
</td>
</tr>
<tr>
<td style="text-align: left">Education
</td>
<td>X
</td>
<td>✓
</td>
<td>✓
</td>
</tr>
<td colspan="4" style="border-bottom: 1px solid black">
</td>
</tr>
<tr>
<td style="text-align: left">Observations
</td>
<td>1,000
</td>
<td>1,000
</td>
<td>1,000
</td>
</tr>
<tr>
<td style="text-align: left">R<sup>2</sup>
</td>
<td>0.027
</td>
<td>0.366
</td>
<td>0.810
</td>
</tr>
<tr>
<td style="text-align: left">Adjusted R<sup>2</sup>
</td>
<td>0.026
</td>
<td>0.363
</td>
<td>0.806
</td>
</tr>
<tr>
<td style="text-align: left">Residual Std. Error
</td>
<td>0.283
</td>
<td>0.229
</td>
<td>0.127
</td>
</tr>
<tr>
<td style="text-align: left">F Statistic
</td>
<td>28.185<sup>***</sup>
</td>
<td>95.682<sup>***</sup>
</td>
<td>208.164<sup>***</sup>
</td>
</tr>
<tr>
<td colspan="4" style="border-bottom: 1px solid black">
</td>
</tr>
<tr>
<td style="text-align: left">Note:
</td>
 <td colspan="3" style="text-align: right">
  <sup>*</sup>p&lt;0.1;
  <sup>**</sup>p&lt;0.05;
  <sup>***</sup>p&lt;0.01
 </td></tr></table>



## How to interpret the results?

Model 1 is the simplest model (the “unadjusted” pay gap), Model 2 adds controls for employee personal characteristics, and Model 3 adds all of our controls (the “adjusted” pay gap). The starts represent thresholds for p-values. The estimates that are statistically significant have *, ** or *** next to them.

**The first row** of the table shows our estimates of the “unadjusted” and “adjusted” gender pay gap. That’s the coefficient on the effect of the male gender. 

- **In `Model 1`**, a coefficient of 0.095 means there is approximately 9.5 percent “unadjusted” gender pay gap in our hypothetical company. Put differently, men on average earn about 9.5 percent more than women on average in this company. We also see this estimate is highly statistically significant. If you remember, this is exactly the same results that we got by simply comparing the salary between male and female base salary. 

- **In `Model 2`**, we add individual controls like age, performance evaluations and education. Here we see the gender pay gap hasn’t changed much. It’s still a 10 percent pay gap, which is still highly statistically significant. Whatever is causing the pay gap in our hypothetical employer isn’t due to differences in education, age or performance evaluations of men and women. 

- **In `Model 3`**, we add all of the controls we have in our dataset. **This is the “adjusted” Gender Pay Gap**. In this case, we see the gender pay gap shrinks to 1.1 percent after controlling for job title, job seniority and company department. More importantly, **this estimate is no longer statistically significant** — so we can’t conclude that the coefficient it’s really different from zero. 

In this case, we say there’s **no evidence of a systematic Gender Pay Gap on an “adjusted” basis**, after controlling for observable differences between male and female workers.

**The last rows** of the table represent different metrics that basically evaluate how good is the model fitting the data, and the overall significance for a regression model. The important metric here $$R^2$$. 

Both $$R^2$$ and $$Adjusted \ R^2$$ are statistical measures that represents the proportion of the variance for a dependent variable that's explained by the independent variables. In this case, **0.81  means that 81% of the observed variation in workers salaries can be explained by their gender, age, evaluation, and the rest of their characteristics included**. Given that there is only 19% of the variation in the salaries that we cannot explain, it's a great model fit. 

## &#128204; Third round of Insights &#128204;

- Once we control for the fact that men and women work in different roles in this company, the remaining pay difference that’s due to **the gender pay gap shrinks to near zero**.

-  Why? **Men are over-represented in higher-paying software engineer and manager roles**, while they are under-represented in lower-paying marketing roles.

## What to do with the results?

Performing a gender pay audit at your company is an important first step. Once you’ve completed your analysis, what’s the next step?

First, **you should decide whether to share your findings internally with employees**. 

Being transparent about your efforts to pay workers fairly can help boost employee engagement, and contribute toward a healthy culture of fair pay and openness. However, internal communication of the results of your gender pay audit may not be right for every employer — it’s a decision leadership teams will approach differently.

Second, **you will have to decide whether the results of your gender pay audit suggest that changes may be necessary in your company’s pay and hiring systems**. 

**If your analysis identifies a large “adjusted” pay gap**, or particularly large gender gaps within certain departments or job titles, your analysis can help target where to invest efforts to improve pay fairness in the future. 

Even **if you prove no evidence “adjusted” pay gap**, it is important to calculate the “unadjusted” pay gap to detect areas of improvements. In our example, a good following question would be why there are only a 9% of women in Software Engeneering teams and what can we do to hire more women in high paid positions.
