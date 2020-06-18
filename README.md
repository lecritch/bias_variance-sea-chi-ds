
# Bias-Variance Tradeoff


## Agenda

1. Explain what bias, variance, and error are in the context of statistical modeling
2. Explain how bias, variance and error are related via the bias-variance tradeoff
3. Explain how a holdout set can be used to evaluate a model
4. Use a test set to estimate model bias, variance and error



# 1. Explain what bias, variance, and error are in the context of statistical modeling

![which model is better](img/which_model_is_better.png)

https://towardsdatascience.com/cultural-overfitting-and-underfitting-or-why-the-netflix-culture-wont-work-in-your-company-af2a62e41288


# What makes a model good?

- We don’t ultimately care about how well your model fits your data.

- What we really care about is how well your model describes the process that generated your data.

- Why? Because the data set you have is but one sample from a universe of possible data sets, and you want a model that would work for any data set from that universe

# What is a “Model”?

 - A “model” is a general specification of relationships among variables and parameters.
E.G. Linear regression, or 

$$\Large Price = \beta_1 X_1 + \beta_0 + \epsilon$$

 - A “trained model” is a particular model with parameters estimated using some training data.



# Remember Expected Value?

- The expected value of a quantity is the weighted average of that quantity across all possible samples

![6 sided die](https://media.giphy.com/media/sRJdpUSr7W0AiQ3RcM/giphy.gif)
- The expected value of a 6-sided die is:


```python
probs = 1/6
rolls = range(1,7)

expected_value = sum([probs * roll for roll in rolls])
expected_value

```




    3.5



Suppose we created a model which always predicted that the die roll would be 3.

The **bias** of our model would be the difference between the our expected prediction (3) and the expected value (3.5).

What would the **variance** of our model be?


# Defining Error

There are 3 types of prediction error: bias, variance, and irreducible error.

$Total Error = Prediction\ Error+ Irreducible\ Error$

![defining error](img/defining_error.png)

$Total Error = Residual = Prediction\ Error+ Irreducible\ Error$
> For regression, “error” usually refers to prediction error or to residuals <br>Prediction errors are approximated by residuals

### Regression fit statistics are often called “error”
 - Sum of Squared Errors (SSE)
 - Mean Squared Error (MSE) 
     - Calculated using residuals


![residuals](img/residuals.png)

# 2. Explain how bias, variance and error are related via the bias-variance tradeoff


**Let's do a thought experiment:**

1. Imagine you've collected 5 different training sets for the same problem.
2. Now imagine using one algorithm to train 5 models, one for each of your training sets.
3. Bias vs. variance refers to the accuracy vs. consistency of the models trained by your algorithm.

![target_bias_variance](img/target.png)

http://scott.fortmann-roe.com/docs/BiasVariance.html

# Defining Model Bias and Variance

**Model Bias** is the expected prediction error from your expected trained model

**Model Variance** is the expected variation in predictions, relative to your expected trained model

**High bias** algorithms tend to be less complex, with simple or rigid underlying structure.

+ They train models that are consistent, but inaccurate on average.
+ These include linear or parametric algorithms such as regression and naive Bayes.

On the other hand, **high variance** algorithms tend to be more complex, with flexible underlying structure.

+ They train models that are accurate on average, but inconsistent.
+ These include non-linear or non-parametric algorithms such as decision trees and nearest neighbors.

### Let's take a look at our familiar King County housing data. 


```python
import pandas as pd
import numpy as np
np.random.seed(42)
df = pd.read_csv('data/kc_housing.csv', index_col='id')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7129300520</th>
      <td>20141013T000000</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>0</td>
      <td>1955</td>
      <td>0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <th>6414100192</th>
      <td>20141209T000000</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>400</td>
      <td>1951</td>
      <td>1991</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <th>5631500400</th>
      <td>20150225T000000</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>0</td>
      <td>1933</td>
      <td>0</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <th>2487200875</th>
      <td>20141209T000000</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>910</td>
      <td>1965</td>
      <td>0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>1954400510</th>
      <td>20150218T000000</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>0</td>
      <td>1987</td>
      <td>0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Let's generate random subsets of our data
df = pd.read_csv('data/kc_housing.csv', index_col='id')

#Date  is not in the correct format so we are dropping it for now.
df_low_var = df.drop(['date', 'zipcode', 'lat', 'long'], axis=1)

r_2 = []
low_var_rmse = []
for i in range(100):
    
    df_sample = df_low_var.sample(5000, replace=True)
    y = df_sample.price
    X = df_sample.drop('price', axis=1)
    
    lr = LinearRegression()
    lr.fit(X, y)
    y_hat = lr.predict(X)
    low_var_rmse.append(np.sqrt(mean_squared_error(y, y_hat)))
    r_2.append(lr.score(X,y))
    
    

```


```python
print(f'low variance sample mean mean {np.mean(sample_means_low_var)}')
print(f'low variance sample mean variance {np.var(sample_means_low_var)}')
```

    low variance sample mean mean 30465617209.508102
    low variance sample mean variance 3.4940984923692687e+18



```python
from sklearn.preprocessing import PolynomialFeatures


df = pd.read_csv('data/kc_housing.csv', index_col='id')
#Date  is not in the correct format so we are dropping it for now.
df = df.drop(['date', 'zipcode', 'lat', 'long'], axis=1)

pf = PolynomialFeatures(2)

df_poly = pd.DataFrame(pf.fit_transform(df.drop('price', axis=1)))
df_poly.index = df.index
df_poly['price'] = df['price']

cols = list(df_poly)
# move the column to head of list using index, pop and insert
cols.insert(0, cols.pop(cols.index('price')))

df_poly = df_poly.loc[:,cols]

df_poly.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>...</th>
      <th>126</th>
      <th>127</th>
      <th>128</th>
      <th>129</th>
      <th>130</th>
      <th>131</th>
      <th>132</th>
      <th>133</th>
      <th>134</th>
      <th>135</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7129300520</th>
      <td>221900.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.00</td>
      <td>1180.0</td>
      <td>5650.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>3822025.0</td>
      <td>0.0</td>
      <td>2619700.0</td>
      <td>11045750.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1795600.0</td>
      <td>7571000.0</td>
      <td>3.192250e+07</td>
    </tr>
    <tr>
      <th>6414100192</th>
      <td>538000.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>2.25</td>
      <td>2570.0</td>
      <td>7242.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>3806401.0</td>
      <td>3884441.0</td>
      <td>3297190.0</td>
      <td>14903689.0</td>
      <td>3964081.0</td>
      <td>3364790.0</td>
      <td>15209249.0</td>
      <td>2856100.0</td>
      <td>12909910.0</td>
      <td>5.835432e+07</td>
    </tr>
    <tr>
      <th>5631500400</th>
      <td>180000.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.00</td>
      <td>770.0</td>
      <td>10000.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>3736489.0</td>
      <td>0.0</td>
      <td>5257760.0</td>
      <td>15583846.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7398400.0</td>
      <td>21928640.0</td>
      <td>6.499584e+07</td>
    </tr>
    <tr>
      <th>2487200875</th>
      <td>604000.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>3.00</td>
      <td>1960.0</td>
      <td>5000.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>...</td>
      <td>3861225.0</td>
      <td>0.0</td>
      <td>2672400.0</td>
      <td>9825000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1849600.0</td>
      <td>6800000.0</td>
      <td>2.500000e+07</td>
    </tr>
    <tr>
      <th>1954400510</th>
      <td>510000.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>2.00</td>
      <td>1680.0</td>
      <td>8080.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>3948169.0</td>
      <td>0.0</td>
      <td>3576600.0</td>
      <td>14908461.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3240000.0</td>
      <td>13505400.0</td>
      <td>5.629501e+07</td>
    </tr>
    <tr>
      <th>7237550310</th>
      <td>1225000.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>4.50</td>
      <td>5420.0</td>
      <td>101930.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>4004001.0</td>
      <td>0.0</td>
      <td>9524760.0</td>
      <td>203961930.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>22657600.0</td>
      <td>485186800.0</td>
      <td>1.038972e+10</td>
    </tr>
    <tr>
      <th>1321400060</th>
      <td>257500.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>2.25</td>
      <td>1715.0</td>
      <td>6819.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>3980025.0</td>
      <td>0.0</td>
      <td>4464810.0</td>
      <td>13603905.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5008644.0</td>
      <td>15260922.0</td>
      <td>4.649876e+07</td>
    </tr>
    <tr>
      <th>2008000270</th>
      <td>291850.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.50</td>
      <td>1060.0</td>
      <td>9711.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>3853369.0</td>
      <td>0.0</td>
      <td>3238950.0</td>
      <td>19062693.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2722500.0</td>
      <td>16023150.0</td>
      <td>9.430352e+07</td>
    </tr>
    <tr>
      <th>2414600126</th>
      <td>229500.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.00</td>
      <td>1780.0</td>
      <td>7470.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>3841600.0</td>
      <td>0.0</td>
      <td>3488800.0</td>
      <td>15901480.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3168400.0</td>
      <td>14441140.0</td>
      <td>6.582077e+07</td>
    </tr>
    <tr>
      <th>3793500160</th>
      <td>323000.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>2.50</td>
      <td>1890.0</td>
      <td>6560.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>4012009.0</td>
      <td>0.0</td>
      <td>4787170.0</td>
      <td>15162710.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5712100.0</td>
      <td>18092300.0</td>
      <td>5.730490e+07</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 137 columns</p>
</div>




```python
r_2 = []
high_var_rmse = []
for i in range(100):
    
    df_sample = df_poly.sample(1000, replace=True)
    y = df_sample.price
    X = df_sample.drop('price', axis=1)
    
    lr = LinearRegression()
    lr.fit(X, y)
    y_hat = lr.predict(X)
    high_var_rmse.append(np.sqrt(mean_squared_error(y, y_hat)))
    r_2.append(lr.score(X,y))
    
```


```python
print(f'low variance mean {np.mean(low_var_mse)}')
print(f'low variance variance {np.var(low_var_mse)}')

print(f'High variance mean {np.mean(high_var_mse)}')
print(f'High variance variance {np.var(high_var_mse)}')
```

    low variance mean 211983.81781738673
    low variance variance 205032983.53003398
    High variance mean 157307.0655871797
    High variance variance 286694300.50911325


$\Large Total Error = Model\ Bias^2 + Model\ Variance + Irreducible\ Error$


![optimal](img/optimal_bias_variance.png)
http://scott.fortmann-roe.com/docs/BiasVariance.html

![which_model](img/which_model_is_better_2.png)

# Train Test Split

It is hard to know if your model is too simple or complex by just using it on training data.

We can hold out part of our training sample, and use it as a test sample and use it to monitor our prediction error.

This allows us to evaluate whether our model has the right balance of bias/variance. 

<img src='img/testtrainsplit.png' width =550 />

* **training set** —a subset to train a model.
* **test set**—a subset to test the trained model.


### Should you ever train on your test set?  


![no](https://media.giphy.com/media/d10dMmzqCYqQ0/giphy.gif)


**Never train on test data.** If you are seeing surprisingly good results on your evaluation metrics, it might be a sign that you are accidentally training on the test set. 

##### [Link](https://datascience.stackexchange.com/questions/38395/standardscaler-before-and-after-splitting-data) about data leakage and scalars

**How do we know if our model is overfitting or underfitting?**


If our model is not performing well on the training  data, we are probably underfitting it.  


To know if our  model is overfitting the data, we need  to test our model on unseen data. 
We then measure our performance on the unseen data. 

If the model performs way worse on the  unseen data, it is probably  overfitting the data.

<img src='https://developers.google.com/machine-learning/crash-course/images/WorkflowWithTestSet.svg' width=500/>

Let's go back to our KC housing data without the polynomial transformation.


```python
df = pd.read_csv('data/kc_housing.csv', index_col='id')

#Date  is not in the correct format so we are dropping it for now.
df = df.drop(['date', 'zipcode', 'lat', 'long'], axis=1)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7129300520</th>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>0</td>
      <td>1955</td>
      <td>0</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <th>6414100192</th>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>400</td>
      <td>1951</td>
      <td>1991</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <th>5631500400</th>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>0</td>
      <td>1933</td>
      <td>0</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <th>2487200875</th>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>910</td>
      <td>1965</td>
      <td>0</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>1954400510</th>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>0</td>
      <td>1987</td>
      <td>0</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
</div>



Now, we create a train-test split via the sklearn model selection package.


```python
from sklearn.model_selection import train_test_split


y = df.price
X = df.drop('price', axis=1)

# Here is the convention for a traditional train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=43, test_size=.25)
```


```python
# Instanstiate your linear regression object
lr = LinearRegression()
```


```python
# fit the model on the training set
lr.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
# Check the R^2 of the training data
lr.score(X_train, y_train)
```




    0.6573692385436587



A .65 R-squared reflects a model that explains a fairly high amount of the total variance in the data. 

### Knowledge check
How would you describe the bias of the model based on the above training R^2?


```python
# Your answer here
```


```python
#__SOLUTION__
"A model with a .65 R^2 is approaching a low bias model."
```




    'A model with a .65 R^2 is approaching a low bias model.'



Next, we test how well the model performs on the unseen test data. Remember, we do not fit the model again. The model has calculated the optimal parameters learning from the training set.  



```python
lr.score(X_test, y_test)
```




    0.641985077406776



The difference between the train and test scores are low.

What does that indicate about variance?


```python
#__SOLUTION__
'The model has low variance'
```




    'The model has low variance'



# Now, let's try the same thing with our complex, polynomial model.


```python
df = pd.read_csv('data/kc_housing.csv', index_col='id')
#Date  is not in the correct format so we are dropping it for now.
df = df.drop(['date', 'zipcode', 'lat', 'long'], axis=1)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7129300520</th>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>0</td>
      <td>1955</td>
      <td>0</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <th>6414100192</th>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>400</td>
      <td>1951</td>
      <td>1991</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <th>5631500400</th>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>0</td>
      <td>1933</td>
      <td>0</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <th>2487200875</th>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>910</td>
      <td>1965</td>
      <td>0</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>1954400510</th>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>0</td>
      <td>1987</td>
      <td>0</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
</div>




```python
poly_2 = PolynomialFeatures(2)

df_poly = pd.DataFrame(
            poly_2.fit_transform(df.drop('price', axis=1))
                      )

X = df_poly
y = df.price

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.25)

# Always fit on the training set
lr.fit(X_train, y_train)

lr.score(X_train, y_train)
```




    0.7551982375870033




```python
# That indicates a lower bias
```


```python
lr.score(X_test, y_test)
```




    0.7100018682109852




```python
# Indicates higher variance
```

# Kfolds 


```python
For a more rigorous cross-validation, we turn to K-folds
```

![kfolds](img/k_folds.png)

[image via sklearn](https://scikit-learn.org/stable/modules/cross_validation.html)

In this process, we split the dataset into train and test as usual, then we perform a shuffling train test split on the train set.  

KFolds holds out one fraction of the dataset, trains on the larger fraction, then calculates a test score on the held out set.  It repeats this process until each group has served as the test set.

We tune our parameters on the training set using kfolds, then validate on the test data.  This allows us to build our model and check to see if it is overfit without touching the test data set.  This protects our model from bias.


```python
X = df.drop('price', axis=1)
y = df.price

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.25)


```


```python
kf = KFold(n_splits=5)

train_r2 = []
test_r2 = []
for train_ind, test_ind in kf.split(X,y):
    
    X_train, y_train = X.iloc[train_ind], y.iloc[train_ind]
    X_test, y_test = X.iloc[test_ind], y.iloc[test_ind]
    
    lr.fit(X_train, y_train)
    train_r2.append(lr.score(X_train, y_train))
    test_r2.append(lr.score(X_test, y_test))
```


```python
# Mean train r_2
np.mean(train_r2)
```




    0.6543164995590857




```python
# Mean test r_2
np.mean(test_r2)
```




    0.6468201186632571




```python
# Test out our polynomial model
poly_2 = PolynomialFeatures(2)

df_poly = pd.DataFrame(
            poly_2.fit_transform(df.drop('price', axis=1))
                      )

X = df_poly
y = df.price

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.25)
```


```python
kf = KFold(n_splits=5)

train_r2 = []
test_r2 = []
for train_ind, test_ind in kf.split(X,y):
    
    X_train, y_train = X.iloc[train_ind], y.iloc[train_ind]
    X_test, y_test = X.iloc[test_ind], y.iloc[test_ind]
    
    lr.fit(X_train, y_train)
    train_r2.append(lr.score(X_train, y_train))
    test_r2.append(lr.score(X_test, y_test))
```


```python
# Mean train r_2
np.mean(train_r2)
```




    0.7530146190048036




```python
# Mean test r_2
np.mean(test_r2)
```




    0.7305072362988075



Once we have an acceptable model, we train our model on the entire training set, and score on the test to validate.


