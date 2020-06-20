
# Bias-Variance Tradeoff


## Agenda

1. Explain what bias, variance, and error are in the context of statistical modeling
2. Defining Error: prediction error and irreducible error
3. Define prediction error as a combination of bias and variance
4. Explore the bias-variance tradeoff
5. Train-test split



# 1. Explain what bias, variance, and error are in the context of statistical modeling

![which model is better](img/which_model_is_better.png)

https://towardsdatascience.com/cultural-overfitting-and-underfitting-or-why-the-netflix-culture-wont-work-in-your-company-af2a62e41288


# What makes a model good?

- We don’t ultimately care about how well your model fits your data.

- What we really care about is how well your model describes the process that generated your data.

- Why? Because the data set you have is but one sample from a universe of possible data sets, and you want a model that would work for any data set from that universe

# What is a “Model”?

 - A “model” is a general specification of relationships among variables. E.G. Linear Regression:

$$\Large Price = \beta_1 X_1 + \beta_0 + \epsilon$$

 - Each model makes assumptions about how the variables interact. 
 - A 'trained model' operates on these assumptions to learn from how best to interact with training data.
 - In linear regression, the learning results in a set of parameters that define the best fit linear equation.
 - The higher the quality of learning form this training data, the more precicely the model will reflect the real world process the data was generated from.
 - The model will then perform more accurately on unseen samples.


# Remember Expected Value?
- The expected value of a quantity is the weighted average of that quantity across all possible samples

![6 sided die](https://media.giphy.com/media/sRJdpUSr7W0AiQ3RcM/giphy.gif)

- for a 6 sided die, another way to think about the expected value is the arithmetic mean of the rolls of a very large number of independent samples.  

### The expected value of a 6-sided die is:


```python
# code
```

Suppose we created a model which always predicted that the die roll would be 3.

The **bias** of our model would be the difference between the our expected prediction (3) and the expected value (3.5).

What would the **variance** of our model be?


# 2. Defining Error: prediction error and irreducible error



### Regression fit statistics are often called “error”
 - Sum of Squared Errors (SSE)
 - Mean Squared Error (MSE) 
 
 Both are calculated using residuals

![residuals](img/residuals.png)


This error can be broken up into parts:

$Total Error = Residual = Prediction\ Error+ Irreducible\ Error$

![defining error](img/defining_error.png)

There will always be some random, irreducible error inherent in the data.  Real data always has noise.

The goal of modeling is to reduce the prediction error, which is the difference between our model and the realworld processes from which our data is generated.

# 3. Define prediction error as a combination of bias and variance

Our prediction error can be further broken down into error due to bias and error due to variance.

$\Large Prediction\ Error = Model\ Bias^2 + Model\ Variance $

So our total error can be thought of as a combination of bias, variance, and irriducile error.

$\Large Total Error = Model\ Bias^2 + Model\ Variance + Irreducible\ Error$


**Model Bias** is the expected prediction error from your expected trained model

> In other words, if you were to train multiple models on different samples, what would be the average prediction error.

**Model Variance** is the expected variation in predictions, relative to your expected trained model

> In other words, it is a measure of how much your model varies for any given point.

**Let's do a thought experiment:**

# Thought Experiment

1. Imagine you've collected 23 different training sets for the same problem.
2. Now imagine using one algorithm to train 23 models, one for each of your training sets.
3. Bias vs. variance refers to the accuracy vs. consistency of the models trained by your algorithm.

![target_bias_variance](img/target.png)

http://scott.fortmann-roe.com/docs/BiasVariance.html



# 4.  Explore Bias Variance Tradeoff

**High bias** algorithms tend to be less complex, with simple or rigid underlying structure.

+ They train models that are consistent, but inaccurate on average.
+ These include linear or parametric algorithms such as regression and naive Bayes.

On the other hand, **high variance** algorithms tend to be more complex, with flexible underlying structure.

+ They train models that are accurate on average, but inconsistent.
+ These include non-linear or non-parametric algorithms such as decision trees and nearest neighbors.



While we build our models, we have to keep this relationship in mind.  If we build complex models, we risk overfitting our models.  Their predictions will vary greatly when introduced to new data.  If our models are too simple, the predictions as a whole will be inaccurate.   

The goal is to build a model with enough complexity to be accurate, but not too much complexity to be erratic.

![optimal](img/optimal_bias_variance.png)
http://scott.fortmann-roe.com/docs/BiasVariance.html

### Let's take a look at our familiar King County housing data. 


```python
import pandas as pd
import numpy as np
np.random.seed(42)
df = pd.read_csv('data/king_county.csv', index_col='id')
df = df.iloc[:,:12]
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
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
np.random.seed(42)

# Let's generate random subsets of our data

#Date  is not in the correct format so we are dropping it for now.

r_2 = []
simple_rmse = []

for i in range(100):
    
    df_sample = df.sample(5000, replace=True)
    y = df_sample.price
    X = df_sample.drop('price', axis=1)
    
    lr = LinearRegression()
    lr.fit(X, y)
    
    y_hat = lr.predict(X)
    simple_rmse.append(np.sqrt(mean_squared_error(y, y_hat)))
    r_2.append(lr.score(X,y))
    
    

```


```python
print(f'simple mean {np.mean(simple_rmse)}')
print(f'simple variance {np.var(simple_rmse)}')
```

    simple mean 228303.4528251759
    simple variance 73238189.28935923



```python
from sklearn.preprocessing import PolynomialFeatures


df = pd.read_csv('data/king_county.csv', index_col='id')

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
      <th>68</th>
      <th>69</th>
      <th>70</th>
      <th>71</th>
      <th>72</th>
      <th>73</th>
      <th>74</th>
      <th>75</th>
      <th>76</th>
      <th>77</th>
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
      <td>9.0</td>
      <td>21.0</td>
      <td>3540.0</td>
      <td>0.0</td>
      <td>49.0</td>
      <td>8260.0</td>
      <td>0.0</td>
      <td>1392400.0</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <td>9.0</td>
      <td>21.0</td>
      <td>6510.0</td>
      <td>1200.0</td>
      <td>49.0</td>
      <td>15190.0</td>
      <td>2800.0</td>
      <td>4708900.0</td>
      <td>868000.0</td>
      <td>160000.0</td>
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
      <td>9.0</td>
      <td>18.0</td>
      <td>2310.0</td>
      <td>0.0</td>
      <td>36.0</td>
      <td>4620.0</td>
      <td>0.0</td>
      <td>592900.0</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <td>25.0</td>
      <td>35.0</td>
      <td>5250.0</td>
      <td>4550.0</td>
      <td>49.0</td>
      <td>7350.0</td>
      <td>6370.0</td>
      <td>1102500.0</td>
      <td>955500.0</td>
      <td>828100.0</td>
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
      <td>9.0</td>
      <td>24.0</td>
      <td>5040.0</td>
      <td>0.0</td>
      <td>64.0</td>
      <td>13440.0</td>
      <td>0.0</td>
      <td>2822400.0</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <td>9.0</td>
      <td>33.0</td>
      <td>11670.0</td>
      <td>4590.0</td>
      <td>121.0</td>
      <td>42790.0</td>
      <td>16830.0</td>
      <td>15132100.0</td>
      <td>5951700.0</td>
      <td>2340900.0</td>
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
      <td>9.0</td>
      <td>21.0</td>
      <td>5145.0</td>
      <td>0.0</td>
      <td>49.0</td>
      <td>12005.0</td>
      <td>0.0</td>
      <td>2941225.0</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <td>9.0</td>
      <td>21.0</td>
      <td>3180.0</td>
      <td>0.0</td>
      <td>49.0</td>
      <td>7420.0</td>
      <td>0.0</td>
      <td>1123600.0</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <td>9.0</td>
      <td>21.0</td>
      <td>3150.0</td>
      <td>2190.0</td>
      <td>49.0</td>
      <td>7350.0</td>
      <td>5110.0</td>
      <td>1102500.0</td>
      <td>766500.0</td>
      <td>532900.0</td>
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
      <td>9.0</td>
      <td>21.0</td>
      <td>5670.0</td>
      <td>0.0</td>
      <td>49.0</td>
      <td>13230.0</td>
      <td>0.0</td>
      <td>3572100.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 79 columns</p>
</div>




```python
np.random.seed(42)
r_2 = []
complex_rmse = []
for i in range(100):
    
    df_sample = df_poly.sample(1000, replace=True)
    y = df_sample.price
    X = df_sample.drop('price', axis=1)
    
    lr = LinearRegression()
    lr.fit(X, y)
    y_hat = lr.predict(X)
    complex_rmse.append(np.sqrt(mean_squared_error(y, y_hat)))
    r_2.append(lr.score(X,y))
    
```


```python
print(f'simpl mean {np.mean(simple_rmse)}')
print(f'compl mean {np.mean(complex_rmse)}')

print(f'simp variance {np.var(simple_rmse)}')
print(f'comp variance {np.var(complex_rmse)}')
```

    simpl mean 228303.4528251759
    compl mean 195246.84153558695
    simp variance 73238189.28935923
    comp variance 1092553181.415185


![which_model](img/which_model_is_better_2.png)

# 5. Train Test Split

It is hard to know if your model is too simple or complex by just using it on training data.

We can hold out part of our training sample, and use it as a test sample and use it to monitor our prediction error.

This allows us to evaluate whether our model has the right balance of bias/variance. 

<img src='img/testtrainsplit.png' width =550 />

* **training set** —a subset to train a model.
* **test set**—a subset to test the trained model.


**How do we know if our model is overfitting or underfitting?**


If our model is not performing well on the training  data, we are probably underfitting it.  


To know if our  model is overfitting the data, we need  to test our model on unseen data. 
We then measure our performance on the unseen data. 

If the model performs way worse on the  unseen data, it is probably  overfitting the data.

<img src='https://developers.google.com/machine-learning/crash-course/images/WorkflowWithTestSet.svg' width=500/>

### Should you ever train on your test set?  


![no](https://media.giphy.com/media/d10dMmzqCYqQ0/giphy.gif)


**Never train on test data.** If you are seeing surprisingly good results on your evaluation metrics, it might be a sign that you are accidentally training on the test set. 



Let's go back to our KC housing data without the polynomial transformation.


```python
df = pd.read_csv('data/king_county.csv', index_col='id')

#Date  is not in the correct format so we are dropping it for now.
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
    </tr>
  </tbody>
</table>
</div>



Now, we create a train-test split via the sklearn model selection package.


```python
from sklearn.model_selection import train_test_split
np.random.seed(42)

y = df.price
X = df[['bedrooms', 'sqft_living']]

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




    0.5132349854445817




```python
lr.coef_
```




    array([-54632.9149931 ,    311.65365556])



A .513 R-squared reflects a model that explains aabout half of the total variance in the data. 

### Knowledge check
How would you describe the bias of the model based on the above training R^2?


```python
# Your answer here
```

Next, we test how well the model performs on the unseen test data. Remember, we do not fit the model again. The model has calculated the optimal parameters learning from the training set.  



```python
lr.score(X_test, y_test)
```




    0.48688154021233165



The difference between the train and test scores are low.

What does that indicate about variance?

# Now, let's try the same thing with our complex, polynomial model.


```python
df = pd.read_csv('data/king_county.csv', index_col='id')
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
    </tr>
  </tbody>
</table>
</div>




```python
poly_2 = PolynomialFeatures(3)

X_poly = pd.DataFrame(
            poly_2.fit_transform(df.drop('price', axis=1))
                      )

y = df.price
X_poly.head()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>354</th>
      <th>355</th>
      <th>356</th>
      <th>357</th>
      <th>358</th>
      <th>359</th>
      <th>360</th>
      <th>361</th>
      <th>362</th>
      <th>363</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.00</td>
      <td>1180.0</td>
      <td>5650.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>...</td>
      <td>343.0</td>
      <td>57820.0</td>
      <td>0.0</td>
      <td>9746800.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.643032e+09</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>2.25</td>
      <td>2570.0</td>
      <td>7242.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>...</td>
      <td>343.0</td>
      <td>106330.0</td>
      <td>19600.0</td>
      <td>32962300.0</td>
      <td>6076000.0</td>
      <td>1120000.0</td>
      <td>1.021831e+10</td>
      <td>1.883560e+09</td>
      <td>347200000.0</td>
      <td>64000000.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.00</td>
      <td>770.0</td>
      <td>10000.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>...</td>
      <td>216.0</td>
      <td>27720.0</td>
      <td>0.0</td>
      <td>3557400.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.565330e+08</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>4.0</td>
      <td>3.00</td>
      <td>1960.0</td>
      <td>5000.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>...</td>
      <td>343.0</td>
      <td>51450.0</td>
      <td>44590.0</td>
      <td>7717500.0</td>
      <td>6688500.0</td>
      <td>5796700.0</td>
      <td>1.157625e+09</td>
      <td>1.003275e+09</td>
      <td>869505000.0</td>
      <td>753571000.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>2.00</td>
      <td>1680.0</td>
      <td>8080.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>8.0</td>
      <td>...</td>
      <td>512.0</td>
      <td>107520.0</td>
      <td>0.0</td>
      <td>22579200.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.741632e+09</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 364 columns</p>
</div>




```python
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, random_state=20, test_size=.25)
lr_poly = LinearRegression()

# Always fit on the training set
lr_poly.fit(X_train, y_train)

lr_poly.score(X_train, y_train)
```




    0.7133044254532317




```python
# That indicates a lower bias
```


```python
lr_poly.score(X_test, y_test)
```




    0.6119026292718351




```python
# Wow, we get a negative r^2.  Performs terribly on new data.
```

# Pair Exercise

##### [Link](https://datascience.stackexchange.com/questions/38395/standardscaler-before-and-after-splitting-data) about data leakage and scalars

The link above explains that if you are going to scale your data, you should only train your scalar on the training data to prevent data leakage.  

Perform the same train test split as shown aboe for the simple model, but now scale your data appropriately.  

The R2 for both train and test should be the same.



```python
from sklearn.preprocessing import StandardScaler
np.random.seed(42)

y = df.price
X = df[['bedrooms', 'sqft_living']]

# Train test split with random_state=43 and test_size=.25


# Scale appropriately

# fit and score the model 

```

# Kfolds: Even More Rigorous Validation  

For a more rigorous cross-validation, we turn to K-folds

![kfolds](img/k_folds.png)

[image via sklearn](https://scikit-learn.org/stable/modules/cross_validation.html)

In this process, we split the dataset into train and test as usual, then we perform a shuffling train test split on the train set.  

KFolds holds out one fraction of the dataset, trains on the larger fraction, then calculates a test score on the held out set.  It repeats this process until each group has served as the test set.

We tune our parameters on the training set using kfolds, then validate on the test data.  This allows us to build our model and check to see if it is overfit without touching the test data set.  This protects our model from bias.

# Fill in the Blank


```python


mccalister = ['Adam', 'Amanda','Chum', 'Dann', 
 'Jacob', 'Jason', 'Johnhoy', 'Karim', 
'Leana','Luluva', 'Matt', 'Maximilian','Syd' ]

choice = np.random.choice(mccalister)
print(choice)
```

    Matt



```python
X = df.drop('price', axis=1)
y = df.price

```


```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)

train_r2 = []
test_r2 = []
for train_ind, test_ind in kf.split(X,y):
    
    X_train, y_train = X.iloc[<fill_in>], y.iloc[<fill_in>]
    X_test, y_test = X.iloc[<fill_in>], y.iloc[<fill_in>]
    
    # fill in fit
    
    
    train_r2.append(lr.score(X_train, y_train))
    test_r2.append(lr.score(X_test, y_test))
```


      File "<ipython-input-166-3b7ed5927e4c>", line 9
        X_train, y_train = X.iloc[<fill_in>], y.iloc[<fill_in>]
                                  ^
    SyntaxError: invalid syntax




```python
from sklearn.model_selection import KFold

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




    0.6056692082295809




```python
# Mean test r_2
np.mean(test_r2)
```




    0.6018833963943026




```python
# Test out our polynomial model
poly_2 = PolynomialFeatures(2)

df_poly = pd.DataFrame(
            poly_2.fit_transform(df.drop('price', axis=1))
                      )

X = df_poly
y = df.price

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




    0.6954523534689443




```python
# Mean test r_2
np.mean(test_r2)
```




    0.6606213729380702



Once we have an acceptable model, we train our model on the entire training set, and score on the test to validate.


