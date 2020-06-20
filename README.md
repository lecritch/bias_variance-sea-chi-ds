
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

probs = 1/6
rolls = range(1,7)

expected_value = sum([probs * roll for roll in rolls])
expected_value

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

Now, we create a train-test split via the sklearn model selection package.

A .513 R-squared reflects a model that explains aabout half of the total variance in the data. 

### Knowledge check
How would you describe the bias of the model based on the above training R^2?


```python
"A model with a .513 R^2 has a relatively high bias."
```




    'A model with a .513 R^2 has a relatively high bias.'



Next, we test how well the model performs on the unseen test data. Remember, we do not fit the model again. The model has calculated the optimal parameters learning from the training set.  


The difference between the train and test scores are low.

What does that indicate about variance?


```python
'The model has low variance'
```




    'The model has low variance'



# Now, let's try the same thing with our complex, polynomial model.

# Pair Exercise

##### [Link](https://datascience.stackexchange.com/questions/38395/standardscaler-before-and-after-splitting-data) about data leakage and scalars

The link above explains that if you are going to scale your data, you should only train your scalar on the training data to prevent data leakage.  

Perform the same train test split as shown aboe for the simple model, but now scale your data appropriately.  

The R2 for both train and test should be the same.



```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=43, test_size=.25)

ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

lr.fit(X_train, y_train)
print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))
```

    0.5132349854445817
    0.48688154021233154


# Kfolds: Even More Rigorous Validation  

For a more rigorous cross-validation, we turn to K-folds

![kfolds](img/k_folds.png)

[image via sklearn](https://scikit-learn.org/stable/modules/cross_validation.html)

In this process, we split the dataset into train and test as usual, then we perform a shuffling train test split on the train set.  

KFolds holds out one fraction of the dataset, trains on the larger fraction, then calculates a test score on the held out set.  It repeats this process until each group has served as the test set.

We tune our parameters on the training set using kfolds, then validate on the test data.  This allows us to build our model and check to see if it is overfit without touching the test data set.  This protects our model from bias.

# Fill in the Blank

Once we have an acceptable model, we train our model on the entire training set, and score on the test to validate.


