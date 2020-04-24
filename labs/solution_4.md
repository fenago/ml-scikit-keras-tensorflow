[Lab 4](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch04.html#linear_models_lab): Training Models
===============================================================================================================================================

1.  If you have a training set with millions of features you can use
    Stochastic Gradient Descent or Mini-batch Gradient Descent, and
    perhaps Batch Gradient Descent if the training set fits in memory.
    But you cannot use the Normal Equation or the SVD approach because
    the computational complexity grows quickly (more than quadratically)
    with the number of features.

2.  If the features in your training set have very different scales, the
    cost function will have the shape of an elongated bowl, so the
    Gradient Descent algorithms will take a long time to converge. To
    solve this you should scale the data before training the model. Note
    that the Normal Equation or SVD approach will work just fine without
    scaling. Moreover, regularized models may converge to a suboptimal
    solution if the features are not scaled: since regularization
    penalizes large weights, features with smaller values will tend to
    be ignored compared to features with larger values.

3.  Gradient Descent cannot get stuck in a local minimum when training a
    Logistic Regression model because the cost function is
    convex.^[1](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/app01.html#idm45728432174040){#idm45728432174040-marker
    .totri-footnote}^

4.  If the optimization problem is convex (such as Linear Regression or
    Logistic Regression), and assuming the learning rate is not too
    high, then all Gradient Descent algorithms will approach the global
    optimum and end up producing fairly similar models. However, unless
    you gradually reduce the learning rate, Stochastic GD and Mini-batch
    GD will never truly converge; instead, they will keep jumping back
    and forth around the global optimum. This means that even if you let
    them run for a very long time, these Gradient Descent algorithms
    will produce slightly different models.

5.  If the validation error consistently goes up after every epoch, then
    one possibility is that the learning rate is too high and the
    algorithm is diverging. If the training error also goes up, then
    this is clearly the problem and you should reduce the learning rate.
    However, if the training error is not going up, then your model is
    overfitting the training set and you should stop training.

6.  Due to their random nature, neither Stochastic Gradient Descent nor
    Mini-batch Gradient Descent is guaranteed to make progress at every
    single training iteration. So if you immediately stop training when
    the validation error goes up, you may stop much too early, before
    the optimum is reached. A better option is to save the model at
    regular intervals; then, when it has not improved for a long time
    (meaning it will probably never beat the record), you can revert to
    the best saved model.

7.  Stochastic Gradient Descent has the fastest training iteration since
    it considers only one training instance at a time, so it is
    generally the first to reach the vicinity of the global optimum (or
    Mini-batch GD with a very small mini-batch size). However, only
    Batch Gradient Descent will actually converge, given enough training
    time. As mentioned, Stochastic GD and Mini-batch GD will bounce
    around the optimum, unless you gradually reduce the learning rate.

8.  If the validation error is much higher than the training error, this
    is likely because your model is overfitting the training set. One
    way to try to fix this is to reduce the polynomial degree: a model
    with fewer degrees of freedom is less likely to overfit. Another
    thing you can try is to regularize the model---for example, by
    adding an ℓ~2~ penalty (Ridge) or an ℓ~1~ penalty (Lasso) to the
    cost function. This will also reduce the degrees of freedom of the
    model. Lastly, you can try to increase the size of the training set.

9.  If both the training error and the validation error are almost equal
    and fairly high, the model is likely underfitting the training set,
    which means it has a high bias. You should try reducing the
    regularization hyperparameter *α*.

10. Let's see:

    -   A model with some regularization typically performs better than
        a model without any regularization, so you should generally
        prefer Ridge Regression over plain Linear Regression.

    -   Lasso Regression uses an ℓ~1~ penalty, which tends to push the
        weights down to exactly zero. This leads to sparse models, where
        all weights are zero except for the most important weights. This
        is a way to perform feature selection automatically, which is
        good if you suspect that only a few features actually matter.
        When you are not sure, you should prefer Ridge Regression.

    -   Elastic Net is generally preferred over Lasso since Lasso may
        behave erratically in some cases (when several features are
        strongly correlated or when there are more features than
        training instances). However, it does add an extra
        hyperparameter to tune. If you want Lasso without the erratic
        behavior, you can just use Elastic Net with an `l1_ratio` close
        to 1.

11. If you want to classify pictures as outdoor/indoor and
    daytime/nighttime, since these are not exclusive classes (i.e., all
    four combinations are possible) you should train two Logistic
    Regression classifiers.

12. See the Jupyter notebooks available at
    [*https://github.com/ageron/handson-ml2*](https://github.com/ageron/handson-ml2).

