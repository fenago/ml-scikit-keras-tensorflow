[Lab 7](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch07.html#ensembles_lab): Ensemble Learning and Random Forests
================================================================================================================================================================

1.  If you have trained five different models and they all achieve 95%
    precision, you can try combining them into a voting ensemble, which
    will often give you even better results. It works better if the
    models are very different (e.g., an SVM classifier, a Decision Tree
    classifier, a Logistic Regression classifier, and so on). It is even
    better if they are trained on different training instances (that's
    the whole point of bagging and pasting ensembles), but if not this
    will still be effective as long as the models are very different.

2.  A hard voting classifier just counts the votes of each classifier in
    the ensemble and picks the class that gets the most votes. A soft
    voting classifier computes the average estimated class probability
    for each class and picks the class with the highest probability.
    This gives high-confidence votes more weight and often performs
    better, but it works only if every classifier is able to estimate
    class probabilities (e.g., for the SVM classifiers in Scikit-Learn
    you must set `probability=True`).

3.  It is quite possible to speed up training of a bagging ensemble by
    distributing it across multiple servers, since each predictor in the
    ensemble is independent of the others. The same goes for pasting
    ensembles and Random Forests, for the same reason. However, each
    predictor in a boosting ensemble is built based on the previous
    predictor, so training is necessarily sequential, and you will not
    gain anything by distributing training across multiple servers.
    Regarding stacking ensembles, all the predictors in a given layer
    are independent of each other, so they can be trained in parallel on
    multiple servers. However, the predictors in one layer can only be
    trained after the predictors in the previous layer have all been
    trained.

4.  With out-of-bag evaluation, each predictor in a bagging ensemble is
    evaluated using instances that it was not trained on (they were held
    out). This makes it possible to have a fairly unbiased evaluation of
    the ensemble without the need for an additional validation set.
    Thus, you have more instances available for training, and your
    ensemble can perform slightly better.

5.  When you are growing a tree in a Random Forest, only a random subset
    of the features is considered for splitting at each node. This is
    true as well for Extra-Trees, but they go one step further: rather
    than searching for the best possible thresholds, like regular
    Decision Trees do, they use random thresholds for each feature. This
    extra randomness acts like a form of regularization: if a Random
    Forest overfits the training data, Extra-Trees might perform better.
    Moreover, since Extra-Trees don't search for the best possible
    thresholds, they are much faster to train than Random Forests.
    However, they are neither faster nor slower than Random Forests when
    making predictions.

6.  If your AdaBoost ensemble underfits the training data, you can try
    increasing the number of estimators or reducing the regularization
    hyperparameters of the base estimator. You may also try slightly
    increasing the learning rate.

7.  If your Gradient Boosting ensemble overfits the training set, you
    should try decreasing the learning rate. You could also use early
    stopping to find the right number of predictors (you probably have
    too many).

For the solutions to exercises 8 and 9, please see the Jupyter notebooks
available at
[*https://github.com/ageron/handson-ml2*](https://github.com/ageron/handson-ml2).
