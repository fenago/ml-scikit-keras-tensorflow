
[Appendix A. ]{.label}Exercise Solutions
========================================

::: {data-type="note" type="note"}
###### Note

Solutions[]{#exsol20} to the coding exercises are available in the
online Jupyter notebooks at
[*https://github.com/ageron/handson-ml2*](https://github.com/ageron/handson-ml2).
:::

::: {.section data-type="sect1" pdf-bookmark="Chapter 1: The Machine Learning Landscape"}
::: {#idm45728432240024 .sect1}
[Chapter 1](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch01.html#landscape_chapter): The Machine Learning Landscape
==========================================================================================================================================================

1.  Machine Learning is about building systems that can learn from data.
    Learning means getting better at some task, given some performance
    measure.

2.  Machine Learning is great for complex problems for which we have no
    algorithmic solution, to replace long lists of hand-tuned rules, to
    build systems that adapt to fluctuating environments, and finally to
    help humans learn (e.g., data mining).

3.  A labeled training set is a training set that contains the desired
    solution (a.k.a. a label) for each instance.

4.  The two most common supervised tasks are regression and
    classification.

5.  Common unsupervised tasks include clustering, visualization,
    dimensionality reduction, and association rule learning.

6.  Reinforcement Learning is likely to perform best if we want a robot
    to learn to walk in various unknown terrains, since this is
    typically the type of problem that Reinforcement Learning tackles.
    It might be possible to express the problem as a supervised or
    semisupervised learning problem, but it would be less natural.

7.  If you don't know how to define the groups, then you can use a
    clustering algorithm (unsupervised learning) to segment your
    customers into clusters of similar customers. However, if you know
    what groups you would like to have, then you can feed many examples
    of each group to a classification algorithm (supervised learning),
    and it will classify all your customers into these groups.

8.  Spam detection is a typical supervised learning problem: the
    algorithm is fed many emails along with their labels (spam or not
    spam).

9.  An online learning system can learn incrementally, as opposed to a
    batch learning system. This makes it capable of adapting rapidly to
    both changing data and autonomous systems, and of training on very
    large quantities of data.

10. Out-of-core algorithms can handle vast quantities of data that
    cannot fit in a computer's main memory. An out-of-core learning
    algorithm chops the data into mini-batches and uses online learning
    techniques to learn from these mini-batches.

11. An instance-based learning system learns the training data by heart;
    then, when given a new instance, it uses a similarity measure to
    find the most similar learned instances and uses them to make
    predictions.

12. A model has one or more model parameters that determine what it will
    predict given a new instance (e.g., the slope of a linear model). A
    learning algorithm tries to find optimal values for these parameters
    such that the model generalizes well to new instances. A
    hyperparameter is a parameter of the learning algorithm itself, not
    of the model (e.g., the amount of regularization to apply).

13. Model-based learning algorithms search for an optimal value for the
    model parameters such that the model will generalize well to new
    instances. We usually train such systems by minimizing a cost
    function that measures how bad the system is at making predictions
    on the training data, plus a penalty for model complexity if the
    model is regularized. To make predictions, we feed the new
    instance's features into the model's prediction function, using the
    parameter values found by the learning algorithm.

14. Some of the main challenges in Machine Learning are the lack of
    data, poor data quality, nonrepresentative data, uninformative
    features, excessively simple models that underfit the training data,
    and excessively complex models that overfit the data.

15. If a model performs great on the training data but generalizes
    poorly to new instances, the model is likely overfitting the
    training data (or we got extremely lucky on the training data).
    Possible solutions to overfitting are getting more data, simplifying
    the model (selecting a simpler algorithm, reducing the number of
    parameters or features used, or regularizing the model), or reducing
    the noise in the training data.

16. A test set is used to estimate the generalization error that a model
    will make on new instances, before the model is launched in
    production.

17. A validation set is used to compare models. It makes it possible to
    select the best model and tune the hyperparameters.

18. The train-dev set is used when there is a risk of mismatch between
    the training data and the data used in the validation and test
    datasets (which should always be as close as possible to the data
    used once the model is in production). The train-dev set is a part
    of the training set that's held out (the model is not trained on
    it). The model is trained on the rest of the training set, and
    evaluated on both the train-dev set and the validation set. If the
    model performs well on the training set but not on the train-dev
    set, then the model is likely overfitting the training set. If it
    performs well on both the training set and the train-dev set, but
    not on the validation set, then there is probably a significant data
    mismatch between the training data and the validation + test data,
    and you should try to improve the training data to make it look more
    like the validation + test data.

19. If you tune hyperparameters using the test set, you risk overfitting
    the test set, and the generalization error you measure will be
    optimistic (you may launch a model that performs worse than you
    expect).
:::
:::

::: {.section data-type="sect1" pdf-bookmark="Chapter 2: End-to-End Machine Learning Project"}
::: {#idm45728432186536 .sect1}
[Chapter 2](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch02.html#project_chapter): End-to-End Machine Learning Project
=============================================================================================================================================================

See the Jupyter notebooks available at
[*https://github.com/ageron/handson-ml2*](https://github.com/ageron/handson-ml2).
:::
:::

::: {.section data-type="sect1" pdf-bookmark="Chapter 3: Classification"}
::: {#idm45728432183288 .sect1}
[Chapter 3](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch03.html#classification_chapter): Classification
===============================================================================================================================================

See the Jupyter notebooks available at
[*https://github.com/ageron/handson-ml2*](https://github.com/ageron/handson-ml2).
:::
:::

::: {.section data-type="sect1" pdf-bookmark="Chapter 4: Training Models"}
::: {#idm45728432179880 .sect1}
[Chapter 4](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch04.html#linear_models_chapter): Training Models
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
:::
:::

::: {.section data-type="sect1" pdf-bookmark="Chapter 5: Support Vector Machines"}
::: {#idm45728432154792 .sect1}
[Chapter 5](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch05.html#svm_chapter): Support Vector Machines
=============================================================================================================================================

1.  The fundamental idea behind Support Vector Machines is to fit the
    widest possible "street" between the classes. In other words, the
    goal is to have the largest possible margin between the decision
    boundary that separates the two classes and the training instances.
    When performing soft margin classification, the SVM searches for a
    compromise between perfectly separating the two classes and having
    the widest possible street (i.e., a few instances may end up on the
    street). Another key idea is to use kernels when training on
    nonlinear datasets.

2.  After training an SVM, a *support vector* is any instance located on
    the "street" (see the previous answer), including its border. The
    decision boundary is entirely determined by the support vectors. Any
    instance that is *not* a support vector (i.e., is off the street)
    has no influence whatsoever; you could remove them, add more
    instances, or move them around, and as long as they stay off the
    street they won't affect the decision boundary. Computing the
    predictions only involves the support vectors, not the whole
    training set.

3.  SVMs try to fit the largest possible "street" between the classes
    (see the first answer), so if the training set is not scaled, the
    SVM will tend to neglect small features (see
    [Figure 5-2](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch05.html#sensitivity_to_feature_scales_plot)).

4.  An SVM classifier can output the distance between the test instance
    and the decision boundary, and you can use this as a confidence
    score. However, this score cannot be directly converted into an
    estimation of the class probability. If you set `probability=True`
    when creating an SVM in Scikit-Learn, then after training it will
    calibrate the probabilities using Logistic Regression on the SVM's
    scores (trained by an additional five-fold cross-validation on the
    training data). This will add the `predict_proba()` and
    `predict_log_proba()` methods to the SVM.

5.  This question applies only to linear SVMs since kernelized SVMs can
    only use the dual form. The computational complexity of the primal
    form of the SVM problem is proportional to the number of training
    instances *m*, while the computational complexity of the dual form
    is proportional to a number between *m*^2^ and *m*^3^. So if there
    are millions of instances, you should definitely use the primal
    form, because the dual form will be much too slow.

6.  If an SVM classifier trained with an RBF kernel underfits the
    training set, there might be too much regularization. To decrease
    it, you need to increase `gamma` or `C` (or both).

7.  Let's call the QP parameters for the hard margin problem **H**′,
    **f**′, **A**′, and **b**′ (see ["Quadratic
    Programming"](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch05.html#quadratic_programming_paragraph)).
    The QP parameters for the soft margin problem have *m* additional
    parameters (*n*~*p*~ = *n* + 1 + *m*) and *m* additional constraints
    (*n*~*c*~ = 2*m*). They can be defined like so:

    -   **H** is equal to **H**′, plus *m* columns of 0s on the right
        and *m* rows of 0s at the bottom: $\mathbf{H} = \begin{pmatrix}
        \mathbf{H}^{'} & 0 & \cdots \\
        0 & 0 & \\
         \vdots & & \ddots \\
        \end{pmatrix}$

    -   **f** is equal to **f**′ with *m* additional elements, all equal
        to the value of the hyperparameter *C*.

    -   **b** is equal to **b**′ with *m* additional elements, all equal
        to 0.

    -   **A** is equal to **A**′, with an extra *m* × *m* identity
        matrix **I**~*m*~ appended to the right, --\*I\*~*m*~ just below
        it, and the rest filled with 0s: $\mathbf{A} = \begin{pmatrix}
        \mathbf{A}^{'} & \mathbf{I}_{m} \\
        0 & {- \mathbf{I}_{m}} \\
        \end{pmatrix}$

For the solutions to exercises 8, 9, and 10, please see the Jupyter
notebooks available at
[*https://github.com/ageron/handson-ml2*](https://github.com/ageron/handson-ml2).
:::
:::

::: {.section data-type="sect1" pdf-bookmark="Chapter 6: Decision Trees"}
::: {#idm45728432092664 .sect1}
[Chapter 6](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch06.html#trees_chapter): Decision Trees
======================================================================================================================================

1.  The depth of a well-balanced binary tree containing *m* leaves is
    equal to
    log~2~(*m*),^[2](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/app01.html#idm45728432088232){#idm45728432088232-marker
    .totri-footnote}^ rounded up. A binary Decision Tree (one that makes
    only binary decisions, as is the case with all trees in
    Scikit-Learn) will end up more or less well balanced at the end of
    training, with one leaf per training instance if it is trained
    without restrictions. Thus, if the training set contains one million
    instances, the Decision Tree will have a depth of log~2~(10^6^) ≈ 20
    (actually a bit more since the tree will generally not be perfectly
    well balanced).

2.  A node's Gini impurity is generally lower than its parent's. This is
    due to the CART training algorithm's cost function, which splits
    each node in a way that minimizes the weighted sum of its children's
    Gini impurities. However, it is possible for a node to have a higher
    Gini impurity than its parent, as long as this increase is more than
    compensated for by a decrease in the other child's impurity. For
    example, consider a node containing four instances of class A and
    one of class B. Its Gini impurity is 1 -- (1/5)^2^ -- (4/5)^2^ =
    0.32. Now suppose the dataset is one-dimensional and the instances
    are lined up in the following order: A, B, A, A, A. You can verify
    that the algorithm will split this node after the second instance,
    producing one child node with instances A, B, and the other child
    node with instances A, A, A. The first child node's Gini impurity is
    1 -- (1/2)^2^ -- (1/2)^2^ = 0.5, which is higher than its parent's.
    This is compensated for by the fact that the other node is pure, so
    its overall weighted Gini impurity is 2/5 × 0.5 + 3/5 × 0 = 0.2,
    which is lower than the parent's Gini impurity.

3.  If a Decision Tree is overfitting the training set, it may be a good
    idea to decrease `max_depth`, since this will constrain the model,
    regularizing it.

4.  Decision Trees don't care whether or not the training data is scaled
    or centered; that's one of the nice things about them. So if a
    Decision Tree underfits the training set, scaling the input features
    will just be a waste of time.

5.  The computational complexity of training a Decision Tree is *O*(*n*
    × *m* log(*m*)). So if you multiply the training set size by 10, the
    training time will be multiplied by *K* = (*n* × 10*m* × log(10*m*))
    / (*n* × *m* × log(*m*)) = 10 × log(10*m*) / log(*m*). If *m* =
    10^6^, then *K* ≈ 11.7, so you can expect the training time to be
    roughly 11.7 hours.

6.  Presorting the training set speeds up training only if the dataset
    is smaller than a few thousand instances. If it contains 100,000
    instances, setting `presort=True` will considerably slow down
    training.

For the solutions to exercises 7 and 8, please see the Jupyter notebooks
available at
[*https://github.com/ageron/handson-ml2*](https://github.com/ageron/handson-ml2).
:::
:::

::: {.section data-type="sect1" pdf-bookmark="Chapter 7: Ensemble Learning and Random Forests"}
::: {#idm45728432066696 .sect1}
[Chapter 7](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch07.html#ensembles_chapter): Ensemble Learning and Random Forests
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
:::
:::

::: {.section data-type="sect1" pdf-bookmark="Chapter 8: Dimensionality Reduction"}
::: {#idm45728432052984 .sect1}
[Chapter 8](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch08.html#dimensionality_chapter): Dimensionality Reduction
=========================================================================================================================================================

1.  The main motivations for dimensionality reduction are:

    -   To speed up a subsequent training algorithm (in some cases it
        may even remove noise and redundant features, making the
        training algorithm perform better)

    -   To visualize the data and gain insights on the most important
        features

    -   To save space (compression)

    The main drawbacks are:

    -   Some information is lost, possibly degrading the performance of
        subsequent training algorithms.

    -   It can be computationally intensive.

    -   It adds some complexity to your Machine Learning pipelines.

    -   Transformed features are often hard to interpret.

2.  The curse of dimensionality refers to the fact that many problems
    that do not exist in low-dimensional space arise in high-dimensional
    space. In Machine Learning, one common manifestation is the fact
    that randomly sampled high-dimensional vectors are generally very
    sparse, increasing the risk of overfitting and making it very
    difficult to identify patterns in the data without having plenty of
    training data.

3.  Once a dataset's dimensionality has been reduced using one of the
    algorithms we discussed, it is almost always impossible to perfectly
    reverse the operation, because some information gets lost during
    dimensionality reduction. Moreover, while some algorithms (such as
    PCA) have a simple reverse transformation
    [procedure]{.keep-together} that can reconstruct a dataset
    relatively similar to the original, other algorithms (such as T-SNE)
    do not.

4.  PCA can be used to significantly reduce the dimensionality of most
    datasets, even if they are highly nonlinear, because it can at least
    get rid of useless dimensions. However, if there are no useless
    dimensions---as in a Swiss roll dataset---then reducing
    dimensionality with PCA will lose too much information. You want to
    unroll the Swiss roll, not squash it.

5.  That's a trick question: it depends on the dataset. Let's look at
    two extreme examples. First, suppose the dataset is composed of
    points that are almost perfectly aligned. In this case, PCA can
    reduce the dataset down to just one dimension while still preserving
    95% of the variance. Now imagine that the dataset is composed of
    perfectly random points, scattered all around the 1,000 dimensions.
    In this case roughly 950 dimensions are required to preserve 95% of
    the variance. So the answer is, it depends on the dataset, and it
    could be any number between 1 and 950. Plotting the explained
    variance as a function of the number of dimensions is one way to get
    a rough idea of the dataset's intrinsic dimensionality.

6.  Regular PCA is the default, but it works only if the dataset fits in
    memory. Incremental PCA is useful for large datasets that don't fit
    in memory, but it is slower than regular PCA, so if the dataset fits
    in memory you should prefer regular PCA. Incremental PCA is also
    useful for online tasks, when you need to apply PCA on the fly,
    every time a new instance arrives. Randomized PCA is useful when you
    want to considerably reduce dimensionality and the dataset fits in
    memory; in this case, it is much faster than regular PCA. Finally,
    Kernel PCA is useful for nonlinear datasets.

7.  Intuitively, a dimensionality reduction algorithm performs well if
    it eliminates a lot of dimensions from the dataset without losing
    too much information. One way to measure this is to apply the
    reverse transformation and measure the reconstruction error.
    However, not all dimensionality reduction algorithms provide a
    reverse transformation. Alternatively, if you are using
    dimensionality reduction as a preprocessing step before another
    Machine Learning algorithm (e.g., a Random Forest classifier), then
    you can simply measure the performance of that second algorithm; if
    dimensionality reduction did not lose too much information, then the
    algorithm should perform just as well as when using the original
    dataset.

8.  It can absolutely make sense to chain two different dimensionality
    reduction algorithms. A common example is using PCA to quickly get
    rid of a large number of useless dimensions, then applying another
    much slower dimensionality reduction algorithm, such as LLE. This
    two-step approach will likely yield the same performance as using
    LLE only, but in a fraction of the time.

For the solutions to exercises 9 and 10, please see the Jupyter
notebooks available at
[*https://github.com/ageron/handson-ml2*](https://github.com/ageron/handson-ml2).
:::
:::

::: {.section data-type="sect1" pdf-bookmark="Chapter 9: Unsupervised Learning Techniques"}
::: {#idm45728432034872 .sect1}
[Chapter 9](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch09.html#unsupervised_learning_chapter): Unsupervised Learning Techniques
========================================================================================================================================================================

1.  In Machine Learning, clustering is the unsupervised task of grouping
    similar instances together. The notion of similarity depends on the
    task at hand: for example, in some cases two nearby instances will
    be considered similar, while in others similar instances may be far
    apart as long as they belong to the same densely packed group.
    Popular clustering algorithms include K-Means, DBSCAN, agglomerative
    clustering, BIRCH, Mean-Shift, affinity propagation, and spectral
    clustering.

2.  The main applications of clustering algorithms include data
    analysis, customer segmentation, recommender systems, search
    engines, image segmentation, semi-supervised learning,
    dimensionality reduction, anomaly detection, and novelty detection.

3.  The elbow rule is a simple technique to select the number of
    clusters when using K-Means: just plot the inertia (the mean squared
    distance from each instance to its nearest centroid) as a function
    of the number of clusters, and find the point in the curve where the
    inertia stops dropping fast (the "elbow"). This is generally close
    to the optimal number of clusters. Another approach is to plot the
    silhouette score as a function of the number of clusters. There will
    often be a peak, and the optimal number of clusters is generally
    nearby. The silhouette score is the mean silhouette coefficient over
    all instances. This coefficient varies from +1 for instances that
    are well inside their cluster and far from other clusters, to --1
    for instances that are very close to another cluster. You may also
    plot the silhouette diagrams and perform a more thorough analysis.

4.  Labeling a dataset is costly and time-consuming. Therefore, it is
    common to have plenty of unlabeled instances, but few labeled
    instances. Label propagation is a technique that consists in copying
    some (or all) of the labels from the labeled instances to similar
    unlabeled instances. This can greatly extend the number of labeled
    instances, and thereby allow a supervised algorithm to reach better
    performance (this is a form of semi-supervised learning). One
    approach is to use a clustering algorithm such as K-Means on all the
    instances, then for each cluster find the most common label or the
    label of the most representative instance (i.e., the one closest to
    the centroid) and propagate it to the unlabeled instances in the
    same cluster.

5.  K-Means and BIRCH scale well to large datasets. DBSCAN and
    Mean-Shift look for regions of high density.

6.  Active learning is useful whenever you have plenty of unlabeled
    instances but labeling is costly. In this case (which is very
    common), rather than randomly selecting instances to label, it is
    often preferable to perform active learning, where human experts
    interact with the learning algorithm, providing labels for specific
    instances when the algorithm requests them. A common approach is
    uncertainty sampling (see the description in ["Active
    Learning"](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch09.html#activelearning_sb)).

7.  Many people use the terms *anomaly detection* and *novelty
    detection* interchangeably, but they are not exactly the same. In
    anomaly detection, the algorithm is trained on a dataset that may
    contain outliers, and the goal is typically to identify these
    outliers (within the training set), as well as outliers among new
    instances. In novelty detection, the algorithm is trained on a
    dataset that is presumed to be "clean," and the objective is to
    detect novelties strictly among new instances. Some algorithms work
    best for anomaly detection (e.g., Isolation Forest), while others
    are better suited for novelty detection (e.g., one-class SVM).

8.  A Gaussian mixture model (GMM) is a probabilistic model that assumes
    that the instances were generated from a mixture of several Gaussian
    distributions whose parameters are unknown. In other words, the
    assumption is that the data is grouped into a finite number of
    clusters, each with an ellipsoidal shape (but the clusters may have
    different ellipsoidal shapes, sizes, orientations, and densities),
    and we don't know which cluster each instance belongs to. This model
    is useful for density estimation, clustering, and anomaly detection.

9.  One way to find the right number of clusters when using a Gaussian
    mixture model is to plot the Bayesian information criterion (BIC) or
    the Akaike information criterion (AIC) as a function of the number
    of clusters, then choose the number of clusters that minimizes the
    BIC or AIC. Another technique is to use a Bayesian Gaussian mixture
    model, which automatically selects the number of clusters.

For the solutions to exercises 10 to 13, please see the Jupyter
notebooks available at
[*https://github.com/ageron/handson-ml2*](https://github.com/ageron/handson-ml2).
:::
:::

::: {.section data-type="sect1" pdf-bookmark="Chapter 10: Introduction to Artificial Neural Networks with Keras"}
::: {#idm45728432016936 .sect1}
[Chapter 10](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch10.html#ann_chapter): Introduction to Artificial Neural Networks with Keras
============================================================================================================================================================================

1.  Visit the [TensorFlow
    Playground](https://playground.tensorflow.org/) and play around with
    it, as described in this exercise.

2.  Here is a neural network based on the original artificial neurons
    that computes *A* ⊕ *B* (where ⊕ represents the exclusive OR), using
    the fact that *A* ⊕ *B* = (*A* ∧ ¬ *B*) ∨ (¬ *A* ∧ *B*). There are
    other solutions---for example, using the fact that *A* ⊕ *B* = (*A*
    ∨ *B*) ∧ ¬(*A* ∧ *B*), or the fact that *A* ⊕ *B* = (*A* ∨ *B*) ∧ (¬
    *A* ∨ ¬ *B*), and so on.

    ![](./A_files/mls2_aain01.png){width="1442" height="650"}

3.  A classical Perceptron will converge only if the dataset is linearly
    separable, and it won't be able to estimate class probabilities. In
    contrast, a Logistic Regression classifier will converge to a good
    solution even if the dataset is not linearly separable, and it will
    output class probabilities. If you change the Perceptron's
    activation function to the logistic activation function (or the
    softmax activation function if there are multiple neurons), and if
    you train it using Gradient Descent (or some other optimization
    algorithm minimizing the cost function, typically cross entropy),
    then it becomes equivalent to a Logistic Regression classifier.

4.  The logistic activation function was a key ingredient in training
    the first MLPs because its derivative is always nonzero, so Gradient
    Descent can always roll down the slope. When the activation function
    is a step function, Gradient Descent cannot move, as there is no
    slope at all.

5.  Popular activation functions include the step function, the logistic
    (sigmoid) function, the hyperbolic tangent (tanh) function, and the
    Rectified Linear Unit (ReLU) function (see
    [Figure 10-8](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch10.html#activation_functions_plot)).
    See
    [Chapter 11](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch11.html#deep_chapter)
    for other examples, such as ELU and variants of the ReLU function.

6.  Considering the MLP described in the question, composed of one input
    layer with 10 passthrough neurons, followed by one hidden layer with
    50 artificial neurons, and finally one output layer with 3
    artificial neurons, where all artificial neurons use the ReLU
    activation function:

    1.  The shape of the input matrix **X** is *m* × 10, where *m*
        represents the training batch size.

    2.  The shape of the hidden layer's weight vector **W**~*h*~ is 10 ×
        50, and the length of its bias vector **b**~*h*~ is 50.

    3.  The shape of the output layer's weight vector **W**~*o*~ is 50 ×
        3, and the length of its bias vector **b**~*o*~ is 3.

    4.  The shape of the network's output matrix **Y** is *m* × 3.

    5.  Y\* = ReLU(ReLU(**X** **W**~*h*~ + **b**~*h*~) **W**~*o*~ +
        **b**~*o*~). Recall that the ReLU function just sets every
        negative number in the matrix to zero. Also note that when you
        are adding a bias vector to a matrix, it is added to every
        single row in the matrix, which is called *broadcasting*.

7.  To classify email into spam or ham, you just need one neuron in the
    output layer of a neural network---for example, indicating the
    probability that the email is spam. You would typically use the
    logistic activation function in the output layer when estimating a
    probability. If instead you want to tackle MNIST, you need 10
    neurons in the output layer, and you must replace the logistic
    function with the softmax activation function, which can handle
    multiple classes, outputting one probability per class. If you want
    your neural network to predict housing prices like in
    [Chapter 2](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch02.html#project_chapter),
    then you need one output neuron, using no activation function at all
    in the output
    layer.^[3](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/app01.html#idm45728431976808){#idm45728431976808-marker
    .totri-footnote}^

8.  Backpropagation is a technique used to train artificial neural
    networks. It first computes the gradients of the cost function with
    regard to every model parameter (all the weights and biases), then
    it performs a Gradient Descent step using these gradients. This
    backpropagation step is typically performed thousands or millions of
    times, using many training batches, until the model parameters
    converge to values that (hopefully) minimize the cost function. To
    compute the gradients, backpropagation uses reverse-mode autodiff
    (although it wasn't called that when backpropagation was invented,
    and it has been reinvented several times). Reverse-mode autodiff
    performs a forward pass through a computation graph, computing every
    node's value for the current training batch, and then it performs a
    reverse pass, computing all the gradients at once (see
    [Appendix D](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/app04.html#autodiff_appendix)
    for more details). So what's the difference? Well, backpropagation
    refers to the whole process of training an artificial neural network
    using multiple backpropagation steps, each of which computes
    gradients and uses them to perform a Gradient Descent step. In
    contrast, reverse-mode autodiff is just a technique to compute
    gradients efficiently, and it happens to be used by backpropagation.

9.  Here is a list of all the hyperparameters you can tweak in a basic
    MLP: the number of hidden layers, the number of neurons in each
    hidden layer, and the activation function used in each hidden layer
    and in the output
    layer.^[4](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/app01.html#idm45728431971048){#idm45728431971048-marker
    .totri-footnote}^ In general, the ReLU activation function (or one
    of its variants; see
    [Chapter 11](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch11.html#deep_chapter))
    is a good default for the hidden layers. For the output layer, in
    general you will want the logistic activation function for binary
    classification, the softmax activation function for multiclass
    classification, or no activation function for regression.

    If the MLP overfits the training data, you can try reducing the
    number of hidden layers and reducing the number of neurons per
    hidden layer.

10. See the Jupyter notebooks available at
    [*https://github.com/ageron/handson-ml2*](https://github.com/ageron/handson-ml2).
:::
:::

::: {.section data-type="sect1" pdf-bookmark="Chapter 11: Training Deep Neural Networks"}
::: {#idm45728431964744 .sect1}
[Chapter 11](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch11.html#deep_chapter): Training Deep Neural Networks
=====================================================================================================================================================

1.  No, all weights should be sampled independently; they should not all
    have the same initial value. One important goal of sampling weights
    randomly is to break symmetry: if all the weights have the same
    initial value, even if that value is not zero, then symmetry is not
    broken (i.e., all neurons in a given layer are equivalent), and
    backpropagation will be unable to break it. Concretely, this means
    that all the neurons in any given layer will always have the same
    weights. It's like having just one neuron per layer, and much
    slower. It is virtually impossible for such a configuration to
    converge to a good solution.

2.  It is perfectly fine to initialize the bias terms to zero. Some
    people like to initialize them just like weights, and that's okay
    too; it does not make much difference.

3.  A few advantages of the SELU function over the ReLU function are:

    -   It can take on negative values, so the average output of the
        neurons in any given layer is typically closer to zero than when
        using the ReLU activation function (which never outputs negative
        values). This helps alleviate the vanishing gradients problem.

    -   It always has a nonzero derivative, which avoids the dying units
        issue that can affect ReLU units.

    -   When the conditions are right (i.e., if the model is sequential,
        and the weights are initialized using LeCun initialization, and
        the inputs are standardized, and there's no incompatible layer
        or regularization, such as dropout or ℓ~1~ regularization), then
        the SELU activation function ensures the model is
        self-normalized, which solves the exploding/vanishing gradients
        problems.

4.  The SELU activation function is a good default. If you need the
    neural network to be as fast as possible, you can use one of the
    leaky ReLU variants instead (e.g., a simple leaky ReLU using the
    default hyperparameter value). The simplicity of the ReLU activation
    function makes it many people's preferred option, despite the fact
    that it is generally outperformed by SELU and leaky ReLU. However,
    the ReLU activation function's ability to output precisely zero can
    be useful in some cases (e.g., see
    [Chapter 17](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch17.html#autoencoders_chapter)).
    Moreover, it can sometimes benefit from optimized implementation as
    well as from hardware acceleration. The hyperbolic tangent (tanh)
    can be useful in the output layer if you need to output a number
    between --1 and 1, but nowadays it is not used much in hidden layers
    (except in recurrent nets). The logistic activation function is also
    useful in the output layer when you need to estimate a probability
    (e.g., for binary classification), but is rarely used in hidden
    layers (there are exceptions---for example, for the coding layer of
    variational autoencoders; see
    [Chapter 17](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch17.html#autoencoders_chapter)).
    Finally, the softmax activation function is useful in the output
    layer to output probabilities for mutually exclusive classes, but it
    is rarely (if ever) used in hidden layers.

5.  If you set the `momentum` hyperparameter too close to 1 (e.g.,
    0.99999) when using an `SGD` optimizer, then the algorithm will
    likely pick up a lot of speed, hopefully moving roughly toward the
    global minimum, but its momentum will carry it right past the
    minimum. Then it will slow down and come back, accelerate again,
    overshoot again, and so on. It may oscillate this way many times
    before converging, so overall it will take much longer to converge
    than with a smaller `momentum` value.

6.  One way to produce a sparse model (i.e., with most weights equal to
    zero) is to train the model normally, then zero out tiny weights.
    For more sparsity, you can apply ℓ~1~ regularization during
    training, which pushes the optimizer toward sparsity. A third option
    is to use the TensorFlow Model Optimization Toolkit.

7.  Yes, dropout does slow down training, in general roughly by a factor
    of two. However, it has no impact on inference speed since it is
    only turned on during training. MC Dropout is exactly like dropout
    during training, but it is still active during inference, so each
    inference is slowed down slightly. More importantly, when using MC
    Dropout you generally want to run inference 10 times or more to get
    better predictions. This means that making predictions is slowed
    down by a factor of 10 or more.

For the solution to exercise 8, please see the Jupyter notebooks
available at
[*https://github.com/ageron/handson-ml2*](https://github.com/ageron/handson-ml2).
:::
:::

::: {.section data-type="sect1" pdf-bookmark="Chapter 12: Custom Models and Training with TensorFlow"}
::: {#idm45728431943752 .sect1}
[Chapter 12](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch12.html#tensorflow_chapter): Custom Models and Training with TensorFlow
========================================================================================================================================================================

1.  TensorFlow is an open-source library for numerical computation,
    particularly well suited and fine-tuned for large-scale Machine
    Learning. Its core is similar to NumPy, but it also features GPU
    support, support for distributed computing, computation graph
    analysis and optimization capabilities (with a portable graph format
    that allows you to train a TensorFlow model in one environment and
    run it in another), an optimization API based on reverse-mode
    autodiff, and several powerful APIs such as tf.keras, tf.data,
    tf.image, tf.signal, and more. Other popular Deep Learning libraries
    include PyTorch, MXNet, Microsoft Cognitive Toolkit, Theano, Caffe2,
    and Chainer.

2.  Although TensorFlow offers most of the functionalities provided by
    NumPy, it is not a drop-in replacement, for a few reasons. First,
    the names of the functions are not always the same (for example,
    `tf.reduce_sum()` versus `np.sum()`). Second, some functions do not
    behave in exactly the same way (for example, `tf.transpose()`
    creates a transposed copy of a tensor, while NumPy's `T` attribute
    creates a transposed view, without actually copying any data).
    Lastly, NumPy arrays are mutable, while TensorFlow tensors are not
    (but you can use a `tf.Variable` if you need a mutable object).

3.  Both `tf.range(10)` and `tf.constant(np.arange(10))` return a
    one-dimensional tensor containing the integers 0 to 9. However, the
    former uses 32-bit integers while the latter uses 64-bit integers.
    Indeed, TensorFlow defaults to 32 bits, while NumPy defaults to 64
    bits.

4.  Beyond regular tensors, TensorFlow offers several other data
    structures, including sparse tensors, tensor arrays, ragged tensors,
    queues, string tensors, and sets. The last two are actually
    represented as regular tensors, but TensorFlow provides special
    functions to manipulate them (in `tf.strings` and `tf.sets`).

5.  When you want to define a custom loss function, in general you can
    just implement it as a regular Python function. However, if your
    custom loss function must support some hyperparameters (or any other
    state), then you should subclass the `keras.losses.Loss` class and
    implement the `__init__()` and `call()` methods. If you want the
    loss function's hyperparameters to be saved along with the model,
    then you must also implement the `get_config()` method.

6.  Much like custom loss functions, most metrics can be defined as
    regular Python functions. But if you want your custom metric to
    support some hyperparameters (or any other state), then you should
    subclass the `keras.metrics.Metric` class. Moreover, if computing
    the metric over a whole epoch is not equivalent to computing the
    mean metric over all batches in that epoch (e.g., as for the
    precision and recall metrics), then you should subclass the
    `keras.metrics.Metric` class and implement the `__init__()`,
    `update_state()`, and `result()` methods to keep track of a running
    metric during each epoch. You should also implement the
    `reset_states()` method unless all it needs to do is reset all
    variables to 0.0. If you want the state to be saved along with the
    model, then you should implement the `get_config()` method as well.

7.  You should distinguish the internal components of your model (i.e.,
    layers or reusable blocks of layers) from the model itself (i.e.,
    the object you will train). The former should subclass the
    `keras.layers.Layer` class, while the latter should subclass the
    `keras.models.Model` class.

8.  Writing your own custom training loop is fairly advanced, so you
    should only do it if you really need to. Keras provides several
    tools to customize training without having to write a custom
    training loop: callbacks, custom regularizers, custom constraints,
    custom losses, and so on. You should use these instead of writing a
    custom training loop whenever possible: writing a custom training
    loop is more error-prone, and it will be harder to reuse the custom
    code you write. However, in some cases writing a custom training
    loop is necessary⁠---for example, if you want to use different
    optimizers for different parts of your neural network, like in the
    [Wide & Deep paper](https://homl.info/widedeep). A custom training
    loop can also be useful when debugging, or when trying to understand
    exactly how training works.

9.  Custom Keras components should be convertible to TF Functions, which
    means they should stick to TF operations as much as possible and
    respect all the rules listed in ["TF Function
    Rules"](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch12.html#tf_functionrules).
    If you absolutely need to include arbitrary Python code in a custom
    component, you can either wrap it in a `tf.py_function()` operation
    (but this will reduce performance and limit your model's
    portability) or set `dynamic=True` when creating the custom layer or
    model (or set `run_eagerly=True` when calling the model's
    `compile()` method).

10. Please refer to ["TF Function
    Rules"](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch12.html#tf_functionrules)
    for the list of rules to respect when creating a TF Function.

11. Creating a dynamic Keras model can be useful for debugging, as it
    will not compile any custom component to a TF Function, and you can
    use any Python debugger to debug your code. It can also be useful if
    you want to include arbitrary Python code in your model (or in your
    training code), including calls to external libraries. To make a
    model dynamic, you must set `dynamic=True` when creating it.
    Alternatively, you can set `run_eagerly=True` when calling the
    model's `compile()` method. Making a model dynamic prevents Keras
    from using any of TensorFlow's graph features, so it will slow down
    training and inference, and you will not have the possibility to
    export the computation graph, which will limit your model's
    portability.

For the solutions to exercises 12 and 13, please see the Jupyter
notebooks available at
[*https://github.com/ageron/handson-ml2*](https://github.com/ageron/handson-ml2).
:::
:::

::: {.section data-type="sect1" pdf-bookmark="Chapter 13: Loading and Preprocessing Data with TensorFlow"}
::: {#idm45728431910072 .sect1}
[Chapter 13](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch13.html#data_chapter): Loading and Preprocessing Data with TensorFlow
======================================================================================================================================================================

1.  Ingesting a large dataset and preprocessing it efficiently can be a
    complex engineering challenge. The Data API makes it fairly simple.
    It offers many features, including loading data from various sources
    (such as text or binary files), reading data in parallel from
    multiple sources, transforming it, interleaving the records,
    shuffling the data, batching it, and prefetching it.

2.  Splitting a large dataset into multiple files makes it possible to
    shuffle it at a coarse level before shuffling it at a finer level
    using a shuffling buffer. It also makes it possible to handle huge
    datasets that do not fit on a single machine. It's also simpler to
    manipulate thousands of small files rather than one huge file; for
    example, it's easier to split the data into multiple subsets.
    Lastly, if the data is split across multiple files spread across
    multiple servers, it is possible to download several files from
    different servers simultaneously, which improves the bandwidth
    usage.

3.  You can use TensorBoard to visualize profiling data: if the GPU is
    not fully utilized then your input pipeline is likely to be the
    bottleneck. You can fix it by making sure it reads and preprocesses
    the data in multiple threads in parallel, and ensuring it prefetches
    a few batches. If this is insufficient to get your GPU to 100% usage
    during training, make sure your preprocessing code is optimized. You
    can also try saving the dataset into multiple TFRecord files, and if
    necessary perform some of the preprocessing ahead of time so that it
    does not need to be done on the fly during training (TF Transform
    can help with this). If necessary, use a machine with more CPU and
    RAM, and ensure that the GPU bandwidth is large enough.

4.  A TFRecord file is composed of a sequence of arbitrary binary
    records: you can store absolutely any binary data you want in each
    record. However, in practice most TFRecord files contain sequences
    of serialized protocol buffers. This makes it possible to benefit
    from the advantages of protocol buffers, such as the fact that they
    can be read easily across multiple platforms and languages and their
    definition can be updated later in a backward-compatible way.

5.  The `Example` protobuf format has the advantage that TensorFlow
    provides some operations to parse it (the `tf.io.parse*example()`
    functions) without you having to define your own format. It is
    sufficiently flexible to represent instances in most datasets.
    However, if it does not cover your use case, you can define your own
    protocol buffer, compile it using `protoc` (setting the
    `--descriptor_set_out` and `--include_imports` arguments to export
    the protobuf descriptor), and use the `tf.io.decode_proto()`
    function to parse the serialized protobufs (see the "Custom
    protobuf" section of the notebook for an example). It's more
    complicated, and it requires deploying the descriptor along with the
    model, but it can be done.

6.  When using TFRecords, you will generally want to activate
    compression if the TFRecord files will need to be downloaded by the
    training script, as compression will make files smaller and thus
    reduce download time. But if the files are located on the same
    machine as the training script, it's usually preferable to leave
    compression off, to avoid wasting CPU for decompression.

7.  Let's look at the pros and cons of each preprocessing option:

    -   If you preprocess the data when creating the data files, the
        training script will run faster, since it will not have to
        perform preprocessing on the fly. In some cases, the
        preprocessed data will also be much smaller than the original
        data, so you can save some space and speed up downloads. It may
        also be helpful to materialize the preprocessed data, for
        example to inspect it or archive it. However, this approach has
        a few cons. First, it's not easy to experiment with various
        preprocessing logics if you need to generate a preprocessed
        dataset for each variant. Second, if you want to perform data
        augmentation, you have to materialize many variants of your
        dataset, which will use a large amount of disk space and take a
        lot of time to generate. Lastly, the trained model will expect
        preprocessed data, so you will have to add preprocessing code in
        your application before it calls the model.

    -   If the data is preprocessed with the tf.data pipeline, it's much
        easier to tweak the preprocessing logic and apply data
        augmentation. Also, tf.data makes it easy to build highly
        efficient preprocessing pipelines (e.g., with multithreading and
        prefetching). However, preprocessing the data this way will slow
        down training. Moreover, each training instance will be
        preprocessed once per epoch rather than just once if the data
        was preprocessed when creating the data files. Lastly, the
        trained model will still expect preprocessed data.

    -   If you add preprocessing layers to your model, you will only
        have to write the preprocessing code once for both training and
        inference. If your model needs to be deployed to many different
        platforms, you will not need to write the preprocessing code
        multiple times. Plus, you will not run the risk of using the
        wrong preprocessing logic for your model, since it will be part
        of the model. On the downside, preprocessing the data will slow
        down training, and each training instance will be preprocessed
        once per epoch. Moreover, by default the preprocessing
        operations will run on the GPU for the current batch (you will
        not benefit from parallel preprocessing on the CPU, and
        prefetching). Fortunately, the upcoming Keras preprocessing
        layers should be able to lift the preprocessing operations from
        the preprocessing layers and run them as part of the tf.data
        pipeline, so you will benefit from multithreaded execution on
        the CPU and prefetching.

    -   Lastly, using TF Transform for preprocessing gives you many of
        the benefits from the previous options: the preprocessed data is
        materialized, each instance is preprocessed just once (speeding
        up training), and preprocessing layers get generated
        automatically so you only need to write the preprocessing code
        once. The main drawback is the fact that you need to learn how
        to use this tool.

8.  Let's look at how to encode categorical features and text:

    -   To encode a categorical feature that has a natural order, such
        as a movie rating (e.g., "bad," "average," "good"), the simplest
        option is to use ordinal encoding: sort the categories in their
        natural order and map each category to its rank (e.g., "bad"
        maps to 0, "average" maps to 1, and "good" maps to 2). However,
        most categorical features don't have such a natural order. For
        example, there's no natural order for professions or countries.
        In this case, you can use one-hot encoding or, if there are many
        categories, embeddings.

    -   For text, one option is to use a bag-of-words representation: a
        sentence is represented by a vector counting the counts of each
        possible word. Since common words are usually not very
        important, you'll want to use TF-IDF to reduce their weight.
        Instead of counting words, it is also common to count *n*-grams,
        which are sequences of *n* consecutive words⁠---nice and simple.
        Alternatively, you can encode each word using word embeddings,
        possibly pretrained. Rather than encoding words, it is also
        possible to encode each letter, or subword tokens (e.g.,
        splitting "smartest" into "smart" and "est"). These last two
        options are discussed in
        [Chapter 16](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch16.html#nlp_chapter).

For the solutions to exercises 9 and 10, please see the Jupyter
notebooks available at
[*https://github.com/ageron/handson-ml2*](https://github.com/ageron/handson-ml2).
:::
:::

::: {.section data-type="sect1" pdf-bookmark="Chapter 14: Deep Computer Vision Using Convolutional Neural Networks"}
::: {#idm45728431881112 .sect1}
[Chapter 14](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch14.html#cnn_chapter): Deep Computer Vision Using Convolutional Neural Networks
===============================================================================================================================================================================

1.  These are the main advantages of a CNN over a fully connected DNN
    for image classification:

    -   Because consecutive layers are only partially connected and
        because it heavily reuses its weights, a CNN has many fewer
        parameters than a fully connected DNN, which makes it much
        faster to train, reduces the risk of overfitting, and requires
        much less training data.

    -   When a CNN has learned a kernel that can detect a particular
        feature, it can detect that feature anywhere in the image. In
        contrast, when a DNN learns a feature in one location, it can
        detect it only in that particular location. Since images
        typically have very repetitive features, CNNs are able to
        generalize much better than DNNs for image processing tasks such
        as classification, using fewer training examples.

    -   Finally, a DNN has no prior knowledge of how pixels are
        organized; it does not know that nearby pixels are close. A
        CNN's architecture embeds this prior knowledge. Lower layers
        typically identify features in small areas of the images, while
        higher layers combine the lower-level features into larger
        features. This works well with most natural images, giving CNNs
        a decisive head start compared to DNNs.

2.  Let's compute how many parameters the CNN has. Since its first
    convolutional layer has 3 × 3 kernels, and the input has three
    channels (red, green, and blue), each feature map has 3 × 3 × 3
    weights, plus a bias term. That's 28 parameters per feature map.
    Since this first convolutional layer has 100 feature maps, it has a
    total of 2,800 parameters. The second convolutional layer has 3 × 3
    kernels and its input is the set of 100 feature maps of the previous
    layer, so each feature map has 3 × 3 × 100 = 900 weights, plus a
    bias term. Since it has 200 feature maps, this layer has 901 × 200 =
    180,200 parameters. Finally, the third and last convolutional layer
    also has 3 × 3 kernels, and its input is the set of 200 feature maps
    of the previous layers, so each feature map has 3 × 3 × 200 = 1,800
    weights, plus a bias term. Since it has 400 feature maps, this layer
    has a total of 1,801 × 400 = 720,400 parameters. All in all, the CNN
    has 2,800 + 180,200 + 720,400 = 903,400 parameters.

    Now let's compute how much RAM this neural network will require (at
    least) when making a prediction for a single instance. First let's
    compute the feature map size for each layer. Since we are using a
    stride of 2 and `"same"` padding, the horizontal and vertical
    dimensions of the feature maps are divided by 2 at each layer
    (rounding up if necessary). So, as the input channels are 200 × 300
    pixels, the first layer's feature maps are 100 × 150, the second
    layer's feature maps are 50 × 75, and the third layer's feature maps
    are 25 × 38. Since 32 bits is 4 bytes and the first convolutional
    layer has 100 feature maps, this first layer takes up 4 × 100 × 150
    × 100 = 6 million bytes (6 MB). The second layer takes up 4 × 50 ×
    75 × 200 = 3 million bytes (3 MB). Finally, the third layer takes up
    4 × 25 × 38 × 400 = 1,520,000 bytes (about 1.5 MB). However, once a
    layer has been computed, the memory occupied by the previous layer
    can be released, so if everything is well optimized, only 6 + 3 = 9
    million bytes (9 MB) of RAM will be required (when the second layer
    has just been computed, but the memory occupied by the first layer
    has not been released yet). But wait, you also need to add the
    memory occupied by the CNN's parameters! We computed earlier that it
    has 903,400 parameters, each using up 4 bytes, so this adds
    3,613,600 bytes (about 3.6 MB). The total RAM required is therefore
    (at least) 12,613,600 bytes (about 12.6 MB).

    Lastly, let's compute the minimum amount of RAM required when
    training the CNN on a mini-batch of 50 images. During training
    TensorFlow uses backpropagation, which requires keeping all values
    computed during the forward pass until the reverse pass begins. So
    we must compute the total RAM required by all layers for a single
    instance and multiply that by 50. At this point, let's start
    counting in megabytes rather than bytes. We computed before that the
    three layers require respectively 6, 3, and 1.5 MB for each
    instance. That's a total of 10.5 MB per instance, so for 50
    instances the total RAM required is 525 MB. Add to that the RAM
    required by the input images, which is 50 × 4 × 200 × 300 × 3 = 36
    million bytes (36 MB), plus the RAM required for the model
    parameters, which is about 3.6 MB (computed earlier), plus some RAM
    for the gradients (we will neglect this since it can be released
    gradually as backpropagation goes down the layers during the reverse
    pass). We are up to a total of roughly 525 + 36 + 3.6 = 564.6 MB,
    and that's really an optimistic bare minimum.

3.  If your GPU runs out of memory while training a CNN, here are five
    things you could try to solve the problem (other than purchasing a
    GPU with more RAM):

    -   Reduce the mini-batch size.

    -   Reduce dimensionality using a larger stride in one or more
        layers.

    -   Remove one or more layers.

    -   Use 16-bit floats instead of 32-bit floats.

    -   Distribute the CNN across multiple devices.

4.  A max pooling layer has no parameters at all, whereas a
    convolutional layer has quite a few (see the previous questions).

5.  A local response normalization layer makes the neurons that most
    strongly activate inhibit neurons at the same location but in
    neighboring feature maps, which encourages different feature maps to
    specialize and pushes them apart, forcing them to explore a wider
    range of features. It is typically used in the lower layers to have
    a larger pool of low-level features that the upper layers can build
    upon.

6.  The main innovations in AlexNet compared to LeNet-5 are that it is
    much larger and deeper, and it stacks convolutional layers directly
    on top of each other, instead of stacking a pooling layer on top of
    each convolutional layer. The main innovation in GoogLeNet is the
    introduction of *inception modules*, which make it possible to have
    a much deeper net than previous CNN architectures, with fewer
    parameters. ResNet's main innovation is the introduction of skip
    connections, which make it possible to go well beyond 100 layers.
    Arguably, its simplicity and consistency are also rather innovative.
    SENet's main innovation was the idea of using an SE block (a
    two-layer dense network) after every inception module in an
    inception network or every residual unit in a ResNet to recalibrate
    the relative importance of feature maps. Finally, Xception's main
    innovation was the use of depthwise separable convolutional layers,
    which look at spatial patterns and depthwise patterns separately.

7.  Fully convolutional networks are neural networks composed
    exclusively of convolutional and pooling layers. FCNs can
    efficiently process images of any width and height (at least above
    the minimum size). They are most useful for object detection and
    semantic segmentation because they only need to look at the image
    once (instead of having to run a CNN multiple times on different
    parts of the image). If you have a CNN with some dense layers on
    top, you can convert these dense layers to convolutional layers to
    create an FCN: just replace the lowest dense layer with a
    convolutional layer with a kernel size equal to the layer's input
    size, with one filter per neuron in the dense layer, and using
    `"valid"` padding. Generally the stride should be 1, but you can set
    it to a higher value if you want. The activation function should be
    the same as the dense layer's. The other dense layers should be
    converted the same way, but using 1 × 1 filters. It is actually
    possible to convert a trained CNN this way by appropriately
    reshaping the dense layers' weight matrices.

8.  The main technical difficulty of semantic segmentation is the fact
    that a lot of the spatial information gets lost in a CNN as the
    signal flows through each layer, especially in pooling layers and
    layers with a stride greater than 1. This spatial information needs
    to be restored somehow to accurately predict the class of each
    pixel.

For the solutions to exercises 9 to 11, please see the Jupyter notebooks
available at
[*https://github.com/ageron/handson-ml2*](https://github.com/ageron/handson-ml2).
:::
:::

::: {.section data-type="sect1" pdf-bookmark="Chapter 15: Processing Sequences Using RNNs and CNNs"}
::: {#idm45728431852088 .sect1}
[Chapter 15](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch15.html#rnn_chapter): Processing Sequences Using RNNs and CNNs
===============================================================================================================================================================

1.  Here are a few RNN applications:

    -   For a sequence-to-sequence RNN: predicting the weather (or any
        other time series), machine translation (using an
        Encoder--Decoder architecture), video captioning, speech to
        text, music generation (or other sequence generation),
        identifying the chords of a song

    -   For a sequence-to-vector RNN: classifying music samples by music
        genre, analyzing the sentiment of a book review, predicting what
        word an aphasic patient is thinking of based on readings from
        brain implants, predicting the probability that a user will want
        to watch a movie based on their watch history (this is one of
        many possible implementations of *collaborative filtering* for a
        recommender system)

    -   For a vector-to-sequence RNN: image captioning, creating a music
        playlist based on an embedding of the current artist, generating
        a melody based on a set of parameters, locating pedestrians in a
        picture (e.g., a video frame from a self-driving car's camera)

2.  An RNN layer must have three-dimensional inputs: the first dimension
    is the batch dimension (its size is the batch size), the second
    dimension represents the time (its size is the number of time
    steps), and the third dimension holds the inputs at each time step
    (its size is the number of input features per time step). For
    example, if you want to process a batch containing 5 time series of
    10 time steps each, with 2 values per time step (e.g., the
    temperature and the wind speed), the shape will be \[5, 10, 2\]. The
    outputs are also three-dimensional, with the same first two
    dimensions, but the last dimension is equal to the number of
    [neurons]{.keep-together}. For example, if an RNN layer with 32
    neurons processes the batch we just discussed, the output will have
    a shape of \[5, 10, 32\].

3.  To build a deep sequence-to-sequence RNN using Keras, you must set
    `return_sequences=True` for all RNN layers. To build a
    sequence-to-vector RNN, you must set `return_sequences=True` for all
    RNN layers except for the top RNN layer, which must have
    `return_sequences=False` (or do not set this argument at all, since
    `False` is the default).

4.  If you have a daily univariate time series, and you want to forecast
    the next seven days, the simplest RNN architecture you can use is a
    stack of RNN layers (all with `return_sequences=True` except for the
    top RNN layer), using seven neurons in the output RNN layer. You can
    then train this model using random windows from the time series
    (e.g., sequences of 30 consecutive days as the inputs, and a vector
    containing the values of the next 7 days as the target). This is a
    sequence-to-vector RNN. Alternatively, you could set
    `return_sequences=True` for all RNN layers to create a
    sequence-to-sequence RNN. You can train this model using random
    windows from the time series, with sequences of the same length as
    the inputs as the targets. Each target sequence should have seven
    values per time step (e.g., for time step *t*, the target should be
    a vector containing the values at time steps *t* + 1 to *t* + 7).

5.  The two main difficulties when training RNNs are unstable gradients
    (exploding or vanishing) and a very limited short-term memory. These
    problems both get worse when dealing with long sequences. To
    alleviate the unstable gradients problem, you can use a smaller
    learning rate, use a saturating activation function such as the
    hyperbolic tangent (which is the default), and possibly use gradient
    clipping, Layer Normalization, or dropout at each time step. To
    tackle the limited short-term memory problem, you can use `LSTM` or
    `GRU` layers (this also helps with the unstable gradients problem).

6.  An LSTM cell's architecture looks complicated, but it's actually not
    too hard if you understand the underlying logic. The cell has a
    short-term state vector and a long-term state vector. At each time
    step, the inputs and the previous short-term state are fed to a
    simple RNN cell and three gates: the forget gate decides what to
    remove from the long-term state, the input gate decides which part
    of the output of the simple RNN cell should be added to the
    long-term state, and the output gate decides which part of the
    long-term state should be output at this time step (after going
    through the tanh activation function). The new short-term state is
    equal to the output of the cell. See
    [Figure 15-9](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch15.html#lstm_cell_diagram).

7.  An RNN layer is fundamentally sequential: in order to compute the
    outputs at time step *t*, it has to first compute the outputs at all
    earlier time steps. This makes it impossible to parallelize. On the
    other hand, a 1D convolutional layer lends itself well to
    parallelization since it does not hold a state between time steps.
    In other words, it has no memory: the output at any time step can be
    computed based only on a small window of values from the inputs
    without having to know all the past values. Moreover, since a 1D
    convolutional layer is not recurrent, it suffers less from unstable
    gradients. One or more 1D convolutional layers can be useful in an
    RNN to efficiently preprocess the inputs, for example to reduce
    their temporal resolution (downsampling) and thereby help the RNN
    layers detect long-term patterns. In fact, it is possible to use
    only convolutional layers, for example by building a WaveNet
    architecture.

8.  To classify videos based on their visual content, one possible
    architecture could be to take (say) one frame per second, then run
    every frame through the same convolutional neural network (e.g., a
    pretrained Xception model, possibly frozen if your dataset is not
    large), feed the sequence of outputs from the CNN to a
    sequence-to-vector RNN, and finally run its output through a softmax
    layer, giving you all the class probabilities. For training you
    would use cross entropy as the cost function. If you wanted to use
    the audio for classification as well, you could use a stack of
    strided 1D convolutional layers to reduce the temporal resolution
    from thousands of audio frames per second to just one per second (to
    match the number of images per second), and concatenate the output
    sequence to the inputs of the sequence-to-vector RNN (along the last
    dimension).

For the solutions to exercises 9 and 10, please see the Jupyter
notebooks available at
[*https://github.com/ageron/handson-ml2*](https://github.com/ageron/handson-ml2).
:::
:::

::: {.section data-type="sect1" pdf-bookmark="Chapter 16: Natural Language Processing with RNNs and Attention"}
::: {#idm45728431824840 .sect1}
[Chapter 16](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch16.html#nlp_chapter): Natural Language Processing with RNNs and Attention
==========================================================================================================================================================================

1.  Stateless RNNs can only capture patterns whose length is less than,
    or equal to, the size of the windows the RNN is trained on.
    Conversely, stateful RNNs can capture longer-term patterns. However,
    implementing a stateful RNN is much harder⁠---especially preparing
    the dataset properly. Moreover, stateful RNNs do not always work
    better, in part because consecutive batches are not independent and
    identically distributed (IID). Gradient Descent is not fond of
    non-IID [datasets]{.keep-together}.

2.  In general, if you translate a sentence one word at a time, the
    result will be terrible. For example, the French sentence "Je vous
    en prie" means "You are welcome," but if you translate it one word
    at a time, you get "I you in pray." Huh? It is much better to read
    the whole sentence first and then translate it. A plain
    sequence-to-sequence RNN would start translating a sentence
    immediately after reading the first word, while an Encoder--Decoder
    RNN will first read the whole sentence and then translate it. That
    said, one could imagine a plain sequence-to-sequence RNN that would
    output silence whenever it is unsure about what to say next (just
    like human translators do when they must translate a live
    broadcast).

3.  Variable-length input sequences can be handled by padding the
    shorter sequences so that all sequences in a batch have the same
    length, and using masking to ensure the RNN ignores the padding
    token. For better performance, you may also want to create batches
    containing sequences of similar sizes. Ragged tensors can hold
    sequences of variable lengths, and tf.keras will likely support them
    eventually, which will greatly simplify handling variable-length
    input sequences (at the time of this writing, it is not the case
    yet). Regarding variable-length output sequences, if the length of
    the output sequence is known in advance (e.g., if you know that it
    is the same as the input sequence), then you just need to configure
    the loss function so that it ignores tokens that come after the end
    of the sequence. Similarly, the code that will use the model should
    ignore tokens beyond the end of the sequence. But generally the
    length of the output sequence is not known ahead of time, so the
    solution is to train the model so that it outputs an end-of-sequence
    token at the end of each sequence.

4.  Beam search is a technique used to improve the performance of a
    trained Encoder--Decoder model, for example in a neural machine
    translation system. The algorithm keeps track of a short list of the
    *k* most promising output sentences (say, the top three), and at
    each decoder step it tries to extend them by one word; then it keeps
    only the *k* most likely sentences. The parameter *k* is called the
    *beam width*: the larger it is, the more CPU and RAM will be used,
    but also the more accurate the system will be. Instead of greedily
    choosing the most likely next word at each step to extend a single
    sentence, this technique allows the system to explore several
    promising sentences simultaneously. Moreover, this technique lends
    itself well to parallelization. You can implement beam search fairly
    easily using TensorFlow Addons.

5.  An attention mechanism is a technique initially used in
    Encoder--Decoder models to give the decoder more direct access to
    the input sequence, allowing it to deal with longer input sequences.
    At each decoder time step, the current decoder's state and the full
    output of the encoder are processed by an alignment model that
    outputs an alignment score for each input time step. This score
    indicates which part of the input is most relevant to the current
    decoder time step. The weighted sum of the encoder output (weighted
    by their alignment score) is then fed to the decoder, which produces
    the next decoder state and the output for this time step. The main
    benefit of using an attention mechanism is the fact that the
    Encoder--Decoder model can successfully process longer input
    sequences. Another benefit is that the alignment scores makes the
    model easier to debug and interpret: for example, if the model makes
    a mistake, you can look at which part of the input it was paying
    attention to, and this can help diagnose the issue. An attention
    mechanism is also at the core of the Transformer architecture, in
    the Multi-Head Attention layers. See the next answer.

6.  The most important layer in the Transformer architecture is the
    Multi-Head Attention layer (the original Transformer architecture
    contains 18 of them, including 6 Masked Multi-Head Attention
    layers). It is at the core of language models such as BERT and
    GPT-2. Its purpose is to allow the model to identify which words are
    most aligned with each other, and then improve each word's
    representation using these contextual clues.

7.  Sampled softmax is used when training a classification model when
    there are many classes (e.g., thousands). It computes an
    approximation of the cross-entropy loss based on the logit predicted
    by the model for the correct class, and the predicted logits for a
    sample of incorrect words. This speeds up training considerably
    compared to computing the softmax over all logits and then
    estimating the cross-entropy loss. After training, the model can be
    used normally, using the regular softmax function to compute all the
    class probabilities based on all the logits.

For the solutions to exercises 8 to 11, please see the Jupyter notebooks
available at
[*https://github.com/ageron/handson-ml2*](https://github.com/ageron/handson-ml2).
:::
:::

::: {.section data-type="sect1" pdf-bookmark="Chapter 17: Representation Learning and Generative Learning Using Autoencoders and GANs"}
::: {#idm45728431807080 .sect1}
[Chapter 17](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch17.html#autoencoders_chapter): Representation Learning and Generative Learning Using Autoencoders and GANs
===========================================================================================================================================================================================================

1.  Here are some of the main tasks that autoencoders are used for:

    -   Feature extraction

    -   Unsupervised pretraining

    -   Dimensionality reduction

    -   Generative models

    -   Anomaly detection (an autoencoder is generally bad at
        reconstructing outliers)

2.  If you want to train a classifier and you have plenty of unlabeled
    training data but only a few thousand labeled instances, then you
    could first train a deep autoencoder on the full dataset (labeled +
    unlabeled), then reuse its lower half for the classifier (i.e.,
    reuse the layers up to the codings layer, included) and train the
    classifier using the labeled data. If you have little labeled data,
    you probably want to freeze the reused layers when training the
    classifier.

3.  The fact that an autoencoder perfectly reconstructs its inputs does
    not necessarily mean that it is a good autoencoder; perhaps it is
    simply an overcomplete autoencoder that learned to copy its inputs
    to the codings layer and then to the outputs. In fact, even if the
    codings layer contained a single neuron, it would be possible for a
    very deep autoencoder to learn to map each training instance to a
    different coding (e.g., the first instance could be mapped to 0.001,
    the second to 0.002, the third to 0.003, and so on), and it could
    learn "by heart" to reconstruct the right training instance for each
    coding. It would perfectly reconstruct its inputs without really
    learning any useful pattern in the data. In practice such a mapping
    is unlikely to happen, but it illustrates the fact that perfect
    reconstructions are not a guarantee that the autoencoder learned
    anything useful. However, if it produces very bad reconstructions,
    then it is almost guaranteed to be a bad autoencoder. To evaluate
    the performance of an autoencoder, one option is to measure the
    reconstruction loss (e.g., compute the MSE, or the mean square of
    the outputs minus the inputs). Again, a high reconstruction loss is
    a good sign that the autoencoder is bad, but a low reconstruction
    loss is not a guarantee that it is good. You should also evaluate
    the autoencoder according to what it will be used for. For example,
    if you are using it for unsupervised pretraining of a classifier,
    then you should also evaluate the classifier's performance.

4.  An undercomplete autoencoder is one whose codings layer is smaller
    than the input and output layers. If it is larger, then it is an
    overcomplete autoencoder. The main risk of an excessively
    undercomplete autoencoder is that it may fail to reconstruct the
    inputs. The main risk of an overcomplete autoencoder is that it may
    just copy the inputs to the outputs, without learning any useful
    features.

5.  To tie the weights of an encoder layer and its corresponding decoder
    layer, you simply make the decoder weights equal to the transpose of
    the encoder weights. This reduces the number of parameters in the
    model by half, often making training converge faster with less
    training data and reducing the risk of overfitting the training set.

6.  A generative model is a model capable of randomly generating outputs
    that resemble the training instances. For example, once trained
    successfully on the MNIST dataset, a generative model can be used to
    randomly generate realistic images of digits. The output
    distribution is typically similar to the training data. For example,
    since MNIST contains many images of each digit, the generative model
    would output roughly the same number of images of each digit. Some
    generative models can be parametrized---for example, to generate
    only some kinds of outputs. An example of a generative autoencoder
    is the variational autoencoder.

7.  A generative adversarial network is a neural network architecture
    composed of two parts, the generator and the discriminator, which
    have opposing objectives. The generator's goal is to generate
    instances similar to those in the training set, to fool the
    discriminator. The discriminator must distinguish the real instances
    from the generated ones. At each training iteration, the
    discriminator is trained like a normal binary classifier, then the
    generator is trained to maximize the
    [discriminator's]{.keep-together} error. GANs are used for advanced
    image processing tasks such as super resolution, colorization, image
    editing (replacing objects with realistic background), turning a
    simple sketch into a photorealistic image, or predicting the next
    frames in a video. They are also used to augment a dataset (to train
    other models), to generate other types of data (such as text, audio,
    and time series), and to identify the weaknesses in other models and
    strengthen them.

8.  Training GANs is notoriously difficult, because of the complex
    dynamics between the generator and the discriminator. The biggest
    difficulty is mode collapse, where the generator produces outputs
    with very little diversity. Moreover, training can be terribly
    unstable: it may start out fine and then suddenly start oscillating
    or diverging, without any apparent reason. GANs are also very
    sensitive to the choice of hyperparameters.

For the solutions to exercises 9, 10, and 11, please see the Jupyter
notebooks available at
[*https://github.com/ageron/handson-ml2*](https://github.com/ageron/handson-ml2).
:::
:::

::: {.section data-type="sect1" pdf-bookmark="Chapter 18: Reinforcement Learning"}
::: {#idm45728431785448 .sect1}
[Chapter 18](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch18.html#rl_chapter): Reinforcement Learning
============================================================================================================================================

1.  Reinforcement Learning is an area of Machine Learning aimed at
    creating agents capable of taking actions in an environment in a way
    that maximizes rewards over time. There are many differences between
    RL and regular supervised and unsupervised learning. Here are a few:

    -   In supervised and unsupervised learning, the goal is generally
        to find patterns in the data and use them to make predictions.
        In Reinforcement Learning, the goal is to find a good policy.

    -   Unlike in supervised learning, the agent is not explicitly given
        the "right" answer. It must learn by trial and error.

    -   Unlike in unsupervised learning, there is a form of supervision,
        through rewards. We do not tell the agent how to perform the
        task, but we do tell it when it is making progress or when it is
        failing.

    -   A Reinforcement Learning agent needs to find the right balance
        between exploring the environment, looking for new ways of
        getting rewards, and exploiting sources of rewards that it
        already knows. In contrast, supervised and unsupervised learning
        systems generally don't need to worry about exploration; they
        just feed on the training data they are given.

    -   In supervised and unsupervised learning, training instances are
        typically independent (in fact, they are generally shuffled). In
        Reinforcement Learning, consecutive observations are generally
        *not* independent. An agent may remain in the same region of the
        environment for a while before it moves on, so consecutive
        observations will be very correlated. In some cases a replay
        memory (buffer) is used to ensure that the training algorithm
        gets fairly independent observations.

2.  Here are a few possible applications of Reinforcement Learning,
    other than those mentioned in
    [Chapter 18](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch18.html#rl_chapter):

    Music personalization

    :   The environment is a user's personalized web radio. The agent is
        the software deciding what song to play next for that user. Its
        possible actions are to play any song in the catalog (it must
        try to choose a song the user will enjoy) or to play an
        advertisement (it must try to choose an ad that the user will be
        interested in). It gets a small reward every time the user
        listens to a song, a larger reward every time the user listens
        to an ad, a negative reward when the user skips a song or an ad,
        and a very negative reward if the user leaves.

    Marketing

    :   The environment is your company's marketing department. The
        agent is the software that defines which customers a mailing
        campaign should be sent to, given their profile and purchase
        history (for each customer it has two possible actions: send or
        don't send). It gets a negative reward for the cost of the
        mailing campaign, and a positive reward for estimated revenue
        generated from this campaign.

    Product delivery

    :   Let the agent control a fleet of delivery trucks, deciding what
        they should pick up at the depots, where they should go, what
        they should drop off, and so on. It will get positive rewards
        for each product delivered on time, and negative rewards for
        late deliveries.

3.  When estimating the value of an action, Reinforcement Learning
    algorithms typically sum all the rewards that this action led to,
    giving more weight to immediate rewards and less weight to later
    rewards (considering that an action has more influence on the near
    future than on the distant future). To model this, a discount factor
    is typically applied at each time step. For example, with a discount
    factor of 0.9, a reward of 100 that is received two time steps later
    is counted as only 0.9^2^ × 100 = 81 when you are estimating the
    value of the action. You can think of the discount factor as a
    measure of how much the future is valued relative to the present: if
    it is very close to 1, then the future is valued almost as much as
    the present; if it is close to 0, then only immediate rewards
    matter. Of course, this impacts the optimal policy tremendously: if
    you value the future, you may be willing to put up with a lot of
    immediate pain for the prospect of eventual rewards, while if you
    don't value the future, you will just grab any immediate reward you
    can find, never investing in the future.

4.  To measure the performance of a Reinforcement Learning agent, you
    can simply sum up the rewards it gets. In a simulated environment,
    you can run many episodes and look at the total rewards it gets on
    average (and possibly look at the min, max, standard deviation, and
    so on).

5.  The credit assignment problem is the fact that when a Reinforcement
    Learning agent receives a reward, it has no direct way of knowing
    which of its previous actions contributed to this reward. It
    typically occurs when there is a large delay between an action and
    the resulting reward (e.g., during a game of Atari's *Pong*, there
    may be a few dozen time steps between the moment the agent hits the
    ball and the moment it wins the point). One way to alleviate it is
    to provide the agent with shorter-term rewards, when possible. This
    usually requires prior knowledge about the task. For example, if we
    want to build an agent that will learn to play chess, instead of
    giving it a reward only when it wins the game, we could give it a
    reward every time it captures one of the opponent's pieces.

6.  An agent can often remain in the same region of its environment for
    a while, so all of its experiences will be very similar for that
    period of time. This can introduce some bias in the learning
    algorithm. It may tune its policy for this region of the
    environment, but it will not perform well as soon as it moves out of
    this region. To solve this problem, you can use a replay memory;
    instead of using only the most immediate experiences for learning,
    the agent will learn based on a buffer of its past experiences,
    recent and not so recent (perhaps this is why we dream at night: to
    replay our experiences of the day and better learn from them?).

7.  An off-policy RL algorithm learns the value of the optimal policy
    (i.e., the sum of discounted rewards that can be expected for each
    state if the agent acts optimally) while the agent follows a
    different policy. Q-Learning is a good example of such an algorithm.
    In contrast, an on-policy algorithm learns the value of the policy
    that the agent actually executes, including both exploration and
    exploitation.

For the solutions to exercises 8, 9, and 10, please see the Jupyter
notebooks available at
[*https://github.com/ageron/handson-ml2*](https://github.com/ageron/handson-ml2).
:::
:::

::: {.section data-type="sect1" pdf-bookmark="Chapter 19: Training and Deploying TensorFlow Models at Scale"}
::: {#idm45728431757480 .sect1}
[Chapter 19](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch19.html#deployment_chapter): Training and Deploying TensorFlow Models [at Scale]{.keep-together}
=================================================================================================================================================================================================

1.  A SavedModel contains a TensorFlow model, including its architecture
    (a computation graph) and its weights. It is stored as a directory
    containing a *saved\_model.pb* file, which defines the computation
    graph (represented as a serialized protocol buffer), and a
    *variables* subdirectory containing the variable values. For models
    containing a large number of weights, these variable values may be
    split across multiple files. A SavedModel also includes an *assets*
    subdirectory that may contain additional data, such as vocabulary
    files, class names, or some example instances for this model. To be
    more accurate, a SavedModel can contain one or more *metagraphs*. A
    metagraph is a computation graph plus some function signature
    definitions (including their input and output names, types, and
    shapes). Each metagraph is identified by a set of tags. To inspect a
    SavedModel, you can use the command-line tool `saved_model_cli` or
    just load it using `tf.saved_model.load()` and inspect it in Python.

2.  TF Serving allows you to deploy multiple TensorFlow models (or
    multiple versions of the same model) and make them accessible to all
    your applications easily via a REST API or a gRPC API. Using your
    models directly in your applications would make it harder to deploy
    a new version of a model across all applications. Implementing your
    own microservice to wrap a TF model would require extra work, and it
    would be hard to match TF Serving's features. TF Serving has many
    features: it can monitor a directory and autodeploy the models that
    are placed there, and you won't have to change or even restart any
    of your applications to benefit from the new model versions; it's
    fast, well tested, and scales very well; and it supports A/B testing
    of experimental models and deploying a new model version to just a
    subset of your users (in this case the model is called a *canary*).
    TF Serving is also capable of grouping individual requests into
    batches to run them jointly on the GPU. To deploy TF Serving, you
    can install it from source, but it is much simpler to install it
    using a Docker image. To deploy a cluster of TF Serving Docker
    images, you can use an orchestration tool such as Kubernetes, or use
    a fully hosted solution such as Google Cloud AI Platform.

3.  To deploy a model across multiple TF Serving instances, all you need
    to do is configure these TF Serving instances to monitor the same
    *models* directory, and then export your new model as a SavedModel
    into a subdirectory.

4.  The gRPC API is more efficient than the REST API. However, its
    client libraries are not as widely available, and if you activate
    compression when using the REST API, you can get almost the same
    performance. So, the gRPC API is most useful when you need the
    highest possible performance and the clients are not limited to the
    REST API.

5.  To reduce a model's size so it can run on a mobile or embedded
    device, TFLite uses several techniques:

    -   It provides a converter which can optimize a SavedModel: it
        shrinks the model and reduces its latency. To do this, it prunes
        all the operations that are not needed to make predictions (such
        as training operations), and it optimizes and fuses operations
        whenever possible.

    -   The converter can also perform post-training quantization: this
        technique dramatically reduces the model's size, so it's much
        faster to download and store.

    -   It saves the optimized model using the FlatBuffer format, which
        can be loaded to RAM directly, without parsing. This reduces the
        loading time and memory footprint.

6.  Quantization-aware training consists in adding fake quantization
    operations to the model during training. This allows the model to
    learn to ignore the quantization noise; the final weights will be
    more robust to quantization.

7.  Model parallelism means chopping your model into multiple parts and
    running them in parallel across multiple devices, hopefully speeding
    up the model during training or inference. Data parallelism means
    creating multiple exact replicas of your model and deploying them
    across multiple devices. At each iteration during training, each
    replica is given a different batch of data, and it computes the
    gradients of the loss with regard to the model parameters. In
    synchronous data parallelism, the gradients from all replicas are
    then aggregated and the optimizer performs a Gradient Descent step.
    The parameters may be centralized (e.g., on parameter servers) or
    replicated across all replicas and kept in sync using AllReduce. In
    asynchronous data parallelism, the parameters are centralized and
    the replicas run independently from each other, each updating the
    central parameters directly at the end of each training iteration,
    without having to wait for the other replicas. To speed up training,
    data parallelism turns out to work better than model parallelism, in
    general. This is mostly because it requires less communication
    across devices. Moreover, it is much easier to implement, and it
    works the same way for any model, whereas model parallelism requires
    analyzing the model to determine the best way to chop it into
    pieces.

8.  When training a model across multiple servers, you can use the
    following distribution strategies:

    -   The `MultiWorkerMirroredStrategy` performs mirrored data
        parallelism. The model is replicated across all available
        servers and devices, and each replica gets a different batch of
        data at each training iteration and computes its own gradients.
        The mean of the gradients is computed and shared across all
        replicas using a distributed AllReduce implementation (NCCL by
        default), and all replicas perform the same Gradient Descent
        step. This strategy is the simplest to use since all servers and
        devices are treated in exactly the same way, and it performs
        fairly well. In general, you should use this strategy. Its main
        limitation is that it requires the model to fit in RAM on every
        replica.

    -   The `ParameterServerStrategy` performs asynchronous data
        parallelism. The model is replicated across all devices on all
        workers, and the parameters are sharded across all parameter
        servers. Each worker has its own training loop, running
        asynchronously with the other workers; at each training
        iteration, each worker gets its own batch of data and fetches
        the latest version of the model parameters from the parameter
        servers, then it computes the gradients of the loss with regard
        to these parameters, and it sends them to the parameter servers.
        Lastly, the parameter servers perform a Gradient Descent step
        using these gradients. This strategy is generally slower than
        the previous strategy, and a bit harder to deploy, since it
        requires managing parameter servers. However, it is useful to
        train huge models that don't fit in GPU
        RAM.[]{#idm45728431731960}

For the solutions to exercises 9, 10, and 11, please see the Jupyter
notebooks available at
[*https://github.com/ageron/handson-ml2*](https://github.com/ageron/handson-ml2).
:::
:::

::: {data-type="footnotes"}
^[1](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/app01.html#idm45728432174040-marker){.totri-footnote}^
If you draw a straight line between any two points on the curve, the
line never crosses the curve.

^[2](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/app01.html#idm45728432088232-marker){.totri-footnote}^
log~2~ is the binary log; log~2~(*m*) = log(*m*) / log(2).

^[3](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/app01.html#idm45728431976808-marker){.totri-footnote}^
When the values to predict can vary by many orders of magnitude, you may
want to predict the logarithm of the target value rather than the target
value directly. Simply computing the exponential of the neural network's
output will give you the estimated value (since exp(log *v*) = *v*).

^[4](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/app01.html#idm45728431971048-marker){.totri-footnote}^
In
[Chapter 11](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch11.html#deep_chapter)
we discuss many techniques that introduce additional hyperparameters:
type of weight initialization, activation function hyperparameters
(e.g., the amount of leak in leaky ReLU), Gradient Clipping threshold,
type of optimizer and its hyperparameters (e.g., the momentum
hyperparameter when using a `MomentumOptimizer`), type of regularization
for each layer and regularization hyperparameters (e.g., dropout rate
when using dropout), and so on.
