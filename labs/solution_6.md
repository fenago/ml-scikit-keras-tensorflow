
[Lab 6](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch06.html#trees_lab): Decision Trees
======================================================================================================================================

1.  The depth of a well-balanced binary tree containing *m* leaves is
    equal to
    log~2~(*m*),^[2](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/app01.html){-marker
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
[*https://github.com/fenago/ml-scikit-keras-tensorflow*](https://github.com/fenago/ml-scikit-keras-tensorflow).
