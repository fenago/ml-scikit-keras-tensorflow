<img align="right" src="../logo-small.png">


**Solution**


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
    [Figure 5-2]

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
    Programming"]
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
[*https://github.com/fenago/ml-scikit-keras-tensorflow*](https://github.com/fenago/ml-scikit-keras-tensorflow).