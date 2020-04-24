
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
