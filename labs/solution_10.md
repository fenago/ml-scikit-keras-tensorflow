[Lab 10](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch10.html#ann_lab): Introduction to Artificial Neural Networks with Keras
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

    ![](./A_files/mls2_aain01.png)

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
    [Figure 10-8]
    See
    [Lab 11]
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
    [Lab 2]
    then you need one output neuron, using no activation function at all
    in the output
    layer.
^[3](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/app01.html){-marker
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
    [Appendix D]
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
    layer.
^[4](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/app01.html){-marker
    .totri-footnote}^ In general, the ReLU activation function (or one
    of its variants; see
    [Lab 11]
    is a good default for the hidden layers. For the output layer, in
    general you will want the logistic activation function for binary
    classification, the softmax activation function for multiclass
    classification, or no activation function for regression.

    If the MLP overfits the training data, you can try reducing the
    number of hidden layers and reducing the number of neurons per
    hidden layer.
    
    10. See the Jupyter notebooks available at
    [*https://github.com/fenago/ml-scikit-keras-tensorflow*](https://github.com/fenago/ml-scikit-keras-tensorflow).