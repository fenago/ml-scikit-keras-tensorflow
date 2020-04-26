[Lab 11](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch11.html#deep_lab): Training Deep Neural Networks
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
    [Lab 17]
    Moreover, it can sometimes benefit from optimized implementation as
    well as from hardware acceleration. The hyperbolic tangent (tanh)
    can be useful in the output layer if you need to output a number
    between --1 and 1, but nowadays it is not used much in hidden layers
    (except in recurrent nets). The logistic activation function is also
    useful in the output layer when you need to estimate a probability
    (e.g., for binary classification), but is rarely used in hidden
    layers (there are exceptions---for example, for the coding layer of
    variational autoencoders; see
    [Lab 17]
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
[*https://github.com/fenago/ml-scikit-keras-tensorflow*](https://github.com/fenago/ml-scikit-keras-tensorflow).
