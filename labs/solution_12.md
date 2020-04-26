<img align="right" src="../logo-small.png">


**Solution**


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
    Rules"]
    If you absolutely need to include arbitrary Python code in a custom
    component, you can either wrap it in a `tf.py_function()` operation
    (but this will reduce performance and limit your model's
    portability) or set `dynamic=True` when creating the custom layer or
    model (or set `run_eagerly=True` when calling the model's
    `compile()` method).

10. Please refer to ["TF Function
    Rules"]
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
[*https://github.com/fenago/ml-scikit-keras-tensorflow*](https://github.com/fenago/ml-scikit-keras-tensorflow).



