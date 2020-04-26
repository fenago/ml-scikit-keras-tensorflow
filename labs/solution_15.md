
[Lab 15](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch15.html#rnn_lab): Processing Sequences Using RNNs and CNNs
===============================================================================================================================================================

1.  Here are a few RNN applications:

    -   For a sequence-to-sequence RNN: predicting the weather (or any
        other time series), machine translation (using an
        Encoder--Decoder architecture), video captioning, speech to
        text, music generation (or other sequence generation),
        identifying the chords of a song

    -   For a sequence-to-vector RNN: classifying music samples by music
        genre, analyzing the sentiment of a course review, predicting what
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
    [neurons]. For example, if an RNN layer with 32
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
    [Figure 15-9]

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
[*https://github.com/fenago/ml-scikit-keras-tensorflow*](https://github.com/fenago/ml-scikit-keras-tensorflow).
