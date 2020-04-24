

[Lab 17](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch17.html#autoencoders_lab): Representation Learning and Generative Learning Using Autoencoders and GANs
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
    [discriminator's] error. GANs are used for advanced
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


