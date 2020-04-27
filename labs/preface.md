
Preface
=======


The Machine Learning Tsunami
============================

In 2006, Geoffrey Hinton et al. published [a
paper](https://homl.info/136) ^[1] showing how to train a deep neural network capable of
recognizing handwritten digits with state-of-the-art precision (\>98%).
They branded this technique "Deep Learning." A deep neural
network is a (very) simplified model of our
cerebral cortex, composed of a stack of layers of artificial neurons.
Training a deep neural net was widely considered impossible at the
time, ^[2] and most researchers had abandoned the idea in the
late 1990s. This paper revived the interest of the scientific community,
and before long many new papers demonstrated that Deep Learning was not
only possible, but capable of mind-blowing achievements that no other
Machine Learning (ML) technique could hope to
match (with the help of tremendous computing power and great amounts of
data). This enthusiasm soon extended to many other areas of Machine
Learning.

A decade or so later, Machine Learning has conquered the industry: it is
at the heart of much of the magic in today's high-tech products, ranking
your web search results, powering your smartphone's speech recognition,
recommending videos, and beating the world champion at the game of Go.
Before you know it, it will be driving your car.

Machine Learning in Your Projects
=================================

So, naturally you are excited about Machine
Learning and would love to join the party!
Perhaps you would like to give your homemade robot a brain of its own?
Make it recognize faces? Or learn to walk around?

Or maybe your company has tons of data (user logs, financial data,
production data, machine sensor data, hotline stats, HR reports, etc.),
and more than likely you could unearth some hidden gems if you just knew
where to look. With Machine Learning, you could accomplish the following
[and more](https://homl.info/usecases):

-   Segment customers and find the best marketing strategy for each
    group.
-   Recommend products for each client based on what similar clients
    bought.
-   Detect which transactions are likely to be fraudulent.
-   Forecast next year's revenue.

Whatever the reason, you have decided to learn Machine Learning and
implement it in your projects. Great idea!

Objective and Approach
======================

This course assumes that you know close to nothing
about Machine Learning. Its goal is to give you the concepts, tools, and
intuition you need to implement programs capable of *learning from
data*.

We will cover a large number of techniques, from the simplest and most
commonly used (such as Linear Regression) to some of the Deep Learning
techniques that regularly win competitions.

Rather than implementing our own toy versions of each algorithm, we will
be using production-ready Python frameworks:

-   [Scikit-Learn](http://scikit-learn.org/) is 
    very easy to use, yet it implements many Machine Learning algorithms
    efficiently, so it makes for a great entry point to learning Machine
    Learning. It was created by David Cournapeau in 2007, and is now led
    by a team of researchers at the French Institute for Research in
    Computer Science and Automation (Inria).

-   [TensorFlow](https://tensorflow.org/) is a
    more complex library for distributed numerical computation. It makes
    it possible to train and run very large neural networks efficiently
    by distributing the computations across potentially hundreds of
    multi-GPU (graphics processing unit) servers. TensorFlow (TF) was
    created at Google and supports many of its large-scale Machine
    Learning applications. It was open sourced in November 2015, and
    version 2.0 was released in September 2019.

-   [Keras](https://keras.io/) is a high-level
    Deep Learning API that makes it very simple to train and run neural
    networks. It can run on top of either TensorFlow, Theano, or
    Microsoft Cognitive Toolkit (formerly known as CNTK). TensorFlow
    comes with its own implementation of this API, called *tf.keras*,
    which provides support for some advanced TensorFlow features (e.g.,
    the ability to efficiently load data).

The course favors a hands-on approach, growing an intuitive understanding
of Machine Learning through concrete working examples and just a little
bit of theory. While you can read this course without picking up your
laptop, I highly recommend you experiment with the code examples
available online as Jupyter notebooks at
[*https://github.com/fenago/ml-scikit-keras-tensorflow*](https://github.com/fenago/ml-scikit-keras-tensorflow).

Prerequisites
=============

This course assumes that you have some Python
programming experience and that you are familiar with Python's main
scientific libraries in particular, [NumPy](http://numpy.org/),
[pandas](http://pandas.pydata.org/), and
[Matplotlib](http://matplotlib.org/).

Also, if you care about what's under the hood, you should have a
reasonable understanding of college-level math as well (calculus, linear
algebra, probabilities, and statistics).

If you don't know Python yet,
[*http://learnpython.org/*](http://learnpython.org/) is a great place to
start. The official tutorial on
[Python.org](https://docs.python.org/3/tutorial/) is also quite good.

If you have never used Jupyter,
[Lab 2]
will guide you through installation and the basics: it is a powerful
tool to have in your toolbox.

If you are not familiar with Python's scientific libraries, the provided
Jupyter notebooks include a few tutorials. There is also a quick math
tutorial for linear algebra.

Roadmap
=======

This course is organized in two parts. 

**Part I, The Fundamentals of Machine Learning** covers the following topics:

-   What Machine Learning is, what problems it tries to solve, and the
    main categories and fundamental concepts of its systems

-   The steps in a typical Machine Learning project

-   Learning by fitting a model to data

-   Optimizing a cost function

-   Handling, cleaning, and preparing data

-   Selecting and engineering features

-   Selecting a model and tuning hyperparameters using cross-validation

-   The challenges of Machine Learning, in particular underfitting and
    overfitting (the bias/variance trade-off)

-   The most common learning algorithms: Linear and Polynomial
    Regression, Logistic Regression, k-Nearest Neighbors, Support Vector
    Machines, Decision Trees, Random Forests, and Ensemble methods

-   Reducing the dimensionality of the training data to fight the "curse
    of dimensionality"

-   Other unsupervised learning techniques, including clustering,
    density estimation, and anomaly detection

**Part II, Neural Networks and Deep Learning** covers the following topics:

-   What neural nets are and what they're good for

-   Building and training neural nets using TensorFlow and Keras

-   The most important neural net architectures: feedforward neural nets
    for tabular data, convolutional nets for computer vision, recurrent
    nets and long short-term memory (LSTM) nets for sequence processing,
    encoder/decoders and Transformers for natural language processing,
    autoencoders and generative adversarial networks (GANs) for
    generative learning

-   Techniques for training deep neural nets

-   How to build an agent (e.g., a bot in a game) that can learn good
    strategies through trial and error, using Reinforcement Learning

-   Loading and preprocessing large amounts of data efficiently

-   Training and deploying TensorFlow models at scale

The first part is based mostly on Scikit-Learn, while the second part
uses TensorFlow and Keras.


###### Caution

Don't jump into deep waters too hastily: while Deep Learning is no doubt
one of the most exciting areas in Machine Learning, you should master
the fundamentals first. Moreover, most problems can be solved quite well
using simpler techniques such as Random Forests and Ensemble methods

Deep Learning is best suited for complex problems such as image
recognition, speech recognition, or natural language processing,
provided you have enough data, computing power, and patience.

Conventions Used in This Course
=============================

The following typographical conventions are used in this course:

*Italic*

Indicates new terms, URLs, email addresses, filenames, and file
    extensions.

`Constant width`

Used for program listings, as well as within paragraphs to refer to
    program elements such as variable or function names, databases, data
    types, environment variables, statements and keywords.

**`Constant width bold`**

Shows commands or other text that should be typed literally by the
    user.

*`Constant width italic`*

Shows text that should be replaced with user-supplied values or by
    values determined by context.


###### Tip
This element signifies a tip or suggestion.

###### Note
This element signifies a general note.

###### Warning
This element indicates a warning or caution.


Code Examples
=============

There is a series of Jupyter notebooks full of
supplemental material, such as code examples and exercises, available
for download at
[*https://github.com/fenago/ml-scikit-keras-tensorflow*](https://github.com/fenago/ml-scikit-keras-tensorflow).

Some of the code examples in the course leave out repetitive sections or
details that are obvious or unrelated to Machine Learning. This keeps
the focus on the important parts of the code and saves space to cover
more topics. If you want the full code examples, they are all available
in the Jupyter notebooks.

Note that when the code examples display some outputs, these code
examples are shown with Python prompts (`>>>` and `...`), as in a Python
shell, to clearly distinguish the code from the outputs. For example,
this code defines the `square()` function, then it computes and displays
the square of 3:

``` {data-type="programlisting" code-language="pycon"}
>>> def square(x):
...     return x ** 2
...
>>> result = square(3)
>>> result
9
```

When code does not display anything, prompts are not used. However, the
result may sometimes be shown as a comment, like this:

``` {data-type="programlisting" code-language="python"}
def square(x):
    return x ** 2

result = square(3)  # result is 9
```

#### References

^[1] Geoffrey E. Hinton et al., "A Fast Learning Algorithm for Deep Belief
Nets," *Neural Computation* 18 (2006): 1527--1554.

^[2] Despite the fact that Yann LeCun's deep convolutional neural networks
had worked well for image recognition since the 1990s, although they
were not as general-purpose.
