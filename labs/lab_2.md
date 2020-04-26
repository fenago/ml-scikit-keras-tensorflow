
<img align="right" src="../logo-small.png">


[Lab 2. ] End-to-End Machine Learning Project
========================================================

In this lab you will work through an example
project end to end, pretending to be a recently hired data scientist at
a real estate
company.
^[1](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch02.html) Here are the main steps you will go through:

1.  Look at the big picture.

2.  Get the data.

3.  Discover and visualize the data to gain insights.

4.  Prepare the data for Machine Learning algorithms.

5.  Select a model and train it.

6.  Fine-tune your model.

7.  Present your solution.

8.  Launch, monitor, and maintain your system.



Working with Real Data
======================

When 
you are learning about Machine Learning, it is best to experiment with
real-world data, not artificial datasets. Fortunately, there are
thousands of open datasets to choose from, ranging across all sorts of
domains. Here are a few places you can look to get data:

-   Popular open data repositories

    -   [UC Irvine Machine Learning
        Repository](http://archive.ics.uci.edu/ml/)

    -   [Kaggle datasets](https://www.kaggle.com/datasets)

    -   [Amazon's AWS datasets](https://registry.opendata.aws/)

-   Meta portals (they list open data repositories)

    -   [Data Portals](http://dataportals.org/)

    -   [OpenDataMonitor](http://opendatamonitor.eu/)

    -   [Quandl](http://quandl.com/)

-   Other pages listing many popular open data repositories

    -   [Wikipedia's list of Machine Learning
        datasets](https://homl.info/9)

    -   [Quora.com](https://homl.info/10)

    -   [The datasets subreddit](https://www.reddit.com/r/datasets)

In this lab we'll use the
 California Housing Prices
dataset from the StatLib
repository^[2](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch02.html) (see
[Figure 2-1]
This dataset is based on data from the 1990 California census. It is not
exactly recent (a nice house in the Bay Area was still affordable at the
time), but it has many qualities for learning, so we will pretend it is
recent data. For teaching purposes I've added a categorical attribute
and removed a few features.

![](./images/mls2_0201.png)




Look at the Big Picture
=======================

Welcome to the Machine Learning Housing
Corporation! Your first task is to use California census data to build a
model of housing prices in the state. This data includes metrics such as
the population, median income, and median housing price for each block
group in California. Block groups are the smallest geographical unit for
which the US Census Bureau publishes sample data (a block group
typically has a population of 600 to 3,000 people). We will call them
"districts" for short.

Your model should learn from this data and be able to predict the median
housing price in any district, given all the other metrics.


###### Tip

Since you are a well-organized data scientist, the first thing you
should do is pull out your 
Machine Learning project checklist. You can start with the one in
[Appendix B]
it should work reasonably well for most Machine Learning projects, but
make sure to adapt it to your needs. In this lab we will go through
many checklist items, but we will also skip a few, either because they
are self-explanatory or because they will be discussed in later
labs.




Frame the Problem
-----------------

The first question to ask your boss is what
exactly the business objective is. Building a model is probably not the
end goal. How does the company expect to use and benefit from this
model? Knowing the objective is important because it will determine how
you frame the problem, which algorithms you will select, which
performance measure you will use to evaluate your model, and how much
effort you will spend tweaking it.

Your boss answers that your model's output (a prediction of a district's
median housing price) will be fed to another Machine Learning system
(see
[Figure 2-2]
along with many other
signals.
^[3](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch02.html) This downstream system will determine whether it is
worth investing in a given area or not. Getting this right is critical,
as it directly affects [revenue].

![](./images/mls2_0202.png)


##### Pipelines

A sequence of data
processing components is called a data *pipeline*. Pipelines are very
common in Machine Learning systems, since there is a lot of data to
manipulate and many data transformations to apply.

Components typically run asynchronously. Each component pulls in a large
amount of data, processes it, and spits out the result in another data
store. Then, some time later, the next component in the pipeline pulls
this data and spits out its own output. Each component is fairly
self-contained: the interface between components is simply the data
store. This makes the system simple to grasp (with the help of a data
flow graph), and different teams can focus on different components.
Moreover, if a component breaks down, the downstream components can
often continue to run normally (at least for a while) by just using the
last output from the broken component. This makes the architecture quite
robust.

On the other hand, a broken component can go unnoticed for some time if
proper monitoring is not implemented. The data gets stale and the
overall system's performance drops.


The next question to ask your boss is what the current solution looks
like (if any). The current situation will often give you a reference for
performance, as well as insights on how to solve the problem. Your boss
answers that the district housing prices are currently estimated
manually by experts: a team gathers up-to-date information about a
district, and when they cannot get the median housing price, they
estimate it using complex rules.

This is costly and time-consuming, and their estimates are not great; in
cases where they manage to find out the actual median housing price,
they often realize that their estimates were off by more than 20%. This
is why the company thinks that it would be useful to train a model to
predict a district's median housing price, given other data about that
district. The census data looks like a great dataset to exploit for this
purpose, since it includes the median housing prices of thousands of
districts, as well as other data.

With all this information, you are now ready to start designing your
system. First, you need to frame the problem: is it supervised,
unsupervised, or Reinforcement Learning? Is it a classification task, a
regression task, or something else? Should you use batch learning or
online learning techniques? Before you read on, pause and try to answer
these questions for yourself.

Have you found the answers? Let's see: it is clearly a typical
supervised learning task, since you are given
*labeled* training examples (each instance comes with the expected
output, i.e., the district's median housing price). It is also a typical
regression task, since you are asked to predict a value. More
specifically, this is a
*multiple regression* problem, since the system will use multiple
features to make a prediction (it will use the district's population,
the median income, etc.). It 
is also a *univariate regression* problem, since we are only trying to
predict a single value for each district. If we were trying to predict
multiple values per district,
it would be a *multivariate
regression* problem. Finally, there is no continuous flow of data coming
into the system, there is no particular need to adjust to changing data
rapidly, and the data is small enough to fit in memory, so plain batch
learning should do just fine.


###### Tip

If the data were huge, you could either split your batch learning work
across multiple servers (using the MapReduce technique) or use an online
learning technique.





Select a Performance Measure
----------------------------

Your 
next step is to select a performance measure. A typical performance
measure for regression problems is the Root Mean Square Error (RMSE). It
gives an idea of how much error the system typically makes in its
predictions, with a higher weight for large errors. [Equation
2-1]
shows the mathematical formula to compute the RMSE.


##### [Equation 2-1. ] Root Mean Square Error (RMSE)

[]{.MathJax_Preview style="color: inherit; display: none;"}


[[[[[[[RMSE]{#MathJax-Span-18 .mtext
style="font-family: MathJax_Main;"}[[(]{#MathJax-Span-20 .mo
style="font-family: MathJax_Main;"}[X]{#MathJax-Span-21 .mi
style="font-family: MathJax_Main-bold;"}[,]{#MathJax-Span-22 .mo
style="font-family: MathJax_Main;"}[h]{#MathJax-Span-23 .mi
style="font-family: MathJax_Math-italic; padding-left: 0.157em;"}[)]{#MathJax-Span-24
.mo style="font-family: MathJax_Main;"}]{#MathJax-Span-19 .mrow
style="padding-left: 0.157em;"}[=]{#MathJax-Span-25 .mo
style="font-family: MathJax_Main; padding-left: 0.26em;"}[[[[[[[[[1]{#MathJax-Span-30
.mn
style="font-family: MathJax_Main;"}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(3.19em, 1000.42em, 4.167em, -1000.01em); top: -4.676em; left: 50%; margin-left: -0.255em;"}[[m]{#MathJax-Span-31
.mi
style="font-family: MathJax_Math-italic;"}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(3.396em, 1000.88em, 4.167em, -1000.01em); top: -3.339em; left: 50%; margin-left: -0.46em;"}[[]{style="display: inline-block; overflow: hidden; vertical-align: 0em; border-top: 1.3px solid; width: 0.979em; height: 0px;"}[]{style="display: inline-block; width: 0px; height: 1.082em;"}]{style="position: absolute; clip: rect(0.877em, 1000.99em, 1.237em, -1000.01em); top: -1.283em; left: 0em;"}]{style="display: inline-block; position: relative; width: 0.979em; height: 0px; margin-right: 0.105em; margin-left: 0.105em;"}]{#MathJax-Span-29
.mfrac}[[[[∑]{#MathJax-Span-33 .mo
style="font-family: MathJax_Size2; vertical-align: 0em;"}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(2.933em, 1001.4em, 4.63em, -1000.01em); top: -4.008em; left: 0em;"}[[[i]{#MathJax-Span-35
.mi
style="font-size: 70.7%; font-family: MathJax_Math-italic;"}[=]{#MathJax-Span-36
.mo
style="font-size: 70.7%; font-family: MathJax_Main;"}[1]{#MathJax-Span-37
.mn
style="font-size: 70.7%; font-family: MathJax_Main;"}]{#MathJax-Span-34
.mrow}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(3.396em, 1001.09em, 4.27em, -1000.01em); top: -2.928em; left: 0.157em;"}[[m]{#MathJax-Span-38
.mi
style="font-size: 70.7%; font-family: MathJax_Math-italic;"}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(3.447em, 1000.63em, 4.167em, -1000.01em); top: -5.139em; left: 0.414em;"}]{style="display: inline-block; position: relative; width: 1.442em; height: 0px;"}]{#MathJax-Span-32
.munderover
style="padding-left: 0.157em;"}[[[[[[(]{style="font-family: MathJax_Size2;"}]{#MathJax-Span-61
.mo style="vertical-align: 0em;"}[h]{#MathJax-Span-42 .mi
style="font-family: MathJax_Math-italic;"}[[[(]{style="font-family: MathJax_Size2;"}]{#MathJax-Span-44
.mo style="vertical-align: 0em;"}[[[[x]{#MathJax-Span-46 .mi
style="font-family: MathJax_Main-bold;"}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(3.396em, 1000.58em, 4.167em, -1000.01em); top: -4.008em; left: 0em;"}[[[(]{#MathJax-Span-48
.mo
style="font-size: 70.7%; font-family: MathJax_Main;"}[i]{#MathJax-Span-49
.mi
style="font-size: 70.7%; font-family: MathJax_Math-italic;"}[)]{#MathJax-Span-50
.mo
style="font-size: 70.7%; font-family: MathJax_Main;"}]{#MathJax-Span-47
.mrow}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; top: -4.419em; left: 0.62em;"}]{style="display: inline-block; position: relative; width: 1.494em; height: 0px;"}]{#MathJax-Span-45
.msup}[[)]{style="font-family: MathJax_Size2;"}]{#MathJax-Span-51 .mo
style="vertical-align: 0em;"}]{#MathJax-Span-43 .mrow
style="padding-left: 0.157em;"}[−]{#MathJax-Span-52 .mo
style="font-family: MathJax_Main; padding-left: 0.208em;"}[[[[y[]{style="display: inline-block; overflow: hidden; height: 1px; width: 0.003em;"}]{#MathJax-Span-54
.mi
style="font-family: MathJax_Math-italic;"}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(3.396em, 1000.47em, 4.373em, -1000.01em); top: -4.008em; left: 0em;"}[[[(]{#MathJax-Span-56
.mo
style="font-size: 70.7%; font-family: MathJax_Main;"}[i]{#MathJax-Span-57
.mi
style="font-size: 70.7%; font-family: MathJax_Math-italic;"}[)]{#MathJax-Span-58
.mo
style="font-size: 70.7%; font-family: MathJax_Main;"}]{#MathJax-Span-55
.mrow}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; top: -4.419em; left: 0.568em;"}]{style="display: inline-block; position: relative; width: 1.442em; height: 0px;"}]{#MathJax-Span-53
.msup
style="padding-left: 0.208em;"}[[)]{style="font-family: MathJax_Size2;"}]{#MathJax-Span-62
.mo style="vertical-align: 0em;"}]{#MathJax-Span-40
.mfenced}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(2.728em, 1007.05em, 4.836em, -1000.01em); top: -4.008em; left: 0em;"}[[2]{#MathJax-Span-60
.mn
style="font-size: 70.7%; font-family: MathJax_Main;"}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; top: -4.882em; left: 7.201em;"}]{style="display: inline-block; position: relative; width: 7.663em; height: 0px;"}]{#MathJax-Span-39
.msup style="padding-left: 0.157em;"}]{#MathJax-Span-28
.mrow}]{#MathJax-Span-27
.mrow}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(2.316em, 1010.65em, 5.35em, -1000.01em); top: -4.008em; left: 0.979em;"}[[[−[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; font-family: MathJax_Main; top: -4.008em; left: -0.1em;"}[−[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; font-family: MathJax_Main; top: -4.008em; left: 9.977em;"}[−[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Main; position: absolute; top: -4.008em; left: 0.465em;"}[−[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Main; position: absolute; top: -4.008em; left: 1.031em;"}[−[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Main; position: absolute; top: -4.008em; left: 1.596em;"}[−[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Main; position: absolute; top: -4.008em; left: 2.111em;"}[−[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Main; position: absolute; top: -4.008em; left: 2.676em;"}[−[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Main; position: absolute; top: -4.008em; left: 3.242em;"}[−[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Main; position: absolute; top: -4.008em; left: 3.807em;"}[−[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Main; position: absolute; top: -4.008em; left: 4.373em;"}[−[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Main; position: absolute; top: -4.008em; left: 4.938em;"}[−[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Main; position: absolute; top: -4.008em; left: 5.504em;"}[−[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Main; position: absolute; top: -4.008em; left: 6.069em;"}[−[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Main; position: absolute; top: -4.008em; left: 6.584em;"}[−[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Main; position: absolute; top: -4.008em; left: 7.149em;"}[−[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Main; position: absolute; top: -4.008em; left: 7.715em;"}[−[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Main; position: absolute; top: -4.008em; left: 8.28em;"}[−[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Main; position: absolute; top: -4.008em; left: 8.846em;"}[−[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Main; position: absolute; top: -4.008em; left: 9.411em;"}]{style="display: inline-block; position: relative; width: 10.645em; height: 0px;"}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(3.602em, 1010.65em, 3.961em, -1000.01em); top: -5.345em; left: 0.979em;"}[[√]{style="font-family: MathJax_Size4;"}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(2.111em, 1001.04em, 5.401em, -1000.01em); top: -3.905em; left: 0em;"}]{style="display: inline-block; position: relative; width: 11.674em; height: 0px;"}]{#MathJax-Span-26
.msqrt style="padding-left: 0.26em;"}]{#MathJax-Span-17
.mrow}]{#MathJax-Span-16
.mrow}[]{style="display: inline-block; width: 0px; height: 2.162em;"}]{style="position: absolute; clip: rect(0.362em, 1018.67em, 3.653em, -1000.01em); top: -2.157em; left: 0em;"}]{style="display: inline-block; position: relative; width: 18.666em; height: 0px; font-size: 103%;"}[]{style="display: inline-block; overflow: hidden; vertical-align: -1.427em; border-left: 0px solid; width: 0px; height: 3.18em;"}]{#MathJax-Span-15
.math
style="width: 19.232em; display: inline-block;"}[$$\text{RMSE}\left( \mathbf{X},h \right) = \sqrt{\frac{1}{m}\sum\limits_{i = 1}^{m}\left( {h\left( \mathbf{x}^{(i)} \right) - y^{(i)}} \right)^{2}}$$]{.MJX_Assistive_MathML
.MJX_Assistive_MathML_Block
role="presentation"}]{#MathJax-Element-2-Frame .MathJax tabindex="0"
mathml="<math xmlns=\"http://www.w3.org/1998/Math/MathML\" display=\"block\"><mrow><mtext>RMSE</mtext><mrow><mo>(</mo><mi mathvariant=\"bold\">X</mi><mo>,</mo><mi>h</mi><mo>)</mo></mrow><mo>=</mo><msqrt><mrow><mfrac><mn>1</mn><mi>m</mi></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msup><mfenced separators=\"\" open=\"(\" close=\")\"><mi>h</mi><mrow><mo>(</mo><msup><mi mathvariant=\"bold\">x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><mo>-</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup></mfenced><mn>2</mn></msup></mrow></msqrt></mrow></math>"
role="presentation" style="text-align: center; position: relative;"}



##### Notations

This equation introduces several very common
Machine Learning notations that we will use throughout this course:

-   *m* is the number of instances in the dataset you are measuring the
    RMSE on.

    -   For example, if you are evaluating the RMSE on a validation set
        of 2,000 districts, then *m* = 2,000.

-   **x**^(*i*)^ is a vector of all the feature values (excluding the
    label) of the *i*^th^ instance in the dataset, and *y*^(*i*)^ is its
    label (the desired output value for that instance).

    -   For example, if the first district in the dataset is located at
        longitude --118.29°, latitude 33.91°, and it has 1,416
        inhabitants with a median income of \$38,372, and the median
        house value is \$156,400 (ignoring the other features for now),
        then:

        
        []{.MathJax_Preview style="color: inherit; display: none;"}
        
        [[[[[[[[[[x]{#MathJax-Span-67 .mi
        style="font-family: MathJax_Main-bold;"}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(3.396em, 1000.58em, 4.167em, -1000.01em); top: -4.008em; left: 0em;"}[[[(]{#MathJax-Span-69
        .mo
        style="font-size: 70.7%; font-family: MathJax_Main;"}[1]{#MathJax-Span-70
        .mn
        style="font-size: 70.7%; font-family: MathJax_Main;"}[)]{#MathJax-Span-71
        .mo
        style="font-size: 70.7%; font-family: MathJax_Main;"}]{#MathJax-Span-68
        .mrow}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; top: -4.419em; left: 0.62em;"}]{style="display: inline-block; position: relative; width: 1.596em; height: 0px;"}]{#MathJax-Span-66
        .msup}[=]{#MathJax-Span-72 .mo
        style="font-family: MathJax_Main; padding-left: 0.26em;"}[[[[⎛[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; font-family: MathJax_Size4; top: -2.877em; left: 0em;"}[⎝[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; font-family: MathJax_Size4; top: 0.62em; left: 0em;"}[⎜[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Size4; position: absolute; top: -1.643em; left: 0em;"}[⎜[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Size4; position: absolute; top: -1.077em; left: 0em;"}[⎜[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Size4; position: absolute; top: -0.512em; left: 0em;"}]{style="display: inline-block; position: relative; width: 0.877em; height: 0px;"}]{#MathJax-Span-75
        .mo
        style="vertical-align: 2.882em;"}[[[[[[[[[−]{#MathJax-Span-80
        .mo style="font-family: MathJax_Main;"}[118.29]{#MathJax-Span-81
        .mn style="font-family: MathJax_Main;"}]{#MathJax-Span-79
        .mrow}]{#MathJax-Span-78 .mrow}]{#MathJax-Span-77
        .mtd}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(3.19em, 1003.51em, 4.27em, -1000.01em); top: -6.116em; left: 50%; margin-left: -1.797em;"}[[[[[33.91]{#MathJax-Span-85
        .mn style="font-family: MathJax_Main;"}]{#MathJax-Span-84
        .mrow}]{#MathJax-Span-83 .mrow}]{#MathJax-Span-82
        .mtd}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(3.19em, 1002.22em, 4.167em, -1000.01em); top: -4.676em; left: 50%; margin-left: -1.129em;"}[[[[[1,416]{#MathJax-Span-89
        .mn style="font-family: MathJax_Main;"}]{#MathJax-Span-88
        .mrow}]{#MathJax-Span-87 .mrow}]{#MathJax-Span-86
        .mtd}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(3.19em, 1002.22em, 4.373em, -1000.01em); top: -3.237em; left: 50%; margin-left: -1.129em;"}[[[[[38,372]{#MathJax-Span-93
        .mn style="font-family: MathJax_Main;"}]{#MathJax-Span-92
        .mrow}]{#MathJax-Span-91 .mrow}]{#MathJax-Span-90
        .mtd}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(3.19em, 1002.74em, 4.373em, -1000.01em); top: -1.797em; left: 50%; margin-left: -1.386em;"}]{style="display: inline-block; position: relative; width: 3.55em; height: 0px;"}[]{style="display: inline-block; width: 0px; height: 6.172em;"}]{style="position: absolute; clip: rect(3.242em, 1003.51em, 8.692em, -1000.01em); top: -6.167em; left: 0em;"}]{style="display: inline-block; position: relative; width: 3.55em; height: 0px;"}]{#MathJax-Span-76
        .mtable
        style="padding-right: 0.157em; padding-left: 0.157em;"}[[[⎞[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; font-family: MathJax_Size4; top: -2.877em; left: 0em;"}[⎠[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; font-family: MathJax_Size4; top: 0.62em; left: 0em;"}[⎟[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Size4; position: absolute; top: -1.643em; left: 0em;"}[⎟[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Size4; position: absolute; top: -1.077em; left: 0em;"}[⎟[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Size4; position: absolute; top: -0.512em; left: 0em;"}]{style="display: inline-block; position: relative; width: 0.877em; height: 0px;"}]{#MathJax-Span-94
        .mo style="vertical-align: 2.882em;"}]{#MathJax-Span-73 .mfenced
        style="padding-left: 0.26em;"}]{#MathJax-Span-65
        .mrow}]{#MathJax-Span-64
        .mrow}[]{style="display: inline-block; width: 0px; height: 4.99em;"}]{style="position: absolute; clip: rect(1.956em, 1008.19em, 7.56em, -1000.01em); top: -4.985em; left: 0em;"}]{style="display: inline-block; position: relative; width: 8.486em; height: 0px; font-size: 103%;"}[]{style="display: inline-block; overflow: hidden; vertical-align: -2.539em; border-left: 0px solid; width: 0px; height: 5.563em;"}]{#MathJax-Span-63
        .math
        style="width: 8.743em; display: inline-block;"}[$$\mathbf{x}^{(1)} = \begin{pmatrix}
        {- 118.29} \\
        33.91 \\
        1,416 \\
        38,372 \\
        \end{pmatrix}$$]{.MJX_Assistive_MathML
        .MJX_Assistive_MathML_Block
        role="presentation"}]{#MathJax-Element-3-Frame .MathJax
        tabindex="0"
        mathml="<math xmlns=\"http://www.w3.org/1998/Math/MathML\" display=\"block\"><mrow><msup><mi mathvariant=\"bold\">x</mi><mrow><mo>(</mo><mn>1</mn><mo>)</mo></mrow></msup><mo>=</mo><mfenced open=\"(\" close=\")\"><mtable><mtr><mtd><mrow><mo>-</mo><mn>118.29</mn></mrow></mtd></mtr><mtr><mtd><mrow><mn>33.91</mn></mrow></mtd></mtr><mtr><mtd><mrow><mn>1,416</mn></mrow></mtd></mtr><mtr><mtd><mrow><mn>38,372</mn></mrow></mtd></mtr></mtable></mfenced></mrow></math>"
        role="presentation"
        style="text-align: center; position: relative;"}
        
        

        and:

        
        []{.MathJax_Preview style="color: inherit; display: none;"}
        
        [[[[[[[[[[y[]{style="display: inline-block; overflow: hidden; height: 1px; width: 0.003em;"}]{#MathJax-Span-99
        .mi
        style="font-family: MathJax_Math-italic;"}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(3.396em, 1000.47em, 4.373em, -1000.01em); top: -4.008em; left: 0em;"}[[[(]{#MathJax-Span-101
        .mo
        style="font-size: 70.7%; font-family: MathJax_Main;"}[1]{#MathJax-Span-102
        .mn
        style="font-size: 70.7%; font-family: MathJax_Main;"}[)]{#MathJax-Span-103
        .mo
        style="font-size: 70.7%; font-family: MathJax_Main;"}]{#MathJax-Span-100
        .mrow}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; top: -4.419em; left: 0.568em;"}]{style="display: inline-block; position: relative; width: 1.545em; height: 0px;"}]{#MathJax-Span-98
        .msup}[=]{#MathJax-Span-104 .mo
        style="font-family: MathJax_Main; padding-left: 0.26em;"}[156,400]{#MathJax-Span-105
        .mn
        style="font-family: MathJax_Main; padding-left: 0.26em;"}]{#MathJax-Span-97
        .mrow}]{#MathJax-Span-96
        .mrow}[]{style="display: inline-block; width: 0px; height: 2.111em;"}]{style="position: absolute; clip: rect(1.031em, 1006.08em, 2.47em, -1000.01em); top: -2.105em; left: 0em;"}]{style="display: inline-block; position: relative; width: 6.121em; height: 0px; font-size: 103%;"}[]{style="display: inline-block; overflow: hidden; vertical-align: -0.262em; border-left: 0px solid; width: 0px; height: 1.274em;"}]{#MathJax-Span-95
        .math
        style="width: 6.327em; display: inline-block;"}[$$y^{(1)} = 156,400$$]{.MJX_Assistive_MathML
        .MJX_Assistive_MathML_Block
        role="presentation"}]{#MathJax-Element-4-Frame .MathJax
        tabindex="0"
        mathml="<math xmlns=\"http://www.w3.org/1998/Math/MathML\" display=\"block\"><mrow><msup><mi>y</mi><mrow><mo>(</mo><mn>1</mn><mo>)</mo></mrow></msup><mo>=</mo><mn>156,400</mn></mrow></math>"
        role="presentation"
        style="text-align: center; position: relative;"}
        
        

-   **X** is a matrix containing all the feature values (excluding
    labels) of all instances in the dataset. There is one row per
    instance, and the *i*^th^ row is equal to the transpose of
    **x**^(*i*)^, noted
    (**x**^(*i*)^)^⊺^.
^[4](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch02.html){-marker
    .totri-footnote}^

    -   For example, if the first district is as just described, then
        the matrix **X** looks like this:

        
        []{.MathJax_Preview style="color: inherit; display: none;"}
        
        [[[[[[[X]{#MathJax-Span-109 .mi
        style="font-family: MathJax_Main-bold;"}[=]{#MathJax-Span-110
        .mo
        style="font-family: MathJax_Main; padding-left: 0.26em;"}[[[[⎛[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; font-family: MathJax_Size4; top: -2.877em; left: 0em;"}[⎝[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; font-family: MathJax_Size4; top: 3.653em; left: 0em;"}[⎜[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Size4; position: absolute; top: -1.694em; left: 0em;"}[⎜[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Size4; position: absolute; top: -1.129em; left: 0em;"}[⎜[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Size4; position: absolute; top: -0.614em; left: 0em;"}[⎜[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Size4; position: absolute; top: -0.1em; left: 0em;"}[⎜[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Size4; position: absolute; top: 0.465em; left: 0em;"}[⎜[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Size4; position: absolute; top: 0.979em; left: 0em;"}[⎜[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Size4; position: absolute; top: 1.494em; left: 0em;"}[⎜[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Size4; position: absolute; top: 2.059em; left: 0em;"}[⎜[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Size4; position: absolute; top: 2.573em; left: 0em;"}]{style="display: inline-block; position: relative; width: 0.877em; height: 0px;"}]{#MathJax-Span-130
        .mo
        style="vertical-align: 4.424em;"}[[[[[[[[[[[[[(]{style="font-family: MathJax_Size1;"}]{#MathJax-Span-118
        .mo style="vertical-align: 0em;"}[[[[x]{#MathJax-Span-120 .mi
        style="font-family: MathJax_Main-bold;"}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(3.396em, 1000.58em, 4.167em, -1000.01em); top: -4.008em; left: 0em;"}[[[(]{#MathJax-Span-122
        .mo
        style="font-size: 70.7%; font-family: MathJax_Main;"}[1]{#MathJax-Span-123
        .mn
        style="font-size: 70.7%; font-family: MathJax_Main;"}[)]{#MathJax-Span-124
        .mo
        style="font-size: 70.7%; font-family: MathJax_Main;"}]{#MathJax-Span-121
        .mrow}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; top: -4.368em; left: 0.62em;"}]{style="display: inline-block; position: relative; width: 1.596em; height: 0px;"}]{#MathJax-Span-119
        .msup}[[)]{style="font-family: MathJax_Size1;"}]{#MathJax-Span-125
        .mo style="vertical-align: 0em;"}]{#MathJax-Span-117
        .mrow}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(2.985em, 1002.38em, 4.527em, -1000.01em); top: -4.008em; left: 0em;"}[[⊺]{#MathJax-Span-127
        .mo
        style="font-size: 70.7%; font-family: MathJax_AMS;"}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; top: -4.625em; left: 2.522em;"}]{style="display: inline-block; position: relative; width: 2.985em; height: 0px;"}]{#MathJax-Span-116
        .msup}]{#MathJax-Span-115 .mrow}]{#MathJax-Span-114
        .mtd}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(2.933em, 1002.99em, 4.527em, -1000.01em); top: -7.504em; left: 50%; margin-left: -1.488em;"}[[[[[[[[[(]{style="font-family: MathJax_Size1;"}]{#MathJax-Span-135
        .mo style="vertical-align: 0em;"}[[[[x]{#MathJax-Span-137 .mi
        style="font-family: MathJax_Main-bold;"}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(3.396em, 1000.58em, 4.167em, -1000.01em); top: -4.008em; left: 0em;"}[[[(]{#MathJax-Span-139
        .mo
        style="font-size: 70.7%; font-family: MathJax_Main;"}[2]{#MathJax-Span-140
        .mn
        style="font-size: 70.7%; font-family: MathJax_Main;"}[)]{#MathJax-Span-141
        .mo
        style="font-size: 70.7%; font-family: MathJax_Main;"}]{#MathJax-Span-138
        .mrow}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; top: -4.368em; left: 0.62em;"}]{style="display: inline-block; position: relative; width: 1.596em; height: 0px;"}]{#MathJax-Span-136
        .msup}[[)]{style="font-family: MathJax_Size1;"}]{#MathJax-Span-142
        .mo style="vertical-align: 0em;"}]{#MathJax-Span-134
        .mrow}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(2.985em, 1002.38em, 4.527em, -1000.01em); top: -4.008em; left: 0em;"}[[⊺]{#MathJax-Span-143
        .mo
        style="font-size: 70.7%; font-family: MathJax_AMS;"}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; top: -4.625em; left: 2.522em;"}]{style="display: inline-block; position: relative; width: 2.985em; height: 0px;"}]{#MathJax-Span-133
        .msup}]{#MathJax-Span-132 .mrow}]{#MathJax-Span-131
        .mtd}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(2.933em, 1002.99em, 4.527em, -1000.01em); top: -5.807em; left: 50%; margin-left: -1.488em;"}[[[[⋮]{#MathJax-Span-146
        .mo style="font-family: MathJax_Main;"}]{#MathJax-Span-145
        .mrow}]{#MathJax-Span-144
        .mtd}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(2.573em, 1000.22em, 4.219em, -1000.01em); top: -3.699em; left: 50%; margin-left: -0.152em;"}[[[[[[[[[(]{style="font-family: MathJax_Size1;"}]{#MathJax-Span-151
        .mo style="vertical-align: 0em;"}[[[[x]{#MathJax-Span-153 .mi
        style="font-family: MathJax_Main-bold;"}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(3.396em, 1000.58em, 4.167em, -1000.01em); top: -4.008em; left: 0em;"}[[[(]{#MathJax-Span-155
        .mo
        style="font-size: 70.7%; font-family: MathJax_Main;"}[1999]{#MathJax-Span-156
        .mn
        style="font-size: 70.7%; font-family: MathJax_Main;"}[)]{#MathJax-Span-157
        .mo
        style="font-size: 70.7%; font-family: MathJax_Main;"}]{#MathJax-Span-154
        .mrow}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; top: -4.368em; left: 0.62em;"}]{style="display: inline-block; position: relative; width: 2.625em; height: 0px;"}]{#MathJax-Span-152
        .msup}[[)]{style="font-family: MathJax_Size1;"}]{#MathJax-Span-158
        .mo style="vertical-align: 0em;"}]{#MathJax-Span-150
        .mrow}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(2.985em, 1003.4em, 4.527em, -1000.01em); top: -4.008em; left: 0em;"}[[⊺]{#MathJax-Span-159
        .mo
        style="font-size: 70.7%; font-family: MathJax_AMS;"}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; top: -4.625em; left: 3.55em;"}]{style="display: inline-block; position: relative; width: 4.013em; height: 0px;"}]{#MathJax-Span-149
        .msup}]{#MathJax-Span-148 .mrow}]{#MathJax-Span-147
        .mtd}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(2.933em, 1004.02em, 4.527em, -1000.01em); top: -2.157em; left: 50%; margin-left: -2.003em;"}[[[[[[[[[(]{style="font-family: MathJax_Size1;"}]{#MathJax-Span-164
        .mo style="vertical-align: 0em;"}[[[[x]{#MathJax-Span-166 .mi
        style="font-family: MathJax_Main-bold;"}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(3.396em, 1000.58em, 4.167em, -1000.01em); top: -4.008em; left: 0em;"}[[[(]{#MathJax-Span-168
        .mo
        style="font-size: 70.7%; font-family: MathJax_Main;"}[2000]{#MathJax-Span-169
        .mn
        style="font-size: 70.7%; font-family: MathJax_Main;"}[)]{#MathJax-Span-170
        .mo
        style="font-size: 70.7%; font-family: MathJax_Main;"}]{#MathJax-Span-167
        .mrow}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; top: -4.368em; left: 0.62em;"}]{style="display: inline-block; position: relative; width: 2.625em; height: 0px;"}]{#MathJax-Span-165
        .msup}[[)]{style="font-family: MathJax_Size1;"}]{#MathJax-Span-171
        .mo style="vertical-align: 0em;"}]{#MathJax-Span-163
        .mrow}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(2.985em, 1003.4em, 4.527em, -1000.01em); top: -4.008em; left: 0em;"}[[⊺]{#MathJax-Span-172
        .mo
        style="font-size: 70.7%; font-family: MathJax_AMS;"}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; top: -4.625em; left: 3.55em;"}]{style="display: inline-block; position: relative; width: 4.013em; height: 0px;"}]{#MathJax-Span-162
        .msup}]{#MathJax-Span-161 .mrow}]{#MathJax-Span-160
        .mtd}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(2.933em, 1004.02em, 4.527em, -1000.01em); top: -0.46em; left: 50%; margin-left: -2.003em;"}]{style="display: inline-block; position: relative; width: 4.013em; height: 0px;"}[]{style="display: inline-block; width: 0px; height: 9.309em;"}]{style="position: absolute; clip: rect(4.733em, 1004.02em, 13.37em, -1000.01em); top: -9.303em; left: 0em;"}]{style="display: inline-block; position: relative; width: 4.013em; height: 0px;"}]{#MathJax-Span-113
        .mtable
        style="padding-right: 0.157em; padding-left: 0.157em;"}[[[⎞[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; font-family: MathJax_Size4; top: -2.877em; left: 0em;"}[⎠[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; font-family: MathJax_Size4; top: 3.653em; left: 0em;"}[⎟[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Size4; position: absolute; top: -1.694em; left: 0em;"}[⎟[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Size4; position: absolute; top: -1.129em; left: 0em;"}[⎟[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Size4; position: absolute; top: -0.614em; left: 0em;"}[⎟[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Size4; position: absolute; top: -0.1em; left: 0em;"}[⎟[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Size4; position: absolute; top: 0.465em; left: 0em;"}[⎟[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Size4; position: absolute; top: 0.979em; left: 0em;"}[⎟[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Size4; position: absolute; top: 1.494em; left: 0em;"}[⎟[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Size4; position: absolute; top: 2.059em; left: 0em;"}[⎟[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="font-family: MathJax_Size4; position: absolute; top: 2.573em; left: 0em;"}]{style="display: inline-block; position: relative; width: 0.877em; height: 0px;"}]{#MathJax-Span-173
        .mo style="vertical-align: 4.424em;"}]{#MathJax-Span-111
        .mfenced style="padding-left: 0.26em;"}[=]{#MathJax-Span-174 .mo
        style="font-family: MathJax_Main; padding-left: 0.26em;"}[[[(]{style="font-family: MathJax_Size4;"}]{#MathJax-Span-176
        .mo style="vertical-align: 0em;"}[[[[[[[[[−]{#MathJax-Span-181
        .mo
        style="font-family: MathJax_Main;"}[118.29]{#MathJax-Span-182
        .mn style="font-family: MathJax_Main;"}]{#MathJax-Span-180
        .mrow}]{#MathJax-Span-179 .mrow}]{#MathJax-Span-178
        .mtd}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(3.19em, 1003.51em, 4.27em, -1000.01em); top: -4.933em; left: 50%; margin-left: -1.797em;"}[[[[⋮]{#MathJax-Span-197
        .mo style="font-family: MathJax_Main;"}]{#MathJax-Span-196
        .mrow}]{#MathJax-Span-195
        .mtd}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(2.573em, 1000.22em, 4.219em, -1000.01em); top: -2.979em; left: 50%; margin-left: -0.152em;"}]{style="display: inline-block; position: relative; width: 3.55em; height: 0px;"}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(2.265em, 1003.51em, 5.195em, -1000.01em); top: -4.008em; left: 0em;"}[[[[[[[33.91]{#MathJax-Span-186
        .mn style="font-family: MathJax_Main;"}]{#MathJax-Span-185
        .mrow}]{#MathJax-Span-184 .mrow}]{#MathJax-Span-183
        .mtd}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(3.19em, 1002.22em, 4.167em, -1000.01em); top: -4.933em; left: 50%; margin-left: -1.129em;"}[[[[⋮]{#MathJax-Span-200
        .mo style="font-family: MathJax_Main;"}]{#MathJax-Span-199
        .mrow}]{#MathJax-Span-198
        .mtd}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(2.573em, 1000.22em, 4.219em, -1000.01em); top: -2.979em; left: 50%; margin-left: -0.152em;"}]{style="display: inline-block; position: relative; width: 2.265em; height: 0px;"}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(2.265em, 1002.22em, 5.195em, -1000.01em); top: -4.008em; left: 4.373em;"}[[[[[[[1,416]{#MathJax-Span-190
        .mn style="font-family: MathJax_Main;"}]{#MathJax-Span-189
        .mrow}]{#MathJax-Span-188 .mrow}]{#MathJax-Span-187
        .mtd}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(3.19em, 1002.22em, 4.373em, -1000.01em); top: -4.933em; left: 50%; margin-left: -1.129em;"}[[[[⋮]{#MathJax-Span-203
        .mo style="font-family: MathJax_Main;"}]{#MathJax-Span-202
        .mrow}]{#MathJax-Span-201
        .mtd}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(2.573em, 1000.22em, 4.219em, -1000.01em); top: -2.979em; left: 50%; margin-left: -0.152em;"}]{style="display: inline-block; position: relative; width: 2.265em; height: 0px;"}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(2.265em, 1002.22em, 5.195em, -1000.01em); top: -4.008em; left: 7.458em;"}[[[[[[[38,372]{#MathJax-Span-194
        .mn style="font-family: MathJax_Main;"}]{#MathJax-Span-193
        .mrow}]{#MathJax-Span-192 .mrow}]{#MathJax-Span-191
        .mtd}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(3.19em, 1002.74em, 4.373em, -1000.01em); top: -4.933em; left: 50%; margin-left: -1.386em;"}[[[[⋮]{#MathJax-Span-206
        .mo style="font-family: MathJax_Main;"}]{#MathJax-Span-205
        .mrow}]{#MathJax-Span-204
        .mtd}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(2.573em, 1000.22em, 4.219em, -1000.01em); top: -2.979em; left: 50%; margin-left: -0.152em;"}]{style="display: inline-block; position: relative; width: 2.779em; height: 0px;"}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(2.265em, 1002.74em, 5.195em, -1000.01em); top: -4.008em; left: 10.491em;"}]{style="display: inline-block; position: relative; width: 13.267em; height: 0px;"}]{#MathJax-Span-177
        .mtable
        style="padding-right: 0.157em; padding-left: 0.157em;"}[[)]{style="font-family: MathJax_Size4;"}]{#MathJax-Span-207
        .mo style="vertical-align: 0em;"}]{#MathJax-Span-175 .mfenced
        style="padding-left: 0.26em;"}]{#MathJax-Span-108
        .mrow}]{#MathJax-Span-107
        .mrow}[]{style="display: inline-block; width: 0px; height: 7.509em;"}]{style="position: absolute; clip: rect(2.933em, 1024.43em, 11.571em, -1000.01em); top: -7.504em; left: 0em;"}]{style="display: inline-block; position: relative; width: 24.681em; height: 0px; font-size: 103%;"}[]{style="display: inline-block; overflow: hidden; vertical-align: -4.075em; border-left: 0px solid; width: 0px; height: 8.688em;"}]{#MathJax-Span-106
        .math
        style="width: 25.453em; display: inline-block;"}[$$\mathbf{X} = \begin{pmatrix}
        \left( \mathbf{x}^{(1)} \right)^{\intercal} \\
        \left( \mathbf{x}^{(2)} \right)^{\intercal} \\
         \vdots \\
        \left( \mathbf{x}^{(1999)} \right)^{\intercal} \\
        \left( \mathbf{x}^{(2000)} \right)^{\intercal} \\
        \end{pmatrix} = \begin{pmatrix}
        {- 118.29} & 33.91 & 1,416 & 38,372 \\
         \vdots & \vdots & \vdots & \vdots \\
        \end{pmatrix}$$]{.MJX_Assistive_MathML
        .MJX_Assistive_MathML_Block
        role="presentation"}]{#MathJax-Element-5-Frame .MathJax
        tabindex="0"
        mathml="<math xmlns=\"http://www.w3.org/1998/Math/MathML\" display=\"block\"><mrow><mi mathvariant=\"bold\">X</mi><mo>=</mo><mfenced open=\"(\" close=\")\"><mtable><mtr><mtd><msup><mrow><mo>(</mo><msup><mi mathvariant=\"bold\">x</mi><mrow><mo>(</mo><mn>1</mn><mo>)</mo></mrow></msup><mo>)</mo></mrow><mo>&#x22BA;</mo></msup></mtd></mtr><mtr><mtd><msup><mrow><mo>(</mo><msup><mi mathvariant=\"bold\">x</mi><mrow><mo>(</mo><mn>2</mn><mo>)</mo></mrow></msup><mo>)</mo></mrow><mo>&#x22BA;</mo></msup></mtd></mtr><mtr><mtd><mo>&#x22EE;</mo></mtd></mtr><mtr><mtd><msup><mrow><mo>(</mo><msup><mi mathvariant=\"bold\">x</mi><mrow><mo>(</mo><mn>1999</mn><mo>)</mo></mrow></msup><mo>)</mo></mrow><mo>&#x22BA;</mo></msup></mtd></mtr><mtr><mtd><msup><mrow><mo>(</mo><msup><mi mathvariant=\"bold\">x</mi><mrow><mo>(</mo><mn>2000</mn><mo>)</mo></mrow></msup><mo>)</mo></mrow><mo>&#x22BA;</mo></msup></mtd></mtr></mtable></mfenced><mo>=</mo><mfenced open=\"(\" close=\")\"><mtable><mtr><mtd><mrow><mo>-</mo><mn>118.29</mn></mrow></mtd><mtd><mrow><mn>33.91</mn></mrow></mtd><mtd><mrow><mn>1,416</mn></mrow></mtd><mtd><mrow><mn>38,372</mn></mrow></mtd></mtr><mtr><mtd><mo>&#x22EE;</mo></mtd><mtd><mo>&#x22EE;</mo></mtd><mtd><mo>&#x22EE;</mo></mtd><mtd><mo>&#x22EE;</mo></mtd></mtr></mtable></mfenced></mrow></math>"
        role="presentation"
        style="text-align: center; position: relative;"}
        
        

-   *h* is your system's prediction function, also called a
    *hypothesis*. When your system is given an instance's feature vector
    **x**^(*i*)^, it outputs a predicted value *ŷ*^(*i*)^ =
    *h*(**x**^(*i*)^) for that instance (*ŷ* is pronounced "y-hat").

    -   For example, if your system predicts that the median housing
        price in the first district is \$158,400, then *ŷ*^(1)^ =
        *h*(**x**^(1)^) = 158,400. The prediction error for this
        district is *ŷ*^(1)^ -- *y*^(1)^ = 2,000.

-   RMSE(**X**,*h*) is the cost function measured on the set of examples
    using your hypothesis *h*.

We use lowercase italic font for scalar values (such as *m* or
*y*^(*i*)^) and function names (such as *h*), lowercase bold font for
vectors (such as **x**^(*i*)^), and uppercase bold font for matrices
(such as **X**).


Even though the RMSE is generally the preferred performance measure for
regression tasks, in some contexts you may prefer to use another
function. For example, suppose that there are many outlier districts. In
that case, you may
consider 
using the *mean absolute error* (MAE, also called
the average absolute deviation; see [Equation
2-2]


##### [Equation 2-2. ] Mean absolute error (MAE)

[]{.MathJax_Preview style="color: inherit; display: none;"}


[[[[[[[MAE]{#MathJax-Span-211 .mtext
style="font-family: MathJax_Main;"}[[(]{#MathJax-Span-213 .mo
style="font-family: MathJax_Main;"}[X]{#MathJax-Span-214 .mi
style="font-family: MathJax_Main-bold;"}[,]{#MathJax-Span-215 .mo
style="font-family: MathJax_Main;"}[h]{#MathJax-Span-216 .mi
style="font-family: MathJax_Math-italic; padding-left: 0.157em;"}[)]{#MathJax-Span-217
.mo style="font-family: MathJax_Main;"}]{#MathJax-Span-212 .mrow
style="padding-left: 0.157em;"}[=]{#MathJax-Span-218 .mo
style="font-family: MathJax_Main; padding-left: 0.26em;"}[[[[1]{#MathJax-Span-220
.mn
style="font-family: MathJax_Main;"}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(3.19em, 1000.42em, 4.167em, -1000.01em); top: -4.676em; left: 50%; margin-left: -0.255em;"}[[m]{#MathJax-Span-221
.mi
style="font-family: MathJax_Math-italic;"}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(3.396em, 1000.88em, 4.167em, -1000.01em); top: -3.339em; left: 50%; margin-left: -0.46em;"}[[]{style="display: inline-block; overflow: hidden; vertical-align: 0em; border-top: 1.3px solid; width: 0.979em; height: 0px;"}[]{style="display: inline-block; width: 0px; height: 1.082em;"}]{style="position: absolute; clip: rect(0.877em, 1000.99em, 1.237em, -1000.01em); top: -1.283em; left: 0em;"}]{style="display: inline-block; position: relative; width: 0.979em; height: 0px; margin-right: 0.105em; margin-left: 0.105em;"}]{#MathJax-Span-219
.mfrac style="padding-left: 0.26em;"}[[[[∑]{#MathJax-Span-223 .mo
style="font-family: MathJax_Size2; vertical-align: 0em;"}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(2.933em, 1001.4em, 4.63em, -1000.01em); top: -4.008em; left: 0em;"}[[[i]{#MathJax-Span-225
.mi
style="font-size: 70.7%; font-family: MathJax_Math-italic;"}[=]{#MathJax-Span-226
.mo
style="font-size: 70.7%; font-family: MathJax_Main;"}[1]{#MathJax-Span-227
.mn
style="font-size: 70.7%; font-family: MathJax_Main;"}]{#MathJax-Span-224
.mrow}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(3.396em, 1001.09em, 4.27em, -1000.01em); top: -2.928em; left: 0.157em;"}[[m]{#MathJax-Span-228
.mi
style="font-size: 70.7%; font-family: MathJax_Math-italic;"}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(3.447em, 1000.63em, 4.167em, -1000.01em); top: -5.139em; left: 0.414em;"}]{style="display: inline-block; position: relative; width: 1.442em; height: 0px;"}]{#MathJax-Span-222
.munderover
style="padding-left: 0.157em;"}[[[[∣[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; font-family: MathJax_Main; top: -3.237em; left: 0em;"}[∣[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; font-family: MathJax_Main; top: -2.465em; left: 0em;"}]{style="display: inline-block; position: relative; width: 0.26em; height: 0px;"}]{#MathJax-Span-230
.mo style="vertical-align: 1.134em;"}[h]{#MathJax-Span-231 .mi
style="font-family: MathJax_Math-italic;"}[[[(]{style="font-family: MathJax_Size2;"}]{#MathJax-Span-233
.mo style="vertical-align: 0em;"}[[[[x]{#MathJax-Span-235 .mi
style="font-family: MathJax_Main-bold;"}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(3.396em, 1000.58em, 4.167em, -1000.01em); top: -4.008em; left: 0em;"}[[[(]{#MathJax-Span-237
.mo
style="font-size: 70.7%; font-family: MathJax_Main;"}[i]{#MathJax-Span-238
.mi
style="font-size: 70.7%; font-family: MathJax_Math-italic;"}[)]{#MathJax-Span-239
.mo
style="font-size: 70.7%; font-family: MathJax_Main;"}]{#MathJax-Span-236
.mrow}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; top: -4.419em; left: 0.62em;"}]{style="display: inline-block; position: relative; width: 1.494em; height: 0px;"}]{#MathJax-Span-234
.msup}[[)]{style="font-family: MathJax_Size2;"}]{#MathJax-Span-240 .mo
style="vertical-align: 0em;"}]{#MathJax-Span-232 .mrow
style="padding-left: 0.157em;"}[−]{#MathJax-Span-241 .mo
style="font-family: MathJax_Main; padding-left: 0.208em;"}[[[[y[]{style="display: inline-block; overflow: hidden; height: 1px; width: 0.003em;"}]{#MathJax-Span-243
.mi
style="font-family: MathJax_Math-italic;"}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; clip: rect(3.396em, 1000.47em, 4.373em, -1000.01em); top: -4.008em; left: 0em;"}[[[(]{#MathJax-Span-245
.mo
style="font-size: 70.7%; font-family: MathJax_Main;"}[i]{#MathJax-Span-246
.mi
style="font-size: 70.7%; font-family: MathJax_Math-italic;"}[)]{#MathJax-Span-247
.mo
style="font-size: 70.7%; font-family: MathJax_Main;"}]{#MathJax-Span-244
.mrow}[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; top: -4.419em; left: 0.568em;"}]{style="display: inline-block; position: relative; width: 1.442em; height: 0px;"}]{#MathJax-Span-242
.msup
style="padding-left: 0.208em;"}[[[∣[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; font-family: MathJax_Main; top: -3.237em; left: 0em;"}[∣[]{style="display: inline-block; width: 0px; height: 4.013em;"}]{style="position: absolute; font-family: MathJax_Main; top: -2.465em; left: 0em;"}]{style="display: inline-block; position: relative; width: 0.26em; height: 0px;"}]{#MathJax-Span-248
.mo style="vertical-align: 1.134em;"}]{#MathJax-Span-229 .mfenced
style="padding-left: 0.157em;"}]{#MathJax-Span-210
.mrow}]{#MathJax-Span-209
.mrow}[]{style="display: inline-block; width: 0px; height: 3.859em;"}]{style="position: absolute; clip: rect(2.162em, 1015.9em, 5.195em, -1000.01em); top: -3.854em; left: 0em;"}]{style="display: inline-block; position: relative; width: 15.992em; height: 0px; font-size: 103%;"}[]{style="display: inline-block; overflow: hidden; vertical-align: -1.268em; border-left: 0px solid; width: 0px; height: 2.968em;"}]{#MathJax-Span-208
.math
style="width: 16.507em; display: inline-block;"}[$$\text{MAE}\left( \mathbf{X},h \right) = \frac{1}{m}\sum\limits_{i = 1}^{m}\left| {h\left( \mathbf{x}^{(i)} \right) - y^{(i)}} \right|$$]{.MJX_Assistive_MathML
.MJX_Assistive_MathML_Block
role="presentation"}]{#MathJax-Element-6-Frame .MathJax tabindex="0"
mathml="<math xmlns=\"http://www.w3.org/1998/Math/MathML\" display=\"block\"><mrow><mtext>MAE</mtext><mrow><mo>(</mo><mi mathvariant=\"bold\">X</mi><mo>,</mo><mi>h</mi><mo>)</mo></mrow><mo>=</mo><mfrac><mn>1</mn><mi>m</mi></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><mfenced separators=\"\" open=\"|\" close=\"|\"><mi>h</mi><mrow><mo>(</mo><msup><mi mathvariant=\"bold\">x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><mo>-</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup></mfenced></mrow></math>"
role="presentation" style="text-align: center; position: relative;"}


Both the RMSE and the MAE are ways to measure the distance between two
vectors: the vector of predictions and the vector of target values.
Various distance measures, or *norms*, are possible:

-   Computing the root of a sum of squares (RMSE)
    corresponds to the *Euclidean norm*: this is
    the notion of distance you are familiar with. It is also called the
    ℓ~2~ *norm*, noted ∥ · ∥~2~ (or just ∥ · ∥).

-   Computing the sum of absolutes (MAE) corresponds to the ℓ~1~ *norm*,
    noted ∥ · ∥~1~. This is sometimes called the
    *Manhattan norm* because it measures the distance between two points
    in a city if you can only travel along orthogonal city blocks.

-   More generally, the ℓ~*k*~ *norm* of a vector **v** containing *n*
    elements is defined as ∥**v**∥~*k*~ = (\|*v*~0~\|^*k*^ +
    \|*v*~1~\|^*k*^ + \... + \|*v*~*n*~\|^*k*^)^1/*k*^. ℓ~0~ gives the
    number of nonzero elements in the vector, and ℓ~∞~ gives the maximum
    absolute value in the vector.

-   The higher the norm index, the more it focuses on large values and
    neglects small ones. This is why the RMSE is more sensitive to
    outliers than the MAE. But when outliers are exponentially rare
    (like in a bell-shaped curve), the RMSE performs very well and is
    generally preferred.




Check the Assumptions
---------------------

Lastly, it is good practice to list and verify the
assumptions that have been made so far (by you or others); this can help
you catch serious issues early on. For example, the district prices that
your system outputs are going to be fed into a downstream Machine
Learning system, and you assume that these prices are going to be used
as such. But what if the downstream system converts the prices into
categories (e.g., "cheap," "medium," or "expensive") and then uses those
categories instead of the prices themselves? In this case, getting the
price perfectly right is not important at all; your system just needs to
get the category right. If that's so, then the problem should have been
framed as a classification task, not a regression task. You don't want
to find this out after working on a regression system for months.

Fortunately, after talking with the team in charge of the downstream
system, you are confident that they do indeed need the actual prices,
not just categories. Great! You're all set, the lights are green, and
you can start coding now!





Get the Data
============

It's time to get your hands dirty. Don't hesitate to pick
up your laptop and walk through the following code examples in a Jupyter
notebook. The full Jupyter notebook is available at
[*https://github.com/fenago/ml-scikit-keras-tensorflow*](https://github.com/fenago/ml-scikit-keras-tensorflow).



Create the Workspace
--------------------

First you will need to have Python installed. It
is probably already installed on your system. If not, you can get it at
[*https://www.python.org/*](https://www.python.org/).
^[5](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch02.html){-marker
.totri-footnote}^

Next you need to create a workspace directory for your Machine Learning
code and datasets. Open a terminal and type the following commands
(after the `$` prompts):

``` {data-type="programlisting" code-language="shell-session"}
$ export ML_PATH="$HOME/ml"      # You can change the path if you prefer
$ mkdir -p $ML_PATH
```

You will need a number of
Python modules: Jupyter, NumPy, pandas, Matplotlib, and Scikit-Learn. If
you already have Jupyter running with all these modules installed, you
can safely skip to ["Download the
Data"]
If you don't have them yet, there are many ways to install them (and
their dependencies). You can use your [system's]
packaging system (e.g., apt-get on Ubuntu, or MacPorts or Homebrew on
macOS), install a Scientific Python distribution such as Anaconda and
use its packaging system, or just use Python's own packaging system,
pip, which is included by default with the Python binary installers
(since Python
2.7.9).
^[6](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch02.html) You can check to see if pip is installed by typing the
following command:

``` {data-type="programlisting" code-language="shell-session"}
$ python3 -m pip --version
pip 19.3.1 from [...]/lib/python3.7/site-packages/pip (python 3.7)
```

You should make sure you have a recent version of pip installed. To
upgrade the pip module, type the following (the exact version may
differ): ^[7](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch02.html){-marker
.totri-footnote}^

``` {data-type="programlisting" code-language="shell-session"}
$ python3 -m pip install --user -U pip
Collecting pip
[...]
Successfully installed pip-19.3.1
```


##### Creating an Isolated Environment

If you would like to work in an isolated
environment (which is strongly recommended so that you can work on
different projects without having conflicting library versions), install
virtualenv^[8](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch02.html) by running the following pip command (again, if you
want virtualenv to be installed for all users on your machine, remove
`--user` and run this command with administrator rights):

``` {data-type="programlisting" code-language="shell-session"}
$ python3 -m pip install --user -U virtualenv
Collecting virtualenv
[...]
Successfully installed virtualenv-16.7.6
```

Now you can create an isolated Python environment by typing this:

``` {data-type="programlisting" code-language="shell-session"}
$ cd $ML_PATH
$ python3 -m virtualenv my_env
Using base prefix '[...]'
New python executable in [...]/ml/my_env/bin/python3
Also creating executable in [...]/ml/my_env/bin/python
Installing setuptools, pip, wheel...done.
```

Now every time you want to activate this environment, just open a
terminal and type the following:

``` {data-type="programlisting" code-language="shell-session"}
$ cd $ML_PATH
$ source my_env/bin/activate # on Linux or macOS
$ .\my_env\Scripts\activate  # on Windows
```

To deactivate this environment, type `deactivate`. While the environment
is active, any package you install using pip will be installed in this
isolated environment, and Python will only have access to these packages
(if you also want access to the system's packages, you should create the
environment using virtualenv's `--system-site-packages` option). Check
out virtualenv's documentation for more information.


Now you can install all the required modules and their dependencies
using this simple pip command (if you are not using a virtualenv, you
will need the `--user` option or administrator rights):

``` {data-type="programlisting" code-language="shell-session"}
$ python3 -m pip install -U jupyter matplotlib numpy pandas scipy scikit-learn
Collecting jupyter
  Downloading https://[...]/jupyter-1.0.0-py2.py3-none-any.whl
Collecting matplotlib
  [...]
```

If you created a virtualenv, you need to register it to Jupyter and give
it a name:

``` {data-type="programlisting" code-language="shell-session"}
$ python3 -m ipykernel install --user --name=python3
```

Now you can fire up Jupyter by typing the following command:

``` {data-type="programlisting" code-language="shell-session"}
$ jupyter notebook
[...] Serving notebooks from local directory: [...]/ml
[...] The Jupyter Notebook is running at:
[...] http://localhost:8888/?token=60995e108e44ac8d8865a[...]
[...]  or http://127.0.0.1:8889/?token=60995e108e44ac8d8865a[...]
[...] Use Control-C to stop this server and shut down all kernels [...]
```

A Jupyter server is now running in your terminal, listening to port
8888. You can visit this server by opening your web browser to
*http://localhost:8888/* (this usually happens automatically when the
server starts). You should see your empty workspace directory
(containing only the *env* directory if you followed the preceding
virtualenv instructions).

Now create a new Python notebook by clicking the New button and
selecting the appropriate Python
version^[9](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch02.html) (see
[Figure 2-3]
Doing that will create a new notebook file called *Untitled.ipynb* in
your workspace, start a Jupyter Python kernel to run the notebook, and
open this notebook in a new tab. You should start by renaming this
notebook to "Housing" (this will automatically rename the file to
*Housing.ipynb*) by clicking Untitled and typing the new name.

![](./images/mls2_0203.png)

A notebook contains a list of cells. Each cell can contain executable
code or formatted text. Right now the notebook contains only one empty
code cell, labeled "In \[1\]:". Try typing `print("Hello world!")` in
the cell and clicking the play button (see
[Figure 2-4]
or pressing Shift-Enter. This sends the current cell to this notebook's
Python kernel, which runs it and returns the output. The result is
displayed below the cell, and since you've reached the end of the
notebook, a new cell is automatically created. Go through the User
Interface Tour from Jupyter's Help menu to learn the basics.

![](./images/mls2_0204.png)




Download the Data
-----------------

In typical environments your data would be
available in a relational database (or some other common data store) and
spread across multiple tables/documents/files. To access it, you would
first need to get your credentials and access
authorizations^[10](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch02.html){-marker}^
and familiarize yourself with the data schema. In this project, however,
things are much simpler: you will just download a single compressed
file, *housing.tgz*, which contains a comma-separated values (CSV) file
called *housing.csv* with all the data.

You could use your web browser to download the file and run
`tar xzf housing.tgz` to decompress it and extract the CSV file, but it
is preferable to create a small function to do that. Having a function
that downloads the data is useful in particular if the data changes
regularly: you can write a small script that uses the function to fetch
the latest data (or you can set up a scheduled job to do that
automatically at regular intervals). Automating the process of fetching
the data is also useful if you need to install the dataset on multiple
machines.

Here is the function to fetch the
data: ^[11](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch02.html){-marker}^

``` {data-type="programlisting" code-language="python"}
import os
import tarfile
import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/fenago/ml-scikit-keras-tensorflow/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
```

Now when you call `fetch_housing_data()`, it creates a
*datasets/housing* directory in your workspace, downloads the
*housing.tgz* file, and extracts the *housing.csv* file from it in this
directory.

Now let's load the data using pandas. Once again, you should write a
small function to load the data:

``` {data-type="programlisting" code-language="python"}
import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
```

This function returns a pandas DataFrame object containing all the data.




Take a Quick Look at the Data Structure
---------------------------------------

Let's take a look at the top five rows using the DataFrame's `head()`
method (see
[Figure 2-5]

![](./images/mls2_0205.png)

Each row represents one district. There are 10 attributes (you can see
the first 6 in the screenshot): `longitude`, `latitude`,
`housing_median_age`, `total_rooms`, `total_bedrooms`, `population`,
`households`, `median_income`, `median_house_value`, and
`ocean_proximity`.

The `info()` method is useful to get a quick description of the data, in
particular the total number of rows, each attribute's type, and the
number of nonnull values (see
[Figure 2-6]

![](./images/mls2_0206.png)

There are 20,640 instances in the dataset, which means that it is fairly
small by Machine Learning standards, but it's perfect to get started.
Notice that the `total_bedrooms` attribute has only 20,433 nonnull
values, meaning that 207 districts are missing this feature. We will
need to take care of this later.

All attributes are numerical, except the `ocean_proximity` field. Its
type is `object`, so it could hold any kind of Python object. But since
you loaded this data from a CSV file, you know that it must be a text
attribute. When you looked at the top five rows, you probably noticed
that the values in the `ocean_proximity` column were repetitive, which
means that it is probably a categorical attribute. You can find out what
categories exist and how many districts belong to each category by using
the `value_counts()` method:

``` {data-type="programlisting" code-language="pycon"}
>>> housing["ocean_proximity"].value_counts()
<1H OCEAN     9136
INLAND        6551
NEAR OCEAN    2658
NEAR BAY      2290
ISLAND           5
Name: ocean_proximity, dtype: int64
```

Let's look at the other fields. The `describe()` method shows a summary
of the numerical attributes
([Figure 2-7]

![](./images/mls2_0207.png)

The `count`, `mean`, `min`, and `max` rows are self-explanatory. Note
that the null values are ignored (so, for example, the `count` of
`total_bedrooms` is 20,433, not 20,640). The `std` row shows the
*standard deviation*, which measures how dispersed the values
are.
^[12](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch02.html){-marker}^
The 25%, 50%, and 75% rows show the corresponding *percentiles*: a
percentile indicates the value below which a given percentage of
observations in a group of observations fall. For example, 25% of the
districts have a `housing_median_age` lower than 18, while 50% are lower
than 29 and 75% are lower than 37. These are often called the 25th
percentile (or first *quartile*), the median, and the 75th percentile
(or third [quartile]).

Another quick way to get a feel of the type of data you are dealing with
is to plot a histogram for each numerical attribute. A histogram shows
the number of instances (on the vertical axis) that have a given value
range (on the horizontal axis). You can either plot this one attribute
at a time, or you can call the `hist()` method on the whole dataset (as
shown in the following code example), and it will plot a histogram for
each numerical attribute (see
[Figure 2-8]

``` {data-type="programlisting" code-language="python"}
%matplotlib inline   # only in a Jupyter notebook
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()
```


###### Note

The `hist()` method relies on Matplotlib, which in
turn relies on a user-specified graphical backend to draw on your
screen. So before you can plot anything, you need to specify which
backend Matplotlib should use. The simplest option is to use Jupyter's
magic command `%matplotlib inline`. This tells Jupyter to set up
Matplotlib so it uses Jupyter's own backend. Plots are then rendered
within the notebook itself. Note that calling `show()` is optional in a
Jupyter notebook, as Jupyter will automatically display plots when a
cell is executed.


![](./images/mls2_0208.png)

There are a few things you might notice in these histograms:

1.  First, the median income attribute does not look like it is
    expressed in US dollars (USD). After checking with the team that
    collected the data, you are told that the data has been scaled and
    capped at 15 (actually, 15.0001) for higher median incomes, and at
    0.5 (actually, 0.4999) for lower median incomes. The numbers
    represent roughly tens of thousands of dollars (e.g., 3 actually
    means about \$30,000). Working with preprocessed attributes is
    common in Machine Learning, and it is not necessarily a problem, but
    you should try to understand how the data was computed.

2.  The housing median age and the median house value were also capped.
    The latter may be a serious problem since it is your target
    attribute (your labels). Your Machine Learning algorithms may learn
    that prices never go beyond that limit. You need to check with your
    client team (the team that will use your system's output) to see if
    this is a problem or not. If they tell you that they need precise
    predictions even beyond \$500,000, then you have two options:

    1.  Collect proper labels for the districts whose labels were
        capped.

    2.  Remove those districts from the training set (and also from the
        test set, since your system should not be evaluated poorly if it
        predicts values beyond \$500,000).

3.  These attributes have very different scales. We will discuss this
    later in this lab, when we explore feature scaling.

4.  Finally, many histograms are *tail-heavy*:
    they extend much farther to the right of the median than to the
    left. This may make it a bit harder for some Machine Learning
    algorithms to detect patterns. We will try transforming these
    attributes later on to have more bell-shaped distributions.

Hopefully you now have a better understanding of the kind of data you
are dealing with.


###### Warning

Wait! Before you look at the data any further, you need to create a test
set, put it aside, and never look at it.





Create a Test Set
-----------------

It may sound strange to
voluntarily set aside part of the data at this stage. After all, you
have only taken a quick glance at the data, and surely you should learn
a whole lot more about it before you decide what algorithms to use,
right? This is true, but your brain is an amazing pattern detection
system, which means that it is highly prone to overfitting: if you look
at the test set, you may stumble upon some seemingly interesting pattern
in the test data that leads you to select a particular kind of Machine
Learning model. When you estimate the generalization error using the
test set, your estimate will be too optimistic, and you will launch a
system that will not perform as well as expected. This is called *data
snooping* bias.

Creating a test set is theoretically simple: pick some instances
randomly, typically 20% of the dataset (or less if your dataset is very
large), and set them aside:

``` {data-type="programlisting" code-language="python"}
import numpy as np

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
```

You can then use this function like
this: ^[13](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch02.html){-marker}^

``` {data-type="programlisting" code-language="pycon"}
>>> train_set, test_set = split_train_test(housing, 0.2)
>>> len(train_set)
16512
>>> len(test_set)
4128
```

Well, this works, but it is not perfect: if you run the program again,
it will generate a different test set! Over time, you (or your Machine
Learning algorithms) will get to see the whole dataset, which is what
you want to avoid.

One solution is to save the test set on the first run and then load it
in subsequent runs. Another option is to set the random number
generator's seed (e.g., with
`np.random.seed(42)`)^[14](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch02.html){-marker}^
before calling `np.random.permutation()` so that it always generates the
same shuffled indices.

But both these solutions will break the next time you fetch an updated
dataset. To have a stable train/test split even after updating the
dataset, a common solution is to use each instance's identifier to
decide whether or not it should go in the test set (assuming instances
have a unique and immutable identifier). For example, you could compute
a hash of each instance's identifier and put that instance in the test
set if the hash is lower than or equal to 20% of the maximum hash value.
This ensures that the test set will remain consistent across multiple
runs, even if you refresh the dataset. The new test set will contain 20%
of the new instances, but it will not contain any instance that was
previously in the training set.

Here is a possible implementation:

``` {data-type="programlisting" code-language="python"}
from zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]
```

Unfortunately, the housing dataset does not have an identifier column.
The simplest solution is to use the row index as the ID:

``` {data-type="programlisting" code-language="python"}
housing_with_id = housing.reset_index()   # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
```

If you use the row index as a unique identifier, you need to make sure
that new data gets appended to the end of the dataset and that no row
ever gets deleted. If this is not possible, then you can try to use the
most stable features to build a unique identifier. For example, a
district's latitude and longitude are guaranteed to be stable for a few
million years, so you could combine them into an ID like
so: ^[15](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch02.html){-marker}^

``` {data-type="programlisting" code-language="python"}
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
```

Scikit-Learn provides a few functions to split
datasets into multiple subsets in various ways. The simplest function is
`train_test_split()`, which does pretty much the same thing as the
function `split_train_test()`, with a couple of additional features.
First, there is a `random_state` parameter that allows you to set the
random generator seed. Second, you can pass it multiple datasets with an
identical number of rows, and it will split them on the same indices
(this is very useful, for example, if you have a separate DataFrame for
labels):

``` {data-type="programlisting" code-language="python"}
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
```

So far we have considered purely random sampling methods. This is
generally fine if your dataset is large enough (especially relative to
the number of attributes), but if it is not, you run the risk of
introducing a significant sampling bias. When a survey company decides
to call 1,000 people to ask them a few questions, they don't just pick
1,000 people randomly in a phone course. They try to ensure that these
1,000 people are representative of the whole population. For example,
the US population is 51.3% females and 48.7% males, so a well-conducted
survey in the US would try to maintain this ratio in the sample: 513
female and 487 male. This is called *stratified
sampling*: the population is divided into homogeneous subgroups called
*strata*, and the right number of instances are sampled from each
stratum to guarantee that the test set is representative of the overall
population. If the people running the survey used purely random
sampling, there would be about a 12% chance of sampling a skewed test
set that was either less than 49% female or more than 54% female. Either
way, the survey results would be significantly biased.

Suppose you chatted with experts who told you that the median income is
a very important attribute to predict median housing prices. You may
want to ensure that the test set is representative of the various
categories of incomes in the whole dataset. Since the median income is a
continuous numerical attribute, you first need to create an income
category attribute. Let's look at the median income histogram more
closely (back in
[Figure 2-8]
most median income values are clustered around 1.5 to 6 (i.e.,
\$15,000--\$60,000), but some median incomes go far beyond 6. It is
important to have a sufficient number of instances in your dataset for
each stratum, or else the estimate of a stratum's importance may be
biased. This means that you should not have too many strata, and each
stratum should be large enough. The following code uses the `pd.cut()`
function to create an income category attribute with five categories
(labeled from 1 to 5): category 1 ranges from 0 to 1.5 (i.e., less than
\$15,000), category 2 from 1.5 to 3, and so on:

``` {data-type="programlisting" code-language="python"}
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
```

These income categories are represented in
[Figure 2-9]

``` {data-type="programlisting" code-language="python"}
housing["income_cat"].hist()
```

![](./images/mls2_0209.png)

Now you are ready to do stratified sampling based
on the income category. For this you can use Scikit-Learn's
`StratifiedShuffleSplit` class:

``` {.pagebreak-before data-type="programlisting" code-language="python"}
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
```

Let's see if this worked as expected. You can start by looking at the
income category proportions in the test set:

``` {data-type="programlisting" code-language="pycon"}
>>> strat_test_set["income_cat"].value_counts() / len(strat_test_set)
3    0.350533
2    0.318798
4    0.176357
5    0.114583
1    0.039729
Name: income_cat, dtype: float64
```

With similar code you can measure the income category proportions in the
full dataset.
[Figure 2-10]
compares the income category proportions in the overall dataset, in the
test set generated with stratified sampling, and in a test set generated
using purely random sampling. As you can see, the test set generated
using stratified sampling has income category proportions almost
identical to those in the full dataset, whereas the test set generated
using purely random sampling is skewed.

![](./images/mls2_0210.png)

Now you should remove the `income_cat` attribute so the data is back to
its original state:

``` {data-type="programlisting" code-language="python"}
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
```

We spent quite a bit of time on test set generation for a good reason:
this is an often neglected but critical part of a Machine Learning
project. Moreover, many of these ideas will be useful later when we
discuss cross-validation. Now it's time to move on to the next stage:
exploring the data. 





Discover and Visualize the Data to Gain Insights
================================================

So far you
have only taken a quick glance at the data to get a general
understanding of the kind of data you are manipulating. Now the goal is
to go into a little more depth.

First, make sure you have put the test set aside and you are only
exploring the training set. Also, if the training set is very large, you
may want to sample an exploration set, to make manipulations easy and
fast. In our case, the set is quite small, so you can just work directly
on the full set. Let's create a copy so that you can play with it
without harming the training set:

``` {data-type="programlisting" code-language="python"}
housing = strat_train_set.copy()
```



Visualizing Geographical Data
-----------------------------

Since there is geographical
information (latitude and longitude), it is a good idea to create a
scatterplot of all districts to visualize the data
([Figure 2-11]

``` {data-type="programlisting" code-language="python"}
housing.plot(kind="scatter", x="longitude", y="latitude")
```

![](./images/mls2_0211.png)

This looks like California all right, but other than that it is hard to
see any particular pattern. Setting the `alpha` option to `0.1` makes it
much easier to visualize the places where there is a high density of
data points
([Figure 2-12]

``` {data-type="programlisting" code-language="python"}
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
```

![](./images/mls2_0212.png)

Now that's much better: you can clearly see the high-density areas,
namely the Bay Area and around Los Angeles and San Diego, plus a long
line of fairly high density in the Central Valley, in particular around
Sacramento and Fresno.

Our brains are very good at spotting patterns in pictures, but you may
need to play around with visualization parameters to make the patterns
stand out.

Now let's look at the housing prices
([Figure 2-13]
The radius of each circle represents the district's population (option
`s`), and the color represents the price (option `c`). We will use a
predefined color map (option `cmap`) called `jet`, which ranges from
blue (low values) to red (high
prices): ^[16](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch02.html){-marker}^

``` {data-type="programlisting" code-language="python"}
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()
```

![](./images/mls2_0213.png)

This image tells you that the housing prices are very much related to
the location (e.g., close to the ocean) and to the population density,
as you probably knew already. A clustering algorithm should be useful
for detecting the main cluster and for adding new features that measure
the proximity to the cluster centers. The ocean proximity attribute may
be useful as well, although in Northern California the housing prices in
coastal districts are not too high, so it is not a simple rule.




Looking for Correlations
------------------------

Since 
the dataset is not too large, you can easily compute the *standard
correlation coefficient* (also called *Pearson's r*) between every pair
of attributes using the `corr()` method:

``` {data-type="programlisting" code-language="python"}
corr_matrix = housing.corr()
```

Now let's look at how much each attribute correlates with the median
house value:

``` {data-type="programlisting" code-language="pycon"}
>>> corr_matrix["median_house_value"].sort_values(ascending=False)
median_house_value    1.000000
median_income         0.687170
total_rooms           0.135231
housing_median_age    0.114220
households            0.064702
total_bedrooms        0.047865
population           -0.026699
longitude            -0.047279
latitude             -0.142826
Name: median_house_value, dtype: float64
```

The correlation coefficient ranges from --1 to 1. When it is close to 1,
it means that there is a strong positive correlation; for example, the
median house value tends to go up when the median income goes up. When
the coefficient is close to --1, it means that there is a strong
negative correlation; you can see a small negative correlation between
the latitude and the median house value (i.e., prices have a slight
tendency to go down when you go north). Finally, coefficients close to 0
mean that there is no linear correlation.
[Figure 2-14]
shows various plots along with the correlation coefficient between their
horizontal and vertical axes.

![](./images/mls2_0214.png)


###### Warning

The correlation coefficient only measures linear correlations ("if *x*
goes up, then *y* generally goes up/down"). It may completely miss out
on nonlinear relationships (e.g., "if *x* is close to 0, then *y*
generally goes up"). Note how all the plots of the bottom row have a
correlation coefficient equal to 0, despite the fact that their axes are
clearly not independent: these are examples of nonlinear relationships.
Also, the second row shows examples where the correlation coefficient is
equal to 1 or --1; notice that this has nothing to do with the slope.
For example, your height in inches has a correlation coefficient of 1
with your height in feet or in nanometers.


Another way to check for correlation between attributes is to use the
pandas [`scatter_matrix()`] function, which plots every
numerical attribute against every other numerical attribute. Since there
are now 11 numerical attributes, you would get 11^2^ = 121 plots, which
would not fit on a page---so let's just focus on a few promising
attributes that seem most correlated with the median housing value
([Figure 2-15]

``` {data-type="programlisting" code-language="python"}
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
```

![](./images/mls2_0215.png)

The main diagonal (top left to bottom right) would be full of straight
lines if pandas plotted each variable against itself, which would not be
very useful. So instead pandas displays a histogram of each attribute
(other options are available; see the pandas documentation for more
details).

The most promising attribute to predict the median house value is the
median income, so let's zoom in on their correlation scatterplot
([Figure 2-16]

``` {data-type="programlisting" code-language="python"}
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
```

![](./images/mls2_0216.png)

This plot reveals a few things. First, the correlation is indeed very
strong; you can clearly see the upward trend, and the points are not too
dispersed. Second, the price cap that we noticed earlier is clearly
visible as a horizontal line at \$500,000. But this plot reveals other
less obvious straight lines: a horizontal line around \$450,000, another
around \$350,000, perhaps one around \$280,000, and a few more below
that. You may want to try removing the corresponding districts to
prevent your algorithms from learning to reproduce these data quirks.




Experimenting with Attribute Combinations
-----------------------------------------

Hopefully the previous sections gave you an idea
of a few ways you can explore the data and gain insights. You identified
a few data quirks that you may want to clean up before feeding the data
to a Machine Learning algorithm, and you found interesting correlations
between attributes, in particular with the target attribute. You also
noticed that some attributes have a tail-heavy distribution, so you may
want to transform them (e.g., by computing their logarithm). Of course,
your mileage will vary considerably with each project, but the general
ideas are similar.

One last thing you may want to do before preparing the data for Machine
Learning algorithms is to try out various attribute combinations. For
example, the total number of rooms in a district is not very useful if
you don't know how many households there are. What you really want is
the number of rooms per household. Similarly, the total number of
bedrooms by itself is not very useful: you probably want to compare it
to the number of rooms. And the population per household also seems like
an interesting attribute combination to look at. Let's create these new
attributes:

``` {data-type="programlisting" code-language="python"}
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
```

And now let's look at the correlation matrix again:

``` {data-type="programlisting" code-language="pycon"}
>>> corr_matrix = housing.corr()
>>> corr_matrix["median_house_value"].sort_values(ascending=False)
median_house_value          1.000000
median_income               0.687160
rooms_per_household         0.146285
total_rooms                 0.135097
housing_median_age          0.114110
households                  0.064506
total_bedrooms              0.047689
population_per_household   -0.021985
population                 -0.026920
longitude                  -0.047432
latitude                   -0.142724
bedrooms_per_room          -0.259984
Name: median_house_value, dtype: float64
```

Hey, not bad! The new `bedrooms_per_room` attribute is much more
correlated with the median house value than the total number of rooms or
bedrooms. Apparently houses with a lower bedroom/room ratio tend to be
more expensive. The number of rooms per household is also more
informative than the total number of rooms in a district---obviously the
larger the houses, the more expensive they are.

This round of exploration does not have to be absolutely thorough; the
point is to start off on the right foot and quickly gain insights that
will help you get a first reasonably good prototype. But this is an
iterative process: once you get a prototype up and running, you can
analyze its output to gain more insights and come back to this
exploration step. 





Prepare the Data for Machine Learning Algorithms
================================================

It's time to prepare the data for
your Machine Learning algorithms. Instead of doing this manually, you
should write functions for this purpose, for several good reasons:

-   This will allow you to reproduce these transformations easily on any
    dataset (e.g., the next time you get a fresh dataset).

-   You will gradually build a library of transformation functions that
    you can reuse in future projects.

-   You can use these functions in your live system to transform the new
    data before feeding it to your algorithms.

-   This will make it possible for you to easily try various
    transformations and see which combination of transformations works
    best.

But first let's revert to a clean training set (by copying
`strat_train_set` once again). Let's also separate the predictors and
the labels, since we don't necessarily want to apply the same
transformations to the predictors and the target values (note that
`drop()` creates a copy of the data and does not affect
`strat_train_set`):

``` {data-type="programlisting" code-language="python"}
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
```



Data Cleaning
-------------

Most Machine Learning algorithms cannot work with
missing features, so let's create a few functions to take care of them.
We saw earlier that the `total_bedrooms` attribute has some missing
values, so let's fix this. You have three options:

1.  Get rid of the corresponding districts.

2.  Get rid of the whole attribute.

3.  Set the values to some value (zero, the mean, the median, etc.).

You can accomplish these easily using DataFrame's `dropna()`, `drop()`,
and `fillna()` methods:

``` {data-type="programlisting" code-language="python"}
housing.dropna(subset=["total_bedrooms"])    # option 1
housing.drop("total_bedrooms", axis=1)       # option 2
median = housing["total_bedrooms"].median()  # option 3
housing["total_bedrooms"].fillna(median, inplace=True)
```

If you choose option 3, you should compute the median value on the
training set and use it to fill the missing values in the training set.
Don't forget to save the median value that you have computed. You will
need it later to replace missing values in the test set when you want to
evaluate your system, and also once the system goes live to replace
missing values in new data.

Scikit-Learn provides a handy class to take care
of missing values: `SimpleImputer`. Here is how to use it. First, you
need to create a `SimpleImputer` instance, specifying that you want to
replace each attribute's missing values with the median of that
attribute:

``` {data-type="programlisting" code-language="python"}
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
```

Since the median can only be computed on numerical attributes, you need
to create a copy of the data without the text attribute
`ocean_proximity`:

``` {data-type="programlisting" code-language="python"}
housing_num = housing.drop("ocean_proximity", axis=1)
```

Now you can fit the `imputer` instance to the training data using the
`fit()` method:

``` {data-type="programlisting" code-language="python"}
imputer.fit(housing_num)
```

The `imputer` has simply computed the median of each attribute and
stored the result in its `statistics_` instance variable. Only the
`total_bedrooms` attribute had missing values, but we cannot be sure
that there won't be any missing values in new data after the system goes
live, so it is safer to apply the `imputer` to all the numerical
attributes:

``` {data-type="programlisting" code-language="pycon"}
>>> imputer.statistics_
array([ -118.51 , 34.26 , 29. , 2119.5 , 433. , 1164. , 408. , 3.5409])
>>> housing_num.median().values
array([ -118.51 , 34.26 , 29. , 2119.5 , 433. , 1164. , 408. , 3.5409])
```

Now you can use this "trained" `imputer` to transform the training set
by replacing missing values with the learned medians:

``` {data-type="programlisting" code-language="python"}
X = imputer.transform(housing_num)
```

The result is a plain NumPy array containing the transformed features.
If you want to put it back into a pandas DataFrame, it's simple:

``` {data-type="programlisting" code-language="python"}
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)
```


##### Scikit-Learn Design

Scikit-Learn's API is remarkably well designed.
These are the [main design
principles](https://homl.info/11): ^[17](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch02.html){-marker}^

Consistency

:   All objects share a consistent and simple interface:

    Estimators

    :   Any object that can estimate some
        parameters based on a dataset is called an *estimator* (e.g., an
        `imputer` is an estimator). The estimation itself is performed
        by the `fit()` method, and it takes only a dataset as a
        parameter (or two for supervised learning algorithms; the second
        dataset contains the labels). Any other parameter needed to
        guide the estimation process is considered a hyperparameter
        (such as an `imputer`'s `strategy`), and it must be set as an
        instance variable (generally via a constructor parameter).

    Transformers

    :   Some estimators (such as an `imputer`) can
        also transform a dataset; these are called *transformers*. Once
        again, the API is simple: the transformation is
        [performed] by the `transform()` method with the
        dataset to transform as a parameter. It returns the transformed
        dataset. This transformation generally relies on the learned
        parameters, as is the case for an `imputer`. All transformers
        also have a convenience method called `fit_transform()` that is
        equivalent to calling `fit()` and then `transform()` (but
        sometimes `fit_transform()` is optimized and runs much faster).

    Predictors

    :   Finally, some estimators, given a dataset,
        are capable of making predictions; they are called *predictors*.
        For example, the `LinearRegression` model in the previous
        lab was a predictor: given a country's GDP per capita, it
        predicted life satisfaction. A predictor has a `predict()`
        method that takes a dataset of new instances and returns a
        dataset of corresponding predictions. It also has a `score()`
        method that measures the quality of the predictions, given a
        test set (and the corresponding labels, in the case of
        supervised learning
        algorithms).
^[18](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch02.html){-marker}^

Inspection

:   All the estimator's hyperparameters are accessible directly via
    public instance variables (e.g., `imputer.strategy`), and all the
    estimator's learned parameters are accessible via public instance
    variables with an underscore suffix (e.g., `imputer.statistics_`).

Nonproliferation of classes

:   Datasets are represented as NumPy arrays or SciPy sparse matrices,
    instead of homemade classes. Hyperparameters are just regular Python
    strings or numbers.

Composition

:   Existing building blocks are reused as much as possible. For
    example, it is easy to create a `Pipeline` estimator from an
    arbitrary sequence of transformers followed by a final estimator, as
    we will see.

Sensible defaults

:   Scikit-Learn provides reasonable default values for most parameters,
    making it easy to quickly create a baseline working system.





Handling Text and Categorical Attributes
----------------------------------------

So far we have only dealt with numerical
attributes, but now let's look at text attributes. In this dataset,
there is just one: the `ocean_proximity` attribute. Let's look at its
value for the first 10 instances:

``` {.pagebreak-before data-type="programlisting" code-language="pycon"}
>>> housing_cat = housing[["ocean_proximity"]]
>>> housing_cat.head(10)
      ocean_proximity
17606       <1H OCEAN
18632       <1H OCEAN
14650      NEAR OCEAN
3230           INLAND
3555        <1H OCEAN
19480          INLAND
8879        <1H OCEAN
13685          INLAND
4937        <1H OCEAN
4861        <1H OCEAN
```

It's not arbitrary text: there are a limited number of possible values,
each of which represents a category. So this attribute is a categorical
attribute. Most Machine Learning algorithms prefer to work with numbers,
so let's convert these categories from text to numbers. For this, we
can use Scikit-Learn's `OrdinalEncoder`
class: ^[19](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch02.html){-marker}^

``` {data-type="programlisting" code-language="pycon"}
>>> from sklearn.preprocessing import OrdinalEncoder
>>> ordinal_encoder = OrdinalEncoder()
>>> housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
>>> housing_cat_encoded[:10]
array([[0.],
       [0.],
       [4.],
       [1.],
       [0.],
       [1.],
       [0.],
       [1.],
       [0.],
       [0.]])
```

You can get the list of categories using the `categories_` instance
variable. It is a list containing a 1D array of categories for each
categorical attribute (in this case, a list containing a single array
since there is just one categorical attribute):

``` {data-type="programlisting" code-language="pycon"}
>>> ordinal_encoder.categories_
[array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
       dtype=object)]
```

One issue with this representation is that ML algorithms will assume
that two nearby values are more similar than two distant values. This
may be fine in some cases (e.g., for ordered categories such as "bad,"
"average," "good," and "excellent"), but it is obviously not the case
for the `ocean_proximity` column (for example, categories 0 and 4 are
clearly more similar than categories 0 and 1). To fix this issue, a
common solution is to create one binary attribute per category: one
attribute equal to 1 when the category is "\<1H OCEAN" (and 0
otherwise), another attribute equal to 1 when the category is "INLAND"
(and 0 otherwise), and so on. This is called
*one-hot encoding*, because only one attribute will be equal to 1 (hot),
while the others will be 0 (cold). The new attributes are sometimes
called *dummy* attributes.
Scikit-Learn provides a `OneHotEncoder` class to
convert categorical values into one-hot
vectors: ^[20](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch02.html){-marker}^

``` {data-type="programlisting" code-language="pycon"}
>>> from sklearn.preprocessing import OneHotEncoder
>>> cat_encoder = OneHotEncoder()
>>> housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
>>> housing_cat_1hot
<16512x5 sparse matrix of type '<class 'numpy.float64'>'
  with 16512 stored elements in Compressed Sparse Row format>
```

Notice that the output is a SciPy *sparse matrix*,
instead of a NumPy array. This is very useful when you have categorical
attributes with thousands of categories. After one-hot encoding, we get
a matrix with thousands of columns, and the matrix is full of 0s except
for a single 1 per row. Using up tons of memory mostly to store zeros
would be very wasteful, so instead a sparse matrix only stores the
location of the nonzero elements. You can use it mostly like a normal 2D
array,^[21](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch02.html){-marker}^
but if you really want to convert it to a (dense)
NumPy array, just call the `toarray()` method:

``` {data-type="programlisting" code-language="pycon"}
>>> housing_cat_1hot.toarray()
array([[1., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1.],
       ...,
       [0., 1., 0., 0., 0.],
       [1., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0.]])
```

Once again, you can get the list of categories using the encoder's
`categories_` instance variable:

``` {data-type="programlisting" code-language="pycon"}
>>> cat_encoder.categories_
[array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
       dtype=object)]
```


###### Tip

If a categorical attribute has a large number of possible categories
(e.g., country code, profession, species), then one-hot encoding will
result in a large number of input features. This may slow down training
and degrade performance. If this happens, you may want to replace the
categorical input with useful numerical features related to the
categories: for example, you could replace the `ocean_proximity` feature
with the distance to the ocean (similarly, a country code could be
replaced with the country's population and GDP per capita).
Alternatively, you could replace each category with a learnable,
low-dimensional vector called an *embedding*. Each
category's representation would be learned during training.
This is an example of *representation learning*
(see Labs
[13]
and
[17]
for more details).





Custom Transformers
-------------------

Although Scikit-Learn
provides many useful
transformers, you will need to write your own for tasks such as custom
cleanup operations or combining specific attributes.
You will want your transformer to work seamlessly
with Scikit-Learn functionalities (such as pipelines), and since
Scikit-Learn relies on duck typing (not
inheritance), all you need to do is create a class and implement three
methods: `fit()` (returning `self`), `transform()`, and
`fit_transform()`.

You can get the last one for free by simply adding `TransformerMixin` as
a base class. If you add `BaseEstimator` as a base class (and avoid
`*args` and `**kargs` in your constructor), you will also get two extra
methods (`get_params()` and `set_params()`) that will be useful for
automatic hyperparameter tuning.

For example, here is a small transformer class that adds the combined
attributes we discussed earlier:

``` {data-type="programlisting" code-language="python"}
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]

        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
```

In this example the transformer has one hyperparameter,
`add_bedrooms_per_room`, set to `True` by default (it is often helpful
to provide sensible defaults). This hyperparameter will allow you to
easily find out whether adding this attribute helps the Machine Learning
algorithms or not. More generally, you can add a hyperparameter to gate
any data preparation step that you are not 100% sure about. The more you
automate these data preparation steps, the more combinations you can
automatically try out, making it much more likely that you will find a
great combination (and saving you a lot of time).




Feature Scaling
---------------

One of the most important
transformations you need to apply to your data is *feature scaling*.
With few exceptions, Machine Learning algorithms don't perform well when
the input numerical attributes have very different scales. This is the
case for the housing data: the total number of rooms ranges from about 6
to 39,320, while the median incomes only range from 0 to 15. Note that
scaling the target values is generally not required.

There are two common ways to
get all attributes to have the same scale: *min-max scaling* and
*standardization*.

Min-max scaling (many people call this
*normalization*) is the simplest: values are shifted and rescaled so
that they end up ranging from 0 to 1. We do this by subtracting the min
value and dividing by the max minus the min. Scikit-Learn provides a
transformer called `MinMaxScaler` for this. It has a `feature_range`
hyperparameter that lets you change the range if, for some reason, you
don't want 0--1.

Standardization is different: first it subtracts the mean value (so
standardized values always have a zero mean), and then it divides by the
standard deviation so that the resulting distribution has unit variance.
Unlike min-max scaling, standardization does not bound values to a
specific range, which may be a problem for some algorithms (e.g., neural
networks often expect an input value ranging from 0 to 1). However,
standardization is much less affected by outliers. For example, suppose
a district had a median income equal to 100 (by mistake). Min-max
scaling would then crush all the other values from 0--15 down to
0--0.15, whereas standardization would not be much affected.
Scikit-Learn provides a transformer called `StandardScaler` for
[standardization].


###### Warning

As with all the transformations, it is important to fit the scalers to
the training data only, not to the full dataset (including the test
set). Only then can you use them to transform the training set and the
test set (and new data).





Transformation Pipelines
------------------------

As you
can see, there are many data transformation steps that need to be
executed in the right order. Fortunately, Scikit-Learn provides the
`Pipeline` class to help with such sequences of transformations. Here is
a small pipeline for the numerical attributes:

``` {data-type="programlisting" code-language="python"}
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)
```

The `Pipeline` constructor takes a list of name/estimator pairs defining
a sequence of steps. All but the last estimator must be transformers
(i.e., they must have a `fit_transform()` method). The names can be
anything you like (as long as they are unique and don't contain double
underscores, `__`); they will come in handy later for hyperparameter
tuning.

When you call the pipeline's `fit()` method, it calls `fit_transform()`
sequentially on all transformers, passing the output of each call as the
parameter to the next call until it reaches the final estimator, for
which it calls the `fit()` method.

The pipeline exposes the same methods as the final estimator. In this
example, the last estimator is a `StandardScaler`, which is a
transformer, so the pipeline has a `transform()` method that applies all
the transforms to the data in sequence (and of course also a
`fit_transform()` method, which is the one we used).

So far, we have handled the categorical columns and the numerical
columns separately. It would be more convenient to have a single
transformer able to handle all columns, applying the appropriate
transformations to each column. In version 0.20, Scikit-Learn introduced
the `ColumnTransformer` for this purpose, and the good news is that it
works great with pandas DataFrames. Let's use it to apply all the
transformations to the housing data:

``` {.pagebreak-before data-type="programlisting" code-language="python"}
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
```

First we import the `ColumnTransformer` class, next we get the list of
numerical column names and the list of categorical column names, and
then we construct a `ColumnTransformer`. The constructor requires a list
of tuples, where each tuple contains a
name,^[22](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch02.html){-marker}^
a transformer, and a list of names (or indices) of columns that the
transformer should be applied to. In this example, we specify that the
numerical columns should be transformed using the `num_pipeline` that we
defined earlier, and the categorical columns should be transformed using
a `OneHotEncoder`. Finally, we apply this `ColumnTransformer` to the
housing data: it applies each transformer to the appropriate columns and
concatenates the outputs along the second axis (the transformers must
return the same number of rows).

Note that the `OneHotEncoder` returns a sparse matrix, while the
`num_pipeline` returns a dense matrix. When there is such a mix of
sparse and dense matrices, the `ColumnTransformer` estimates the density
of the final matrix (i.e., the ratio of nonzero cells), and it returns a
sparse matrix if the density is lower than a given threshold (by
default, `sparse_threshold=0.3`). In this example, it returns a dense
matrix. And that's it! We have a preprocessing pipeline that takes the
full housing data and applies the appropriate transformations to each
column.


###### Tip

Instead of using a transformer, you can specify the string `"drop"` if
you want the columns to be dropped, or you can specify `"passthrough"`
if you want the columns to be left untouched. By default, the remaining
columns (i.e., the ones that were not listed) will be dropped, but you
can set the `remainder` hyperparameter to any transformer (or to
`"passthrough"`) if you want these columns to be handled differently.


If you are using Scikit-Learn 0.19 or earlier, you can use a third-party
library such as `sklearn-pandas`, or you can roll out your own custom
transformer to get the same functionality as the `ColumnTransformer`.
Alternatively, you can use the `FeatureUnion` class, which can apply
different transformers and concatenate their outputs. But you cannot
specify different columns for each transformer; they all apply to the
whole data. It is possible to work around this limitation using a custom
transformer for column selection (see the Jupyter notebook for an
example). 





Select and Train a Model
========================

At last!
You 
framed the problem, you got the data and explored it, you sampled a
training set and a test set, and you wrote transformation pipelines to
clean up and prepare your data for Machine Learning algorithms
automatically. You are now ready to select and train a Machine Learning
model.



Training and Evaluating on the Training Set
-------------------------------------------

The good news is that thanks to all these previous steps, things are now
going to be much simpler than you might think. Let's first train a
Linear Regression model, like we did in the previous lab:

``` {data-type="programlisting" code-language="python"}
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
```

Done! You now have a working Linear Regression model. Let's try it out
on a few instances from the training set:

``` {data-type="programlisting" code-language="pycon"}
>>> some_data = housing.iloc[:5]
>>> some_labels = housing_labels.iloc[:5]
>>> some_data_prepared = full_pipeline.transform(some_data)
>>> print("Predictions:", lin_reg.predict(some_data_prepared))
Predictions: [ 210644.6045  317768.8069  210956.4333  59218.9888  189747.5584]
>>> print("Labels:", list(some_labels))
Labels: [286600.0, 340600.0, 196900.0, 46300.0, 254500.0]
```

It works, although the predictions are not exactly accurate (e.g., the
first prediction is off by close to 40%!). Let's measure this regression
model's RMSE on the whole training set using 
Scikit-Learn's `mean_squared_error()` function:

``` {data-type="programlisting" code-language="pycon"}
>>> from sklearn.metrics import mean_squared_error
>>> housing_predictions = lin_reg.predict(housing_prepared)
>>> lin_mse = mean_squared_error(housing_labels, housing_predictions)
>>> lin_rmse = np.sqrt(lin_mse)
>>> lin_rmse
68628.19819848922
```

This is better than nothing, but clearly not a great score: most
districts' `median_housing_values` range between \$120,000 and
\$265,000, so a typical prediction error of \$68,628 is not very
satisfying. This is an example of a model underfitting the training
data. When this happens it can mean that the features do not provide
enough [information] to make good predictions, or that
the model is not powerful enough. As we saw in the previous lab, the
main ways to fix underfitting are to select a more powerful model, to
feed the training algorithm with better features, or to reduce the
constraints on the model. This model is not regularized, which rules out
the last option. You could try to add more features (e.g., the log of
the population), but first let's try a more complex model to see how it
does.

Let's train a `DecisionTreeRegressor`. This is a powerful model, capable
of finding complex nonlinear relationships in the data (Decision Trees
are presented in more detail in
[Lab 6]
The code should look familiar by now:

``` {data-type="programlisting" code-language="python"}
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
```

Now that the model is trained, let's evaluate it on the training set:

``` {data-type="programlisting" code-language="pycon"}
>>> housing_predictions = tree_reg.predict(housing_prepared)
>>> tree_mse = mean_squared_error(housing_labels, housing_predictions)
>>> tree_rmse = np.sqrt(tree_mse)
>>> tree_rmse
0.0
```

Wait, what!? No error at all? Could this model really be absolutely
perfect? Of course, it is much more likely that the model has badly
overfit the data. How can you be sure? As we saw earlier, you don't want
to touch the test set until you are ready to launch a model you are
confident about, so you need to use part of the training set for
training and part of it for model validation.




Better Evaluation Using Cross-Validation
----------------------------------------

One way to evaluate the
Decision Tree model would be to use the `train_test_split()` function to
split the training set into a smaller training set and a validation set,
then train your models against the smaller training set and evaluate
them against the validation set. It's a bit of work, but nothing too
difficult, and it would work fairly well.

A 
great alternative is to use Scikit-Learn's *K-fold cross-validation*
feature. The following code randomly splits the training set into 10
distinct subsets called *folds*, then it trains and evaluates the
Decision Tree model 10 times, picking a different fold for evaluation
every time and training on the other 9 folds. The result is an array
containing the 10 evaluation scores:

``` {data-type="programlisting" code-language="python"}
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
```


###### Warning

Scikit-Learn's cross-validation features expect a utility function
(greater is better) rather than a cost function (lower is better), so
the scoring function is actually the opposite of the MSE (i.e., a
negative value), which is why the preceding code computes `-scores`
before calculating the square root.


Let's look at the results:

``` {data-type="programlisting" code-language="pycon"}
>>> def display_scores(scores):
...     print("Scores:", scores)
...     print("Mean:", scores.mean())
...     print("Standard deviation:", scores.std())
...
>>> display_scores(tree_rmse_scores)
Scores: [70194.33680785 66855.16363941 72432.58244769 70758.73896782
 71115.88230639 75585.14172901 70262.86139133 70273.6325285
 75366.87952553 71231.65726027]
Mean: 71407.68766037929
Standard deviation: 2439.4345041191004
```

Now the Decision Tree doesn't look as good as it did earlier. In fact,
it seems to perform worse than the Linear Regression model! Notice that
cross-validation allows you to get not only an estimate of the
performance of your model, but also a measure of how precise this
estimate is (i.e., its standard deviation). The Decision Tree has a
score of approximately 71,407, generally ±2,439. You would not have this
information if you just used one validation set. But cross-validation
comes at the cost of training the model several times, so it is not
always possible.

Let's compute the same scores for the Linear Regression model just to be
sure:

``` {data-type="programlisting" code-language="pycon"}
>>> lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
...                              scoring="neg_mean_squared_error", cv=10)
...
>>> lin_rmse_scores = np.sqrt(-lin_scores)
>>> display_scores(lin_rmse_scores)
Scores: [66782.73843989 66960.118071   70347.95244419 74739.57052552
 68031.13388938 71193.84183426 64969.63056405 68281.61137997
 71552.91566558 67665.10082067]
Mean: 69052.46136345083
Standard deviation: 2731.674001798348
```

That's right: the Decision Tree model is overfitting so badly that it
performs worse than the Linear Regression model.

Let's try one last model now: the `RandomForestRegressor`. As we will
see in
[Lab 7]
Random Forests work by training many Decision Trees on random subsets of
the features, then averaging out their predictions. Building
a model on top of many other models is called
*Ensemble Learning*, and it is often a great way to push ML algorithms
even further. We will skip most of the code since it is essentially the
same as for the other models:

``` {data-type="programlisting" code-language="pycon"}
>>> from sklearn.ensemble import RandomForestRegressor
>>> forest_reg = RandomForestRegressor()
>>> forest_reg.fit(housing_prepared, housing_labels)
>>> [...]
>>> forest_rmse
18603.515021376355
>>> display_scores(forest_rmse_scores)
Scores: [49519.80364233 47461.9115823  50029.02762854 52325.28068953
 49308.39426421 53446.37892622 48634.8036574  47585.73832311
 53490.10699751 50021.5852922 ]
Mean: 50182.303100336096
Standard deviation: 2097.0810550985693
```

Wow, this is much better: Random Forests look very promising. However,
note that the score on the training set is still much lower than on the
validation sets, meaning that the model is still overfitting the
training set. Possible solutions for overfitting are to simplify the
model, constrain it (i.e., regularize it), or get a lot more training
data. Before you dive much deeper into Random Forests, however, you
should try out many other models from various categories of Machine
Learning algorithms (e.g., several Support Vector Machines with
different kernels, and possibly a neural network), without spending too
much time tweaking the hyperparameters. The goal is to shortlist a few
(two to five) promising models.


###### Tip

You should save every model you experiment with so that you can come
back easily to any model you want. Make sure you save both the
hyperparameters and the trained parameters, as well as the
cross-validation scores and perhaps the actual predictions as well. This
will allow you to easily compare scores across model types, and compare
the types of errors they make. You can easily save
Scikit-Learn models by using Python's `pickle` module or by using the
`joblib` library, which is more efficient at
serializing large NumPy arrays (you can install this library using pip):

``` {data-type="programlisting" code-language="python"}
import joblib

joblib.dump(my_model, "my_model.pkl")
# and later...
my_model_loaded = joblib.load("my_model.pkl")
```






Fine-Tune Your Model
====================

Let's assume that you now
have a shortlist of promising models. You now need to fine-tune them.
Let's look at a few ways you can do that.



Grid Search
-----------

One option would be to fiddle with the hyperparameters manually, until
you find a great combination of hyperparameter values. This would be
very tedious work, and you may not have time to explore many
combinations.

Instead, you should get Scikit-Learn's
`GridSearchCV` to search for you. All you need to do is tell it which
hyperparameters you want it to experiment with and what values to try
out, and it will use cross-validation to evaluate all the possible
combinations of hyperparameter values. For example, the following code
searches for the best combination of hyperparameter values for the
`RandomForestRegressor`:

``` {data-type="programlisting" code-language="python"}
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)
```


###### Tip

When you have no idea what value a hyperparameter should have, a simple
approach is to try out consecutive powers of 10 (or a smaller number if
you want a more fine-grained search, as shown in this example with the
`n_estimators` hyperparameter).


This `param_grid` tells Scikit-Learn to first evaluate all 3 × 4 = 12
combinations of `n_estimators` and `max_features` hyperparameter values
specified in the first `dict` (don't worry about what these
hyperparameters mean for now; they will be explained in
[Lab 7]
then try all 2 × 3 = 6 combinations of hyperparameter values in the
second `dict`, but this time with the `bootstrap` hyperparameter set to
`False` instead of `True` (which is the default value for this
hyperparameter).

The grid search will explore 12 + 6 = 18 combinations of
`RandomForestRegressor` hyperparameter values, and it will train each
model 5 times (since we are using five-fold cross validation). In other
words, all in all, there will be 18 × 5 = 90 rounds of training! It may
take quite a long time, but when it is done you can get the best
combination of parameters like this:

``` {data-type="programlisting" code-language="pycon"}
>>> grid_search.best_params_
{'max_features': 8, 'n_estimators': 30}
```


###### Tip

Since 8 and 30 are the maximum values that were evaluated, you should
probably try searching again with higher values; the score may continue
to improve.


You can also get the best estimator directly:

``` {data-type="programlisting" code-language="pycon"}
>>> grid_search.best_estimator_
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features=8, max_leaf_nodes=None, min_impurity_decrease=0.0,
           min_impurity_split=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=30, n_jobs=None, oob_score=False, random_state=None,
           verbose=0, warm_start=False)
```


###### Note

If `GridSearchCV` is initialized with `refit=True` (which is the
default), then once it finds the best estimator using cross-validation,
it retrains it on the whole training set. This is usually a good idea,
since feeding it more data will likely improve its
[performance].


And of course the evaluation scores are also available:

``` {data-type="programlisting" code-language="pycon"}
>>> cvres = grid_search.cv_results_
>>> for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
...     print(np.sqrt(-mean_score), params)
...
63669.05791727153 {'max_features': 2, 'n_estimators': 3}
55627.16171305252 {'max_features': 2, 'n_estimators': 10}
53384.57867637289 {'max_features': 2, 'n_estimators': 30}
60965.99185930139 {'max_features': 4, 'n_estimators': 3}
52740.98248528835 {'max_features': 4, 'n_estimators': 10}
50377.344409590376 {'max_features': 4, 'n_estimators': 30}
58663.84733372485 {'max_features': 6, 'n_estimators': 3}
52006.15355973719 {'max_features': 6, 'n_estimators': 10}
50146.465964159885 {'max_features': 6, 'n_estimators': 30}
57869.25504027614 {'max_features': 8, 'n_estimators': 3}
51711.09443660957 {'max_features': 8, 'n_estimators': 10}
49682.25345942335 {'max_features': 8, 'n_estimators': 30}
62895.088889905004 {'bootstrap': False, 'max_features': 2, 'n_estimators': 3}
54658.14484390074 {'bootstrap': False, 'max_features': 2, 'n_estimators': 10}
59470.399594730654 {'bootstrap': False, 'max_features': 3, 'n_estimators': 3}
52725.01091081235 {'bootstrap': False, 'max_features': 3, 'n_estimators': 10}
57490.612956065226 {'bootstrap': False, 'max_features': 4, 'n_estimators': 3}
51009.51445842374 {'bootstrap': False, 'max_features': 4, 'n_estimators': 10}
```

In this example, we obtain the best solution by setting the
`max_features` hyperparameter to `8` and the `n_estimators`
hyperparameter to `30`. The RMSE score for this combination is 49,682,
which is slightly better than the score you got earlier using the
default hyperparameter values (which was 50,182). Congratulations, you
have successfully fine-tuned your best model!


###### Tip

Don't forget that you can treat some of the data preparation steps as
hyperparameters. For example, the grid search will automatically find
out whether or not to add a feature you were not sure about (e.g., using
the `add_bedrooms_per_room` hyperparameter of your
`CombinedAttributesAdder` transformer). It may similarly be used to
automatically find the best way to handle outliers, missing features,
feature selection, and more.





Randomized Search
-----------------

The grid search approach is fine when you are exploring relatively few
combinations, like in the previous example, but when the hyperparameter
search space is large, it is often preferable to use
`RandomizedSearchCV` instead. This class can be used in much the same
way as the `GridSearchCV` class, but instead of trying out all possible
combinations, it evaluates a given number of random combinations by
selecting a random value for each hyperparameter at every iteration.
This approach has two main [benefits]:

-   If you let the randomized search run for, say, 1,000 iterations,
    this approach will explore 1,000 different values for each
    hyperparameter (instead of just a few values per hyperparameter with
    the grid search approach).

-   Simply by setting the number of iterations, you have more control
    over the computing budget you want to allocate to hyperparameter
    search.




Ensemble Methods
----------------

Another way to fine-tune your system is to try to combine the models
that perform best. The group (or "ensemble") will often perform better
than the best individual model (just like Random Forests perform better
than the individual Decision Trees they rely on), especially if the
individual models make very different types of errors. We will cover
this topic in more detail in
[Lab 7]




Analyze the Best Models and Their Errors
----------------------------------------

You will often gain good insights on the problem by inspecting the best
models. For example, the `RandomForestRegressor` can indicate the
relative importance of each attribute for making accurate predictions:

``` {data-type="programlisting" code-language="pycon"}
>>> feature_importances = grid_search.best_estimator_.feature_importances_
>>> feature_importances
array([7.33442355e-02, 6.29090705e-02, 4.11437985e-02, 1.46726854e-02,
       1.41064835e-02, 1.48742809e-02, 1.42575993e-02, 3.66158981e-01,
       5.64191792e-02, 1.08792957e-01, 5.33510773e-02, 1.03114883e-02,
       1.64780994e-01, 6.02803867e-05, 1.96041560e-03, 2.85647464e-03])
```

Let's display these importance scores next to their corresponding
attribute names:

``` {data-type="programlisting" code-language="pycon"}
>>> extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
>>> cat_encoder = full_pipeline.named_transformers_["cat"]
>>> cat_one_hot_attribs = list(cat_encoder.categories_[0])
>>> attributes = num_attribs + extra_attribs + cat_one_hot_attribs
>>> sorted(zip(feature_importances, attributes), reverse=True)
[(0.3661589806181342, 'median_income'),
 (0.1647809935615905, 'INLAND'),
 (0.10879295677551573, 'pop_per_hhold'),
 (0.07334423551601242, 'longitude'),
 (0.0629090704826203, 'latitude'),
 (0.05641917918195401, 'rooms_per_hhold'),
 (0.05335107734767581, 'bedrooms_per_room'),
 (0.041143798478729635, 'housing_median_age'),
 (0.014874280890402767, 'population'),
 (0.014672685420543237, 'total_rooms'),
 (0.014257599323407807, 'households'),
 (0.014106483453584102, 'total_bedrooms'),
 (0.010311488326303787, '<1H OCEAN'),
 (0.002856474637320158, 'NEAR OCEAN'),
 (0.00196041559947807, 'NEAR BAY'),
 (6.028038672736599e-05, 'ISLAND')]
```

With this information, you may want to try dropping some of the less
useful features (e.g., apparently only one `ocean_proximity` category is
really useful, so you could try dropping the others).

You should also look at the specific errors that your system makes, then
try to understand why it makes them and what could fix the problem
(adding extra features or getting rid of uninformative ones, cleaning up
outliers, etc.).




Evaluate Your System on the Test Set
------------------------------------

After tweaking your models for a while, you eventually have a system
that performs sufficiently well. Now is the time to evaluate the final
model on the test set. There is nothing special about this process; just
get the predictors and the labels from your test set, run your
`full_pipeline` to transform the data (call `transform()`, *not*
`fit_transform()`---you do not want to fit the test set!), and evaluate
the final model on the test set:

``` {data-type="programlisting" code-language="python"}
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)   # => evaluates to 47,730.2
```

In some cases, such a point estimate of the generalization error will
not be quite enough to convince you to launch: what if it is just 0.1%
better than the model currently in production? You might want to have an
idea of how precise this estimate is. For this, you can compute a 95%
*confidence interval* for the generalization error using
`scipy.stats.t.interval()`:

``` {data-type="programlisting" code-language="pycon"}
>>> from scipy import stats
>>> confidence = 0.95
>>> squared_errors = (final_predictions - y_test) ** 2
>>> np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
...                          loc=squared_errors.mean(),
...                          scale=stats.sem(squared_errors)))
...
array([45685.10470776, 49691.25001878])
```

If you did a lot of hyperparameter tuning, the performance will usually
be slightly worse than what you measured using cross-validation (because
your system ends up fine-tuned to perform well on the validation data
and will likely not perform as well on unknown datasets). It is not the
case in this example, but when this happens you must resist the
temptation to tweak the hyperparameters to make the numbers look good on
the test set; the improvements would be unlikely to generalize to new
data.

Now comes the project prelaunch phase: you need to present your solution
(highlighting what you have learned, what worked and what did not, what
assumptions were made, and what your system's limitations are), document
everything, and create nice presentations with clear visualizations and
easy-to-remember statements (e.g., "the median income is the number one
predictor of housing prices"). In this California housing example, the
final performance of the system is not better than the experts' price
estimates, which were often off by about 20%, but it may still be a good
idea to launch it, especially if this frees up some time for the experts
so they can work on more interesting and productive
tasks. 





Launch, Monitor, and Maintain Your System
=========================================

Perfect, you got approval to
launch! You now need to get your solution ready for production (e.g.,
polish the code, write documentation and tests, and so on). Then you can
deploy your model to your production environment. One way to do this is
to save the trained Scikit-Learn model (e.g., using `joblib`), including
the full preprocessing and prediction pipeline, then load this trained
model within your production environment and use it to make predictions
by calling its `predict()` method. For example, perhaps the model will
be used within a website: the user will type in some data about a new
district and click the Estimate Price button. This will send a query
containing the data to the web server, which will forward it to your web
application, and finally your code will simply call the model's
`predict()` method (you want to load the model upon server startup,
rather than every time the model is used). Alternatively, you can wrap
the model within a dedicated web service that your web application can
query through a REST
API^[23](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch02.html){-marker}^
(see
[Figure 2-17]
This makes it easier to upgrade your model to new versions without
interrupting the main application. It also simplifies scaling, since you
can start as many web services as needed and load-balance the requests
coming from your web application across these web services. Moreover, it
allows your web application to use any language, not just Python.

![](./images/mls2_0217.png)

Another popular strategy is to deploy your model on the cloud, for
example on Google Cloud AI Platform (formerly known as Google Cloud ML
Engine): just save your model using `joblib` and upload it to Google
Cloud Storage (GCS), then head over to Google Cloud AI Platform and
create a new model version, pointing it to the GCS file. That's it! This
gives you a simple web service that takes care of load balancing and
scaling for you. It take JSON requests containing the input data (e.g.,
of a district) and returns JSON responses containing the predictions.
You can then use this web service in your website (or whatever
production environment you are using). As we will see in
[Lab 19]
deploying TensorFlow models on AI Platform is not
much different from deploying Scikit-Learn models.

But deployment is not the end of the story. You also need to write
monitoring code to check your system's live performance at regular
intervals and trigger alerts when it drops. This could be a steep drop,
likely due to a broken component in your infrastructure, but be aware
that it could also be a gentle decay that could easily go unnoticed for
a long time. This is quite common because models tend to "rot" over
time: indeed, the world changes, so if the model was trained with last
year's data, it may not be adapted to today's data.


###### Warning

Even a model trained to classify pictures of cats and dogs may need to
be retrained regularly, not because cats and dogs will mutate overnight,
but because cameras keep changing, along with image formats, sharpness,
brightness, and size ratios. Moreover, people may love different breeds
next year, or they may decide to dress their pets with tiny hats---who
knows?


So you need to monitor your model's live performance. But how do you
that? Well, it depends. In some cases, the model's performance can be
inferred from downstream metrics. For example, if your model is part of
a recommender system and it suggests products that the users may be
interested in, then it's easy to monitor the number of recommended
products sold each day. If this number drops (compared to
non-recommended products), then the prime suspect is the model. This may
be because the data pipeline is broken, or perhaps the model needs to be
retrained on fresh data (as we will discuss shortly).

However, it's not always possible to determine the model's performance
without any human analysis. For example, suppose you trained an image
classification model (see
[Lab 3]
to detect several product defects on a production line. How can you get
an alert if the model's performance drops, before thousands of defective
products get shipped to your clients? One solution is to send to human
raters a sample of all the pictures that the model classified
(especially pictures that the model wasn't so sure about). Depending on
the task, the raters may need to be experts, or they could be
nonspecialists, such as workers on a crowdsourcing platform (e.g.,
Amazon Mechanical Turk). In some applications they could even be the
users themselves, responding for example via surveys or repurposed
captchas.
^[24](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch02.html){-marker}^

Either way, you need to put in place a monitoring system (with or
without human raters to evaluate the live model), as well as all the
relevant processes to define what to do in case of failures and how to
prepare for them. Unfortunately, this can be a lot of work. In fact, it
is often much more work than building and training a model.

If the data keeps evolving, you will need to update your datasets and
retrain your model regularly. You should probably automate the whole
process as much as possible. Here are a few things you can automate:

-   Collect fresh data regularly and label it (e.g., using human
    raters).

-   Write a script to train the model and fine-tune the hyperparameters
    automatically. This script could run automatically, for example
    every day or every week, depending on your needs.

-   Write another script that will evaluate both the new model and the
    previous model on the updated test set, and deploy the model to
    production if the performance has not decreased (if it did, make
    sure you investigate why).

You should also make sure you evaluate the model's input data quality.
Sometimes performance will degrade slightly because of a poor-quality
signal (e.g., a malfunctioning sensor sending random values, or another
team's output becoming stale), but it may take a while before your
system's performance degrades enough to trigger an alert. If you monitor
your model's inputs, you may catch this earlier. For example, you could
trigger an alert if more and more inputs are missing a feature, or if
its mean or standard deviation drifts too far from the training set, or
a categorical feature starts containing new categories.

Finally, make sure you keep backups of every model you create and have
the process and tools in place to roll back to a previous model quickly,
in case the new model starts failing badly for some reason. Having
backups also makes it possible to easily compare new models with
previous ones. Similarly, you should keep backups of every version of
your datasets so that you can roll back to a previous dataset if the new
one ever gets corrupted (e.g., if the fresh data that gets added to it
turns out to be full of outliers). Having backups of your datasets also
allows you to evaluate any model against any previous dataset.


###### Tip

You may want to create several subsets of the test set in order to
evaluate how well your model performs on specific parts of the data. For
example, you may want to have a subset containing only the most recent
data, or a test set for specific kinds of inputs (e.g., districts
located inland versus districts located near the ocean). This will give
you a deeper understanding of your model's strengths and weaknesses.


As you can see, Machine Learning involves quite a lot of infrastructure,
so don't be surprised if your first ML project takes a lot of effort and
time to build and deploy to production. Fortunately, once all the
infrastructure is in place, going from idea to production will be much
faster.




Try It Out!
===========

Hopefully this lab gave you a good idea of what a Machine Learning
project looks like as well as showing you some of the tools you can use
to train a great system. As you can see, much of the work is in the data
preparation step: building monitoring tools, setting up human evaluation
pipelines, and automating regular model training. The Machine Learning
algorithms are important, of course, but it is probably preferable to be
comfortable with the overall process and know three or four algorithms
well rather than to spend all your time exploring advanced algorithms.

So, if you have not already done so, now is a good time to pick up a
laptop, select a dataset that you are interested in, and try to go
through the whole process from A to Z. A good place to start is on a
competition website such as [*http://kaggle.com/*](http://kaggle.com/):
you will have a dataset to play with, a clear goal, and people to share
the experience with. Have fun!




Exercises
=========

The following exercises are all based on this lab's housing dataset:

1.  Try a Support Vector Machine regressor (`sklearn.svm.SVR`) with
    various hyperparameters, such as `kernel="linear"` (with various
    values for the `C` hyperparameter) or `kernel="rbf"` (with various
    values for the `C` and `gamma` hyperparameters). Don't worry about
    what these hyperparameters mean for now. How does the best `SVR`
    predictor perform?

2.  Try replacing `GridSearchCV` with `RandomizedSearchCV`.

3.  Try adding a transformer in the preparation pipeline to select only
    the most important attributes.

4.  Try creating a single pipeline that does the full data preparation
    plus the final prediction.

5.  Automatically explore some preparation options using `GridSearchCV`.

Solutions to these exercises can be found in the Jupyter notebooks
available at
[*https://github.com/fenago/ml-scikit-keras-tensorflow*](https://github.com/fenago/ml-scikit-keras-tensorflow).



^[1]

The example project is fictitious; the goal is to illustrate the main
steps of a Machine Learning project, not to learn anything about the
real estate business.

^[2]

The original dataset appeared in R. Kelley Pace and Ronald Barry,
"Sparse Spatial Autoregressions," *Statistics & Probability Letters* 33,
no. 3 (1997): 291--297.

^[3]

A piece of information fed to a Machine Learning system is often called
a *signal*, in reference to Claude Shannon's information theory, which
he developed at Bell Labs to improve telecommunications. His theory: you
want a high signal-to-noise ratio.

^[4]

Recall that the transpose operator flips a column vector into a row
vector (and vice versa).

^[5]

The latest version of Python 3 is recommended. Python 2.7+ may work too,
but now that it's deprecated, all major scientific libraries are
dropping support for it, so you should migrate to Python 3 as soon as
possible.

^[6]

I'll show the installation steps using pip in a bash shell on a Linux or
macOS system. You may need to adapt these commands to your own system.
On Windows, I recommend installing Anaconda instead.

^[7]

If you want to upgrade pip for all users on your machine rather than
just your own user, you should remove the `--user` option and make sure
you have administrator rights (e.g., by adding `sudo` before the whole
command on Linux or macOS).

^[8]

Alternative tools include venv (very similar to virtualenv and included
in the standard library), virtualenvwrapper (provides extra
functionalities on top of virtualenv), pyenv (allows easy switching
between Python versions), and pipenv (a great packaging tool by the same
author as the popular `requests` library, built on top of pip and
virtualenv).

^[9]

Note that Jupyter can handle multiple versions of Python, and even many
other languages such as R or Octave.

^[10]
You might also need to check legal constraints, such as private fields
that should never be copied to unsafe data stores.

^[11]
In a real project you would save this code in a Python file, but for now
you can just write it in your Jupyter notebook.

^[12]
The standard deviation is generally denoted *σ* (the Greek letter
sigma), and it is the square root of the *variance*, which is the
average of the squared deviation from the mean. When a feature has a
bell-shaped *normal distribution* (also called a *Gaussian
distribution*), which is very common, the "68-95-99.7" rule applies:
about 68% of the values fall within 1*σ* of the mean, 95% within 2*σ*,
and 99.7% within 3*σ*.

^[13]
In this course, when a code example contains a mix of code and outputs, as
is the case here, it is formatted like in the Python interpreter, for
better readability: the code lines are prefixed with `>>>` (or `...` for
indented blocks), and the outputs have no prefix.

^[14]
You will often see people set the random seed to 42. This number has no
special property, other than to be the Answer to the Ultimate Question
of Life, the Universe, and Everything.

^[15]
The location information is actually quite coarse, and as a result many
districts will have the exact same ID, so they will end up in the same
set (test or train). This introduces some unfortunate sampling bias.

^[16]
If you are reading this in grayscale, grab a red pen and scribble over
most of the coastline from the Bay Area down to San Diego (as you might
expect). You can add a patch of yellow around Sacramento as well.

^[17]
For more details on the design principles, see Lars Buitinck et al.,
"API Design for Machine Learning Software: Experiences from the
Scikit-Learn Project" ," arXiv preprint arXiv:1309.0238 (2013).

^[18]
Some predictors also provide methods to measure the confidence of their
predictions.

^[19]
This class is available in Scikit-Learn 0.20 and later. If you use an
earlier version, please consider upgrading, or use the pandas
`Series.factorize()` method.

^[20]
Before Scikit-Learn 0.20, the method could only encode integer
categorical values, but since 0.20 it can also handle other types of
inputs, including text categorical inputs.

^[21]
See SciPy's documentation for more details.

^[22]
Just like for pipelines, the name can be anything as long as it does not
contain double underscores.

^[23]
In a nutshell, a REST (or RESTful) API is an HTTP-based API that follows
some conventions, such as using standard HTTP verbs to read, update,
create, or delete resources (GET, POST, PUT, and DELETE) and using JSON
for the inputs and outputs.

^[24]
A captcha is a test to ensure a user is not a robot. These tests have
often been used as a cheap way to label training data.
