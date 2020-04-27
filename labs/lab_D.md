Autodiff
========

This appendix explains how
TensorFlow's autodifferentiation (autodiff) feature works, and how it
compares to other solutions.

Suppose you define a function *f*(*x*, *y*) = *x*^2^*y* + *y* + 2, and
you need its partial derivatives ∂*f*/∂*x* and ∂*f*/∂*y*, typically to
perform Gradient Descent (or some other optimization algorithm). Your
main options are manual differentiation, finite difference
approximation, forward-mode autodiff, and reverse-mode autodiff.
TensorFlow implements reverse-mode autodiff, but to understand it, it's
useful to look at the other options first. So let's go through each of
them, starting with manual differentiation.



Manual Differentiation
======================

The first approach to compute derivatives is to
pick up a pencil and a piece of paper and use your calculus knowledge to
derive the appropriate equation. For the function *f*(*x*, *y*) just
defined, it is not too hard; you just need to use five rules:

-   The derivative of a constant is 0.

-   The derivative of *λx* is *λ* (where *λ* is a constant).

-   The derivative of *x*^λ^ is *λx*^*λ*\ --\ 1^, so the derivative of
    *x*^2^ is 2*x*.

-   The derivative of a sum of functions is the sum of these functions'
    derivatives.

-   The derivative of *λ* times a function is *λ* times its derivative.

From these rules, you can derive [Equation
D-1]


##### [Equation D-1. ] Partial derivatives of *f*(*x*, *y*)

$$\begin{aligned}
\frac{\partial f}{\partial x} & {= \frac{\partial\left( x^{2}y \right)}{\partial x} + \frac{\partial y}{\partial x} + \frac{\partial 2}{\partial x} = y\frac{\partial\left( x^{2} \right)}{\partial x} + 0 + 0 = 2xy} \\
\frac{\partial f}{\partial y} & {= \frac{\partial\left( x^{2}y \right)}{\partial y} + \frac{\partial y}{\partial y} + \frac{\partial 2}{\partial y} = x^{2} + 1 + 0 = x^{2} + 1} \\
\end{aligned}$$


This approach can become very tedious for more complex functions, and
you run the risk of making mistakes. Fortunately, there are other
options. Let's look at finite difference approximation now.




Finite Difference Approximation
===============================

Recall that the derivative *h*′(*x*~0~) of a
function *h*(*x*) at a point *x*~0~ is the slope of the function at that
point. More precisely, the derivative is defined as the limit of the
slope of a straight line going through this point *x*~0~ and another
point *x* on the function, as *x* gets infinitely close to *x*~0~ (see
[Equation
D-2]


##### [Equation D-2. ] Definition of the derivative of a function *h*(*x*) at point *x*~0~

$$\begin{array}{cl}
{h^{'}\left( x_{0} \right)} & {= \lim\limits_{x\rightarrow x_{0}}\frac{h\left( x \right) - h\left( x_{0} \right)}{x - x_{0}}} \\
 & {= \lim\limits_{\varepsilon\rightarrow 0}\frac{h\left( x_{0} + \varepsilon \right) - h\left( x_{0} \right)}{\varepsilon}} \\
\end{array}$$


So, if we wanted to calculate the partial derivative of *f*(*x*, *y*)
with regard to *x* at *x* = 3 and *y* = 4, we could compute *f*(3 + *ε*,
4) -- *f*(3, 4) and divide the result by *ε*, using a very small value
for *ε*. This type of numerical approximation of the derivative is
called a *finite difference approximation*, and
this specific equation is called *Newton's
difference quotient*. That's exactly what the following code does:

``` {data-type="programlisting" code-language="python"}
def f(x, y):
    return x**2*y + y + 2

def derivative(f, x, y, x_eps, y_eps):
    return (f(x + x_eps, y + y_eps) - f(x, y)) / (x_eps + y_eps)

df_dx = derivative(f, 3, 4, 0.00001, 0)
df_dy = derivative(f, 3, 4, 0, 0.00001)
```

Unfortunately, the result is imprecise (and it gets worse for more
complicated functions). The correct results are respectively 24 and 10,
but instead we get:

``` {data-type="programlisting" code-language="pycon"}
>>> print(df_dx)
24.000039999805264
>>> print(df_dy)
10.000000000331966
```

Notice that to compute both partial derivatives, we have to call `f()`
at least three times (we called it four times in the preceding code, but
it could be optimized). If there were 1,000 parameters, we would need to
call `f()` at least 1,001 times. When you are dealing with large neural
networks, this makes finite difference approximation way too
inefficient.

However, this method is so simple to implement that it is a great tool
to check that the other methods are implemented correctly. For example,
if it disagrees with your manually derived function, then your function
probably contains a mistake.

So far, we have considered two ways to compute gradients: using manual
differentiation and using finite difference approximation.
Unfortunately, both were fatally flawed to train a large-scale neural
network. So let's turn to autodiff, starting with forward mode.




Forward-Mode Autodiff
=====================

[Figure D-1]
shows how forward-mode autodiff works on an even
simpler function, *g*(*x*, *y*) = 5 + *xy*. The graph for that function
is represented on the left. After forward-mode autodiff, we get the
graph on the right, which represents the partial derivative ∂*g*/∂*x* =
0 + (0 × *x* + *y* × 1) = *y* (we could similarly obtain the partial
derivative with regard to *y*).

![](./D_files/mls2_ad01.png)

The algorithm will go through the computation graph from the inputs to
the outputs (hence the name "forward mode"). It starts by getting the
partial derivatives of the leaf nodes. The constant node (5) returns the
constant 0, since the derivative of a constant is always 0. The variable
*x* returns the constant 1 since ∂*x*/∂*x* = 1, and the variable *y*
returns the constant 0 since ∂*y*/∂*x* = 0 (if we were looking for the
partial derivative with regard to *y*, it would be the reverse).

Now we have all we need to move up the graph to the multiplication node
in function *g*. Calculus tells us that the derivative of the product of
two functions *u* and *v* is [∂(*u* × *v*)/∂*x*] =
∂*v*/∂*x* × *u* + *v* × ∂*u*/∂*x*. We can therefore construct a large
part of the graph on the right, representing 0 × *x* + *y* × 1.

Finally, we can go up to the addition node in function *g*. As
mentioned, the derivative of a sum of functions is the sum of these
functions' derivatives. So we just need to create an addition node and
connect it to the parts of the graph we have already computed. We get
the correct partial derivative: ∂*g*/∂*x* = 0 + (0 × *x* + *y* × 1).

However, this equation can be simplified (a lot). A few pruning steps
can be applied to the computation graph to get rid of all unnecessary
operations, and we get a much smaller graph with just one node:
∂*g*/∂*x* = *y*. In this case simplification is fairly easy, but for a
more complex function forward-mode autodiff can produce a huge graph
that may be tough to simplify and lead to suboptimal performance.

Note that we started with a computation graph, and forward-mode autodiff
produced another computation graph. This is called
*symbolic differentiation*, and it has two nice features: first, once
the computation graph of the derivative has been produced, we can use it
as many times as we want to compute the derivatives of the given
function for any value of *x* and *y*; second, we can run forward-mode
autodiff again on the resulting graph to get second-order derivatives if
we ever need to (i.e., derivatives of derivatives). We could even
compute third-order derivatives, and so on.

But it is also possible to run forward-mode autodiff without
constructing a graph (i.e., numerically, not symbolically), just by
computing intermediate results on the fly. One way
to do this is to use *dual numbers*, which are weird but fascinating
numbers of the form *a* + *bε*, where *a* and *b* are real numbers and
*ε* is an infinitesimal number such that *ε*^2^ = 0 (but *ε* ≠ 0). You
can think of the dual number 42 + 24*ε* as something akin to
42.0000⋯000024 with an infinite number of 0s (but of course this is
simplified just to give you some idea of what dual numbers are). A dual
number is represented in memory as a pair of floats. For example, 42 +
24*ε* is represented by the pair (42.0, 24.0).

Dual numbers can be added, multiplied, and so on, as shown in [Equation
D-3]


##### [Equation D-3. ] A few operations with dual numbers

$$\begin{array}{cl}
 & {\lambda\left( a + b\varepsilon \right) = \lambda a + \lambda b\varepsilon} \\
 & {\left( a + b\varepsilon \right) + \left( c + d\varepsilon \right) = \left( a + c \right) + \left( b + d \right)\varepsilon} \\
 & {\left( a + b\varepsilon \right) \times \left( c + d\varepsilon \right) = ac + \left( ad + bc \right)\varepsilon + \left( bd \right)\varepsilon^{2} = ac + \left( ad + bc \right)\varepsilon} \\
\end{array}$$


Most importantly, it can be shown that *h*(*a* + *bε*) = *h*(*a*) + *b*
× *h*′(*a*)*ε*, so computing *h*(*a* + *ε*) gives you both *h*(*a*) and
the derivative *h*′(*a*) in just one shot.
[Figure D-2]
shows that the partial derivative of *f*(*x*, *y*) with regard to *x* at
*x* = 3 and *y* = 4 (which we will write ∂*f*/∂*x* (3, 4)) can be
computed using dual numbers. All we need to do is compute *f*(3 + *ε*,
4); this will output a dual number whose first component is equal to
*f*(3, 4) and whose second component is equal to ∂*f*/∂*x* (3, 4).

![](./D_files/mls2_ad02.png)

To compute ∂*f*/∂*x* (3, 4) we would have to go through the graph again,
but this time with *x* = 3 and *y* = 4 + *ε*.

So forward-mode autodiff is much more accurate than finite difference
approximation, but it suffers from the same major flaw, at least when
there are many inputs and few outputs (as is the case when dealing with
neural networks): if there were 1,000 parameters, it would require 1,000
passes through the graph to compute all the partial derivatives. This is
where reverse-mode autodiff shines: it can compute all of them in just
two passes through the graph. Let's see how.




Reverse-Mode Autodiff
=====================

Reverse-mode autodiff is the solution implemented
by TensorFlow. It first goes through the graph in the forward direction
(i.e., from the inputs to the output) to compute the value of each node.
Then it does a second pass, this time in the reverse direction (i.e.,
from the output to the inputs), to compute all the partial derivatives.
The name "reverse mode" comes from this second pass through the graph,
where gradients flow in the reverse direction.
[Figure D-3]
represents the second pass. During the first pass, all the node values
were computed, starting from *x* = 3 and *y* = 4. You can see those
values at the bottom right of each node (e.g., *x* × *x* = 9). The nodes
are labeled *n*~1~ to *n*~7~ for clarity. The output node is *n*~7~:
*f*(3, 4) = *n*~7~ = 42.

![](./D_files/mls2_ad03.png)

The idea is to gradually go down the graph, computing the partial
derivative of *f*(*x*, *y*) with regard to each consecutive node, until
we reach the variable nodes. For this, reverse-mode autodiff relies
heavily on the *chain rule*, shown in [Equation
D-4]


##### [Equation D-4. ] Chain rule

$$\frac{\partial f}{\partial x} = \frac{\partial f}{\partial n_{i}} \times \frac{\partial n_{i}}{\partial x}$$


Since *n*~7~ is the output node, *f* = *n*~7~ so ∂*f*/∂*n*~7~ = 1.

Let's continue down the graph to *n*~5~: how much does *f* vary when
*n*~5~ varies? The answer is ∂*f*/∂*n*~5~ = ∂*f*/∂*n*~7~ ×
∂*n*~7~/∂*n*~5~. We already know that ∂*f*/∂*n*~7~ = 1, so all we need
is ∂*n*~7~/∂*n*~5~. Since *n*~7~ simply performs the sum *n*~5~ +
*n*~6~, we find that ∂*n*~7~/∂*n*~5~ = 1, so ∂*f*/∂*n*~5~ = 1 × 1 = 1.

Now we can proceed to node *n*~4~: how much does *f* vary when *n*~4~
varies? The answer is ∂*f*/∂*n*~4~ = ∂*f*/∂*n*~5~ × ∂*n*~5~/∂*n*~4~.
Since *n*~5~ = *n*~4~ × *n*~2~, we find that ∂*n*~5~/∂*n*~4~ = *n*~2~,
so ∂*f*/∂*n*~4~ = 1 × *n*~2~ = 4.

The process continues until we reach the bottom of the graph. At that
point we will have calculated all the partial derivatives of *f*(*x*,
*y*) at the point *x* = 3 and *y* = 4. In this example, we find
∂*f*/∂*x* = 24 and ∂*f*/∂*y* = 10. Sounds about right!

Reverse-mode autodiff is a very powerful and accurate technique,
especially when there are many inputs and few outputs, since it requires
only one forward pass plus one reverse pass per output to compute all
the partial derivatives for all outputs with regard to all the inputs.
When training neural networks, we generally want to minimize the loss,
so there is a single output (the loss), and hence only two passes
through the graph are needed to compute the gradients. Reverse-mode
autodiff can also handle functions that are not entirely differentiable,
as long as you ask it to compute the partial derivatives at points that
are differentiable.

In
[Figure D-3]
the numerical results are computed on the fly, at each node. However,
that's not exactly what TensorFlow does: instead, it creates a new
computation graph. In other words, it implements *symbolic* reverse-mode
autodiff. This way, the computation graph to compute the gradients of
the loss with regard to all the parameters in the neural network only
needs to be generated once, and then it can be executed over and over
again, whenever the optimizer needs to compute the gradients. Moreover,
this makes it possible to compute higher-order derivatives if needed.


###### Tip

If you ever want to implement a new type of low-level TensorFlow
operation in C++, and you want to make it compatible with autodiff, then
you will need to provide a function that returns the partial derivatives
of the function's outputs with regard to its inputs. For example,
suppose you implement a function that computes the square of its input:
*f*(*x*) = *x*^2^. In that case you would need to provide the
corresponding derivative function: *f*′(*x*) =
2*x*. 
