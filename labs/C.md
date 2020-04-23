
[Appendix C. ]{.label}SVM Dual Problem
======================================

To[]{#idm45728431592488}[]{#idm45728431591064}[]{#idm45728431590216}
understand *duality*, you first need to understand the *Lagrange
multipliers* method. The general idea is to transform a constrained
optimization objective into an unconstrained one, by moving the
constraints into the objective function. Let's look at a simple example.
Suppose you want to find the values of *x* and *y* that minimize the
function *f*(*x*, *y*) = *x*^2^ + 2*y*, subject to an *equality
constraint*: 3*x* + 2*y* + 1 = 0. Using the Lagrange multipliers method,
we start by defining a new function called the *Lagrangian* (or
*Lagrange function*): *g*(*x*, *y*, *α*) = *f*(*x*, *y*) -- *α*(3*x* +
2*y* + 1). Each constraint (in this case just one) is subtracted from
the original objective, multiplied by a new variable called a Lagrange
multiplier.

Joseph-Louis Lagrange[]{#idm45728431591544} showed that if
$\left( \hat{x},\hat{y} \right)$ is a solution to the constrained
optimization problem, then there must exist an $\hat{\alpha}$ such that
$\left( \hat{x},\hat{y},\hat{\alpha} \right)$ is a *stationary point* of
the Lagrangian (a stationary point is a point where all partial
derivatives are equal to zero). In other words, we can compute the
partial derivatives of *g*(*x*, *y*, *α*) with regard to *x*, *y*, and
*α*; we can find the points where these derivatives are all equal to
zero; and the solutions to the constrained optimization problem (if they
exist) must be among these stationary points.

In this example the partial derivatives are: $\left\{ \begin{array}{l}
{\frac{\partial}{\partial x}g\left( x,y,\alpha \right) = 2x - 3\alpha} \\
{\frac{\partial}{\partial y}g\left( x,y,\alpha \right) = 2 - 2\alpha} \\
{\frac{\partial}{\partial\alpha}g\left( x,y,\alpha \right) = - 3x - 2y - 1} \\
\end{array} \right.$

When all these partial derivatives are equal to 0, we find that
$2\hat{x} - 3\hat{\alpha} = 2 - 2\hat{\alpha} = -3\hat{x} - 2\hat{y} - 1 = 0$,
from which we can easily find that $\hat{x} = \frac{3}{2}$,
$\hat{y} = - \frac{11}{4}$, and $\hat{\alpha} = 1$. This is the only
stationary point, and as it respects the constraint, it must be the
solution to the constrained optimization problem.

However,
this[]{#idm45728431497896}[]{#idm45728431497192}[]{#idm45728431496488}
method applies only to equality constraints. Fortunately, under some
regularity conditions (which are respected by the SVM objectives), this
method can be generalized to *inequality constraints* as well (e.g.,
3*x* + 2*y* + 1 ≥ 0). The *generalized Lagrangian* for the hard margin
problem is given by [Equation
C-1](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/app03.html#generalized_lagrangian_hard_margin),
where the *α*^(*i*)^ variables are called the *Karush--Kuhn--Tucker*
(KKT) multipliers, and they must be greater or equal to zero.

::: {#generalized_lagrangian_hard_margin data-type="equation"}
##### [Equation C-1. ]{.label}Generalized Lagrangian for the hard margin problem

$$\begin{array}{r}
{\mathcal{L}\left( \mathbf{w},b,\alpha \right) = \frac{1}{2}\mathbf{w}^{\intercal}\mathbf{w} - \sum\limits_{i = 1}^{m}{\alpha^{(i)}\left( {t^{(i)}\left( \mathbf{w}^{\intercal}\mathbf{x}^{(i)} + b \right) - 1} \right)}} \\
{\text{with}\alpha^{(i)} \geq 0\text{for} i = 1,2,\cdots,m} \\
\end{array}$$
:::

Just like with the Lagrange multipliers method, you can compute the
partial derivatives and locate the stationary points. If there is a
solution, it will necessarily be among the stationary points
$\left( \hat{\mathbf{w}},\hat{b},\hat{\alpha} \right)$ that respect the
*KKT conditions*:

-   Respect the problem's constraints:
    $t^{(i)}\left( {\hat{\mathbf{w}}}^{\intercal}\mathbf{x}^{(i)} + \hat{b} \right) \geq 1~\text{for~}i = 1,2,\ldots,m$.

-   Verify ${\hat{\alpha}}^{(i)} \geq 0\text{for} i = 1,2,\cdots,m$.

-   Either[]{#idm45728431420584} ${\hat{\alpha}}^{(i)} = 0$ or the
    *i*^th^ constraint must be an *active constraint*, meaning it must
    hold by equality:
    $t^{(i)}\left( {\hat{\mathbf{w}}}^{\intercal}\mathbf{x}^{(i)} + \hat{b} \right) = 1$.
    This[]{#idm45728431401624} condition is called the *complementary
    slackness* condition. It implies that either
    ${\hat{\alpha}}^{(i)} = 0$ or the *i*^th^ instance lies on the
    boundary (it is a support vector).

Note that the KKT conditions are necessary conditions for a stationary
point to be a solution of the constrained optimization problem. Under
some conditions, they are also sufficient conditions. Luckily, the SVM
optimization problem happens to meet these conditions, so any stationary
point that meets the KKT conditions is guaranteed to be a solution to
the constrained optimization problem.

We can compute the partial derivatives of the generalized Lagrangian
with regard to **w** and *b* with [Equation
C-2](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/app03.html#partial_derivatives_of_generalized_lagrangian).

::: {#partial_derivatives_of_generalized_lagrangian .fifty-percent data-type="equation"}
##### [Equation C-2. ]{.label}Partial derivatives of the generalized Lagrangian

$$\begin{array}{r}
{\nabla_{\mathbf{w}}\mathcal{L}\left( \mathbf{w},b,\alpha \right) = \mathbf{w} - \sum\limits_{i = 1}^{m}\alpha^{(i)}t^{(i)}\mathbf{x}^{(i)}} \\
{\frac{\partial}{\partial b}\mathcal{L}\left( \mathbf{w},b,\alpha \right) = - \sum\limits_{i = 1}^{m}\alpha^{(i)}t^{(i)}} \\
\end{array}$$
:::

When these partial derivatives are equal to zero, we have [Equation
C-3](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/app03.html#stationary_points_properties).

::: {#stationary_points_properties .fifty-percent data-type="equation"}
##### [Equation C-3. ]{.label}Properties of the stationary points

$$\begin{array}{r}
{\hat{\mathbf{w}} = \sum\limits_{i = 1}^{m}{\hat{\alpha}}^{(i)}t^{(i)}\mathbf{x}^{(i)}} \\
{\sum\limits_{i = 1}^{m}{\hat{\alpha}}^{(i)}t^{(i)} = 0} \\
\end{array}$$
:::

If we plug these results into the definition of the generalized
Lagrangian, some terms disappear and we find [Equation
C-4](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/app03.html#dual_form_generalized_lagrangian).

::: {#dual_form_generalized_lagrangian data-type="equation"}
##### [Equation C-4. ]{.label}Dual form of the SVM problem

$$\begin{array}{r}
{\mathcal{L}\left( \hat{\mathbf{w}},\hat{b},\alpha \right) = \frac{1}{2}\sum\limits_{i = 1}^{m}{\sum\limits_{j = 1}^{m}{\alpha^{(i)}\alpha^{(j)}t^{(i)}t^{(j)}{\mathbf{x}^{(i)}}^{\intercal}\mathbf{x}^{(j)}}} - \sum\limits_{i = 1}^{m}\alpha^{(i)}} \\
{\text{with}\alpha^{(i)} \geq 0\text{for} i = 1,2,\cdots,m} \\
\end{array}$$
:::

The goal is now to find the vector $\hat{\mathbf{\alpha}}$ that
minimizes this function, with ${\hat{\alpha}}^{(i)} \geq 0$ for all
instances. This constrained optimization problem is the dual problem we
were looking for.

Once you find the optimal $\hat{\mathbf{\alpha}}$, you can compute
$\hat{\mathbf{w}}$ using the first line of [Equation
C-3](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/app03.html#stationary_points_properties).
To compute $\hat{b}$, you can use the fact that a support vector must
verify *t*^(*i*)^($\hat{\mathbf{w}}$^⊺^ **x**^(*i*)^ + $\hat{b}$) = 1,
so if the *k*^th^ instance is a support vector (i.e.,
${\hat{\alpha}}^{(k)} > 0$), you can use it to compute
$\hat{b} = t^{(k)} - {{\hat{\mathbf{w}}}^{\intercal}\mathbf{x}^{(k)}}$.
However, it is often preferred to compute the average over all support
vectors to get a more stable and precise value, as in [Equation
C-5](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/app03.html#bias_term_dual_form).

::: {#bias_term_dual_form .fifty-percent data-type="equation"}
##### [Equation C-5. ]{.label}Bias term estimation using the dual form

$$\hat{b} = \frac{1}{n_{s}}\sum\limits_{\binom{i = 1}{{\hat{\alpha}}^{(i)} > 0}}^{m}\left\lbrack {t^{(i)} - {{\hat{\mathbf{w}}}^{\intercal}\mathbf{x}^{(i)}}} \right\rbrack$$
