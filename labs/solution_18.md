
[Chapter 18](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch18.html#rl_chapter): Reinforcement Learning
============================================================================================================================================

1.  Reinforcement Learning is an area of Machine Learning aimed at
    creating agents capable of taking actions in an environment in a way
    that maximizes rewards over time. There are many differences between
    RL and regular supervised and unsupervised learning. Here are a few:

    -   In supervised and unsupervised learning, the goal is generally
        to find patterns in the data and use them to make predictions.
        In Reinforcement Learning, the goal is to find a good policy.

    -   Unlike in supervised learning, the agent is not explicitly given
        the "right" answer. It must learn by trial and error.

    -   Unlike in unsupervised learning, there is a form of supervision,
        through rewards. We do not tell the agent how to perform the
        task, but we do tell it when it is making progress or when it is
        failing.

    -   A Reinforcement Learning agent needs to find the right balance
        between exploring the environment, looking for new ways of
        getting rewards, and exploiting sources of rewards that it
        already knows. In contrast, supervised and unsupervised learning
        systems generally don't need to worry about exploration; they
        just feed on the training data they are given.

    -   In supervised and unsupervised learning, training instances are
        typically independent (in fact, they are generally shuffled). In
        Reinforcement Learning, consecutive observations are generally
        *not* independent. An agent may remain in the same region of the
        environment for a while before it moves on, so consecutive
        observations will be very correlated. In some cases a replay
        memory (buffer) is used to ensure that the training algorithm
        gets fairly independent observations.

2.  Here are a few possible applications of Reinforcement Learning,
    other than those mentioned in
    [Chapter 18](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch18.html#rl_chapter):

    Music personalization

    :   The environment is a user's personalized web radio. The agent is
        the software deciding what song to play next for that user. Its
        possible actions are to play any song in the catalog (it must
        try to choose a song the user will enjoy) or to play an
        advertisement (it must try to choose an ad that the user will be
        interested in). It gets a small reward every time the user
        listens to a song, a larger reward every time the user listens
        to an ad, a negative reward when the user skips a song or an ad,
        and a very negative reward if the user leaves.

    Marketing

    :   The environment is your company's marketing department. The
        agent is the software that defines which customers a mailing
        campaign should be sent to, given their profile and purchase
        history (for each customer it has two possible actions: send or
        don't send). It gets a negative reward for the cost of the
        mailing campaign, and a positive reward for estimated revenue
        generated from this campaign.

    Product delivery

    :   Let the agent control a fleet of delivery trucks, deciding what
        they should pick up at the depots, where they should go, what
        they should drop off, and so on. It will get positive rewards
        for each product delivered on time, and negative rewards for
        late deliveries.

3.  When estimating the value of an action, Reinforcement Learning
    algorithms typically sum all the rewards that this action led to,
    giving more weight to immediate rewards and less weight to later
    rewards (considering that an action has more influence on the near
    future than on the distant future). To model this, a discount factor
    is typically applied at each time step. For example, with a discount
    factor of 0.9, a reward of 100 that is received two time steps later
    is counted as only 0.9^2^ × 100 = 81 when you are estimating the
    value of the action. You can think of the discount factor as a
    measure of how much the future is valued relative to the present: if
    it is very close to 1, then the future is valued almost as much as
    the present; if it is close to 0, then only immediate rewards
    matter. Of course, this impacts the optimal policy tremendously: if
    you value the future, you may be willing to put up with a lot of
    immediate pain for the prospect of eventual rewards, while if you
    don't value the future, you will just grab any immediate reward you
    can find, never investing in the future.

4.  To measure the performance of a Reinforcement Learning agent, you
    can simply sum up the rewards it gets. In a simulated environment,
    you can run many episodes and look at the total rewards it gets on
    average (and possibly look at the min, max, standard deviation, and
    so on).

5.  The credit assignment problem is the fact that when a Reinforcement
    Learning agent receives a reward, it has no direct way of knowing
    which of its previous actions contributed to this reward. It
    typically occurs when there is a large delay between an action and
    the resulting reward (e.g., during a game of Atari's *Pong*, there
    may be a few dozen time steps between the moment the agent hits the
    ball and the moment it wins the point). One way to alleviate it is
    to provide the agent with shorter-term rewards, when possible. This
    usually requires prior knowledge about the task. For example, if we
    want to build an agent that will learn to play chess, instead of
    giving it a reward only when it wins the game, we could give it a
    reward every time it captures one of the opponent's pieces.

6.  An agent can often remain in the same region of its environment for
    a while, so all of its experiences will be very similar for that
    period of time. This can introduce some bias in the learning
    algorithm. It may tune its policy for this region of the
    environment, but it will not perform well as soon as it moves out of
    this region. To solve this problem, you can use a replay memory;
    instead of using only the most immediate experiences for learning,
    the agent will learn based on a buffer of its past experiences,
    recent and not so recent (perhaps this is why we dream at night: to
    replay our experiences of the day and better learn from them?).

7.  An off-policy RL algorithm learns the value of the optimal policy
    (i.e., the sum of discounted rewards that can be expected for each
    state if the agent acts optimally) while the agent follows a
    different policy. Q-Learning is a good example of such an algorithm.
    In contrast, an on-policy algorithm learns the value of the policy
    that the agent actually executes, including both exploration and
    exploitation.

For the solutions to exercises 8, 9, and 10, please see the Jupyter
notebooks available at
[*https://github.com/ageron/handson-ml2*](https://github.com/ageron/handson-ml2).


