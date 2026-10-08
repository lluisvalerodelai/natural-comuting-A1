# Particle Swarm Optimization (PSO)

## 1. What PSO is

Particle Swarm Optimization (PSO) is one of the most common swarm-based algorithms.

- Based on a model of the social behavior of bird flocks.
- An optimization technique for continuous optimization problems.
- Developed by Kennedy and Eberhart in 1995.
- A population-based optimization technique inspired by the social behavior of bird flocking or fish schooling.
- Individual swarm members can profit from the discoveries and previous experience of other members of the swarm.

## 2. From Boids to PSO

The lecture first presents Boids as a model of swarming behavior.

Boids use three simple rules:

- **Separation:** move away from neighbors if these are too close.
- **Alignment:** steer towards the average heading of neighbors.
- **Cohesion:** try to move toward the average position of neighbors.

**Boids is not an optimization algorithm.**

Kennedy and Eberhart extended a simplified Boids-like simulation by adding a **roost**.

Each agent:

- Is attracted to the location of the roost.
- Remembers where it was closer to the roost.
- Shares information with its neighbors about its closest location to the roost.

Eventually, all agents land on the roost.

The lecture then asks what happens if the notion of distance to the roost is replaced by an unknown function. This leads into PSO as an optimization method.

---

## 3. General PSO concept

PSO uses a **swarm of particles**.

Each particle:

- Resides at a position in the search space.
- Has a fitness value representing the quality of its position.
- Moves through the search space with a certain velocity.

The velocity contains both:

- Direction.
- Speed.

The velocity of each particle is influenced by:

1. Its own best position found so far.
2. The best solution found so far by its neighbors.

Eventually, the swarm will converge to optimal positions.

---

## 4. Basic PSO algorithm

The basic algorithm given in the lecture is:

```text
Algorithm: Particle Swarm Optimization

Randomly initialize particles (x_i) and their velocities (v_i) in the search space

while termination criteria are not met do

    for each particle i do

        Evaluate the fitness y_i at current position x_i

        if y_i is better than its personal best pbest_i then
            update p_i and pbest_i
        end if

        if y_i is better than its global/neighborhood best gbest_i then
            update g_i and gbest_i
        end if

    end for

    for each particle i do

        Update the velocity v_i based on p_i, g_i and v_i

        Update the position:
            x_i <- x_i + v_i

    end for

end while
```

The lecture later emphasizes that the best solution used in the social component is determined by the particle's **neighborhood**.

---

## 5. Original PSO notation

For each particle \(i\):

- \(x_i\): vector denoting the particle's **position**.
- \(v_i\): vector denoting the particle's **velocity**.
- \(y_i\): fitness score of \(x_i\).
- \(p_i\): best position that particle \(i\) has found so far.
- \(pbest_i\): fitness of \(p_i\).
- \(g_i\): best position that has been found so far in the neighborhood of particle \(i\).
- \(gbest_i\): fitness of \(g_i\).

### Random vectors

\(U(0,\phi_i)\) denotes a random vector uniformly distributed in:

\[
[0,\phi_i]
\]

It is generated:

- At each generation.
- For each particle.

### Acceleration coefficients

- \(\phi_1\): controls the scale of the force in the direction of the particle's own best position \(p_i\).
- \(\phi_2\): controls the scale of the force in the direction of the neighborhood best position \(g_i\).

The operator \(\otimes\) denotes **element-wise multiplication**.

---

## 6. Original velocity update

The lecture gives the original PSO velocity update as:

\[
v_i
\leftarrow
v_i
+
U(0,\phi_1)\otimes(p_i-x_i)
+
U(0,\phi_2)\otimes(g_i-x_i)
\]

The velocity update has three components.

### 6.1 Momentum

\[
v_i
\]

The force pulling the particle to continue in its current direction.

### 6.2 Cognitive component

\[
U(0,\phi_1)\otimes(p_i-x_i)
\]

The force emerging from the tendency to return to the particle's own best solution found so far.

### 6.3 Social component

\[
U(0,\phi_2)\otimes(g_i-x_i)
\]

The force emerging from attraction to the best solution found so far in the particle's neighborhood.

After the velocity is updated, the position is updated by:

\[
x_i \leftarrow x_i + v_i
\]

---

## 7. Neighborhoods

The neighborhood of each particle defines its **communication structure**, or social network.

The lecture distinguishes two general types.

### 7.1 Geographical neighborhood topologies

- Based on Euclidean proximity in the search space.
- Close to the real-world paradigm.
- Computationally expensive.

### 7.2 Communication network topologies

The neighborhood is defined by a connection graph architecture.

Examples include:

- Rings.
- Stars.
- von Neumann networks.
- Random graphs.

These are favored over geographical neighborhoods because they involve:

- Better convergence properties.
- Less computation.

---

## 8. Neighborhood topologies

Examples shown in the lecture include:

- **Ring topology**
  - Local-best topology.
- **Global-best topology**
- **Random graph**
- **Star topology**

Other considered topologies include:

- Clustering topologies / islands.
- Dynamic topologies.
- Other variants.

The lecture states that there is **no clear way of saying which topology is the best**.

Neighborhood topology affects the balance between exploration and exploitation.

According to the lecture:

- Some neighborhood topologies are better for local search.
- Others are better for global search.
- **lbest** neighborhood topologies seem better for global search.
- **gbest** topologies seem better for local search.

---

## 9. lbest and gbest PSO

### lbest PSO

Each particle uses the best solution found in a more local neighborhood.

The lecture associates lbest with stronger exploration behavior.

### gbest PSO

Particles use the global best solution available through the chosen topology.

The lecture associates gbest with stronger exploitation behavior.

---

## 10. Synchronous and asynchronous updates

### 10.1 Synchronous updates

- Personal best and neighborhood best values are updated separately from the position and velocity vectors.
- Slower.
- Give "perfect" feedback about best positions.
- Better for **gbest PSO**.
- Associated with exploitation.

### 10.2 Asynchronous updates

- New best positions are updated after each particle position update.
- Immediate feedback.
- Feedback is imperfect because other particles may not yet have been updated.
- Better for **lbest PSO**.
- Associated with exploration.

---

## 11. Acceleration coefficients

The acceleration coefficients determine the scale distribution of the random vectors making up:

- The cognitive component.
- The social component.

### Cases discussed in the lecture

#### \(\phi_1 > 0,\ \phi_2 = 0\)

Particles are independent hill-climbers.

#### \(\phi_1 = 0,\ \phi_2 > 0\)

The swarm is one stochastic hill-climber.

#### \(\phi_1 = \phi_2 > 0\)

Particles are attracted toward the average of \(p_i\) and \(g_i\).

#### \(\phi_2 > \phi_1\)

More beneficial for unimodal problems.

#### \(\phi_1 > \phi_2\)

More beneficial for multimodal problems.

### Magnitude of the coefficients

Low \(\phi_1,\phi_2\):

- Smooth particle trajectories.

High \(\phi_1,\phi_2\):

- More acceleration.
- More abrupt movements.

Adaptive acceleration coefficients have also been proposed, for example decreasing \(\phi_1\) and \(\phi_2\) over time.

---

## 12. Problems with original PSO

The lecture states:

- The acceleration coefficients should be set sufficiently high.
- Higher acceleration coefficients result in less stable systems.
- Particle velocity can have a tendency to **explode**.

A common fix is to limit velocity to a maximum range using \(v_{\max}\).

However:

- Limiting velocity does not necessarily prevent particles from leaving the search space.
- It does not guarantee convergence.

---

## 13. Inertia-weighted PSO

An inertia weight \(\omega\) was introduced to control velocity explosion.

The update becomes:

\[
v_i
\leftarrow
\omega v_i
+
U(0,\phi_1)\otimes(p_i-x_i)
+
U(0,\phi_2)\otimes(g_i-x_i)
\]

If \(\omega\), \(\phi_1\), and \(\phi_2\) are set correctly, this update rule allows convergence without using \(v_{\max}\).

The inertia weight controls the balance between exploration and exploitation.

### If \(\omega \ge 1\)

- Velocities increase over time.
- The swarm diverges.

### If \(0 < \omega < 1\)

- Particles decelerate.
- Convergence depends on \(\phi_1\) and \(\phi_2\).

### Rule-of-thumb settings

\[
\omega = 0.7298
\]

\[
\phi_1 = \phi_2 = 1.49618
\]

---

## 14. Time-decreasing inertia weight

Eberhart and Shi proposed decreasing \(\omega\) over time.

A typical schedule is:

\[
0.9 \rightarrow 0.4
\]

This gradually changes the behavior of the swarm from:

\[
\text{exploration} \rightarrow \text{exploitation}
\]

Other dynamically changing inertia-weight schemes are also possible.

The lecture compares this to the **cooling scheme in Simulated Annealing**.

---

## 15. Constricted Coefficients PSO

Constriction is introduced to:

- Reduce guesswork in parameter setting.
- Prevent velocity explosion.
- Ensure convergence.
- Eliminate \(v_{\max}\).

The velocity update becomes:

\[
v_i
\leftarrow
\chi
\left(
v_i
+
U(0,\phi_1)\otimes(p_i-x_i)
+
U(0,\phi_2)\otimes(g_i-x_i)
\right)
\]

Define:

\[
\phi = \phi_1 + \phi_2 > 4
\]

The constriction coefficient is:

\[
\chi =
\frac{2}
{\phi+\sqrt{\phi^2-4\phi}}
\]

---

## 16. Fully Informed Particle Swarms (FIPS)

In FIPS, each particle is affected by **all of its \(K\) neighbors**.

The velocity update given in the lecture is:

\[
v_i
\leftarrow
\chi
\left(
v_i
+
\frac{1}{K_i}
\sum_{n=1}^{K_i}
U(0,\phi)
\otimes
(p_{nbr_n}-x_i)
\right)
\]

The position update remains:

\[
x_i \leftarrow x_i + v_i
\]

The lecture states:

- FIPS outperforms canonical PSO on most test problems.
- FIPS performance is generally more dependent on neighborhood topology.
- Global-best neighborhood topology is recommended for FIPS.

---

## 17. Bare Bones PSO

Bare Bones PSO removes the velocity update.

The motivation is to eliminate the tricky parameter tuning associated with particle velocity.

Instead of moving particles by adding velocity, particles move according to a probability distribution.

The lecture uses a Gaussian distribution.

For the \(j\)-th component of the \(i\)-th particle:

\[
x_{ij} = N(\mu_{ij}, \sigma_{ij})
\]

where:

\[
\mu_{ij}
=
\frac{p_{ij}+g_{ij}}{2}
\]

and:

\[
\sigma_{ij}
=
|p_{ij}-g_{ij}|
\]

So the Gaussian has:

- Mean equal to the midpoint between the particle's personal best and neighborhood best.
- Standard deviation equal to their absolute distance.

The lecture notes:

- Bare Bones PSO works fairly well.
- The Gaussian distribution does not seem to be the best probability distribution.
- The Lévy distribution has also been tried.

---

## 18. Binary / discrete PSO

Binary/discrete PSO modifies PSO for binary search spaces.

The velocity remains **continuous** and still uses the original velocity update rule.

The position is binary.

A random threshold is sampled:

\[
\tau \sim U(0,1)
\]

The velocity is converted into a probability using:

\[
s(v_{ij})
=
\frac{1}{1+\exp(-v_{ij})}
\]

Then:

\[
x_{ij}
=
\begin{cases}
1 & \text{if } \tau < s(v_{ij})\\
0 & \text{otherwise}
\end{cases}
\]

Thus, the continuous velocity determines the probability that a binary component becomes 1.

---

## 19. Lecture summary

The lecture summarizes PSO as:

- A swarm-based algorithm for continuous optimization.
- A velocity update consisting of:
  - Inertia.
  - Cognitive component.
  - Social component.
- A social component determined by neighborhood topology / communication network.
- An algorithm whose behavior is strongly affected by acceleration coefficients.
- An algorithm with many variants.
