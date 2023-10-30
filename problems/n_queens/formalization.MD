# N-Queens problems

## World state

The world is a list $X = s = [n_1,...,n_n]$ donde $1 \leq n_i \leq n$ . La reina i la denotaremos como $s_i$ cuyo valor sera $n_i$

## Initial states

$S = [x_0,..., x_n]$  donde  $0\leq x \leq n$

## Final states

$amenaza(i,j,s) = (n_i=n_j)\vee(|n_i-n_j|=|i-j|)$

$amenazas = \sum\limits_{i=0}^{n-1} \sum\limits_{j=i+1}^{n} amenaza(i,j,s)$

## Transition function

$[i,b](s) = \begin{cases}
    n_i=n_i & \text{if } i \neq j, \\
    n_i=n_i+b & \text{if } i = j.
\end{cases}$
 donde $b \in \{1,-1\}$