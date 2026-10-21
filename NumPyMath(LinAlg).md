# NumPyMath(LinAlg) 

## Explained np.zeros array shape

np.zeros((1, 4)) creates a 2D NumPy array   
with 1 row and 4 columns, all initialized to 0.0 (float64 by default):

| import numpy as np np.zeros((1, 4))  ←  output  array(\[\[0., 0., 0., 0.\]\])   → |
| :---- |

The shape tuple (1, 4\) means 1 row × 4 columns.   
If you need integers instead, pass dtype=int.

## This is the forward pass of your single-hidden-layer network:

1. z1 \= X @ W1 \+ b1 — linear combination for the hidden layer (inputs × weights \+ biases)  
2. a1 \= sigmoid(z1) — squash each value into (0, 1), introducing non-linearity  
3. z2 \= a1 @ W2 \+ b2 — linear combination for the output layer  
4. a2 \= sigmoid(z2) — final output, also in (0, 1\)

Since you're using sigmoid at the output, a2 is a probability<br>
in (0, 1\) — typical for binary classification.  
The shapes work out:

| Step | Shape |
| :---- | :---- |
| X | (n, 2\) |
| z1 | (n, 4\) |
| a1 | (n, 4\) |
| z2 | (n, 1\) |
| a2 | (n, 1\) |

where n is the number of samples.   
*You'd next compute a loss (e.g., cross-entropy),*   
*backpropagate gradients, and update W1, b1, W2, b2.*<br>
<br>
*z1 contains a matrix multiplication. <br>
is the output another array?*<br>
<br>
Yes. X @ W1 is a matrix multiplication that produces <br>
a new array of shape (n, 4). Adding b1 <br>
(broadcast from (1, 4\) to (n, 4)) keeps it at (n, 4).  

So z1 is a fresh (n, 4\) NumPy array <br>
— it doesn't modify X or W1 in place. <br>
Every operation in NumPy returns a new array<br>
(unless you use in-place variants like \+= or \*=).<br>
<br>
*can you show the liner combination? <br>
the z1   sigmoid input?  math refresher style*<br>
<br>
Sure. Let's break down z1 \= X @ W1 \+ b1 <br>
for a single sample (one row of X):<br>

## The linear combination

Each hidden neuron j computes a weighted sum<br>
of the inputs plus a bias:<br>
<br>
z1,j = x1​⋅w1j + x2⋅w2j + bj <br>
​ <br>
For 4 hidden neurons, that's 4 such sums:

| Neuron | Formula |
| :---- | :---- |
| z₁,₁ | x₁·w₁₁ \+ x₂·w₂₁ \+ b₁ |
| z₁,₂ | x₁·w₁₂ \+ x₂·w₂₂ \+ b₂ |
| z₁,₃ | x₁·w₁₃ \+ x₂·w₂₃ \+ b₃ |
| z₁,₄ | x₁·w₁₄ \+ x₂·w₂₄ \+ b₄ |

## Concrete mini-example

Say one sample is x \= \[3, \-1\], and <br>
(hypothetically) the first column of <br>
W1 is \[0.5, \-0.2\], with b₁ \= 0:<br>
<br>
z1,1=3(0.5)+(−1)(−0.2)+0=1.5+0.2=1.7<br>
<br>
That single number 1.7 is what gets <br>
fed into sigmoid(1.7) ≈ 0.846.

## Why "linear combination"?

It's a sum of scaled inputs<br>
— each input is multiplied by a weight (scaled), <br>
then all products are added together (combined). <br>
<br>
The bias is just an extra constant added at the end.<br>
No powers, no products of inputs with each other <br>
— that's what makes it linear.<br>
The sigmoid applied afterward is what introduces<br>
the non-linearity.
