# Neural net tracking

It was recommended to make these changes.  
Setup a repo so we can track and revert, if needed.

*Could the NN be written better? A few small improvements:*

| Issue | Why | Better |
| :---- | :---- | :---- |
| pred\[0\]\[0\] in the print | Fragile, relies on knowing the shape | pred.item() or float(pred) — always gives a Python scalar |
| MSE loss \+ sigmoid output | Gradients vanish when the network is confident-wrong | Cross-entropy loss: dz2 \= a2 \- y is actually already the CE gradient, so your code is *already* fine here — just know that the printed "loss" is MSE while the gradient is CE |
| b1 \= np.zeros((1, 4)) | Works but makes everything 2-D | np.zeros(4) — lets numpy broadcast and keeps shapes 1-D, so pred\[0\] would have worked as-is |
| No np.clip or gradient clipping | With a higher lr this could diverge | Minor, but lr \= 0.5 is already aggressive for this problem |

*The biggest single change: make b1 and b2 1-D (np.zeros(4) and np.zeros(1)). Then every intermediate is 1-D, pred is shape (1,), and pred\[0\] is a plain float — no .item() needed.*

## First working output

**d@mac81**:**\~/Downloads/Repo/Kate/NeuralNetwork**\$ python3 nn\_xor.py  
Epoch     0  loss \= 0.283190  
Epoch  2000  loss \= 0.000002  
Epoch  4000  loss \= 0.000000  
Epoch  6000  loss \= 0.000000  
Epoch  8000  loss \= 0.000000

Input  →  Output  
 \[0 0\] → 0.0001  (target: 0\)  
 \[0 1\] → 0.9998  (target: 1\)  
 \[1 0\] → 0.9997  (target: 1\)  
 \[1 1\] → 0.0003  (target: 0\)  
**d@mac81**:**\~/Downloads/Repo/Kate/NeuralNetwork**\$  

