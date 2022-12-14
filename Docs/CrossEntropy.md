# Cross Entropy

`One-Hot Encoding` workds very well for most problems until you get situation when you have millions of classes, in that case, your vector becomes very large with a lot of zeros every where.

Using embedings, we can mesure how well we are doing by comparing the distance $D$ between 2 vectors. One that comes up from the classifier (Logistic Classifier) that contains the probabilities of our clases and the One-Hot encoded vector that correspons to the labels.

|S(Y)|L|
|---|---|
|0.7|1|
|0.2|0|
|0.1|0|

$$ D(S,L) = -\sum_{j}L_{i}*log(S_{i}) $$

The *Cross Entropy* is not symetric, because of the nasted $log$ in the equation. There will be 0 in $L$ vector where in the $S(Y)$ vector is guaranted that you always have a litle bit of probability every where, so you will never take a $log(0)$

$$ D(S,L) \neq D(L,S) $$ 