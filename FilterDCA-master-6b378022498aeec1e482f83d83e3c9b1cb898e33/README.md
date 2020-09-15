### FilterDCA 
### interpretable supervised contact prediction using inter-domain coevolution

FilterDCA used 2 features to compute a probability of being a contact for a couple (i,j) in domain1 and domain2.
The first feature is the result of the method plmDCA.
The second one is a pattern score which is computed by apply severals maps on the dca score matrix and keeping the best correlation.

To use the script you need:
- the result of plmDCA for the join-MSA of the 2 domains ;
- the lengh of the 2 domains ;
- the size of filters you want to use : 5, 13, 21, 37, 45 or 69 ;
- and to set the size of the M effictive ('medium' if under 200 and 'big' otherwise).

In the 2 folders you can find:
- the 6 maps (3 corresponding to helix-helix contact, and 3 for strand-strand contacts) for each of the possible sizes (5, 13, 21, 37, 45 or 69)
- the classifier (logistic regression) and the 'min'/'max' values to normalise the correlation/ pattern score

You can then produice :
- the pattern score: the best correlation score matrix
- the matrix of probabililty of contact
- and finaly the predicted contact map

The code is a iPython3 notebook, severals package are needed: pandas, numpy, scipy, matplotlib, pickle and sklearn.

