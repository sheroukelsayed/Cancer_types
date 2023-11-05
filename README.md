# Cancer_types
 classify cells to whether the samples are benign or malignant

we compare 5 popular machine learning algorithms:KNN, Decision Tree, Logistic Regression, Random Forest, and SVM
​Here is a comparison of the 5 machine learning algorithms with explanations merged with key equations:

K-Nearest Neighbors (KNN):
KNN makes predictions by searching the training set for the K most similar instances (neighbors) and outputting the majority class among the neighbors. It does not have an explicit mathematical model, and instead relies on similarity measures:

Prediction = majority vote of K closest training samples 

Decision Tree:
Decision trees work by recursively splitting the data on feature values that result in the largest information gain at each node. Information gain is based on reducing entropy in the split data:

Information Gain = Entropy(parent) - Σ[(Probability of split) x Entropy(child)]

At each node, choose the feature that maximizes information gain. Predictions are made by following decisions from root to leaf node.

Logistic Regression:
Logistic regression calculates the probability P(Y=1|X) using the logistic function:

P(Y=1|X) = 1/(1+e^(-wx+b))

It optimizes the weights w and bias b to maximize the likelihood of the training data. The decision boundary is formed by thresholding the probability.

Random Forest:
Random forest creates multiple decision trees on random subsets of data and features. Each tree makes a prediction and the final prediction is the majority vote:

Prediction = mode of predictions from all decision trees

By aggregating predictions across diverse trees, overfitting is reduced.

Support Vector Machine (SVM):
SVM constructs a hyperplane f(x) = w^Tx + b to separate the classes with maximum margin. For non-linearly separable classes, it maps data to a high-dim space using kernels. The prediction is made by determining which side of the hyperplane a data point falls on. The optimization objective is to maximize the margin between classes.
Load the Cancer data
The example is based on a dataset that is publicly available from the UCI Machine Learning Repository (Asuncion and Newman, 2007)[http://mlearn.ics.uci.edu/MLRepository.html]. The dataset consists of several hundred human cell sample records, each of which contains the values of a set of cell characteristics. The fields in each record are:
Field name	Description
ID	        Clump thickness
Clump	        Clump thickness
UnifSize	Uniformity of cell size
UnifShape	Uniformity of cell shape
MargAdh	Marginal adhesion
SingEpiSize	Single epithelial cell size
BareNuc	Bare    nuclei
BlandChrom	Bland chromatin
NormNucl	Normal nucleoli
Mit	        Mitoses
Class	        Benign or malignant


For the purposes of this example, we're using a dataset that has a relatively small number of predictors in each record. To download the data, we will use !wget to download it from IBM Object Storage.
