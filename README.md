# Week-1-Decision-Trees
Learn the most basic basicest decision tree in Python and use it to predict cool stuff! 

Decisions.

![Car](assets/jgmm-cat-meme.gif)

We all have to make them.

But what if we can program something to make deicisions for us. 

That's not what a decision tree actually does, but it's pretty close. A decision tree is a model that basically splits data into "branches" based on the values of specific features to make predictions. At each node, it chooses the feature that best separates the data, continuing until it reaches a leaf node that assigns a class or value (ultimate output). Essentially, you give it a bunch of data, it then splits that data and learns from it, and then can make predictions based on the matching the features of the inputs you give it to what it knows to be the most likely item that has all those features. 

### Week 1 Objectives:

- Learn how to make decision trees in Python using Scikit-Learn
- Get used to using datasets, maybe dig around and try to figure out how to make one of your own
- Learn about choosing features & visualizing trees

## 1) Make sure you have Python and Scikit-Learn set up

[computer thing]

VS Code: https://code.visualstudio.com
youtube wisdom: https://www.youtube.com/watch?v=GM_xE2GDsfo

## 2) Pick a dataset from SciKit-Learn

https://scikit-learn.org/stable/datasets/toy_dataset.html

(You don't HAVE to use one of these datasets, but they're available for use without downloading anything from some sus website)

## 3) Train your tree

Using the iris set as an example, making and training a tree would look a little like this:

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#loading the data
data = load_iris()
X = data['data']
y = data['target']

#splitting data between training sets and testing sets, sets to learn from and sets to evaluate model performance with 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#training the decsion tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

## 3) Make predictions on a new set of data 

Now, to actually be able to use our decision tree, we're going to make an imaginary new flower with the following features:

flower1 = [[5.1, 3.5, 1.4, 0.2]] #[sepal length, sepal width, petal length, petal width]

And now, we can predict which flower we're looking at!

prediction = clf.predict(new_flower)
print("Predicted class:", prediction)
print("Predicted species name:", data['target_names'][prediction[0]])

## 4) Try it yourself!!!!

[the thingy]

Please. 

Try it out with a different dataset! Experiment! Find new ways to visualize the trees, maybe solve a few problems. 

Some interesting decision trees that have been programmed over the years:

Healthcare:

https://pmc.ncbi.nlm.nih.gov/articles/PMC4251295/

COVID-19:

https://intjem.biomedcentral.com/articles/10.1186/s12245-024-00681-7

Finance:

https://arxiv.org/abs/2007.06617



