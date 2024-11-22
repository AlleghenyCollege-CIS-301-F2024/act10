# Activity 10

You Teach Me: Scikit-Learn Methods

## Assigned: Friday, 22nd November 2024

## Due: Monday, 25th November 2024 __by the end of class__

---

## Learning Objectives

- Follow tutorials to learn core concepts.
- Teach yourself how to apply these concepts through coding.
- Present your findings, introducing others to your concepts and code.
- Gain experience using Python and SciKit-Learn.

## Instructions

Individual work: You will edit two files to complete this activity: `src/tutorialCode.py`, and `writing/reflection.md`. Below are tutorials made available from [Scikit-Learn](https://scikit-learn.org/) that explain and demonstrate how to use different types of machine methods. You are to pick a tutorial to follow from the below menu, capture and execute the presented code, and then explain its fundamental mechanism to the class as a Lighting Talk during next class.

Note: If you cannot get the `src/tutorialCode.py` to run, then you can submit a Jupyter notebook file of the same name. You can find a working Jupyter notebook at [https://www.oliverbonhamcarter.com/live/lab/index.html](https://www.oliverbonhamcarter.com/live/lab/index.html). Be be aware that if you save your work to this Jupyter Notebook, then you will need to make a backup to store somewhere else as Jupyter will save to your browser's cookies.

## Menu of Machine leaning tutorials

Please be sure that at least one group covers the Neural Networks method from below. Also, please be sure to pick a tutorial that comes with code to execute.

### __Supervised Learning Methods (Including Neural Networks):__

1. __Neural Networks (MLPClassifier/MLPRegressor)__  
   - __Description__: A supervised learning method that mimics the structure of the human brain, with layers of interconnected neurons to model complex relationships. Multi-layer perceptrons (MLPs) are a common type of feedforward neural network.  
   - __Problem Type__: Suitable for both classification and regression tasks, especially when dealing with large datasets or complex patterns, such as image recognition, speech recognition, and predictive modeling.
   - [URL](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)  

2. __Linear Regression__  
   - __Description__: A regression algorithm that models the relationship between a dependent variable and one or more independent variables using a linear equation.  
   - __Problem Type__: Suitable for predicting continuous values, such as predicting house prices or stock prices.
   - [URL](https://scikit-learn.org/stable/modules/linear_model.html#linear-regression)  

3. __Logistic Regression__  
   - __Description__: A classification algorithm that models the probability of a binary outcome using a logistic function.  
   - __Problem Type__: Ideal for binary classification problems, such as spam detection or disease diagnosis.
   - [URL](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)  

4. __Support Vector Machines (SVM)__  
   - __Description__: A powerful classification algorithm that finds the hyperplane that best separates data points of different classes.  
   - __Problem Type__: Suitable for both binary and multiclass classification problems, particularly with high-dimensional data like image classification.
   - [URL](https://scikit-learn.org/stable/modules/svm.html)  

5. __Decision Trees__  
   - __Description__: A model that splits data into subsets based on feature values, creating a tree-like structure for decision making.  
   - __Problem Type__: Suitable for both classification and regression tasks, like predicting customer churn or classifying diseases.
   - [URL](https://scikit-learn.org/stable/modules/tree.html)  

6. __Random Forest__  
   - __Description__: An ensemble method that builds multiple decision trees and combines their results to improve performance and reduce overfitting.  
   - __Problem Type__: Effective for classification and regression problems, such as customer segmentation or sales forecasting.
   - [URL](https://scikit-learn.org/stable/modules/ensemble.html#random-forest)  

7. __K-Nearest Neighbors (KNN)__  
   - __Description__: A non-parametric method that classifies data based on the majority label of its nearest neighbors in the feature space.  
   - __Problem Type__: Useful for classification tasks with small to medium datasets, such as handwritten digit recognition or recommendation systems.
   - [URL](https://scikit-learn.org/stable/modules/neighbors.html#k-nearest-neighbors)  

8. __Naive Bayes__  
   - __Description__: A probabilistic classifier based on Bayes' theorem that assumes independence between features.  
   - __Problem Type__: Well-suited for text classification tasks, such as email filtering or sentiment analysis.
   - [URL](https://scikit-learn.org/stable/modules/naive_bayes.html)  

---

### __Unsupervised Learning Methods:__

1. __K-Means Clustering__  
   - __Description__: A clustering algorithm that partitions data into K distinct clusters by minimizing the variance within each cluster.  
   - __Problem Type__: Used for segmenting data, such as customer segmentation or grouping similar documents.
   - [URL](https://scikit-learn.org/stable/modules/clustering.html#k-means)  

2. __DBSCAN (Density-Based Spatial Clustering of Applications with Noise)__  
   - __Description__: A density-based clustering algorithm that can find arbitrarily shaped clusters and detect outliers in the data.  
   - __Problem Type__: Suitable for identifying clusters in spatial or geographic data, and can handle noise, such as in geospatial analysis.
   - [URL](https://scikit-learn.org/stable/modules/clustering.html#dbscan)  

3. __Principal Component Analysis (PCA)__  
   - __Description__: A dimensionality reduction technique that transforms the data into a lower-dimensional space while preserving variance.  
   - __Problem Type__: Useful for data preprocessing and reducing features before applying other machine learning algorithms, often in high-dimensional data like images.
   - [URL](https://scikit-learn.org/stable/modules/decomposition.html#pca)  

4. __Gaussian Mixture Models (GMM)__  
   - __Description__: A probabilistic model that assumes all data points are generated from a mixture of several Gaussian distributions.  
   - __Problem Type__: Suitable for clustering problems where the data is expected to come from multiple distributions, such as in image segmentation or anomaly detection.
   - [URL](https://scikit-learn.org/stable/modules/mixture.html#gaussian-mixture)  

5. __Hierarchical Clustering__  
   - __Description__: A clustering method that builds a tree of clusters in a hierarchical manner, either through agglomerative or divisive strategies.  
   - __Problem Type__: Useful for exploratory data analysis, and finding hierarchical relationships in the data, such as in biology (species classification).
   - [URL](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)  

6. __Isolation Forest__  
   - __Description__: An anomaly detection algorithm that isolates observations by randomly selecting features and splitting data, making it well-suited for outlier detection.  
   - __Problem Type__: Effective for identifying anomalies or outliers in data, like fraud detection or identifying rare events.
   - [URL](https://scikit-learn.org/stable/modules/ensemble.html#isolation-forest)  

7. __t-Distributed Stochastic Neighbor Embedding (t-SNE)__  
   - __Description__: A technique for dimensionality reduction that is particularly good at preserving local structure in high-dimensional data.  
   - __Problem Type__: Suitable for visualizing high-dimensional data, such as in visualizing embeddings of neural networks or exploring clustering results.
   - [URL](https://scikit-learn.org/stable/modules/manifold.html#t-sne)  

### Code

#### `tutorialCode.py`

Follow the code in the tutorial exactly as presented. Your goal is to get it running on your machine by copying and pasting the code. After execution, the code should produce meaningful output, such as plots, tables, or other forms of analysis. Be sure to place this tutorial code into the `src/tutorialCode.py` file.

### Writing

#### `reflection.md`

Once you have completed and tested your code (with proper citations and comments), move on to the reflection document. Write your reflections in the `writing/reflection.md` file.

### Lightning Talk During Lab Session

During next class, we will be holding lightning talks for you to present your work. Your presentation should be 3 to 5 minutes long and consist of 5 to 7 slides. The slides should cover the following points;

- __Overview__: Which tutorial did you use? What is this tutorial used for?
  - __Method__: What is the method?
  - __Type__: Supervised or unsupervised? What makes this method one or the other type?
  - __Application__: What kinds of problems can it address?
  - __Demo of Code__: Run the code from the tutorial and explain the general output.
  - __Conclusions__: What conclusions can you draw about this method?

## Project Assessment

This assignment will be graded on a checkmark scale.

## GatorGrade Checking

You can use `gatorgrade` to check if you meet the baseline requirements for this assignment. To do this, run the following command in your terminal:

```
gatorgrade --config config/gatorgrade.yml
```

If `gatorgrade` confirms that all checks pass, you will know that your project meets the basic requirements.

Once you have pushed your work to your repository, visit the repository on GitHub (you may need to log in) to verify that your files have been successfully uploaded.

### Installing GatorGrade

1. First, [install `pipx`](https://pypa.github.io/pipx/installation/).
2. Then, install `gatorgrade` by running the following command:

```
pipx install gatorgrade
```

Additionally, you may want to install [VSCode](https://code.visualstudio.com/docs/setup/setup-overview) (or another text editor) to assist with editing your code files.

## Seeking Assistance

If you have questions about the project outside of class or lab time, please feel free to ask them in the course's Discord channel or during the instructor’s or teaching assistant's office hours.
