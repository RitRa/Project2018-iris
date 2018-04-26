
# coding: utf-8

# # Research
# 
# The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper "The use of multiple measurements in taxonomic problems" as an example of linear discriminant analysis.
# This famous iris data set gives the measurements in centimeters of the variables sepal length and width and petal length and width, respectively, for 50 flowers from each of 3 species of iris. The species are Iris setosa, versicolor, and virginica. Based on the combination of these four features, Fisher developed a linear discriminant model to distinguish the species from each other.
# 
# The dataset contains a set of 150 records under 5 attributes -
# 
# 1. sepal length in cm 
# 2. sepal width in cm 
# 3. petal length in cm 
# 4. petal width in cm 
# 5. Species: 
# -- Iris Setosa 
# -- Iris Versicolour 
# -- Iris Virginica
# 

# ![irises.png](attachment:irises.png)

# Importing the libaries for this project: Pandas, Numpy, Holoviews.
# 
# Pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools.
# 
# NumPy is the fundamental package for scientific computing with Python
# 
# HoloViews is an open-source Python library designed to make data analysis and visualization seamless and simple.
# 
# Seaborn is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics.
# 
# I also used the Jupyter Notebook for this project. 

# In[48]:


import pandas as pd
import numpy as np
#import holoviews as hv
import seaborn as sns
#hv.extension('bokeh', 'matplotlib')


# # Data
# Import the iris.csv using the panda library and examine first few rows of data

# In[49]:


iris_data = pd.read_csv('assets/iris.csv')

iris_data.columns = ['sepal_length', 'sepal_width' , 'petal_length', 'petal_width', 'species']

#you can specific the number to show here
iris_data.head(10)


# # Discovering the Shape of the table

# In[50]:


iris_data.shape


# # Find out unique classification/type of iris flower and the amount

# In[51]:


iris_data['species'].unique()


# In[52]:


print(iris_data.groupby('species').size())


# # Investigating the data: Min, Max, Mean, Median and Standard Deviation
# Get the minimum value of all the column in python pandas
# 

# In[53]:


iris_data.min()


# Get the maximum value of all the column in python pandas

# In[54]:


iris_data.max()


# 
# 
# Get the mean value of all the column in python pandas

# In[55]:


iris_data.mean()


# Get the median value of all the column in python pandas

# In[56]:


iris_data.median()


# Get the standard deviation value of all the column in python pandas

# In[57]:


iris_data.std()


# # Presenting the Summary Statistics a more readable way
# DataFrame.describe: Generates descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
# 
# R has a faster way of getting the data with summary
# 

# In[58]:


summary = iris_data.describe()
summary = summary.transpose()
summary.head()


# From the above summary, we can see there is  huge range in the size of the Sepal Length and Petal Length. We will use a scatter plot to see if the size is related to the species of Iris.

# Let's investigate the various species and see if there are any obvious differences.
# 
# # Boxplot
# The boxplot is a quick way of visually summarizing one or more groups of numerical data through their quartiles.
# Comparing the distributions of:
# - Sepal Length
# - Sepal Width
# - Petal Length
# - Petal Width 
# 
# 

# In[59]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid", palette="GnBu_d", rc={'figure.figsize':(11.7,8.27)})

title="Compare the Distributions of Sepal Length"

sns.boxplot(x="species", y="sepal_length", data=iris_data)

# increasing font size
plt.title(title, fontsize=26)
# Show the plot
plt.show()


# In[60]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid", palette="GnBu_d", rc={'figure.figsize':(11.7,8.27)})

title="Compare the Distributions of Sepal Width"

sns.boxplot(x="species", y="sepal_width", data=iris_data)

# increasing font size
plt.title(title, fontsize=26)
# Show the plot
plt.show()


# In[61]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid", palette="GnBu_d", rc={'figure.figsize':(11.7,8.27)})

title="Compare the Distributions of Petal Length"

sns.boxplot(x="species", y="petal_length", data=iris_data)

# increasing font size
plt.title(title, fontsize=26)
# Show the plot
plt.show()


# In[62]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid", palette="GnBu_d", rc={'figure.figsize':(11.7,8.27)})

title="Compare the distributions of Petal Width"

sns.boxplot(x="species", y="petal_width", data=iris_data)

# increasing font size
plt.title(title, fontsize=26)
# Show the plot
plt.show()


# From the boxplot chart analysis, there are clear differences in the size of the Sepal Length, Petal Length and Petal Width among the different species. 

# # Comparing the Petal Width and Petal Length across the different Species 
# Using different colours it is clear that the three species have very different petal sizes.

# In[63]:


from bokeh.plotting import figure, output_file, show
output_file("test1.html")

color1 = '#fcc5c0'
color2 = '#f768a1'
color3 = '#7a0177'
    
#adding colors
colormap = {'Iris-setosa': color1, 'Iris-versicolor': color2, 'Iris-virginica': color3}
colors = [colormap[x] for x in iris_data['species']]

#adding labels
p = figure(title = "Petal Width and Petal Length")
p.xaxis.axis_label = 'Petal Length'
p.yaxis.axis_label = 'Petal Width'
p.legend.location = "top_left"

p.diamond(iris_data["petal_length"], iris_data["petal_width"],color=colors, fill_alpha=0.2, size=10)


show(p)


# # Comparing the Petal Width and Sepal Length across the different Species 

# In[64]:


from bokeh.plotting import figure, output_file, show
output_file("test2.html")

#adding colors
colormap = {'Iris-setosa': color1, 'Iris-versicolor': color2, 'Iris-virginica': color3}
colors = [colormap[x] for x in iris_data['species']]

#adding labels
p = figure(title = "Petal Width and Sepal Length")
p.xaxis.axis_label = 'Sepal Length'
p.yaxis.axis_label = 'Petal Width'


p.circle(iris_data["sepal_length"], iris_data["petal_width"],
         color=colors, fill_alpha=0.2, size=10)


show(p)


# # Pairplot
# Looking for relationships between variables across multiple dimensions

# In[65]:


# My favourite colors = palettes GnBu_d
import seaborn as sns
# Scatter plots for the features and histograms
# custom markers also applied
sns.pairplot(iris_data, hue="species", palette="GnBu_d", markers=["o", "s", "D"])

#Remove the top and right spines from plot
sns.despine()

#show plot
import matplotlib.pyplot as plt
plt.show()




# In[66]:


#setting the background color
import seaborn as sns
sns.set(style="whitegrid")

sns.pairplot(iris_data, hue="species", palette="GnBu_d", diag_kind="kde", markers=["o", "s", "D"])

#Remove the top and right spines from plot
sns.despine()

import matplotlib.pyplot as plt
plt.show()


# In[67]:


# plotting regression and confidence intervals
import seaborn as sns
sns.pairplot(iris_data,hue="species", kind='reg', palette="GnBu_d")

#Remove the top and right spines from plot
sns.despine()

#show plot
import matplotlib.pyplot as plt
plt.show()

#strong relationship between petal length and petal width and petal length and sepal length


# In[68]:


import seaborn as sns
#setting the background color and size of graph
sns.set(style="whitegrid", palette="GnBu_d", rc={'figure.figsize':(11.7,8.27)})

# "Melt" the dataset
iris2 = pd.melt(iris_data, "species", var_name="measurement")

# Draw a categorical scatterplot
sns.swarmplot(x="measurement", y="value", hue="species",palette="GnBu_d", data=iris2)

#Remove the top and right spines from plot
sns.despine()

#show plot
import matplotlib.pyplot as plt
plt.show()


# In[69]:


import seaborn as sns
#setting the background color and size of graph
sns.set(style="whitegrid", palette="GnBu_d", rc={'figure.figsize':(11.7,8.27)})


sns.violinplot(x="species", y="petal_length", palette="GnBu_d", data=iris_data)

#Remove the top and right spines from plot
sns.despine()

#show plot
import matplotlib.pyplot as plt
plt.show()


# In[70]:


import seaborn as sns
#setting the background color and size of graph
sns.set(style="whitegrid", palette="GnBu_d", rc={'figure.figsize':(11.7,8.27)})

sns.violinplot(x="species", y="petal_width", palette="GnBu_d", data=iris_data)
#Remove the top and right spines from plot
sns.despine()


#show plot
import matplotlib.pyplot as plt
plt.show()


# In[71]:


import matplotlib.pyplot as plt
iris_data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()


# # Machine Learning using scikit-learn
# Machine Learning in Python. This is a tutorial I found online. 
# 
# The iris data set already exisits in sklearn so I'm going to reuse it.

# In[109]:


# import load_iris function from datasets module
from sklearn.datasets import load_iris


# In[110]:


# save "bunch" object containing iris dataset and its attributes
iris = load_iris()
type(iris)


# In[111]:


# print the iris data
print(iris.data)


# In[112]:


# print integers representing the species of each observation
print(iris.target)

#print the names of the targets
print(iris.target_names)

# print the names of the four features
print(iris.feature_names)


# In[113]:


from sklearn.datasets import load_iris
iris = load_iris()

from matplotlib import pyplot as plt

# The indices of the features that we are plotting
x_index = 0
y_index = 1

# this formatter will label the colorbar with the correct target names
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

plt.figure(figsize=(5, 4))
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])

plt.tight_layout()
plt.show()


# In[114]:


# print the names of the four features
print(iris.target.shape)


# In[115]:


# store feature matrix in "X"
x = iris.data

# store response vector in "y"
y = iris.target

print(x.shape)

print(y.shape)
print(np.unique(y))


# Using the powerful classification algorithm called K-Nearest-Neighbors (KNN). KNN falls in the supervised learning family of algorithms. 

# In[116]:


# loading library
from sklearn.neighbors import KNeighborsClassifier

# instantiate learning model (k = 1)
knn = KNeighborsClassifier(n_neighbors=1)

print(knn)


# In[117]:


# fitting the model
knn.fit(x, y)


# In[118]:


# What kind of iris has 3cm x 5cm sepal and 4cm x 2cm petal?

result = knn.predict([[3, 5, 4, 2]])

print(iris.target_names[result])


# In[119]:


X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]

# predict the response for new
knn.predict(X_new)


# In[88]:


import numpy as np
from matplotlib import pyplot as plt
from sklearn import neighbors, datasets
from matplotlib.colors import ListedColormap

# Create color maps for 3-class classification problem, as with iris
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features. We could
                    # avoid this ugly slicing by using a two-dim dataset
y = iris.target

knn = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)

x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.axis('tight')
plt.show()


# # References
# Background info
# https://en.wikipedia.org/wiki/Iris_flower_data_set
# https://archive.ics.uci.edu/ml/datasets/iris
# 
# 
# Summary values
# https://stackoverflow.com/questions/33889310/r-summary-equivalent-in-numpy
# 
# R iris project
# https://rstudio-pubs-static.s3.amazonaws.com/205883_b658730c12d14aa6996fe2f6c612c65f.html
# 
# python iris project
# https://rajritvikblog.wordpress.com/2017/06/29/iris-dataset-analysis-python/
# 
# min value
# http://www.datasciencemadesimple.com/get-minimum-value-column-python-pandas/
# 
# A histogram with Iris Dataset: Sora Jin June 21st, 2015
# https://rpubs.com/Sora/developing-data-product
# 
# Plot 2D views of the iris dataset
# http://www.scipy-lectures.org/packages/scikit-learn/auto_examples/plot_iris_scatter.html
# 
# Statistics in Python
# http://www.scipy-lectures.org/packages/statistics/index.html#statistics
# 
# Python - IRIS Data visualization and explanation
# https://www.kaggle.com/abhishekkrg/python-iris-data-visualization-and-explanation
# 
# Visualization with Seaborn (Python)
# https://www.kaggle.com/rahulm7/visualization-with-seaborn-python
# 
# Iris Data Visualization using Python
# https://www.kaggle.com/aschakra/iris-data-visualization-using-python
# 
# Seaborn Understanding the Weird Parts: pairplot
# https://www.youtube.com/watch?v=cpZExlOKFH4
# 
# Docs
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
# 
# http://holoviews.org/gallery/demos/bokeh/boxplot_chart.html
# 
# Machine Learning Tutorial
# http://scikit-learn.org/stable/tutorial/basic/tutorial.html
# 
# https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/
# 
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# 
# https://github.com/whatsrupp/iris-classification/blob/master/petal_classifier.py
# 
# https://diwashrestha.com/2017/09/18/machine-learning-on-iris/
# 
# https://www.youtube.com/watch?v=rNHKCKXZde8
# 
# http://seaborn.pydata.org/examples/scatterplot_categorical.html
# 
# IRIS DATASET ANALYSIS (PYTHON)
# http://d4t4.biz/ml-with-scikit-learn/support-vector-machines-project-wip/
# 
# Getting started in scikit-learn with the famous iris dataset
# https://www.youtube.com/watch?v=hd1W4CyPX58
# http://blog.kaggle.com/2015/04/22/scikit-learn-video-3-machine-learning-first-steps-with-the-iris-dataset/
