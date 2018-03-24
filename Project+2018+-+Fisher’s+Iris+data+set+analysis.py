
# coding: utf-8

# # Research
# 
# The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper "The use of multiple measurements in taxonomic problems" as an example of linear discriminant analysis.
# This famous iris data set gives the measurements in centimeters of the variables sepal length and width and petal length and width, respectively, for 50 flowers from each of 3 species of iris. The species are Iris setosa, versicolor, and virginica.
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

# Importing the libaries for this project: Pandas, Numpy, Holoviews.
# Pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools.
# NumPy is the fundamental package for scientific computing with Python
# HoloViews is an open-source Python library designed to make data analysis and visualization seamless and simple.

# In[1]:


import pandas as pd
import numpy as np
import holoviews as hv
hv.extension('bokeh')


# Import the iris.csv using the panda library

# In[37]:


iris_data = pd.read_csv('assets/iris.csv')

iris_data.columns = ['sepal_length', 'sepal_width' , 'petal_length', 'petal_width', 'species']

iris_data.head()


# In[40]:


from bokeh.plotting import figure, show

#adding colors
colormap = {'Iris-setosa': 'red', 'Iris-versicolor': 'green', 'Iris-virginica': 'blue'}
colors = [colormap[x] for x in iris_data['species']]

#adding labels
p = figure(title = "Iris Morphology")
p.xaxis.axis_label = 'Petal Length'
p.yaxis.axis_label = 'Petal Width'


p.circle(iris_data["petal_length"], iris_data["petal_width"],
         color=colors, fill_alpha=0.2, size=10)


show(p)

