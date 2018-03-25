
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

# In[66]:


import pandas as pd
import numpy as np
import holoviews as hv
hv.extension('bokeh', 'matplotlib')


# # Data
# Import the iris.csv using the panda library and examine first few rows of data

# In[81]:


iris_data = pd.read_csv('assets/iris.csv')

iris_data.columns = ['sepal_length', 'sepal_width' , 'petal_length', 'petal_width', 'species']

iris_data.head()


# Shape of the table

# In[68]:


iris_data.shape


# # Investigating the data: Min, Max, Mean, Median and Standard Deviation
# Get the minimum value of all the column in python pandas
# 

# In[69]:


iris_data.min()


# Get the maximum value of all the column in python pandas

# In[51]:


iris_data.max()


# 
# 
# Get the mean value of all the column in python pandas

# In[47]:


iris_data.mean()


# Get the median value of all the column in python pandas

# In[49]:


iris_data.median()


# Get the standard deviation value of all the column in python pandas

# In[50]:


iris_data.std()


# # Calculate the summary statistics in a different way
# DataFrame.describe:Generates descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
# 
# R has a faster way of getting the data with summary
# 

# In[90]:


summary = iris_data.describe()
summary = summary.transpose()
summary.head()


# # Boxplot
# Comparing Sepal Length, Sepal Width, Petal Length, Petal Width 

# In[100]:


title = "Compare the distributions of Sepal Length"

boxwhisker = hv.BoxWhisker(iris_data, ['species'], 'sepal_length', label=title)

plot_opts = dict(show_legend=False, width=400)
style = dict(color='species')

boxwhisker(plot=plot_opts, style=style)


# In[101]:


title = "Compare the distributions of Sepal Width"

boxwhisker = hv.BoxWhisker(iris_data, ['species'], 'sepal_width', label=title)

plot_opts = dict(show_legend=False, width=400)
style = dict(color='species')

boxwhisker(plot=plot_opts, style=style)


# In[102]:


title = "Compare the distributions of Petal Length"

boxwhisker = hv.BoxWhisker(iris_data, ['species'], 'petal_length', label=title)

plot_opts = dict(show_legend=False, width=400)
style = dict(color='species')

boxwhisker(plot=plot_opts, style=style)


# In[103]:


title = "Compare the distributions of Petal Width"

boxwhisker = hv.BoxWhisker(iris_data, ['species'], 'petal_width', label=title)

plot_opts = dict(show_legend=False, width=400)
style = dict(color='species')

boxwhisker(plot=plot_opts, style=style)


# # Histogram
# As there is a big difference is the min and max of Sepal Length. Let's see the distribution of Sepal Length and Species
# 

# In[75]:





# In[80]:





# In[71]:


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


# # References
# https://en.wikipedia.org/wiki/Iris_flower_data_set
# https://archive.ics.uci.edu/ml/datasets/iris
# 
# https://stackoverflow.com/questions/33889310/r-summary-equivalent-in-numpy
# https://rstudio-pubs-static.s3.amazonaws.com/205883_b658730c12d14aa6996fe2f6c612c65f.html
# 
# min value
# http://www.datasciencemadesimple.com/get-minimum-value-column-python-pandas/
# 
# Docs
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
# http://holoviews.org/gallery/demos/bokeh/boxplot_chart.html
# 
