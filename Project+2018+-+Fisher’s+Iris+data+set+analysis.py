
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
# I am using the Jupyter Notebook for this project.

# In[211]:


import pandas as pd
import numpy as np
import holoviews as hv
hv.extension('bokeh', 'matplotlib')


# # Data
# Import the iris.csv using the panda library and examine first few rows of data

# In[315]:


iris_data = pd.read_csv('assets/iris.csv')

iris_data.columns = ['sepal_length', 'sepal_width' , 'petal_length', 'petal_width', 'species']

#you can specific the number to show here
iris_data.head(10)


# # Discovering the Shape of the table

# In[213]:


iris_data.shape


# # Find out unique classification/type of iris flower and the amount

# In[310]:


iris_data['species'].unique()


# In[313]:


print(iris_data.groupby('species').size())


# # Investigating the data: Min, Max, Mean, Median and Standard Deviation
# Get the minimum value of all the column in python pandas
# 

# In[214]:


iris_data.min()


# Get the maximum value of all the column in python pandas

# In[215]:


iris_data.max()


# 
# 
# Get the mean value of all the column in python pandas

# In[167]:


iris_data.mean()


# Get the median value of all the column in python pandas

# In[168]:


iris_data.median()


# Get the standard deviation value of all the column in python pandas

# In[169]:


iris_data.std()


# # Presenting the Summary Statistics a more readable way
# DataFrame.describe: Generates descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
# 
# R has a faster way of getting the data with summary
# 

# In[243]:


summary = iris_data.describe()
summary = summary.transpose()
summary.head()


# From the above summary, we can see there is  huge range in the size of the Sepal Length and Petal Length. We will use a scatter plot to see if the size is related to the species of Iris.

# Let's investigate the various species and see if there are any obvious differences.
# 
# # Boxplot
# Comparing the distributions of:
# - Sepal Length
# - Sepal Width
# - Petal Length
# - Petal Width 
# 
# 

# In[265]:


# A BoxWhisker Element is a quick way of visually summarizing one or more groups of numerical data 
#through their quartiles.

# Using holoviews Boxplot

title = "Compare the Distributions of Sepal Length"

boxwhisker = hv.BoxWhisker(iris_data, ['species'], 'sepal_length', label=title )

plot_opts = dict(show_legend=True, width=600, height=600)
style = dict(color='species')

boxwhisker(plot=plot_opts, style=style )



# In[266]:


title = "Compare the distributions of Sepal Width"

boxwhisker = hv.BoxWhisker(iris_data, ['species'], 'sepal_width', label=title)

plot_opts = dict(show_legend=True, width=600, height=600)
style = dict(color='species')

boxwhisker(plot=plot_opts, style=style)


# In[267]:


title = "Compare the distributions of Petal Length"

boxwhisker = hv.BoxWhisker(iris_data, ['species'], 'petal_length', label=title)

plot_opts = dict(show_legend=True, width=600, height=600)
style = dict(color='species')

boxwhisker(plot=plot_opts, style=style)


# In[268]:


title = "Compare the distributions of Petal Width"

boxwhisker = hv.BoxWhisker(iris_data, ['species'], 'petal_width', label=title)

plot_opts = dict(show_legend=True, width=600, height=600)
style = dict(color='species')

boxwhisker(plot=plot_opts, style=style)


# From the boxplot chart analysis, there are clear differences in the size of the Sepal Length, Petal Length and Petal Width.

# # Histogram
# As there is a big difference is the min and max of Sepal Length. Let's see the distribution of Sepal Length and Species
# 

# In[330]:


import seaborn as sns
sns.set()

# Scatter plots for the features
sns.pairplot(iris_data, hue="species")
plt.show()


# # Comparing the Petal Width and Petal Length across the different Species 
# Using different colours it is clear that the three species have very different petal sizes.

# In[204]:


from bokeh.plotting import figure


#adding colors
colormap = {'Iris-setosa': color1, 'Iris-versicolor': color2, 'Iris-virginica': color3}
colors = [colormap[x] for x in iris_data['species']]

#adding labels
p = figure(title = "Petal Width and Petal Length")
p.xaxis.axis_label = 'Petal Length'
p.yaxis.axis_label = 'Petal Width'
p.legend

p.diamond(iris_data["petal_length"], iris_data["petal_width"],color=colors, fill_alpha=0.2, size=10)


show(p)


# # Comparing the Petal Width and Petal Length across the different Species 

# In[203]:


from bokeh.plotting import figure

#adding colors
colormap = {'Iris-setosa': color1, 'Iris-versicolor': color2, 'Iris-virginica': color3}
colors = [colormap[x] for x in iris_data['species']]

#adding labels
p = figure(title = "Petal Width and Petal Length")
p.xaxis.axis_label = 'Petal Length'
p.yaxis.axis_label = 'Petal Width'


p.circle(iris_data["petal_length"], iris_data["petal_width"],
         color=colors, fill_alpha=0.2, size=10)


show(p)


# In[207]:



iris_data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()


# In[308]:


import numpy as np
from bokeh.io import show, output_file
from bokeh.plotting import figure

data = np.random.normal(0, 0.5, 1000)
hist, edges = np.histogram(data, density=True, bins=50)

x = np.linspace(-2, 2, 1000)

p = figure()
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="white")

output_file("hist.html")
show(p)


# In[216]:





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
# Statistics in Python
# http://www.scipy-lectures.org/packages/statistics/index.html#statistics
# 
# Python - IRIS Data visualization and explanation
# https://www.kaggle.com/abhishekkrg/python-iris-data-visualization-and-explanation
# 
# Docs
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
# http://holoviews.org/gallery/demos/bokeh/boxplot_chart.html
# 
# Machine Learning Tutorial
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# https://github.com/whatsrupp/iris-classification/blob/master/petal_classifier.py
# https://diwashrestha.com/2017/09/18/machine-learning-on-iris/
# https://www.youtube.com/watch?v=rNHKCKXZde8
