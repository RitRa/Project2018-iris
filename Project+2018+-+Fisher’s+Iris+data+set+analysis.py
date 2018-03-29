
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

# In[189]:


import pandas as pd
import numpy as np
import holoviews as hv
hv.extension('bokeh', 'matplotlib')


# # Data
# Import the iris.csv using the panda library and examine first few rows of data

# In[163]:


iris_data = pd.read_csv('assets/iris.csv')

iris_data.columns = ['sepal_length', 'sepal_width' , 'petal_length', 'petal_width', 'species']

iris_data.head()


# # Discovering the Shape of the table

# In[164]:


iris_data.shape


# # Investigating the data: Min, Max, Mean, Median and Standard Deviation
# Get the minimum value of all the column in python pandas
# 

# In[165]:


iris_data.min()


# Get the maximum value of all the column in python pandas

# In[166]:


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


# # Presenting the summary statistics a more readable way
# DataFrame.describe: Generates descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
# 
# R has a faster way of getting the data with summary
# 

# In[170]:


summary = iris_data.describe()
summary = summary.transpose()
summary.head()


# From the above summary, we can see there is  huge range in the size of the Sepal Length and Petal Length. We will use a scatter plot to see if the size is related to the species of irish.

# # Boxplot
# Comparing the distributions of:
# - Sepal Length
# - Sepal Width
# - Petal Length
# - Petal Width 
# 
# This should give us a clearer picture in the differences between the species.

# In[171]:


title = "Compare the distributions of Sepal Length"

boxwhisker = hv.BoxWhisker(iris_data, ['species'], 'sepal_length', label=title)

plot_opts = dict(show_legend=False, width=400)
style = dict(color='species')

boxwhisker(plot=plot_opts, style=style)


# In[172]:


title = "Compare the distributions of Sepal Width"

boxwhisker = hv.BoxWhisker(iris_data, ['species'], 'sepal_width', label=title)

plot_opts = dict(show_legend=False, width=400)
style = dict(color='species')

boxwhisker(plot=plot_opts, style=style)


# In[173]:


title = "Compare the distributions of Petal Length"

boxwhisker = hv.BoxWhisker(iris_data, ['species'], 'petal_length', label=title)

plot_opts = dict(show_legend=False, width=400)
style = dict(color='species')

boxwhisker(plot=plot_opts, style=style)


# In[174]:


title = "Compare the distributions of Petal Width"

boxwhisker = hv.BoxWhisker(iris_data, ['species'], 'petal_width', label=title)

plot_opts = dict(show_legend=False, width=400)
style = dict(color='species')

boxwhisker(plot=plot_opts, style=style)


# # Histogram
# As there is a big difference is the min and max of Sepal Length. Let's see the distribution of Sepal Length and Species
# 

# In[176]:


import matplotlib.pyplot as plt

data = iris_data

data.hist(figsize=(10, 10))
plt.show()


# In[202]:


from pandas.plotting import scatter_matrix

data = iris_data

color1 = '#fcc5c0'
color2 = '#f768a1'
color3 = '#7a0177'

colormap = {'Iris-setosa': color1, 'Iris-versicolor': color2, 'Iris-virginica': color3}
colors = [colormap[x] for x in iris_data['species']]

scatter_matrix(data, alpha=0.5, color=colors, figsize=(10, 10))

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


# In[185]:


hv.help(hv.Histogram)


# In[205]:


from bokeh.plotting import figure


hist, edges = np.histogram(iris_data, density=True, bins=50)



p = figure()
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="white")

output_file("hist.html")
show(p)




# In[199]:


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
# Docs
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
# http://holoviews.org/gallery/demos/bokeh/boxplot_chart.html
# 
