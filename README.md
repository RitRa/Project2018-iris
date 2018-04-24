# Project2018-iris

The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper "The use of multiple measurements in taxonomic problems" as an example of linear discriminant analysis.
This famous iris data set gives the measurements in centimeters of the variables sepal length and width and petal length and width, respectively, for 50 flowers from each of 3 species of iris. The species are Iris setosa, versicolor, and virginica.
 
The dataset contains a set of 150 records under 5 attributes -

1. sepal length in cm 
2. sepal width in cm 
3. petal length in cm 
4. petal width in cm 
5. Species: 
-- Iris Setosa 
-- Iris Versicolour 
-- Iris Virginica

![iris](assets/irises.png)

## Libraries Used
Importing the libaries for this project: Pandas, Numpy, Holoviews.

Pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools.

NumPy is the fundamental package for scientific computing with Python

HoloViews is an open-source Python library designed to make data analysis and visualization seamless and simple.

Seaborn is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics.

I also used the Jupyter Notebook for this project. 

## Data Import
Import the iris.csv using the panda library and examine first few rows of data

## Discovering the Shape of the table
Find out what the size of rows and columns in the table

## Find out unique classification/type of iris flower and the amount
'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'

## Investigating the data
Min, Max, Mean, Median and Standard Deviation

## Summary Statistics Table
This statistics table is a much nicer, cleaner way to present the data. We can see there is huge range in the size of the Sepal Length and Petal Length. We will use box plots and scatter plots to see if the size is related to the species of Iris.

## Boxplots
The boxplot is a quick way of visually summarizing one or more groups of numerical data through their quartiles. Comparing the distributions of:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

![Boxplot Petal Length](assets/boxplot-petal_length.png)
![Boxplot Petal Width](assets/boxplot-petal_width.png)
![Boxplot Sepal Length](assets/boxplot-sepal_length.png)
![Boxplot Petal Width](assets/boxplot-sepal_width.png)

From the Boxplot, we can see that there are distinct differences between the Petal Length, Petal Width and Sepal Length across the Species. 

## Scatterplots
Here we can use to variables to show that there is distinct difference in sizes between the species. Firstly, we look at the Petal width and Petal length across the species. Is it clear to see that the iris Setosa has a significantly smaller petal width and petal length than the other two species. This difference occurs again for the Petal width and Sepal length. And in both cases we can see that the Iris Viginica is the largest species.

![scatterplot](assets/bokeh_plot1.png)
![scatterplot](assets/bokeh_plot2.png)

## Pairplot
This chart enables us to quickly see the relationships between variables across multiple dimensions usings scatterplots and histograms.

![pairplot](assets/pairplot1.png)

Plotting regression and confidence intervals
![pairplot](assets/pairplot2.png)

Use kernel density estimates for univariate plots
![pairplot](assets/pairplot3.png)

# Swarm plot

![Swarmplot](assets/swarmplot1.png)

# Violin plot

A voilin plot is used to visualise the distribution of the data and its probability density. The thick black bar in the center represents the interquartile range, the thin black line extended from it represents the 95% confidence intervals, and the white dot is the median.

Petal Length
![Swarmplot](assets/violinplot.png)

Petal Width
![Swarmplot](assets/vp_pw.png)

# Box plot
![Simple Boxplot](assets/Figure_1.png)

# Machine Learning using scikit-learn


# References
Background info
https://en.wikipedia.org/wiki/Iris_flower_data_set
https://archive.ics.uci.edu/ml/datasets/iris

Summary values
https://stackoverflow.com/questions/33889310/r-summary-equivalent-in-numpy

R iris project
https://rstudio-pubs-static.s3.amazonaws.com/205883_b658730c12d14aa6996fe2f6c612c65f.html

python iris project
https://rajritvikblog.wordpress.com/2017/06/29/iris-dataset-analysis-python/

min value
http://www.datasciencemadesimple.com/get-minimum-value-column-python-pandas/

A histogram with Iris Dataset: Sora Jin June 21st, 2015
https://rpubs.com/Sora/developing-data-product

Plot 2D views of the iris dataset
http://www.scipy-lectures.org/packages/scikit-learn/auto_examples/plot_iris_scatter.html

Statistics in Python
http://www.scipy-lectures.org/packages/statistics/index.html#statistics

Python - IRIS Data visualization and explanation
https://www.kaggle.com/abhishekkrg/python-iris-data-visualization-and-explanation


Visualization with Seaborn (Python)
https://www.kaggle.com/rahulm7/visualization-with-seaborn-python

Iris Data Visualization using Python
https://www.kaggle.com/aschakra/iris-data-visualization-using-python

Seaborn Understanding the Weird Parts: pairplot
https://www.youtube.com/watch?v=cpZExlOKFH4

Docs
https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html

http://holoviews.org/gallery/demos/bokeh/boxplot_chart.html

Machine Learning Tutorial
http://scikit-learn.org/stable/tutorial/basic/tutorial.html

https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

https://github.com/whatsrupp/iris-classification/blob/master/petal_classifier.py

https://diwashrestha.com/2017/09/18/machine-learning-on-iris/

https://www.youtube.com/watch?v=rNHKCKXZde8

http://seaborn.pydata.org/examples/scatterplot_categorical.html

IRIS DATASET ANALYSIS (PYTHON)
http://d4t4.biz/ml-with-scikit-learn/support-vector-machines-project-wip/

Getting started in scikit-learn with the famous iris dataset
https://www.youtube.com/watch?v=hd1W4CyPX58
http://blog.kaggle.com/2015/04/22/scikit-learn-video-3-machine-learning-first-steps-with-the-iris-dataset/
