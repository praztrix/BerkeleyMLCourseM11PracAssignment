# What drives the price of a car?

![](images/kurt.jpeg)

**OVERVIEW**

In this application, you will explore a dataset from Kaggle. The original dataset contained information on 3 million used cars. The provided dataset contains information on 426K cars to ensure speed of processing.  Your goal is to understand what factors make a car more or less expensive.  As a result of your analysis, you should provide clear recommendations to your client -- a used car dealership -- as to what consumers value in a used car.

**Files**
1. README.md
2. data/vehicles.csv - contains used car data
3. images/ - contains a few images for the problem statement
4. used_car_sales_modeling.ipynb - Jupyter notebook for used car sales modeling 
5. prac2app_utils.py - util functions for plotting, modeling, etc

### CRISP-DM Framework

<center>
    <img src = images/crisp.png width = 50%/>
</center>


To frame the task, throughout our practical applications, we will refer back to a standard process in industry for data projects called CRISP-DM.  This process provides a framework for working through a data problem.  Your first step in this application will be to read through a brief overview of CRISP-DM [here](https://mo-pcco.s3.us-east-1.amazonaws.com/BH-PCMLAI/module_11/readings_starter.zip).  After reading the overview, answer the questions below.


### Business Understanding

From a business perspective, we are tasked with identifying key drivers for used car prices.  In the CRISP-DM overview, we are asked to convert this business framing to a data problem definition.  Using a few sentences, reframe the task as a data task with the appropriate technical vocabulary. 

<b> Data Problem Definition</b><br/>

We need to devise a machine learning model to predict used car prices. We should also identify  features of different types of automobiles that affect its sale price. In order to accomplish this task, we have been given used car sales data from 1900 to 2022. We are going to use multiple prediction techniques to model the used car sales data in order to predict sale prices in future

For our modeling purposes, I am going to consider only about the last 20 years of data as car prices earlier than that may not affect used car prices today due to different economic conditions.

Based on the modeling, we should identify car features valued by customers.


### Data Understanding

After considering the business understanding, we want to get familiar with our data.  Write down some steps that you would take to get to know the dataset and identify any quality issues within.  Take time to get to know the dataset and explore what information it contains and how this could be used to inform your business understanding.

Used car sales data contains the following information <br/>

<b>id</b> - recordid <br/>
<b>region</b> - US region where the car sales occurred<br/>
<b>price</b> - sale price of the car<br/>
<b>year</b> - year of manufacture<br/>
<b>manufacturer</b> - car brand<br/>
<b>model</b> - car model<br/>
<b>condition</b> - car condition<br/>
<b>cylinders</b> - number of cylinders in the car<br/>
<b>fuel</b> - fuel type used by the car<br/>
<b>odometer</b> - number of miles driven by the car<br/>
<b>title_status</b> - status of the car title<br/>
<b>transmission</b> - transmission type of the car<br/>
<b>VIN</b> - VIN number of the car<br/>
<b>drive</b> - drive train for the car<br/>
<b>size</b>  - size of the car<br/>
<b>type</b> - type of the car<br/>
<b>paint_color</b> - color of the car<br/>
<b>state</b> - state where the car was registered<br/>


<b>id</b> and <b>VIN</b> are row and car identifiers respectively. These two variables don't affect the sale price.<br/>
<b>region</b>, <b>manufacturer</b>, <b>model</b>, <b>condition</b>, <b>cylinders</b>, <b>fuel</b>, <b>odometer</b>, <b>title_status</b>,<b>transmission</b>, <b>drive</b>, <b>size</b>, <b>type</b>, <b>paint_color</b> and <b>state</b> are categorical variables.
<br/>

I am going to calculate the age of the car to understand its impact on the model.

Target variable for our modeling is <b>price</b>

<b>Exception</b> :  I am <b>not</b> using <b>'model'</b> as one of the features due to the amount of computation needed for processing this column. This is done due to practical reasons as including it in feature independence tests,encoding  and modeling results in this notebook hanging for a long time.


### Data Preparation

After our initial exploration and fine-tuning of the business understanding, it is time to construct our final dataset prior to modeling.  Here, we want to make sure to handle any integrity issues and cleaning, the engineering of new features, any transformations that we believe should happen (scaling, logarithms, normalization, etc.), and general preparation for modeling with `sklearn`. 

#### Data Preparation Steps

1. Filter out data before 2004  and copy it to a new DataFrame<br/>
2. Calculate age of the car and add it to the DataFrame
3. Handle NULL Data <br/>
4. Drop unwanted columns
5. Ensure that the target and odometer columns contain non zero values.
6. Replace null values or missing values with appropirate values for remaining columns.
7. Analyze  the target variable <b>price</b> and  continuous variables odometer & age to remove outliers.
