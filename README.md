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
1. Filter out data before 2004  and copy it to a new DataFrame. Data before 2004 may not be relevant due to different economic conditions.<br/>
2. Calculate age of the car and add it to the DataFrame
3. Handle NULL Data <br/>
4. Drop unwanted columns
5. Ensure that the target and odometer columns contain non zero values.
6. Replace null values or missing values with appropirate values for remaining columns.
7. Analyze  the target variable <b>price</b> and  continuous variables odometer & age to remove outliers.
8. Feature independence test for categorical variables

#### Manual Feature Selection
1. Skip fuel as 84% of entries use gas fuelled vehicle
2. Skip transmission as 78% of the vehicle is automatic.
3. Skip cylinder as this info can be reasonably deduced from the type of vehicle

Typically,  car buyers look at the <b>combination</b> of manufacturer, condition, type, drive, odometer and the age of the car for making decisions about buying a car. State is also a factor as taxes and  dealer fees are different for different states.

#### Encode categorical features using James Stein Encoding

I am using the James Stein encoder to ensure that the number of features are manageable. Column transformers are used in the Modeling section to use JamesStein encoder to encode categorical variables.
 

### Modeling

With your (almost?) final dataset in hand, it is now time to build some models.  Here, you should build a number of different regression models with the price as the target.  In building your models, you should explore different parameters and be sure to cross-validate your findings.

#### Modeling Details

1. Build four models - Linear, Ridge, Lasso and GridCVSearch
2. Do simple cross validation for polynomial degrees [1,2,3] for Linear, Ridge and Lasso Regressions
3. Use Root Mean Squared Error as an evaluation metric for all these models. RMSE carries the same unit as the target variable which makes it easy to evaluate these models.

#### Linear Regression:

<b>Evaluation Metric: Root Mean Squared Error </b>

Cross validate Ridge regression across (1,2,3) degrees of polynomial and choose the model with the lease RMSE. 

#### Linear Regression Interpretation

After simple cross validation across polynomial degrees (1,2,3), the model with polynomial degree 1 has the least RMSE score. Cross validation across (1,2,3) degrees of polynomial test is returning the same train and test rmses. This is due to SequenceFeatureSelector selecting the same columns for Linear Regression of all 3 polynomial degrees.

 <b>Evaluation Metric: Best RMSE for test set is 9093.95198 </b>


 |    | Features                        |   Coefficients |
|---:|:--------------------------------|---------------:|
|  0 | polynomialfeatures__odometer    |       -2976.07 |
|  1 | polynomialfeatures__age         |       -5659.39 |
|  2 | jamessteinencoder__manufacturer |        2550.84 |
|  3 | jamessteinencoder__type         |        3126.88 |
|  4 | jamessteinencoder__drive        |        2477.95 |

Based on observed coefficients, I see that (manufacturer, type and drive) of the car seems to proportionately affect the price of the car. odometer and age shows a negative relationship which shows that older the car, lesser the price and vice versa.


#### Ridge Regression:

<b>Evaluation Metric: Root Mean Squared Error </b>

Cross validate Ridge regression across (1,2,3) degrees of polynomial and choose the model with the lease RMSE. 

### Ridge Regression Interpretation

After simple cross validation across polynomial degrees (1,2,3), the model with polynomial degree 3 has the least RMSE score. <br/>

 <b>Evaluation Metric: Best RMSE for polynomial degree 3 test set is 8876.502631 </b>

|    | Features                           |   Coefficients |
|---:|:-----------------------------------|---------------:|
|  0 | polynomialfeatures__odometer       |        238.842 |
|  1 | polynomialfeatures__age            |     -16562.8   |
|  2 | polynomialfeatures__odometer^2     |       4182     |
|  3 | polynomialfeatures__odometer age   |     -15246.7   |
|  4 | polynomialfeatures__age^2          |      20702     |
|  5 | polynomialfeatures__odometer^3     |       -137.201 |
|  6 | polynomialfeatures__odometer^2 age |      -2452.79  |
|  7 | polynomialfeatures__odometer age^2 |      11378.8   |
|  8 | polynomialfeatures__age^3          |      -9779.25  |
|  9 | jamessteinencoder__manufacturer    |       2499.77  |
| 10 | jamessteinencoder__condition       |        418.028 |
| 11 | jamessteinencoder__type            |       2934.12  |
| 12 | jamessteinencoder__drive           |       2472.08  |
| 13 | jamessteinencoder__state           |       1154.02  |

Based on observed coefficients, I see that (manufacturer, condition, type, drive and state ) of the car positively correlates with the price of the car. Age shows a negative relationship which shows that older the car, lesser the price and vice versa. It is surprising to see that the odometer doesn't have a negative coefficient.

### Lasso Regression:

<b>Evaluation Metric: Root Mean Squared Error </b>

Cross validate Ridge regression across (1,2,3) degrees of polynomial and choose the model with the lease RMSE. 

#### Lasso Regression Interpretation

After simple cross validation across polynomial degrees (1,2,3), model with polynomial degree 3 has the least RMSE score. <br/>

 <b>Evaluation Metric: Best RMSE for polynomial degree 3 test set is 8881.619973   </b>


|    | Features                           |   Coefficients |
|---:|:-----------------------------------|---------------:|
|  0 | polynomialfeatures__odometer       |      -3540.1   |
|  1 | polynomialfeatures__age            |     -10693.4   |
|  2 | polynomialfeatures__odometer^2     |       2343.2   |
|  3 | polynomialfeatures__odometer age   |      -2151.64  |
|  4 | polynomialfeatures__age^2          |       3958.17  |
|  5 | polynomialfeatures__odometer^3     |        190.569 |
|  6 | polynomialfeatures__odometer^2 age |      -1198.49  |
|  7 | polynomialfeatures__odometer age^2 |       1891.68  |
|  8 | polynomialfeatures__age^3          |       1327.9   |
|  9 | jamessteinencoder__manufacturer    |       2499.81  |
| 10 | jamessteinencoder__condition       |        413.916 |
| 11 | jamessteinencoder__type            |       2943.58  |
| 12 | jamessteinencoder__drive           |       2474.49  |
| 13 | jamessteinencoder__state           |       1155.83  |

Based on observed coefficients, I see that (manufacturer, condition, type, drive, state ) of the car positively correlates with the price of the car. odometer and age shows a negative relationship which shows that older the car, lesser the price and vice versa.


### GridCVSearch 

<b>Evaluation Metric: Root Mean Squared Error.</b>

Use Ridge regression and a polynomial degree of 3 along with 5 fold cross validation. Do hyperparameter selection by iterating over alpha for the ridge regression. 

#### GridCVSearch Interpretation

With GridCVSearch for RidgeRegression, the optimal value for the hyperparameter alpha is 0.1 with the RMSE score of 8913.213270988472 for polynomial degree 3.

GridCVSearch also runs the K-Fold algorithm to find the best test set.

 |    | Features                           |   Coefficients |
|---:|:-----------------------------------|---------------:|
|  0 | polynomialfeatures__odometer       |        397.433 |
|  1 | polynomialfeatures__age            |     -16907.5   |
|  2 | polynomialfeatures__odometer^2     |       4218.6   |
|  3 | polynomialfeatures__odometer age   |     -15734.7   |
|  4 | polynomialfeatures__age^2          |      21588.9   |
|  5 | polynomialfeatures__odometer^3     |       -145.471 |
|  6 | polynomialfeatures__odometer^2 age |      -2471.57  |
|  7 | polynomialfeatures__odometer age^2 |      11710.6   |
|  8 | polynomialfeatures__age^3          |     -10328.3   |
|  9 | jamessteinencoder__manufacturer    |       2499.98  |
| 10 | jamessteinencoder__condition       |        418.259 |
| 11 | jamessteinencoder__type            |       2933.86  |
| 12 | jamessteinencoder__drive           |       2471.94  |
| 13 | jamessteinencoder__state           |       1153.97  |




Based on the coefficients, I see that (manufacturer, condition, type, drive abd state ) of the car positively correlates with the price of the car. Age shows a negative relationship which shows that older the car, lesser the price and vice versa. It is surprising to see that the odometer doesn't have a negative coefficient. The observation are similar to Ridge Regression results without GridCVSearch.


Evaluation
With some modeling accomplished, we aim to reflect on what we identify as a high-quality model and what we are able to learn from this. We should review our business objective and explore how well we can provide meaningful insight into drivers of used car prices. Your goal now is to distill your findings and determine whether the earlier phases need revisitation and adjustment or if you have information of value to bring back to your client.


#### Best Model Selection

|    | ModelName         |   RMSEScore |   PolynomialDegree |
|---:|:------------------|------------:|-------------------:|
|  0 | LinearRegression  |     9093.95 |                  1 |
|  1 | RidgeRegression   |     8876.5  |                  3 |
|  2 | LassoRegression   |     8881.62 |                  3 |
|  3 | RidgeGridCVSearch |     8913.21 |                  3 |

Based on RMSE scores of test sets, Ridge Regression has the least RMSE value and therefore is the best model.

Manufacturer, condition, type, drive and state of the car influence the car price significantly. Even though the odometer has a positive coefficient, I am skeptical of its positive correlation to the price. This requires further investigation.

I am not sure how to interpret odometer^2, odometer age, etc


### Deployment

Now that we've settled on our models and findings, it is time to deliver the information to the client.  You should organize your work as a basic report that details your primary findings.  Keep in mind that your audience is a group of used car dealers interested in fine-tuning their inventory.

#### Findings

Based on our analysis, customers value (manufacturer, condition, type, drive) of the car. State of purchase affects the car price due to different taxation, dealership fees and other state specific fees.

As a future work, I would fine tune models by running  models on a state basis. 
