# Data Scientist Salary Estimator
## Overview of the project
* Created this tool which helps data scientist negotiate their salary with their employer by estimating DS salaries for them. (with Mean Absolute Error of ~$11K)
* Scraped 1000+ data scientist job descriptions from GlassDoor using selenium and python
* Engineered Features from text to quantify the value companies put on different technologies like Python, AWS , Excel and Spark etc.
* Build and Optimized Linear Regressiona , Lasso Regression and Random forest using gridSearchCV to get the best out of the models.
* Built an API using Flask for clients to request the model for new data.


## Resoruces and Code used
**Python Version:** 3.7  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, selenium, flask, json, pickle  
**For Web Framework Requirements:**  ```pip install -r requirements.txt```  
**Scraper Github:** https://github.com/arapfaik/scraping-glassdoor-selenium  
**Scraper Article:** https://towardsdatascience.com/selenium-tutorial-scraping-glassdoor-com-in-10-minutes-3d0915c6d905 

## Scrapping Data from Glass Door
Tweaked the web scraper github repo (above) because the website of the glassdoor has been changed and needed lots of playing around the website to scrap the data that I needed, I scraped 1000+ job postings from glassdoor.com. Each job got the following:
*	Job title
*	Salary Estimate
*	Job Description
*	Rating
*	Company 
*	Location
*	Company Headquarters 
*	Company Size
*	Company Founded Date
*	Type of Ownership 
*	Industry
*	Sector
*	Revenue
*	Competitors 


## Data Cleaning
After getting the data from the glassdoor.com, I needed to clean the data so it was usable for my model. Few nwe columns were introduced to the dataset.
* Removed rows which had null/nan values for salary
* New columns for each type of skill were introduced (skills were extracted from the job description)
  * Python
  * Excel
  * AWS
  * Spark
  * R
* Created age of company column by finding the age using founded date of the company
* Simplified the job title and Seniority both of these columns were added

## EDA
I looked at different numerical values to get the sense of the data and what will be useful feature for our model. I also analyzed different categorical data.

## Model building
As with every project there are many ways to deal with categorical data, in this project I transformed categorical data into dummy varibales. The dataset was then split into train and test sets with 20% of it being the test data.

As mentioned in the overview, I used 3 different models, these models were evaluated using `Mean Absolute Error`, why? because its easy to interpret.

The models used are as follows:
*	**Multiple Linear Regression** – Baseline for the model
*	**Lasso Regression** – Because of the sparse data from the many categorical variables, I thought a normalized regression like lasso would be effective.
*	**Random Forest** – Again, with the sparsity associated with the data, I thought that this would be a good fit. 

## Model performance
It was noticed that Random Forest model far outperformed the other models on both test and validations sets. It was of no surprise as It was expected, because Random forest
is far way good then Linear Regression. The Lasso modle did improve but it was still not good enough.

## Productionization 
In this step, I built flask API endpoint that
In this step, I built a flask API endpoint that was hosted on a local webserver. The API endpoint takes in a request with a list of values from a job listing and returns an estimated salary. 
