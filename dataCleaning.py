# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 16:11:25 2022

@author: Zain Khan

In this program we are going to clean the dataset that
we aquired from scrapping glassdoor website. The process
involve are as follows.
these steps looks important for me after going through the dataset

clean salary estimate column
company name text only
state field
company age or how old the company is
parse job description for important key words
"""

import pandas as pd

df = pd.read_csv("glassdoor_jobs.csv")

# first lets create new columns so its easy for us to process salary estimate column
# If you look to our dataset it has some values with hourly rate, lets create a new column from those values
df['hourly'] =  df['Salary Estimate'].apply(lambda x: 1 if "per hour" in x.lower() else 0)

# now lets do the same for employer provided salary 
df['employer_provided'] =  df['Salary Estimate'].apply(lambda x: 1 if "employer provided salary" in x.lower() else 0)


# lets remove those values with -1 for salary from our dataframe

df = df[df['Salary Estimate'] != '-1']

# removing (Glassdoor est.), K and $ signs  from the Salary Estimate attribute
salary =  df['Salary Estimate'].apply(lambda k: k.split("(")[0].replace("K", "").replace("$", ""))

# remove per hour/employer provided salary from the salaryt estimate attribute
salary = salary.apply(lambda k: k.lower().replace('per hour', "").replace('employer provided salary:', ""))

# now lets create min, max and avg salary column
df['min_salary'] = salary.apply(lambda k: int(k.split('-')[0]))
df['max_salary'] = salary.apply(lambda k: int(k.split('-')[1]))
df['avg_salart'] =(df.min_salary + df.max_salary)/2

# now that we have dealt with the salary estimate column we should probably process companies name text from any kind of impurities.
df['company_text'] = df.apply(lambda k: k['Company Name'] if k['Rating'] == -1 else k['Company Name'][:-3], axis=1)

# now that we have cleaned comapny name lets go to the next step

df['job_state'] = df['Location'].apply(lambda k: k.split(',')[1])
# =============================================================================
# print("jobs in each state ")
# print(df['job_state'].value_counts())
# =============================================================================
 
# lets see if the job is in headquater state
df['same_state'] = df.apply(lambda k: 1 if k.Location == k.Headquarters else 0, axis=1)

# now lets do the next step which to find the age of the company
df['age']= df['Founded'].apply(lambda k: k if k < 1 else 2021 - k)

# Above we found how old each comapny is,  now the last thing that I thing is relevant is parsing job description for some keyword such python etc
# for now we will look for python, r studio, spark , excel and aws
# for python
df['python'] = df['Job Description'].apply(lambda k: 1 if "python" in k.lower() else 0)

# for r studio
df['rstudio'] = df['Job Description'].apply(lambda k: 1 if "r studio" in k.lower() or 'r-studio' in k.lower() else 0)

# for spart
df['spark'] = df['Job Description'].apply(lambda k: 1 if "spark" in k.lower() else 0)

# for excel
df['excel '] = df['Job Description'].apply(lambda k: 1 if "excel" in k.lower() else 0)

# for python
df['aws'] = df['Job Description'].apply(lambda k: 1 if "aws" in k.lower() else 0)


# drop the unnamed colum
df.drop('Unnamed: 0', axis=1, inplace=True)

# save to a file
df.to_csv("cleaned_glassdoor_data.csv", index=False)





