# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 17:20:09 2022

@author: Zain Khan
"""

import requests
from data_input import data_in

url = "http://127.0.0.1:5000/predict"  ## change it to your URL
header = {"Content-Type": "application/json"}
data = {'input': data_in}

result = requests.get(url, headers=header, json=data)

print(result.json())