#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install requests==2.26.0')
get_ipython().system('pip install json==2.0.9')


# In[ ]:


import requests
import json


# In[10]:


# Normal API call with all inputs in correct format

url = 'https://deep-learning-pipeline.azurewebsites.net/get_license_status'
params = {
    'APPLICATION_TYPE': 'RENEW',
    'CONDITIONAL_APPROVAL': 'N',
    'LICENSE_CODE': 1010,
    'SSA': None,
    'LEGAL_NAME': 'ALL-BRY CONSTRUCTION CO.',
    'DOING_BUSINESS_AS_NAME': 'ALL-BRY CONSTRUCTION CO.',
    'ZIP_CODE': 60439,
    'APPLICATION_REQUIREMENTS_COMPLETE': '2004-02-10T00:00:00',
    'LICENSE_DESCRIPTION': 'Limited Business License'
}

response = requests.post(url, data=json.dumps(params))
print(response.text)


# In[ ]:




