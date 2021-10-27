```python
import sys
sys.path.append('/path/to/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/')

import pandas as pd
pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 1000)

from help_doc.help_doc import HelpDoc
```


```python
# Try to load an existing HelpDoc instance. If it doesn't exist, create a new one
# By default, homepage_url is set as "www.amazon.com" and language is set as None (US marketplace + EN language)
try:
    us_en_help_doc = HelpDoc('us_en_help_doc', '/path/to/local/working/dir', load=True)
    us_en_help_doc.set_logger_level('DEBUG')
except FileNotFoundError:
    us_en_help_doc = HelpDoc('us_en_help_doc', '/path/to/local/working/dir', logger_level='DEBUG', load=False)

print('The created / loaded instance contains the HTML data pulled at the following times:')
us_en_help_doc.time_list()
```

# HTML Pulling


```python
# Another backend you can use is "selenium", but it is recommended to use "requests"
pulled_data = us_en_help_doc.pull('https://www.amazon.com/gp/help/customer/display.html?nodeId=GSNBBJP63SM65UDB', 
                                  backend='requests', max_n_failure=1, n_cpu=1)
print(pulled_data.keys())
pulled_data
```


```python
pulled_data = us_en_help_doc.pull([
    'https://www.amazon.com/gp/help/customer/display.html?nodeId=GSNBBJP63SM65UDB', 
    'http://www.amazon.com/gp/help/customer/display.html?nodeId=G6E3B2E8QPHQ88KF', 
    'https://www.amazon.com/gp/help/customer/display.html?nodeId=G6YZ2FWM3PD2CMYV'
], backend='requests', max_n_failure=1, n_cpu=1)
print(len(pulled_data))
pulled_data
```


```python
us_en_help_doc.pull('GSNBBJP63SM65UDB', 
                    backend='requests', max_n_failure=1, n_cpu=1)
```


```python
us_en_help_doc.pull([
    'GSNBBJP63SM65UDB', 
    'https://www.amazon.com/gp/help/customer/display.html?nodeId=G6E3B2E8QPHQ88KF', 
    'G6YZ2FWM3PD2CMYV'
], backend='requests', max_n_failure=1, n_cpu=1)
```


```python
us_en_help_doc.pull('www.amazon.com', 
                    backend='requests', max_n_failure=1, n_cpu=1)
```


```python
us_en_help_doc.pull([
    'www.amazon.com', 
    'abc', 
    'https://www.amazon.com/gp/help/customer/display.html?nodeId=abc'
], backend='requests', max_n_failure=1, n_cpu=1)
```


```python
# The above operations have not "saved" the pulled HTML to HelpDoc instance
# To save the pulled HTML to HelpDoc instance, call the "pull_and_save" method
time_pulled, df_html = us_en_help_doc.pull_and_save([
    'https://www.amazon.com/gp/help/customer/display.html?nodeId=GSNBBJP63SM65UDB', 
    'http://www.amazon.com/gp/help/customer/display.html?nodeId=G6E3B2E8QPHQ88KF', 
    'https://www.amazon.com/gp/help/customer/display.html?nodeId=G6YZ2FWM3PD2CMYV'
], max_n_raw_data=3, backend='requests', max_n_failure=1, n_cpu=1)

# Save the HelpDoc to hard disk if you want
# us_en_help_doc.save()

print(time_pulled)
print(us_en_help_doc.time_list())
df_html
```

# HTML Pulling + Parsing


```python
parsed_data = us_en_help_doc.parse(
    us_en_help_doc.pull('https://www.amazon.com/gp/help/customer/display.html?nodeId=GSNBBJP63SM65UDB'), 
    n_cpu=1
)

print(parsed_data.keys())
us_en_help_doc.log_data_parsed(parsed_data, print, width=100)
```


```python
parsed_data = us_en_help_doc.parse(
    us_en_help_doc.pull([
        'https://www.amazon.com/gp/help/customer/display.html?nodeId=GSNBBJP63SM65UDB', 
        'G6E3B2E8QPHQ88KF', 
        'www.amazon.com'
    ]), 
    n_cpu=1
)

print(len(parsed_data))
for data in parsed_data:
    us_en_help_doc.log_data_parsed(data, print, width=80)
    print('')
```


```python
# To parse the HTML data obtained above by "pull_and_save" (i.e., the HTML data in "df_html") and save the parsed data to the HelpDoc instance, call the "parse_and_save" method
# The "parse" and "parse_and_save" methods can perfectly parse HTMLs of ~2500 help documents, while handling tons of corner cases and even mistakes in HTMLs
time_pulled, df_html_parsed = us_en_help_doc.parse_and_save(-1, n_cpu=1) # "-1" means parsing the "last" pulled HTML data

print(time_pulled)
df_html_parsed
```

# Access, Delete, Insert, Update Existing Data


```python
us_en_help_doc.time_list()
```

## Parsed HTML Data 


```python
us_en_help_doc[-1]
```


```python
del us_en_help_doc[-2]
```


```python
# If the time stamp doesn't exist, you will insert the new dataframe of parsed HTML data "df_new"
# If the time stamp already exists, you will update the old dataframe of parsed HTML data with the new one "df_new"
us_en_help_doc['2021-08-30-11-00-25-PDT'] = df_new
```

## Raw HTML Data


```python
us_en_help_doc.df_raw_data('2021-08-30-11-00-25-PDT')
```


```python
us_en_help_doc.del_raw_data(-1)
```

# Compare Help Documents Pulled & Parsed at Different Times


```python
df_compare = us_en_help_doc.compare(-3, -2)

print(df_compare.shape)
df_compare
```


```python
df_compare = us_en_help_doc.compare(-3, -2, col_compare=('text', 'sidebar', 'breadcrumb'))

print(df_compare.shape)
df_compare
```


```python
df_compare = us_en_help_doc.compare(-3, -2, col_compare=('text', ))

print(df_compare.shape)
df_compare
```

# Check Validity of Links in Documents


```python
error = us_en_help_doc.check('https://www.amazon.com/gp/help/customer/display.html?nodeId=G2202016320', 
                             backend='requests', max_n_failure=1, n_cpu=1)
error
```


```python
error = us_en_help_doc.check([
    'www.google.com',
    'https://s3-us-west-2.amazonaws.com/customerdocumentation/EK2/Get+Started+with+Kindle+(2nd+Generation).pdf', 
    'mailto:copyright@amazon.com', 
    'https://www.amazon.com/gp/help/customer/display.html/?nodeId=GV38326YW5JX9V9X', 
    'https://www.amazon.com/gp/help/customer/display.html?nodeId=G6YZ2FWM3PD2CMYV'
], backend='requests', max_n_failure=1, n_cpu=1)

error
```


```python
time_pulled, df_url_in_doc = us_en_help_doc.check_and_save(-1, backend='requests', max_n_failure=2, n_cpu=1)
df_url_in_doc
```


```python
sorted(df_url_in_doc['url_in_doc'].tolist())
```


```python
df_url_in_doc['error'].value_counts()
```


```python
# For example, 
# "Alexa Help Videos" in https://www.amazon.com/gp/help/customer/display.html?nodeId=GENXY8NAJRLAXGBT is an invalid link
# "Marketplace Returns and Refunds" in https://www.amazon.com/gp/help/customer/display.html?nodeId=GKUY8GVEZ5DZ89QR is an invalid link
df_url_in_doc.loc[df_url_in_doc['error'] == 'HelpDocNotFound', :]
```

# Wrap Up Everything to Automatically & Regularly Update Help Document Collection


```python
# Create a cron job on GPU Machine 2 (see below) to automatically run the following scripts every 6 hours to update the help document collection:

# SHELL=/bin/bash
# 0 */6 * * * PYTHONPATH=/path/to/python/site-packages /path/to/bin/python /path/to/the/following/scripts.py

###################################################################################################################################################
import os
import shutil
from help_doc.help_doc import HelpDoc, help_doc_nodeid_from_s3

# Create/load the help doc module
try:
    us_en_help_doc = HelpDoc('us_en_help_doc', '/local/dir/on/GPU/Machine', load=True)
except FileNotFoundError:
    us_en_help_doc = HelpDoc('us_en_help_doc', '/local/dir/on/GPU/Machine', logger_level='INFO', load=False)

# Get the list of NodeID
us_en_nodeid, _ = help_doc_nodeid_from_s3(
    us_en_help_doc.root_dir,
    s3_file_name_prefix='US-en',
    row_filter=lambda x: x.loc['breadcrumb'].startswith('Help & Customer Service'),
    s3_bucket_name='cs-help-index-data'
)

# Delete the NodeID downloaded from S3 to save space, but please always double check to make sure this is the correct path you want to delete
shutil.rmtree(os.path.join(us_en_help_doc.root_dir, 'cs-help-index-data'))  

# Pull HTMLs, parse HTMLs, and check invalid URLs in each document
us_en_help_doc(us_en_nodeid, s3_bucket_name='your_s3_bucket_name', s3_dir='remote/dir/in/your/s3/bucket', 
               max_n_raw_data=2, n_cpu_pull=1, n_cpu_parse=10, n_cpu_check=1)

# Save the help doc module
us_en_help_doc.save()
```


```python
us_en_nodeid = [
    'https://www.amazon.com/gp/help/customer/display.html?nodeId=GSNBBJP63SM65UDB', 
    'https://www.amazon.com/gp/help/customer/display.html?nodeId=G6E3B2E8QPHQ88KF', 
    'https://www.amazon.com/gp/help/customer/display.html?nodeId=G6YZ2FWM3PD2CMYV'
]
us_en_help_doc(us_en_nodeid, s3_bucket_name='qyouran', s3_dir='help_doc_production_demo/help_doc', 
               max_n_raw_data=2, n_cpu_pull=1, n_cpu_parse=1, n_cpu_check=1)
```

# Other Marketplaces


```python
# You should be able to do all above mentioned for any other marketplaces, possibly after making minor changes
```


```python
# GB marketplace + EN language
help_doc = HelpDoc('gb_en_help_doc', '/Users/qyouran/Documents/QA/QABotHelpDocDemo', logger_level='DEBUG', 
                   homepage_url='www.amazon.co.uk', language=None)
help_doc.log_data_parsed(help_doc.parse(help_doc.pull('EF528GN65XSJ7V8')), print)
```


```python
# JP marketplace + JA language
help_doc = HelpDoc('jp_ja_help_doc', '/Users/qyouran/Documents/QA/QABotHelpDocDemo', logger_level='DEBUG', 
                   homepage_url='www.amazon.co.jp', language=None)
help_doc.log_data_parsed(help_doc.parse(help_doc.pull('EF528GN65XSJ7V8')), print)
```


```python
# JP marketplace + EN language
help_doc = HelpDoc('jp_en_help_doc', '/Users/qyouran/Documents/QA/QABotHelpDocDemo', logger_level='DEBUG', 
                   homepage_url='www.amazon.co.jp', language='en_US')
help_doc.log_data_parsed(help_doc.parse(help_doc.pull('EF528GN65XSJ7V8')), print)
```


```python
# DE marketplace + DE language
help_doc = HelpDoc('de_de_help_doc', '/Users/qyouran/Documents/QA/QABotHelpDocDemo', logger_level='DEBUG', 
                   homepage_url='www.amazon.de', language=None)
help_doc.log_data_parsed(help_doc.parse(help_doc.pull('EF528GN65XSJ7V8')), print)
```


```python
# DE marketplace + NL language
help_doc = HelpDoc('de_nl_help_doc', '/Users/qyouran/Documents/QA/QABotHelpDocDemo', logger_level='DEBUG', 
                   homepage_url='www.amazon.de', language='nl_NL')
help_doc.log_data_parsed(help_doc.parse(help_doc.pull('EF528GN65XSJ7V8')), print)
```


```python
# CN marketplace + ZH language
help_doc = HelpDoc('cn_zh_help_doc', '/Users/qyouran/Documents/QA/QABotHelpDocDemo', logger_level='DEBUG', 
                   homepage_url='www.amazon.cn', language=None)
help_doc.log_data_parsed(help_doc.parse(help_doc.pull('201956070')), print)
```


```python
# IN marketplace + HI language
help_doc = HelpDoc('in_hi_help_doc', '/Users/qyouran/Documents/QA/QABotHelpDocDemo', logger_level='DEBUG', 
                   homepage_url='www.amazon.in', language='hi_IN')
help_doc.log_data_parsed(help_doc.parse(help_doc.pull('201956070')), print)
```
