# Parser


```python
import re
import requests
import time
import random
from bs4 import BeautifulSoup

USER_AGENT = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}

## GET THE HTML DOM ELEMENT
def fetch_results(node_id):
    amazon_help_url = 'https://www.amazon.com/gp/help/customer/display.html?nodeId={}&pop-up=1'.format(node_id)
    response = requests.get(amazon_help_url, headers=USER_AGENT)
    response.raise_for_status()
    return response.text

def extractTable(table):
    data = ''
    colnames = []
    for children in table:
        if children.name == "caption":           
            data = data + ' ' + children.text.strip() + ': '
        if children.name == "thead":
            cols = children.find_all('th')
            for c in cols:
                colnames.append(c.text.strip())
        if children.name == "tbody":
            rows = children.find_all('tr')
            for r in rows:
                cols = r.find_all('td')
                count = 0
                sentinel = ''
                if len(colnames) == len(cols):
                    for c in cols:
                        data = data + sentinel + colnames[count] + '=' + getData(c).strip()
                        sentinel = ', '
                        count += 1
                    data = data + '. '
                else:
                    for c in cols:
                        data = data + sentinel +getData(c).strip()
                        sentinel = ', '
                        count += 1
                    data = data + '. '
                
                
    return data

def getData(node, ignore_tags=("script", "style")):
    if node.name is None:
        return node
    text = ''
    for children in node:
        if children.name in ignore_tags:
            continue;
        elif children.name == "h1" or children.name == "h2" or children.name == "h3":
            temp = children.text
            if len(temp) > 0 and temp[len(temp) - 1] in (':', '?', '.'):
                text = text.rstrip() + ' ' + temp[:-1].rstrip() + ': '
            else:
                text = text.rstrip() + ' ' + temp.rstrip() + ': '
        elif children.name == "a" and "href" in children.attrs and len(children["href"]) > 0:
            text = text.rstrip() + ' [' + children.text.strip() + '](' + ('www.amazon.com' + children["href"] if children["href"][0] == "/" else children["href"]) + ')'
        elif children.name == "table":
            text = text.rstrip() + extractTable(children)
        else:
            temp = getData(children)
            if len(temp) > 1 and temp[0:2] == " [":
                text = text.rstrip() + temp
            else:
                text = text + temp
    return text



def get_text(node_id):
    html = fetch_results(node_id)
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.title.text.replace('Amazon.com Help: ','').strip()
    soup = soup.find_all('div', attrs={'class': ['help-content','cs-help-content']})[0]
    a = re.sub("\s*[\.]+\s*[\n]+\s*",". ",getData(soup).strip())
    a = re.sub("[:]+\s*[\n]+\s*",": ",a)
    a = re.sub("[\?]+\s*[\n]+\s*","? ",a)
    a = re.sub(r"(\s*[\n]+\s*)([A-Z])",r". \2",a)
    a = re.sub("\s*[\n]+\s*"," ",a)
    return title,a
```

## Testing & Debugging Parser


```python
#get_text('GMDKZSFRSX7NKKN3')
html = fetch_results('201937120')

soup = BeautifulSoup(html, 'html.parser')
title = soup.title.text.replace('Amazon.com Help: ','').strip()
soup = soup.find_all('div', attrs={'class': ['help-content','cs-help-content']})[0]
a = re.sub("\s*[\.]+\s*[\n]+\s*",". ",getData(soup).strip())
a = re.sub("[:]+\s*[\n]+\s*",": ",a)
a = re.sub("[\?]+\s*[\n]+\s*","? ",a)
a = re.sub(r"(\s*[\n]+\s*)([A-Z])",r". \2",a)
a = re.sub("\s*[\n]+\s*"," ",a)
print(title,a)



```

## Break Answer for Answer Extraction


```python
class Blurb:
    def __init__(self,type,value,relativeUrl,desktopRelativeUrl):
        self.type = type
        self.value = value
        self.relativeUrl = relativeUrl
        self.desktopRelativeUrl = desktopRelativeUrl
        self.buttons = []
        
    def addButton(self,x):
        self.buttons.append(vars(x))
        
class Button:
    def __init__(self,text,type,value):
        self.text = text
        self.type = type
        self.value = value



### devide by peroid and merge
def split_url_ans(text, pattern = r'\]\([^(]+\)'):
#     pattern = r'\((?:https:\/\/)?www.[^(]+\)'
    url_list = [a[2:][:-1] for a in re.findall(pattern, text)]
    ans_list = []
    ans_raw = re.split(pattern, text)
    for i, a in enumerate(ans_raw):
        if i == len(ans_raw) - 1:
            ans_list.append(a.strip(','))
        else: 
            ans_list.append((a + ']').strip(',').strip())
    return ans_list, url_list


def get_answer_list(answers):
    alist = answers.strip().split('. ')
    out_alist=[]
    for a in alist:
        answer = a.strip()
        ans_list, url_list = split_url_ans(answer)
        if len(url_list) > 0:
            for i, u in enumerate(url_list):
                a = re.sub(r'\[(.*)\]', r'<a>\1</a>', ans_list[i])
                if i == len(url_list)-1:
                    ans = (a+ans_list[i+1]).strip()
                    highlights = re.sub(r'\<a\>(.*)\<\/a\>', r'\1', ans)
                    if len(highlights) > 0:
                        out_alist.append([ans,u])
                else: 
                    out_alist.append([a.strip(),u])
        else: 
            out_alist.append([answer,''])
    return out_alist


def reformat_answer(text,max_ans_len = 300):
    answer_list = get_answer_list(text)
    i=0
    j=1
    ans = answer_list[0][0]
    url = answer_list[0][1]
    while j < len(answer_list):
        next_ans = answer_list[j][0]
        next_url = answer_list[j][1]
        url_check = 1 if len(url) > 0 and len(next_url) > 0 else 0
        if len(ans) + len(next_ans) < max_ans_len and url_check == 0:
            ans += '. ' + next_ans
            url += next_url
            if j == len(answer_list)-1:
                answer_list[i] = [ans,url]
        else:
            answer_list[i] = [ans,url]
            ans = answer_list[j][0]
            url = answer_list[j][1]
            if j == len(answer_list)-1:
                answer_list[i+1] = [ans,url]
            i += 1
        j+=1

    formatted_answer = answer_list[:i+1]
    final_answer = [vars(Blurb('ANSWER',blurb[0],blurb[1],blurb[1])) for blurb in formatted_answer]
    return final_answer

reformat_answer('')


```

# Reading Top Help Documents


```python
import pandas as pd
import pickle

#top_160_documents = pd.read_csv('/data/qyouran/QABot/QAData/HelpDoc_InitialLaunch/doc_coverage/df_help_document_2_top160.tsv',sep='\t')
#top_queries = pd.read_csv('/data/qyouran/QABot/QAData/HelpDoc_InitialLaunch/doc_coverage/prediction_2_top160.tsv',sep='\t')
top_200_answers = pd.read_csv('/data/qyouran/QABot/QAData/HelpDoc_InitialLaunch/doc_coverage/df_help_document_2_answer_top200.tsv',sep='\t')
top_answers = pd.read_csv('/data/qyouran/QABot/QAData/HelpDoc_InitialLaunch/doc_coverage/prediction_2_answer_top200.tsv',sep='\t')
with open('node_id_added_from_top160', 'rb') as f:
    node_id_added_from_top160 = pickle.load(f)

```


```python
# queries = top_answers.head(10)['query'].to_list()
# ## s3 initialize
# os.environ['AWS_ACCESS_KEY_ID']='AKIAUOVCCZMYRCESLFEO'
# os.environ['AWS_SECRET_ACCESS_KEY']='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'
# BUCKET_NAME_INPUT='qabot-annotation-input-ir-b'
# s3_client = boto3.client('s3')

# for i,query in enumerate(queries):
#     filename=str(i+1)
#     body=query
#     s3_client.put_object(Body=body,Bucket=BUCKET_NAME_INPUT,Key=filename+'.txt',)
```

## Collecting all documents


```python
from random import randint
from time import sleep
import json

output = dict()
offset = 1
new_list = ['W3L8JX7Q9FH8ALB', 'TZVXNQGEL3K9CSH']
nodes = node_id_added_from_top160 + top_200_answers.node_id.tolist() + new_list


for i,node_id in enumerate(nodes):
    try:
        nid = 'G'+str(node_id)
        url = 'https://www.amazon.com/gp/help/customer/display.html?nodeId={}&pop-up=1'.format(nid)
        pid = offset + i
        title, passage = get_text(nid)
        output[pid] = [passage, nid, url, title]
        
    except:
        print('retrying for pid: {}, node_id: {}'.format(offset + i, node_id))
        sleep(randint(2,4)) 
        nid = 'G'+str(node_id)
        url = 'https://www.amazon.com/gp/help/customer/display.html?nodeId={}&pop-up=1'.format(nid)
        pid = offset + i
        title, passage = get_text(nid)
        output[pid] = [passage, nid, url, title]

with open("/data/QAData/InformationRetrievalData/amazon/helpdocuments_collection_new.json", "w") as outfile:  
    json.dump(output, outfile)
```

## Creating Production Document Collection


```python
import sys
import re
sys.path.append('/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/')
from factory.word_tokenizer_factory import english_preprocessor

def get_preprocessed_document_ir(text):
    english_preprocessor.do_apostr_contract=True
    english_preprocessor.do_apostr_possess = True
    english_preprocessor.noise_regex['URL'] = (re.compile(r'(\(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^)\s]{2,}|www\.[a-zA-Z0-9]' \
                                                      r'[a-zA-Z0-9-]+[a-zA-Z0-9]\.[^)\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^)\s]{2,}|www\.' \
                                                      r'[a-zA-Z0-9]+\.[^)\s]{2,})'), )
    passage_ir = english_preprocessor.preprocess_single(text)
    return passage_ir
```


```python
import json

collection = json.load(open('/data/QAData/InformationRetrievalData/amazon/helpdocuments_collection_new.json','r'))
outfile = '/data/QAData/InformationRetrievalData/amazon/production_collection.json'
outfile_url = '/data/QAData/InformationRetrievalData/amazon/production_collection_url.json'
# prod_pid_excluded = [1, 4, 5, 12, 13, 31, 35, 75, 98, 103, 112, 149, 183, 193] #122, 38 removed from this list
# digital_pid = [2, 9, 10, 11, 22, 30, 32, 36, 38, 44, 46, 47, 65, 69, 80, 91, 95, 96, 129, 131, 137, 138, 148, 150, 151, 155, 173, 178, 184, 185, 189, 191, 196, 199, 225, 237, 243]
ignored_node_ids = ['G4KFSMKGWZPYNW9X', 'G202185710', 'G8P3YMHU34R3XX4N', 'G202120810', 'G201911080', 'G201910410', 'GZ3PDPNA7U6R6UMK', 
                    'GAEJPJ8E5TY8TTNL', 'GWGDSNXVPJ93UW5V', 'GSD587LKW72HKU2V', 'G3BQ95AL4WZLV9ZG', 'G200127470', 'GL4263XHAGWBSC8R', 
                    'G64ENL4SCTZ4EXSY']
digital_node_ids = ['GJT6X5TZUW8AB9Y9', 'G201567520', 'G9JFV7VRANZDKKWG', 'GW5P2J5UV6EHRZCL', 'GE8EWPM8E5QG8E8E', 'G201555990', 'G201890100', 
                    'GXQJG7FBB6SJD22T', 'G3CHA35W7N58VG5B', 'GTQEND3RFAFNLKU5', 'G201609150', 'GSNFRLBMD26UK9PA', 'GQ8QNEVL5FCKBGH2', 
                    'G8637923FFWAR2YH', 'GZCHXL8CUW3VWJQP', 'G202070170', 'GT4SYLY6SVG9QWBV', 'G937D322PWZ6L9BL', 'GEA2QYCTZQXKRG4Z', 
                    'G3EWVFQZCVKH53TB', 'G201755180', 'G202020180', 'GDMMKH7RJP2HU2P2', 'GLSQ4722655M4ZEJ', 'GCRZL3F2UZMNP3T3', 
                    'GRTHB5PBJ32UUSMT', 'G202161160', 'GHANKAZLVY3SKWMV', 'G7CF8KV3YPXE285Y', 'GQ2FT94RCVV5BA3Z', 'GWWLWFYVTYQTLLL9', 
                    'G3AJT9URG45M44HB', 'GAZ2TKL8VEEQUVRC', 'G202200040', 'GLSQSWPWZMLR3RA5', 'G201820380', 'G8XJWJRQCYN3PG6L']

output = dict()
output_url = dict()
length_threshold = 420

for key,val in collection.items():
    if val[1] in ignored_node_ids+digital_node_ids or len(val[0].split()) > length_threshold:
        print('Dropping: {}'.format(val[2]))
        continue
    output[key] = [val[0],get_preprocessed_document_ir(val[0])]
    output_url[key] = [val[2],val[3]]

with open(outfile, "w") as f:  
    json.dump(output, f)
    
with open(outfile_url, "w") as f:  
    json.dump(output_url, f)
```


```python
import json
import pandas as pd

collection = json.load(open('/data/QAData/InformationRetrievalData/amazon/helpdocuments_collection_new.json','r'))
outfile = '/data/QAData/InformationRetrievalData/amazon/production_collection.json'
outfile_url = '/data/QAData/InformationRetrievalData/amazon/production_collection_url.json'
#prod_pid_excluded = [1, 4, 5, 12, 13, 31, 35, 75, 98, 103, 112, 149, 183, 193] #122, 38 removed from this list
# digital_pid = [2, 9, 10, 11, 22, 30, 32, 36, 38, 44, 46, 47, 65, 69, 80, 91, 95, 96, 129, 131, 137, 138, 148, 150, 151, 155, 173, 178, 184, 185, 189, 191, 196, 199, 225, 237, 243]
ignored_node_ids = ['G4KFSMKGWZPYNW9X', 'G202185710', 'G8P3YMHU34R3XX4N', 'G202120810', 'G201911080', 'G201910410', 'GZ3PDPNA7U6R6UMK', 
                    'GAEJPJ8E5TY8TTNL', 'GWGDSNXVPJ93UW5V', 'GSD587LKW72HKU2V', 'G3BQ95AL4WZLV9ZG', 'G200127470', 'GL4263XHAGWBSC8R', 
                    'G64ENL4SCTZ4EXSY']
digital_node_ids = ['GJT6X5TZUW8AB9Y9', 'G201567520', 'G9JFV7VRANZDKKWG', 'GW5P2J5UV6EHRZCL', 'GE8EWPM8E5QG8E8E', 'G201555990', 'G201890100', 
                    'GXQJG7FBB6SJD22T', 'G3CHA35W7N58VG5B', 'GTQEND3RFAFNLKU5', 'G201609150', 'GSNFRLBMD26UK9PA', 'GQ8QNEVL5FCKBGH2', 
                    'G8637923FFWAR2YH', 'GZCHXL8CUW3VWJQP', 'G202070170', 'GT4SYLY6SVG9QWBV', 'G937D322PWZ6L9BL', 'GEA2QYCTZQXKRG4Z', 
                    'G3EWVFQZCVKH53TB', 'G201755180', 'G202020180', 'GDMMKH7RJP2HU2P2', 'GLSQ4722655M4ZEJ', 'GCRZL3F2UZMNP3T3', 
                    'GRTHB5PBJ32UUSMT', 'G202161160', 'GHANKAZLVY3SKWMV', 'G7CF8KV3YPXE285Y', 'GQ2FT94RCVV5BA3Z', 'GWWLWFYVTYQTLLL9', 
                    'G3AJT9URG45M44HB', 'GAZ2TKL8VEEQUVRC', 'G202200040', 'GLSQSWPWZMLR3RA5', 'G201820380', 'G8XJWJRQCYN3PG6L']

output = dict()
output_url = dict()
length_threshold = 420
docs = []

for key,val in collection.items():
    if val[1] in ignored_node_ids or val[1] in digital_node_ids or len(val[0].split()) > length_threshold:
        print('Dropping: {}'.format(val[2]))
        continue
    output[key] = [val[0],get_preprocessed_document_ir(val[0])]
    output_url[key] = val[2]
    doc = dict()
    doc['pid'] = key
    doc['title'] = val[3]
    doc['url'] = val[2]
    doc['passage'] = val[0]
    docs.append(doc)
documents = pd.DataFrame(docs)
documents.to_csv('production_documents.tsv',sep='\t')
```


```python
len(output)
```


```python
## Length Distribution of collection

import json
import numpy as np
import matplotlib.pyplot as plt
output = json.load(open('/data/QAData/InformationRetrievalData/amazon/production_collection.json','r'))
x = []
for key, val in output.items():
    x.append(len(val[0].split()))
plt.hist(x)
z = np.array(x)
np.quantile(z,0.80)
len(output)
```

## Creating AE Annotation into S3


```python
import json
import json
import pandas as pd

collection = json.load(open('/data/QAData/InformationRetrievalData/amazon/helpdocuments_collection_new.json','r'))
outfile = '/data/QAData/InformationRetrievalData/amazon/production_collection.json'
#prod_pid_excluded = [1, 4, 5, 12, 13, 31, 35, 75, 98, 103, 112, 149, 183, 193] #122, 38 removed from this list
# digital_pid = [2, 9, 10, 11, 22, 30, 32, 36, 38, 44, 46, 47, 65, 69, 80, 91, 95, 96, 129, 131, 137, 138, 148, 150, 151, 155, 173, 178, 184, 185, 189, 191, 196, 199, 225, 237, 243]
ignored_node_ids = ['G4KFSMKGWZPYNW9X', 'G202185710', 'G8P3YMHU34R3XX4N', 'G202120810', 'G201911080', 'G201910410', 'GZ3PDPNA7U6R6UMK', 
                    'GAEJPJ8E5TY8TTNL', 'GWGDSNXVPJ93UW5V', 'GSD587LKW72HKU2V', 'G3BQ95AL4WZLV9ZG', 'G200127470', 'GL4263XHAGWBSC8R', 
                    'G64ENL4SCTZ4EXSY']
digital_node_ids = ['GJT6X5TZUW8AB9Y9', 'G201567520', 'G9JFV7VRANZDKKWG', 'GW5P2J5UV6EHRZCL', 'GE8EWPM8E5QG8E8E', 'G201555990', 'G201890100', 
                    'GXQJG7FBB6SJD22T', 'G3CHA35W7N58VG5B', 'GTQEND3RFAFNLKU5', 'G201609150', 'GSNFRLBMD26UK9PA', 'GQ8QNEVL5FCKBGH2', 
                    'G8637923FFWAR2YH', 'GZCHXL8CUW3VWJQP', 'G202070170', 'GT4SYLY6SVG9QWBV', 'G937D322PWZ6L9BL', 'GEA2QYCTZQXKRG4Z', 
                    'G3EWVFQZCVKH53TB', 'G201755180', 'G202020180', 'GDMMKH7RJP2HU2P2', 'GLSQ4722655M4ZEJ', 'GCRZL3F2UZMNP3T3', 
                    'GRTHB5PBJ32UUSMT', 'G202161160', 'GHANKAZLVY3SKWMV', 'G7CF8KV3YPXE285Y', 'GQ2FT94RCVV5BA3Z', 'GWWLWFYVTYQTLLL9', 
                    'G3AJT9URG45M44HB', 'GAZ2TKL8VEEQUVRC', 'G202200040', 'GLSQSWPWZMLR3RA5', 'G201820380', 'G8XJWJRQCYN3PG6L']

output = dict()
length_threshold = 420


for key,val in collection.items():
    if val[1] in ignored_node_ids or val[1] in digital_node_ids or len(val[0].split()) > length_threshold:
        print('Dropping: {}'.format(val[2]))
        continue
    output[key] = val[2] + '\t' + val[0]

with open("annotation_collection.json", "w") as outfile:  
    json.dump(output, outfile)
```


```python
len(output)
```


```python
import json
import os
import boto3
from json import JSONDecodeError

## load to s3 for annotation

annotation_file = 'annotation_collection.json'
try:
    with open(annotation_file, 'r') as f:
        collection = json.load(f)
except (JSONDecodeError, FileNotFoundError) as error:
    print('Error: please make sure the path is correct and the file is a json file')
    raise error


## s3 initialize
os.environ['AWS_ACCESS_KEY_ID']='<replace with access key CS-ML-ANALYTICS account>'
os.environ['AWS_SECRET_ACCESS_KEY']='<replace with secret key of CS-ML-ANALYTICS account>'
BUCKET_NAME_INPUT='qabot-annotation-input-ae'
BUCKET_NAME_OUTPUT='qabot-annotation-output-ae'
s3_client = boto3.client('s3')

for k,v in collection.items():
    filename=str(k)
    body=v
    s3_client.put_object(Body=body,Bucket=BUCKET_NAME_INPUT,Key=filename+'.txt',)
```

### Processing Annotated Data from S3


```python
import pandas as pd
import json
import re
pd.set_option('colwidth',None)
def remove_urls(df):
    my_url_regex = r"(\(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^)\s]{2,}|www\.[a-zA-Z0-9]" \
                   r"[a-zA-Z0-9-]+[a-zA-Z0-9]\.[^)\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^)\s]{2,}|www\." \
                   r"[a-zA-Z0-9]+\.[^)\s]{2,})"

    print("----------------------------------------------------------------------------------")
    print("----- Will remove all urls in passage and answer and replace them with (URL) -----")
    print("----------------------------------------------------------------------------------")

    for index, row in df.iterrows():
        df.at[index, "passage"] = re.sub(my_url_regex, "URL", row["passage"])
        df.at[index, "answer"] = re.sub(my_url_regex, "URL", row["answer"])
    return df

def process_annotation_data(out_file_path:str):
    ## s3 initialize
    os.environ['AWS_ACCESS_KEY_ID']='<replace with access key CS-ML-ANALYTICS account>'
    os.environ['AWS_SECRET_ACCESS_KEY']='<replace with secret key of CS-ML-ANALYTICS account>'
    BUCKET_NAME_INPUT='qabot-annotation-input-ae'
    BUCKET_NAME_OUTPUT='qabot-annotation-output-ae'
    s3_client = boto3.client('s3')

    try:
        s3_obj = s3_client.get_object(Bucket=BUCKET_NAME_OUTPUT,Key='count.txt')
        s3_data = s3_obj['Body'].read()
        count = int(s3_data)
    except:
        count = 0 

    out = []
    for key in range(1,count+1):
        try:
            s3_obj = s3_client.get_object(Bucket=BUCKET_NAME_OUTPUT,Key=str(key)+'.txt')
            s3_data = s3_obj['Body']._raw_stream.readline().decode('utf-8')
            data = json.loads(s3_data)
            temp = dict()
            temp['qid'] = key
            temp['query'] = data['question']
            temp['answer'] = data['answer']
            temp['passage'] = data['content']
            out.append(temp)
        except:
            continue
    out_df = pd.DataFrame(out)
    remove_urls(out_df)
    out_df.to_csv(out_file_path,sep='\t',index=None)
    return out_df

result = process_annotation_data('/data/QAData/AnswerExtractionData/amazon/train_finetune.tsv')
```


```python
result
```


```python

```
