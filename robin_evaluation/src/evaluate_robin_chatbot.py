import os
import pandas as pd
import requests
import json
import time


ROB_DATASET_PATH = os.getenv('ROB_DATASET_PATH')
BASE_URL = 'http://chatbot_api:8000'
FILE_ENDPOINT = '/robin-file-agent'
ROB_EVALUATION_OUTPUT = '/data/output/'

rob_dataset = pd.read_csv(ROB_DATASET_PATH)

print(rob_dataset.columns)
for i, row in rob_dataset.iterrows():
    if os.path.exists(ROB_EVALUATION_OUTPUT + 'evaluation_' + str(row['index']) + '.json'):
        print("Already evaluated")
        continue

    filepath = '/data/pmc/' + row['filename']
    if not os.path.exists(filepath):
        print("File not found")
        continue

    data = {
        'query_text': row['question'],
        'filename': row['filename'],
    }
    file = {'uploaded_file': (row['filename'], open(filepath, 'rb'))}
    req = requests.post(BASE_URL + FILE_ENDPOINT, files=file, params=data, verify=False)
    resp = req.json()
    result = {
        'index': row['index'],
        'question': row['question'],
        'filename': row['filename'],
        'query_text': resp['query_text'],
        'answer': resp['answer']['output'],
        'ground_truth': row['label'],
        'intermediate_steps': resp['answer']['intermediate_steps']
    }

    with open(ROB_EVALUATION_OUTPUT + 'evaluation_' + str(row['index']) + '.json', 'w') as f:
        json.dump(result, f, indent=4)
