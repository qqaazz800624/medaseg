#%%

import json
import os

data_root = '/data2/open_dataset/AMOS22'
datalist_ct_path = 'amos22_ct.json'
datalist_all_path = 'dataset.json'

json_all_path = os.path.join(data_root, datalist_all_path)

with open(json_all_path, 'r') as f:
    datalist_all = json.load(f)


json_ct_path = os.path.join(data_root, datalist_ct_path)

with open(json_ct_path, 'r') as f:
    datalist_ct = json.load(f)

#%%

datalist_ct_mr = {}
datalist_ct_mr['training'] = datalist_all['training']
datalist_ct_mr['validation'] = datalist_all['validation']
datalist_ct_mr['testing'] = datalist_all['test']


#%%

datalist_ct

#%%

datalist_ct_mr




#%%