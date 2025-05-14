import warnings
from urllib3.exceptions import InsecureRequestWarning
from mne.io import BaseRaw
import os

def find_raw_objects(data, types=(BaseRaw), path=""):
    found = []

    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{path}.{key}" if path else key
            found.extend(find_raw_objects(value, types, new_path))

    elif isinstance(data, list):
        for idx, item in enumerate(data):
            new_path = f"{path}[{idx}]"
            found.extend(find_raw_objects(item, types, new_path))

    elif isinstance(data, types):
        found.append((path, data))

    return found

print()
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=InsecureRequestWarning)
print("WARNING : All 'FutureWarning' and 'InsecureRequestWarning' have been disabled")

import numpy as np
from moabb.datasets import *

datasets = [Schirrmeister2017(), GrosseWentrup2009()]
sample_size = 490

datas = []
for dataset in datasets:
    try:
        print("\n#####")
        print(dataset)
        print("#####\n")
        data = dataset.get_data(subjects=dataset.subject_list[:2])
        for raw_path, raw_data in find_raw_objects(data):
            print()
            npdata = raw_data.pick_types(eeg=True).get_data()
            print(f"Shape of data: {npdata.shape}")

            print(f"Splitting into samples of size {sample_size}")
            length = npdata.shape[1]
            samples = []
            i=0
            while (i + 1) * sample_size < length:
                sample = npdata[:, i * sample_size : (i+1) * sample_size]
                samples.append(sample)
                i += 1
            
            res = np.array(samples)
            print(f"New shape: {res.shape}")


            whole_path = 'mne_data/'+dataset.code+"/"+raw_path+'.npy'
            print(f"Saving in {whole_path}")

            os.makedirs(os.path.dirname(whole_path), exist_ok=True)
            np.save(whole_path, res)
            print()

    except Exception as e:
        print(e)
