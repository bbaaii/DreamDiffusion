import torch
import numpy as np

data = torch.load("eeg_5_95_std.pth")

print(f"data: dict with keys {data.keys()}")
print(f"data['dataset']: list of length {len(data['dataset'])}")
print(f"data['dataset'][0]['eeg']: {data['dataset'][0]['eeg'].shape}")

res1 = [sample['eeg'] for sample in data['dataset']]
print(len(res), [res[i].shape for i in range(5)])

np.save("output.npy", np.array(res1, dtype=object))
