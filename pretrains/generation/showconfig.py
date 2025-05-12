import torch
from collections import OrderedDict
import sys

data = torch.load(sys.argv[1])
print(f"checkpoint keys: {data.keys()}")

print(f"printing checkpoint['model_state_dict']")
model_state_dict = data['model_state_dict']
#for k, v in model_state_dict.items():
#        print(k, v.cpu().size())  # or v.cpu().numpy() if you want the values
print("-----")

print(f"printing checkpoint['config']")
config = data['config']
print(config)

print()
for key, value in config.__dict__.items():
    print(f"{key}: {value}")
print("-----")


print(f"printing checkpoint['state']")
state = data['state']
print(state.shape)
print("-----")
