import sys
import torch

in_file = sys.argv[1]
out_file = sys.argv[2]

checkpoint = torch.load(in_file, map_location='cpu')
state_dict = checkpoint['state_dict']
for k in list(state_dict.keys()):
    # keep backbone
    if k.startswith('module.encoder_q.backbone.'):
        # remove backbone prefix
        state_dict[k.replace('module.encoder_q.', '')] = state_dict[k]
    # delete renamed or unused k
    del state_dict[k]

torch.save(state_dict, out_file)