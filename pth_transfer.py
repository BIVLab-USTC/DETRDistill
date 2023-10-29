# -*- coding: utf-8 -*-

import torch
import argparse
from collections import OrderedDict

def change_model(args):
    distill_model = torch.load(args.ckpt_path)
    all_name = []
    for name, v in distill_model["state_dict"].items():
        if name.startswith("student."):
            all_name.append((name[8:], v))
        else:
            continue
    state_dict = OrderedDict(all_name)
    distill_model['state_dict'] = state_dict
    distill_model.pop('optimizer')
    torch.save(distill_model, args.output_path) 

           
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer CKPT')
    parser.add_argument('--ckpt_path', type=str, default='work_dirs/', 
                        metavar='N',help='distill_model path')
    parser.add_argument('--output_path', type=str, default='work_dirs/', 
                        help = 'pair path')
    args = parser.parse_args()
    change_model(args)
