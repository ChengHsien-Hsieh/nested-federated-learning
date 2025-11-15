'''
NeFL with Dual Resource Heterogeneity (FedFold-style)
- Data Heterogeneity: Non-IID data distribution
- Resource Heterogeneity: Strong/Weak device ratio with different trainable width
'''
from torchvision import datasets, transforms
from torchvision.models import resnet18 as Presnet18
from torchvision.models import resnet34 as Presnet34
from torchvision.models import ResNet18_Weights, ResNet34_Weights

import argparse
import os
import wandb
from datetime import datetime
import numpy as np
import random
import torch
import torch.nn as nn
import copy

from models import *
from utils.fed import *
from utils.getData import *
from utils.util import test_img, extract_submodel_weight_from_globalM, get_logger
from utils.NeFedAvg import NeFedAvg

parser = argparse.ArgumentParser()
parser.add_argument('--num_users', type=int, default=100)
parser.add_argument('--noniid', type=str, default='iid') # iid, noniid, noniiddir
parser.add_argument('--class_per_each_client', type=int, default=2)

parser.add_argument('--frac', type=float, default=0.1)
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--local_bs', type=int, default=32)
parser.add_argument('--momentum', type=float, default=0)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--local_ep', type=int, default=5)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--mode', type=str, default='normal') # normal worst
parser.add_argument('--rs', type=int, default=2)

parser.add_argument('--model_num', type=int, default=5)

# ============ NEW: Dual Resource Heterogeneity Parameters ============
parser.add_argument('--device_ratio', type=str, default='S2-W8', 
                    help='Device ratio: S2-W8 means 2 strong : 8 weak')
parser.add_argument('--strong_models', type=str, default='2,3,4',
                    help='Model indices that strong devices can train (comma-separated)')
parser.add_argument('--weak_models', type=str, default='0,1,2',
                    help='Model indices that weak devices can train (comma-separated)')
parser.add_argument('--only_strong', action='store_true',
                    help='Train only strong devices')
parser.add_argument('--only_weak', action='store_true',
                    help='Train only weak devices')
parser.add_argument('--use_dual_hetero', action='store_true',
                    help='Enable dual resource heterogeneity mode')
# ====================================================================

parser.add_argument('--num_experiment', type=int, default=1, help="the number of experiments")
parser.add_argument('--model_name', type=str, default='resnet18')
parser.add_argument('--device_id', type=str, default='0')
parser.add_argument('--learnable_step', type=bool, default=True)
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--wandb', type=bool, default=True)

parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--method', type=str, default='WD') # DD, W, WD

args = parser.parse_args()
args.device = 'cuda:' + args.device_id

# Parse strong/weak model indices
args.strong_model_indices = [int(x) for x in args.strong_models.split(',')]
args.weak_model_indices = [int(x) for x in args.weak_models.split(',')]

dataset_train, dataset_test = getDataset(args)

if args.noniid == 'noniid':
    dict_users = cifar_noniid(args, dataset_train)
elif args.noniid == 'noniiddir':
    dict_users = cifar_noniiddir(args, 0.5, dataset_train)
else:
    dict_users = cifar_iid(dataset_train, args.num_users, args.rs)


def parse_device_ratio(device_ratio_str, num_users, frac):
    """
    Parse device ratio string and assign device types
    Example: 'S2-W8' with 100 users and frac=0.1 (10 selected per round)
    -> 2 strong, 8 weak per round
    -> In total: 20 strong devices, 80 weak devices
    """
    device_types = []
    device_counts = {}
    
    ratios = device_ratio_str.split('-')
    total_ratio = sum([int(r[1:]) for r in ratios])
    
    for ratio in ratios:
        device_type = ratio[0]  # 'S', 'M', or 'W'
        count = int(ratio[1:])
        num_devices = int(num_users * (count / total_ratio))
        device_types.extend([device_type] * num_devices)
        device_counts[device_type] = num_devices
    
    # Fill remaining devices with weak type
    while len(device_types) < num_users:
        device_types.append('W')
        device_counts['W'] = device_counts.get('W', 0) + 1
    
    return device_types, device_counts


def get_model_indices_for_device(device_type, args):
    """
    Determine which models a device should train based on its type
    """
    if device_type == 'S':  # Strong device
        return args.strong_model_indices
    elif device_type == 'W':  # Weak device
        return args.weak_model_indices
    elif device_type == 'M':  # Medium device (if needed)
        # Medium devices train middle-sized models
        mid_idx = len(args.strong_model_indices) // 2
        return args.strong_model_indices[:mid_idx+1]
    else:
        return args.weak_model_indices


def main():
    args.ps, args.s2D = get_submodel_info(args)
    args.num_models = len(args.s2D)

    # Initialize local models
    local_models = []
    if args.model_name == 'resnet18':
        for i in range(args.num_models):
            local_models.append(resnet18wd(args.s2D[i][0], args.ps[i], args.learnable_step, args.num_classes))
    elif args.model_name == 'resnet34':
        for i in range(args.num_models):
            local_models.append(resnet34wd(args.s2D[i][0], args.ps[i], args.learnable_step, args.num_classes))
    elif args.model_name == 'resnet56':
        for i in range(args.num_models):
            local_models.append(resnet56wd(args.s2D[i][0], args.ps[i], args.learnable_step, args.num_classes))

    BN_layers = []
    Steps = []

    for i in range(len(local_models)):
        local_models[i].to(args.device)
        local_models[i].train()
        BN = {}
        Step = {}
        w = copy.deepcopy(local_models[i].state_dict())
        for key in w.keys():
            if len(w[key].shape)<=1 and key!='fc.bias' and not 'step' in key:
                BN[key] = w[key]
            elif 'step' in key:
                Step[key] = w[key]
        BN_layers.append(copy.deepcopy(BN))
        Steps.append(copy.deepcopy(Step))

    # Initialize global model
    if args.model_name == 'resnet18':
        net_glob = resnet18wd(args.s2D[-1][0], 1, True, num_classes=args.num_classes)
        if args.pretrained:
            w_glob = net_glob.state_dict()
            net_glob_temp = Presnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            net_glob_temp.fc = nn.Linear(512 * 1, 10)
            net_glob_temp.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            w_glob_temp = net_glob_temp.state_dict()
            for key in w_glob_temp.keys():
                w_glob[key] = w_glob_temp[key]
            net_glob.load_state_dict(w_glob)
    elif args.model_name == 'resnet34':
        net_glob = resnet34wd(args.s2D[-1][0], 1, True, num_classes=args.num_classes)
        if args.pretrained:
            w_glob = net_glob.state_dict()
            net_glob_temp = Presnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            net_glob_temp.fc = nn.Linear(512 * 1, 10)
            net_glob_temp.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            w_glob_temp = net_glob_temp.state_dict()
            for key in w_glob_temp.keys():
                w_glob[key] = w_glob_temp[key]
            net_glob.load_state_dict(w_glob)
    elif args.model_name == 'resnet56':
        net_glob = resnet56wd(args.s2D[-1][0], 1, True, num_classes=args.num_classes)

    net_glob.to(args.device)
    net_glob.train()

    w_glob = net_glob.state_dict()
    
    com_layers = []
    sing_layers = []
    bn_keys = []
    step_keys = []
            
    for i in w_glob.keys():
        if 'bn' not in i and 'downsample.1' not in i and 'step' not in i:
            if 'layer' in i:
                sing_layers.append(i)
            else:
                com_layers.append(i)
        elif 'step' in i:
            step_keys.append(i)
        else:
            bn_keys.append(i)

    loss_train = []

    # Method naming
    if args.method == 'W':
        method_name = 'NeFLW' if args.learnable_step else 'FjORD'
    elif args.method == 'DD':
        method_name = 'NeFLADD' if args.learnable_step else 'NeFLDD'
    elif args.method == 'WD':
        method_name = 'NeFLWD' if args.learnable_step else 'NeFLWDnL'
    
    if args.noniid == 'noniid':
        niid_name = '[niid]'
    elif args.noniid == 'noniiddir':
        niid_name = '[dir]'
    else:
        niid_name = '[iid]'
        
    if args.pretrained:
        model_name = 'P' + args.model_name
    else:
        model_name = args.model_name

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Add dual heterogeneity info to experiment name
    if args.use_dual_hetero:
        hetero_tag = f"[DH-{args.device_ratio}]"
    else:
        hetero_tag = ""
    
    args.name = '[' + str(args.dataset) + ']' + '[' + model_name + ']' + method_name + niid_name + hetero_tag + '[frac' + str(args.frac) + ']'
    filename = './output/nefl/'+ timestamp + str(args.name) + '[rs' + str(args.rs) + ']'
    if not os.path.exists(filename):
        os.makedirs(filename)

    if args.wandb:
        run = wandb.init(dir=filename, project='NeFL-DualHetero', name= str(args.name)+ str(args.rs), 
                        reinit=True, settings=wandb.Settings(code_dir="."))
        wandb.config.update(args)
    logger = get_logger(logpath=os.path.join(filename, 'logs'), filepath=os.path.abspath(__file__))

    # ============ NEW: Device type assignment ============
    if args.use_dual_hetero:
        device_types, device_counts = parse_device_ratio(args.device_ratio, args.num_users, args.frac)
        print(f"Device distribution: {device_counts}")
        print(f"Strong devices train models: {args.strong_model_indices}")
        print(f"Weak devices train models: {args.weak_model_indices}")
    # ====================================================

    lr = args.lr
    mlist = [_ for _ in range(args.num_models)]

    # Track model usage statistics
    model_usage_count = [0] * args.num_models

    for iter in range(1, args.epochs+1):
        if iter == args.epochs/2:
            lr = lr*0.1
        elif iter == 3*args.epochs/4:
            lr = lr*0.1
        
        loss_locals = []
        w_locals = []
        w_locals.append([w_glob, args.num_models-1])
        
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        for idx in idxs_users:
            # ============ NEW: Dual Resource Heterogeneity Logic ============
            if args.use_dual_hetero:
                device_type = device_types[idx]
                
                # Filter devices based on only_strong/only_weak flags
                if args.only_strong and device_type != 'S':
                    continue
                if args.only_weak and device_type != 'W':
                    continue
                
                # Get allowed model indices for this device
                allowed_models = get_model_indices_for_device(device_type, args)
                
                # Select a model from allowed models
                model_idx = random.choice(allowed_models)
                
                print(f"Round {iter}, Device {idx} (Type: {device_type}): Training model {model_idx}")
            else:
                # Original NeFL logic
                if args.mode == 'worst':
                    dev_spec_idx = 0
                    model_idx = 0
                else:
                    dev_spec_idx = min(idx//(args.num_users//args.num_models), args.num_models-1)
                    model_idx = random.choice(mlist[max(0,dev_spec_idx-2):min(len(args.ps),dev_spec_idx+1+2)])
            # ==============================================================
            
            model_usage_count[model_idx] += 1
            p_select = args.ps[model_idx]
            
            p_select_weight = extract_submodel_weight_from_globalM(
                net=copy.deepcopy(net_glob), 
                BN_layer=BN_layers, 
                Step_layer=Steps, 
                p=p_select, 
                model_i=model_idx
            )
            
            model_select = local_models[model_idx]
            model_select.load_state_dict(p_select_weight)
            local = LocalUpdateM(args, dataset=dataset_train, idxs=dict_users[idx])
            weight, loss = local.train(net=copy.deepcopy(model_select).to(args.device), learning_rate=lr)
            
            w_locals.append([copy.deepcopy(weight), model_idx])
            loss_locals.append(copy.deepcopy(loss))
        
        w_glob, BN_layers, Steps = NeFedAvg(w_locals, BN_layers, Steps, args, com_layers, sing_layers)
        net_glob.load_state_dict(w_glob)
        loss_avg = sum(loss_locals) / len(loss_locals)

        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        print(f'Model usage count: {model_usage_count}')
        
        # Log training loss every round
        if args.wandb:
            wandb.log({"Training loss": loss_avg}, step=iter)

        loss_train.append(loss_avg)
        
        if iter % 10 == 0:
            ti = 1 if args.mode == 'worst' else args.num_models

            for ind in range(ti):
                p = args.ps[ind]
                model_e = copy.deepcopy(local_models[ind])
                
                f = extract_submodel_weight_from_globalM(
                    net=copy.deepcopy(net_glob), 
                    BN_layer=BN_layers, 
                    Step_layer=Steps, 
                    p=p, 
                    model_i=ind
                )
                model_e.load_state_dict(f)
                model_e.eval()
                acc_test, loss_test = test_img(model_e, dataset_test, args)
                print("Testing accuracy " + str(ind) + ": {:.2f}".format(acc_test))
                if args.wandb:
                    wandb.log({
                        "Local model " + str(ind) + " test accuracy": acc_test,
                        "Local model " + str(ind) + " test loss": loss_test
                    }, step=iter)

    # Save models
    filename = './output/nefl/'+ timestamp + str(args.name) + str(args.rs) + '/models'
    if not os.path.exists(filename):
        os.makedirs(filename)

    ti = 1 if args.mode == 'worst' else args.num_models
    for ind in range(ti):
        p = args.ps[ind]
        model_e = copy.deepcopy(local_models[ind])       
        f = extract_submodel_weight_from_globalM(
            net=copy.deepcopy(net_glob), 
            BN_layer=BN_layers, 
            Step_layer=Steps, 
            p=p, 
            model_i=ind
        )
        torch.save(f, os.path.join(filename, 'model' + str(ind) + '.pt'))

    # Final testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    print(f"Final model usage count: {model_usage_count}")
    
    # Save summary
    summary_path = './output/nefl/'+ timestamp + str(args.name) + '[rs' + str(args.rs) + ']' + '/summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Experiment Configuration:\n")
        f.write(f"========================\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Method: {method_name}\n")
        f.write(f"Non-IID: {args.noniid}\n")
        f.write(f"Device Ratio: {args.device_ratio if args.use_dual_hetero else 'N/A'}\n")
        f.write(f"Strong Models: {args.strong_model_indices if args.use_dual_hetero else 'N/A'}\n")
        f.write(f"Weak Models: {args.weak_model_indices if args.use_dual_hetero else 'N/A'}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Frac: {args.frac}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Random Seed: {args.rs}\n")
        f.write(f"\nFinal Results:\n")
        f.write(f"========================\n")
        f.write(f"Training Accuracy: {acc_train:.2f}%\n")
        f.write(f"Testing Accuracy: {acc_test:.2f}%\n")
        f.write(f"\nModel Usage Count:\n")
        f.write(f"========================\n")
        for i, count in enumerate(model_usage_count):
            f.write(f"Model {i}: {count} times\n")
    
    print(f"Summary saved to: {summary_path}")
  
    if args.wandb:
        # Log final results
        wandb.log({
            "Final/Train Accuracy": acc_train,
            "Final/Test Accuracy": acc_test,
            "Final/Train Loss": loss_train,
            "Final/Test Loss": loss_test
        }, step=args.epochs)
        
        # Log model usage statistics
        for i, count in enumerate(model_usage_count):
            wandb.log({f"Model Usage/Model {i}": count}, step=args.epochs)
        
        run.finish()


if __name__ == "__main__":
    for i in range(args.num_experiment):
        torch.manual_seed(args.rs)
        torch.cuda.manual_seed(args.rs)
        torch.cuda.manual_seed_all(args.rs)
        np.random.seed(args.rs)
        random.seed(args.rs)
        main()
        args.rs = args.rs+1
