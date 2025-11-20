'''
saves trained models at the last
'''
from torchvision import datasets, transforms
from torchvision.models import resnet18 as Presnet18
from torchvision.models import resnet34 as Presnet34
from torchvision.models import resnet101 as Presnet101
from torchvision.models import wide_resnet101_2 as Pwide_resnet101_2
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet101_Weights, Wide_ResNet101_2_Weights

import argparse
import os
import time
import torchsummary
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
# from AutoAugment.autoaugment import ImageNetPolicy

parser = argparse.ArgumentParser()
parser.add_argument('--num_users', type=int, default=100)
parser.add_argument('--noniid', type=str, default='iid') # iid, noniid, noniiddir
parser.add_argument('--class_per_each_client', type=int, default=10)

parser.add_argument('--frac', type=float, default=0.1)
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--local_bs', type=int, default=32)
parser.add_argument('--momentum', type=float, default=0)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--local_ep', type=int, default=5)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-1)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--mode', type=str, default='normal') # normal worst
parser.add_argument('--rs', type=int, default=0)

parser.add_argument('--train_ratio', type=str, default='16-1', help="Training ratio between large and small submodel, e.g., '16-1'")
parser.add_argument('--device_ratio', type=str, default='S2-W8', help="Device ratio between strong and weak devices, e.g., 'S2-W8'")

parser.add_argument('--num_experiment', type=int, default=3, help="the number of experiments")
parser.add_argument('--model_name', type=str, default='resnet34') # wide_resnet101_2
parser.add_argument('--device_id', type=str, default='3')
parser.add_argument('--learnable_step', type=bool, default=False) # False: FjORD / HeteroFL / DepthFL / ScaleFL
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--wandb', type=bool, default=True)

parser.add_argument('--dataset', type=str, default='cifar10') # stl10, cifar10, svhn, cinic
parser.add_argument('--method', type=str, default='WD') # DD, W, WD / fjord

# ============ Computation & Communication Heterogeneity ============
parser.add_argument('--strong_comp', type=float, default=2.01, help='Strong device computation time (GPU: NVIDIA 2080 Ti)')
parser.add_argument('--weak_comp', type=float, default=2.56, help='Weak device computation time (CPU: Intel Core i5)')
parser.add_argument('--strong_bw', type=float, default=10, help='Strong device bandwidth in Mbps (weak comm for strong device)')
parser.add_argument('--weak_bw', type=float, default=100, help='Weak device bandwidth in Mbps (strong comm for weak device)')
parser.add_argument('--random_bw', action='store_true', help='Enable random bandwidth sampling')
parser.add_argument('--comm_scenario', type=str, default='fixed_fast', 
                    help='Communication scenario: fixed_fast, random_fast, fixed_slow, random_slow')
parser.add_argument('--measure_wall_time', action='store_true', 
                    help='Enable actual wall time measurement (for calibrating strong_comp/weak_comp parameters)')
parser.add_argument('--force_cpu', action='store_true',
                    help='Force using CPU for training (to measure weak device computation time)')
# ==================================================================

# parser.add_argument('--name', type=str, default='[cifar10][NeFLADD2][R56]') # L-A: bad character
args = parser.parse_args()

# Device selection: Force CPU if measuring weak device time
if args.force_cpu:
    args.device = 'cpu'
    print("\n⚠️  FORCE CPU MODE: Training on CPU to measure weak device computation time")
else:
    args.device = 'cuda:' + args.device_id
    print(f"\n✓ GPU MODE: Training on CUDA device {args.device_id}")

# Set communication scenario presets
if args.comm_scenario == 'fixed_fast':
    args.strong_bw = 10  # Strong device with weak comm
    args.weak_bw = 100   # Weak device with strong comm
    args.random_bw = False
elif args.comm_scenario == 'random_fast':
    args.strong_bw = 10
    args.weak_bw = 100
    args.random_bw = True
elif args.comm_scenario == 'fixed_slow':
    args.strong_bw = 3
    args.weak_bw = 30
    args.random_bw = False
elif args.comm_scenario == 'random_slow':
    args.strong_bw = 3
    args.weak_bw = 30
    args.random_bw = True

print(f"\n=== System Heterogeneity Configuration ===")
print(f"Computation: Strong={args.strong_comp}s, Weak={args.weak_comp}s")
print(f"Communication: {args.comm_scenario}")
print(f"  Strong device: {args.strong_bw} Mbps (min)")
print(f"  Weak device: {args.weak_bw} Mbps (max)")
print(f"  Random sampling: {args.random_bw}")

if args.measure_wall_time:
    print(f"\n🔬 WALL TIME MEASUREMENT MODE ENABLED")
    print(f"   This mode will measure ACTUAL training time on your hardware.")
    print(f"   Use this to calibrate --strong_comp and --weak_comp parameters.")
    print(f"   Current device: {'CPU (weak)' if args.force_cpu else 'GPU (strong)'}")

print(f"==========================================\n")

# args.ps = [sqrt(0.2), sqrt(0.4), sqrt(0.6), sqrt(0.8), 1] # only width -> param. size [0.2, 0.4, 0.6, 0.8, 1]
# args.ps = [0.2, 0.4, 0.6, 0.8, 1]
# parameter size gets 1/r^2 [r1, r2, r3, r4, r5]

def simulate_time(device_type, model_size_bytes, args):
    """
    Simulate computation and communication time for heterogeneous devices.
    
    Args:
        device_type: 'S' (Strong) or 'W' (Weak)
        model_size_bytes: Size of model in bytes
        args: Arguments containing bandwidth and computation settings
    
    Returns:
        (computation_time, communication_time) in seconds
    """
    # Convert Mbps to bytes per second
    min_bw_bytes = args.strong_bw * 1e6 / 8
    max_bw_bytes = args.weak_bw * 1e6 / 8
    
    if args.random_bw:
        # Random bandwidth sampling
        bandwidth = np.random.uniform(min_bw_bytes, max_bw_bytes)
        if device_type == 'S':
            comp_time = args.strong_comp
            comm_time = model_size_bytes / bandwidth
        else:
            comp_time = args.weak_comp
            comm_time = model_size_bytes / bandwidth
    else:
        # Fixed bandwidth: Strong device gets weak comm, Weak device gets strong comm
        if device_type == 'S':
            comp_time = args.strong_comp
            comm_time = model_size_bytes / min_bw_bytes  # Strong device with weak bandwidth
        else:
            comp_time = args.weak_comp
            comm_time = model_size_bytes / max_bw_bytes  # Weak device with strong bandwidth
    
    return comp_time, comm_time

""" Vaying width of the network """
'''
network keys
- conv1.weight / bn1.weight/bias / bn1.running_mean / bn1.running_var / bn1.num_batches_tracked
- layerx.x. conv1.weight / bn1.weight/bias / bn1.running_mean / bn1.running_var / bn1.num_batches_tracked
            conv2.weight / bn2.weight/bias / bn2.running_mean / bn2.running_var / bn2.num_batches_tracked
- linear.weight/bias => fc.weight/bias
shape = w[key].shape

len(shape) = 4: (conv1) / (layer.conv1 / layer.conv2)
len(shape) = 2: (linear.weight)
len(shape) = 1: bn1.weight/bias/running_mean/var [16/32/...] / (linear.bias) [10] / step_size
len(shape) = 0: bn1.num_batches_tracked
'''

dataset_train, dataset_test = getDataset(args)

if args.noniid == 'noniid':
    dict_users = cifar_noniid(args, dataset_train)
elif args.noniid == 'noniiddir':
    dict_users = cifar_noniiddir(args, 0.5, dataset_train)
else:
    dict_users = cifar_iid(dataset_train, args.num_users, args.rs)
# img_size = dataset_train[0][0].shape


def main():
    # args.ps = [sqrt(0.2), sqrt(0.4), sqrt(0.6), sqrt(0.8), 1]
    # args.ps = [0.2, 0.4, 0.6, 0.8, 1]

    args.ps, args.s2D = get_submodel_info(args)
    args.num_models = len(args.s2D)  # Should be 2 now
    
    # Parse device_ratio (e.g., "S2-W8" means Strong:Weak = 2:8)
    device_parts = args.device_ratio.split('-')
    strong_ratio = int(device_parts[0][1:])  # Remove 'S' and get number
    weak_ratio = int(device_parts[1][1:])    # Remove 'W' and get number
    total_ratio = strong_ratio + weak_ratio
    
    # Calculate number of strong and weak devices
    args.num_strong_devices = int(args.num_users * strong_ratio / total_ratio)
    args.num_weak_devices = args.num_users - args.num_strong_devices

    local_models = []
    if args.model_name == 'resnet18':
        for i in range(args.num_models):
            local_models.append(resnet18wd(args.s2D[i][0], args.ps[i], args.learnable_step, args.num_classes))
    elif args.model_name == 'resnet56':
        for i in range(args.num_models):
            local_models.append(resnet56wd(args.s2D[i][0], args.ps[i], args.learnable_step, args.num_classes))
            # local_models.append(resnet56_DW7(args.s2D[i][0], args.ps[i], args.learnable_step, args.num_classes))
    elif args.model_name == 'resnet110':
        # args.epochs=800
        for i in range(args.num_models):
            local_models.append(resnet110wd(args.s2D[i][0], args.ps[i], args.learnable_step, args.num_classes))
    elif args.model_name == 'resnet34':
        for i in range(args.num_models):
            local_models.append(resnet34wd(args.s2D[i][0], args.ps[i], args.learnable_step, args.num_classes))
    elif args.model_name == 'resnet101':
        for i in range(args.num_models):
            local_models.append(resnet101wd(args.s2D[i][0], args.ps[i], args.learnable_step, args.num_classes))
    elif args.model_name == 'wide_resnet101_2':
        for i in range(args.num_models):
            local_models.append(resnet101_2wd(args.s2D[i][0], args.ps[i], args.learnable_step, args.num_classes))

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
    elif args.model_name== 'resnet34':
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
    elif args.model_name== 'resnet101':
        net_glob = resnet101wd(args.s2D[-1][0], 1, True, num_classes=args.num_classes)
        if args.pretrained:
            w_glob = net_glob.state_dict()
            net_glob_temp = Presnet101(weights=ResNet101_Weights.IMAGENET1K_V2) ########
            net_glob_temp.fc = nn.Linear(512 * 4, 10)

            w_glob_temp = net_glob_temp.state_dict()
            for key in w_glob_temp.keys():
                w_glob[key] = w_glob_temp[key]
            net_glob.load_state_dict(w_glob)        
    elif args.model_name == 'resnet56':
        net_glob = resnet56wd(args.s2D[-1][0], 1, True, num_classes=args.num_classes)
    elif args.model_name == 'resnet110':
        net_glob = resnet110wd(args.s2D[-1][0], 1, True, num_classes=args.num_classes)
    elif args.model_name == 'wide_resnet101_2':
        net_glob = resnet101_2wd(args.s2D[-1][0], 1, True, num_classes=args.num_classes)
        if args.pretrained:
            net_glob_temp = Pwide_resnet101_2(weights=Wide_ResNet101_2_Weights.IMAGENET1K_V1)
            w_glob = net_glob.state_dict()
            net_glob_temp.fc = nn.Linear(2048, args.num_classes)
            w_glob_temp = net_glob_temp.state_dict()
            for key in w_glob_temp.keys():
                w_glob[key] = w_glob_temp[key]
            net_glob.load_state_dict(w_glob)
    # torchsummary.summary(net_glob, (3, 32, 32), device='cpu')

    # if args.pretrained:
    #     for i in range(len(local_models)):
    #         model_idx = i
    #         p_select = args.ps[model_idx]
    #         p_select_weight = extract_submodel_weight_from_globalM(net = copy.deepcopy(net_glob), BN_layer=BN_layers, Step_layer=Steps, p=p_select, model_i=model_idx)
    #         local_models[model_idx].load_state_dict(p_select_weight)

    net_glob.to(args.device)
    # torchsummary.summary(local_models[0], (3, 32, 32)) # device='cpu'
    net_glob.train()

    w_glob = net_glob.state_dict()
    
    com_layers = []  # common layers: conv1, bn1, linear
    sing_layers = []  # singular layers: layer1.0.~ 
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

    if args.method == 'W':
        if args.learnable_step:
            method_name = 'NeFLW'
        else:
            method_name = 'FjORD'
    elif args.method == 'DD':
        if args.learnable_step:
            method_name = 'NeFLADD'
        else:
            method_name = 'NeFLDD'
    elif args.method == 'OD':
        if args.learnable_step:
            method_name = 'NeFLAOD'
        else:
            method_name = 'NeFLOD'
    elif args.method == 'WD':
        if args.learnable_step:
            method_name = 'NeFLWD'
        else:
            method_name = 'NeFLWDnL'
    
    if args.noniid == 'noniid': # noniid, noniiddir
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
    args.name = '[' + str(args.dataset) + ']' + '[' + model_name + ']' + method_name + niid_name + str(args.frac) + '[' + args.train_ratio + ']' + '[' + args.device_ratio + ']'
    filename = './output/nefl/'+ timestamp + str(args.name) + str(args.rs)
    if not os.path.exists(filename):
        os.makedirs(filename)

    if args.wandb:
        # wandb.init(dir=filename, project='fjord_psel__', name='fjord' + args.mode)
        run = wandb.init(dir=filename, project='NeFL-240426', name= str(args.name)+ str(args.rs), reinit=True)
        # wandb.run.name = str(stepSize2D) + timestamp
        wandb.config.update(args)
    logger = get_logger(logpath=os.path.join(filename, 'logs'), filepath=os.path.abspath(__file__))

    # ============ Device Type Assignment (System Heterogeneity) ============
    device_types = []
    for i in range(args.num_users):
        if i < args.num_strong_devices:
            device_types.append('S')  # Strong device
        else:
            device_types.append('W')  # Weak device
    
    print(f"\n=== Device Assignment ===")
    print(f"Total devices: {args.num_users}")
    print(f"Strong devices (GPU): {args.num_strong_devices}")
    print(f"Weak devices (CPU): {args.num_weak_devices}")
    print(f"Device ratio: {args.device_ratio}")
    print(f"========================\n")
    
    # Calculate model sizes for communication time simulation
    model_sizes = []  # in bytes
    for i in range(args.num_models):
        model_size = 0
        for param in local_models[i].parameters():
            model_size += param.data.nelement() * param.data.element_size()
        model_sizes.append(model_size)
        print(f"Model {i} size: {model_size / (1024*1024):.2f} MB")
    # ======================================================================

    lr = args.lr
    
    # For tracking system heterogeneity metrics
    total_wall_time = 0.0
    total_computation_time = 0.0
    total_communication_time = 0.0

    for iter in range(1,args.epochs+1):
        if iter == args.epochs/2:
            lr = lr*0.1
        elif iter == 3*args.epochs/4:
            lr = lr*0.1
        loss_locals = []
        # w_glob = net_glob.state_dict()
        w_locals = []
        w_locals.append([w_glob, args.num_models-1])
        
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        # Track max wall time for this round (straggler effect)
        round_max_wall_time = 0.0
        round_computation_times = []
        round_communication_times = []
        # weight_for_bn = [] ####
        # step_weights = []
        
        for idx in idxs_users:
            device_type = device_types[idx]  # Get device type (S or W)
            
            if args.mode == 'worst':
                dev_spec_idx = 0
                model_idx = 0
            else:
                # Original NeFL device-model assignment logic
                dev_spec_idx = min(idx//(args.num_users//args.num_models), args.num_models-1)
                # model_idx = random.choice(mlist[max(0,dev_spec_idx-args.min_flex_num):min(len(args.ps),dev_spec_idx+1+args.max_flex_num)])
                
                # Simple assignment for 2-model setup:
                # Strong devices (S) train large model (index 1)
                # Weak devices (W) train small model (index 0)
                if args.num_models == 2:
                    model_idx = 1 if device_type == 'S' else 0
                else:
                    # Original flexible assignment for 5-model setup
                    mlist = [_ for _ in range(args.num_models)]
                    model_idx = random.choice(mlist[max(0,dev_spec_idx-2):min(len(args.ps),dev_spec_idx+1+2)])
                    
            p_select = args.ps[model_idx]
            
            p_select_weight = extract_submodel_weight_from_globalM(net = copy.deepcopy(net_glob), BN_layer=BN_layers, Step_layer=Steps, p=p_select, model_i=model_idx)
            # p_select_weight = p_submodel(net = copy.deepcopy(net_glob), BN_layer=BN_layers, p=p_select)
            model_select = local_models[model_idx]
            model_select.load_state_dict(p_select_weight)
            local = LocalUpdateM(args, dataset=dataset_train, idxs=dict_users[idx])
            
            # ============ Measure Actual Wall Time ============
            if args.measure_wall_time:
                actual_wall_start = time.time()
            # =================================================
            
            weight, loss = local.train(net=copy.deepcopy(model_select).to(args.device), learning_rate=lr)
            
            # ============ Record Actual Wall Time ============
            if args.measure_wall_time:
                actual_wall_time = time.time() - actual_wall_start
                device_wall_time = actual_wall_time
                print(f"    Device {idx} ({device_type}) Model {model_idx}: Actual wall time = {actual_wall_time:.4f}s")
                round_computation_times.append(device_wall_time)
            else:
                # Simulate computation and communication time
                comp_time, comm_time = simulate_time(device_type, model_sizes[model_idx], args)
                device_wall_time = comp_time * args.local_ep  # Total computation time
                round_computation_times.append(device_wall_time)
                round_communication_times.append(comm_time)
            # ================================================
            
            # Track max wall time (straggler effect)
            round_max_wall_time = max(round_max_wall_time, device_wall_time)
            
            w_locals.append([copy.deepcopy(weight), model_idx])
            loss_locals.append(copy.deepcopy(loss))
        
        # Update total time metrics
        total_wall_time += round_max_wall_time
        total_computation_time += sum(round_computation_times)
        total_communication_time += sum(round_communication_times)
        
        w_glob, BN_layers, Steps = NeFedAvg(w_locals, BN_layers, Steps, args, com_layers, sing_layers)
        net_glob.load_state_dict(w_glob)
        loss_avg = sum(loss_locals) / len(loss_locals)

        if args.measure_wall_time:
            print('Round {:3d}, Average loss {:.3f}, Measured wall time: {:.2f}s (Max: {:.2f}s, Avg: {:.2f}s)'.format(
                iter, loss_avg, round_max_wall_time,
                max(round_computation_times) if round_computation_times else 0,
                np.mean(round_computation_times) if round_computation_times else 0))
        else:
            print('Round {:3d}, Average loss {:.3f}, Round wall time: {:.2f}s (Max comp: {:.2f}s, Avg comm: {:.2f}s)'.format(
                iter, loss_avg, round_max_wall_time, 
                max(round_computation_times) if round_computation_times else 0, 
                np.mean(round_communication_times) if round_communication_times else 0))

        loss_train.append(loss_avg)
        #loss_train에 loss_avg 값 추가
        
        if args.mode == 'worst': ##########
            ti = 1
        else: ##########
            ti = args.num_models

        # ============ Test accuracy every round ============
        # Test global model
        net_glob.eval()
        acc_test_global, loss_test_global = test_img(net_glob, dataset_test, args)
        net_glob.train()
        
        if iter % 10 == 0:
            print("Global model test accuracy: {:.2f}".format(acc_test_global))
        
        # Prepare wandb log dictionary
        wandb_log_dict = {
            "Global model test accuracy": acc_test_global,
            "Global model test loss": loss_test_global,
        }
        
        # Test each local (sub)model
        for ind in range(ti):
            p = args.ps[ind]
            model_e = copy.deepcopy(local_models[ind])
            
            f = extract_submodel_weight_from_globalM(net = copy.deepcopy(net_glob), BN_layer=BN_layers, Step_layer=Steps, p=p, model_i=ind)
            model_e.load_state_dict(f)
            model_e.eval()
            acc_test, loss_test = test_img(model_e, dataset_test, args)
            
            # Print every 10 rounds to avoid cluttering output
            if iter % 10 == 0:
                print("  Local model " + str(ind) + " test accuracy: {:.2f}".format(acc_test))
            
            # Add to wandb log dictionary
            wandb_log_dict["Local model " + str(ind) + " test accuracy"] = acc_test
            wandb_log_dict["Local model " + str(ind) + " test loss"] = loss_test
        
        # Log to wandb with explicit step (communication round)
        if args.wandb:
            wandb.log(wandb_log_dict, step=iter)
        # ==================================================

    filename = './output/nefl/'+ timestamp + str(args.name) + str(args.rs) + '/models'
    if not os.path.exists(filename):
        os.makedirs(filename)

    for ind in range(ti):
        p = args.ps[ind]
        model_e = copy.deepcopy(local_models[ind])       
        f = extract_submodel_weight_from_globalM(net = copy.deepcopy(net_glob), BN_layer=BN_layers, Step_layer=Steps, p=p, model_i=ind)
        torch.save(f, os.path.join(filename, 'model' + str(ind) + '.pt'))

    # testing
    net_glob.eval()

    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    
    # ============ System Heterogeneity Summary ============
    print(f"\n{'='*60}")
    if args.measure_wall_time:
        print(f"Wall Time Measurement Results")
        print(f"{'='*60}")
        print(f"Device: {'CPU (weak)' if args.force_cpu else 'GPU (strong)'}")
        print(f"Model: {args.model_name}")
        print(f"Dataset: {args.dataset}")
        print(f"Local epochs: {args.local_ep}")
        print(f"\nMeasurement Results:")
        print(f"  Total wall time: {total_wall_time:.2f}s")
        print(f"  Avg wall time per round: {total_wall_time / args.epochs:.4f}s")
        print(f"  Avg wall time per device per round: {total_computation_time / (args.epochs * args.frac * args.num_users):.4f}s")
        print(f"\n💡 Recommended parameter:")
        if args.force_cpu:
            print(f"  --weak_comp {total_wall_time / args.epochs:.4f}")
        else:
            print(f"  --strong_comp {total_wall_time / args.epochs:.4f}")
    else:
        print(f"System Heterogeneity Analysis")
        print(f"{'='*60}")
        print(f"Communication Scenario: {args.comm_scenario}")
        print(f"  Strong device: {args.strong_comp}s compute, {args.strong_bw} Mbps comm")
        print(f"  Weak device: {args.weak_comp}s compute, {args.weak_bw} Mbps comm")
        print(f"  Random sampling: {args.random_bw}")
        print(f"\nTotal Training Metrics:")
        print(f"  Total wall time: {total_wall_time:.2f}s")
        print(f"  Total computation time: {total_computation_time:.2f}s")
        print(f"  Total communication time: {total_communication_time:.2f}s")
        print(f"  Avg wall time per round: {total_wall_time / args.epochs:.2f}s")
    print(f"{'='*60}\n")
    # ====================================================
  
    if args.wandb:
        # Log final system heterogeneity metrics
        wandb.log({
            "Final/Total Wall Time": total_wall_time,
            "Final/Total Computation Time": total_computation_time,
            "Final/Total Communication Time": total_communication_time,
            "Final/Avg Wall Time per Round": total_wall_time / args.epochs,
            "Final/Train Accuracy": acc_train,
            "Final/Test Accuracy": acc_test
        })
        run.finish()


if __name__ == "__main__":
    for i in range(args.num_experiment):
        torch.manual_seed(args.rs)
        torch.cuda.manual_seed(args.rs)
        torch.cuda.manual_seed_all(args.rs) # if use multi-GPU
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        np.random.seed(args.rs)
        random.seed(args.rs)
        main()
        args.rs = args.rs+1
