from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch
import torch.nn.functional as F
from torchvision import transforms

from torch.optim.lr_scheduler import LambdaLR
import math

class DatasetSplit(Dataset):

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class AugmentedDatasetSplit(Dataset):
    """
    Dataset wrapper that supports:
    1. Strong data augmentation for small datasets
    2. Virtual size expansion (repeat samples with different augmentations)
    
    This is useful when num_users is large and each client has very few samples.
    
    NOTE: Handles both PIL Images (CIFAR-10) and Tensors (HAR) automatically.
    """

    def __init__(self, dataset, idxs, virtual_size=0, augment=True):
        """
        Args:
            dataset: Original dataset (PIL Images for CIFAR-10, Tensors for HAR)
            idxs: Indices for this client's data
            virtual_size: If > 0, expand dataset to this size by repeating samples
                          If 0, use original size
            augment: Whether to apply strong augmentation
        """
        self.dataset = dataset
        self.idxs = list(idxs)
        self.augment = augment
        
        # Virtual size: minimum of virtual_size or original size
        if virtual_size > 0:
            self.virtual_size = max(virtual_size, len(self.idxs))
        else:
            self.virtual_size = len(self.idxs)
        
        # Check if dataset returns tensors (HAR) or PIL Images (CIFAR-10)
        sample_image, _ = dataset[0]
        self.is_tensor_dataset = isinstance(sample_image, torch.Tensor)
        
        # Setup transforms based on dataset type
        if self.is_tensor_dataset:
            # HAR dataset: already tensors, already normalized
            self.strong_transform = None  # Will apply noise augmentation in __getitem__
        else:
            # CIFAR-10 dataset: PIL Images
            if augment:
                self.strong_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(
                        brightness=0.2, 
                        contrast=0.2, 
                        saturation=0.2, 
                        hue=0.1
                    ),
                    transforms.RandomAffine(
                        degrees=0, 
                        translate=(0.1, 0.1), 
                        scale=(0.9, 1.1)
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), 
                        (0.2023, 0.1994, 0.2010)
                    ),
                ])
            else:
                self.strong_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), 
                        (0.2023, 0.1994, 0.2010)
                    ),
                ])

    def __len__(self):
        return self.virtual_size

    def __getitem__(self, item):
        # Cycle through original indices if virtual_size > len(idxs)
        real_idx = self.idxs[item % len(self.idxs)]
        image, label = self.dataset[real_idx]
        
        if self.is_tensor_dataset:
            # HAR dataset: image is already a tensor, apply random noise
            if self.augment:
                noise = torch.randn_like(image) * 0.01
                image = image + noise
            return image, label
        else:
            # CIFAR-10 dataset: image is PIL Image, apply transform
            image = self.strong_transform(image)
            return image, label

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=True)

    def train(self, net, learning_rate):
        net.train()

        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        epoch_loss = []

        for iter in range(self.args.local_ep):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits, log_probs = net(images)
                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class LocalUpdateM(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        
        # Check if virtual size augmentation is enabled
        use_virtual_augment = hasattr(args, 'virtual_size') and args.virtual_size > 0
        
        if use_virtual_augment:
            # Use AugmentedDatasetSplit for virtual size expansion
            # Dataset should return PIL Images (transform=None in getDataset)
            self.ldr_train = DataLoader(
                AugmentedDatasetSplit(
                    dataset, 
                    idxs, 
                    virtual_size=args.virtual_size,
                    augment=True
                ), 
                batch_size=args.local_bs, 
                shuffle=True
            )
        else:
            # Standard DatasetSplit - augmentation is handled in getDataset's transform
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=True)

    def train(self, net, learning_rate):
        net.train()

        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        epoch_loss = []

        for iter in range(self.args.local_ep):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits, log_probs = net(images)
                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class LocalUpdateM_niid(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=True)
        self.total_steps = len(idxs)*args.epochs/args.local_bs*args.local_ep # dataset_qunaity * commun_round / local batch size * local epoch
        self.step_per_round = len(idxs)//args.local_bs

    def train(self, net, learning_rate, load_scheduler=None, load_optimizer=None):
        net.train()

        # train and update
        # torch.optim.AdamW
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        epoch_loss = []
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=self.args.warmup_steps, t_total=self.total_steps)        

        if load_scheduler:
            scheduler.load_state_dict(load_scheduler)
        if load_optimizer:
            optimizer.load_state_dict(load_optimizer)
            
        for iter in range(self.args.local_ep):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits, log_probs = net(images)
                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1) ##########
                optimizer.step()
                scheduler.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), scheduler.state_dict(), optimizer.state_dict()
    

class LocalUpdateM_ft(object): # Fine-tuning for ResNet
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=True)
        self.total_steps = len(idxs)*args.epochs/args.local_bs*args.local_ep # dataset_qunaity * commun_round / local batch size * local epoch
        self.step_per_round = len(idxs)//args.local_bs


    def train(self, net, learning_rate, load_scheduler=None, load_optimizer=None):
        net.train()

        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        epoch_loss = []
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=self.args.warmup_steps, t_total = self.total_steps)
        if load_scheduler:
            scheduler.load_state_dict(load_scheduler)
        if load_optimizer:
            optimizer.load_state_dict(load_optimizer)

        for iter in range(self.args.local_ep):
            batch_loss = []
                
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits, log_probs = net(images)
                loss = F.cross_entropy(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1)

                optimizer.step()
                scheduler.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        net.to('cpu')
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), scheduler.state_dict(), optimizer.state_dict()
    

class LocalUpdateM_pyvit(object): # Only difference: No log_probs
    '''
    preMAFL-pyViT.py
    '''
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=True)
        self.total_steps = len(idxs)*args.epochs/args.local_bs*args.local_ep # dataset_qunaity * commun_round / local batch size * local epoch
        self.step_per_round = len(idxs)//args.local_bs
        
    def train(self, net, round, learning_rate, load_scheduler=None, load_optimizer=None):
        net.train()
        net.to(self.args.device)

        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        epoch_loss = []
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=self.args.warmup_steps, t_total=self.total_steps)
        # scheduler._step_count = 1 + self.step_per_round*self.args.local_ep*(round-1)
        if load_scheduler:
            scheduler.load_state_dict(load_scheduler)
        if load_optimizer:
            optimizer.load_state_dict(load_optimizer)
            
        for iter in range(self.args.local_ep):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                # net.zero_grad()
                logits, log_probs = net(images)
                loss = F.cross_entropy(logits, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1) ##########
                
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                # print(optimizer.param_groups[0]['lr'], scheduler._step_count, scheduler._last_lr)
                
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        net.to('cpu')
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), scheduler.state_dict(), optimizer.state_dict()
            

class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress))) # 0.0 / 0.024
        

class LocalUpdateH(object):
    '''
    HeteroFL
    '''
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=True)

    def train(self, net, learning_rate):
        net.train()

        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        epoch_loss = []

        for iter in range(self.args.local_ep):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits, log_probs = net(images)
                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1) ### HeteroFL
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    
def sBN(model, dataset, args):
    dl = DataLoader(dataset, batch_size=args.local_bs, shuffle=True)
    with torch.no_grad():
        model.train()
        for i, (images, labels) in enumerate(dl):
            images, labels = images.to(args.device), labels.to(args.device)
            logits, log_probs = model(images)
        
    return model


class LocalUpdate_wr(object):
    '''
    Weight Regularization
    Additional Experiment
    '''
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=True)

    def train(self, net, learning_rate, sub_net_weight, wl=1e-5):
        net.train()
        # p_net_weight=p_select_weight, sub_net_weight=sub_weight
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        epoch_loss = []

        for iter in range(self.args.local_ep):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits, log_probs = net(images)
                if sub_net_weight != -1:
                    wdiff = weight_loss(net.state_dict(), sub_net_weight)
                else:
                    wdiff = 0
                loss = F.cross_entropy(logits, labels) + wl*wdiff
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    
def weight_loss(mother_net, sub_net):
    loss = 0
    for key in mother_net.keys():
        tmp = {}
        sub_sh = sub_net[key].shape
        mot_sh = mother_net[key].shape
        keyFlag = False
        if len(sub_sh) == 4:
            keyFlag = True
            if key == 'conv1.weight':
                tmp[key] = mother_net[key][sub_sh[0]:mot_sh[0], :, :, :]
            else:
                tmp[key] = mother_net[key][sub_sh[0]:mot_sh[0], sub_sh[1]:mot_sh[1], :, :]
        elif len(sub_sh) == 2:
            keyFlag = True
            tmp[key] = mother_net[key][:, sub_sh[1]:mot_sh[1]]
        if keyFlag:
            # tmp_l = sum(abs(tmp[key].reshape(-1)))
            # tmp_l = tmp[key].reshape(-1)
            tmp_l = torch.norm(tmp[key])
            loss += tmp_l
    return loss