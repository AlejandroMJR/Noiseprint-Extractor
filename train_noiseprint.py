# Libraries import #
from typing import List
import argparse
import datetime
import torch
torch.manual_seed(123)
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
from torch.multiprocessing import cpu_count
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset_general import CustomBatchSampler, MyDataset
from loss import DistanceBasedLogisticLoss
from model import DnCNN, Uformer
from model_retormer import Restormer
import torch
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader, BatchSampler
from pprint import pprint
from torch import nn as nn
import torchvision.transforms as transforms
import numpy as np
torch.set_default_tensor_type("torch.cuda.FloatTensor")

def save_model(net: torch.nn.Module, optimizer: torch.optim.Optimizer,
               train_loss: float, val_loss: float,
               batch_size: int, epoch: int,
               path: str):
    path = str(path)
    state = dict(net=net.state_dict(),
                 opt=optimizer.state_dict(),
                 train_loss=train_loss,
                 val_loss=val_loss,
                 batch_size=batch_size,
                 epoch=epoch)
    torch.save(state, path)


def make_train_tag(net_class: nn.Module,
                   lr: float,
                   batch_size: int,
                   ):
    # Training parameters and tag
    now = datetime.datetime.now()
    tag_params = dict(net=net_class.name,
                      lr=lr,
                      batch_size=batch_size,
                      time=now,
                      )
    print('Parameters')
    pprint(tag_params)
    tag = ''
    tag += '_'.join(['-'.join([key, str(tag_params[key])]) for key in tag_params])
    print('Tag: {:s}'.format(tag))
    return tag


def batch_forward(net: torch.nn.Module, device, criterion, data: torch.Tensor, labels: torch.Tensor) -> (
        torch.Tensor, float, int):
    if torch.cuda.is_available():
        #print("start loading data to gpu")
        data = data.to(device)
        #print("data loaded to gpu")
        labels = labels.to(device)
    out = net(data)
    #print("pass end")
    loss = criterion(out, labels)
    return loss


def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        #print(X.size())
        y = y.to(device)
        pred = model(X)
        #print(y.size())
        loss = loss_fn(y, pred)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #if batch % 20 == 0:
        loss, current = loss.item(), (batch + 1) * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def collate_fn(samples):
    """
    Collate function to handle a list of samples with different shapes
    """
    # Assume that each sample is a tuple (data, label)
    data = [s for s in samples[0]]
    label = samples[1]
    #print("collate: " , data[10][1,10,10])
    # Convert to tensor and stack
    data = torch.stack(data, dim=0)
    label = torch.tensor(label)

    return data, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-8)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--es_patience', type=int, default=15, help='Patience for stopping the training if no improvement'
                                                                    'on the validation loss is seen')
    parser.add_argument('--workers', type=int, default=cpu_count() // 2 )

    parser.add_argument('--log_dir', type=str, help='Directory for saving the training logs',
                        default="logs_new")

    parser.add_argument('--model', type=str, help='Denoising model',
                        default="Restormer")

    parser.add_argument('--models_dir', type=str, help='Directory for saving the models weights',
                        default="weigths_new")
    parser.add_argument('--init', type=str, help='Weight initialization file')
    parser.add_argument('--scratch', action='store_true', help='Train from scratch')

    args = parser.parse_args()

    # Parse arguments
    gpu = args.gpu
    batch_size = args.batch_size
    lr = args.lr
    min_lr = args.min_lr
    es_patience = args.es_patience
    epochs = args.epochs
    workers = args.workers
    denoiser = args.model
    weights_folder = args.models_dir
    logs_folder = args.log_dir
    #initial_model = args.init
    initial_model = "model_zoo/gaussian_gray_denoising_sigma25.pth"
    train_from_scratch = args.scratch
    #train_from_scratch = True

    # GPU configuration
    #device = 'cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #uncomment for using several gpu
    #torch.cuda.set_device(1)

    ps = 48 # 128 for Uformer, 48 for DnCNN,
    # Instantiate network
    if denoiser == "DnCNN":
        model = DnCNN()
        num_channels = 1
    if denoiser == "Restormer":
        model = Restormer()
        num_channels = 1
    if denoiser == "Uformer":
        model = Uformer(img_size=ps,embed_dim=32,win_size=8,token_projection='linear',token_mlp='leff',
            depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],modulator=True,dd_in=3)
        num_channels = 3



    model = nn.DataParallel(model) #uncomment for using several gpu
    model = model.to(device)
    # Print model's state_dict
    #print("Model's state_dict:")
    #for param_tensor in model.state_dict():
    #    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Transformer and augmentation
    #net_normalizer = model.get_normalizer()
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=np.mean([0.485, 0.456, 0.406]),
                             std=np.mean([0.229, 0.224, 0.225]))
    ])
    """

    #root_path = "/nas/public/dataset/Vision"

    #read csv files
    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("val.csv")

    n = 5
    ni = 4

    # load sampler
    train_sampler = CustomBatchSampler(train_df, batch_size, 1, n, ni)
    val_sampler = CustomBatchSampler(val_df, batch_size, 1, n, ni)

    #print("\nstart of dataset\n")
    train_ds = MyDataset(img_labels=train_df, patch_size=(ps,ps), num_channels=num_channels, num_classes=n, img_per_class=ni)
    #print("\n end of dataset1 \n")
    val_ds = MyDataset(img_labels=val_df, patch_size=(ps,ps), num_channels=num_channels, num_classes=n, img_per_class=ni)
    #print("\n end of dataset2 \n")
    train_dl = DataLoader(dataset=train_ds, batch_size=None, num_workers=workers, shuffle=False, sampler=train_sampler, collate_fn=collate_fn)
    #print("\n end of dataloader1 \n")
    val_dl = DataLoader(dataset=val_ds, batch_size=None, num_workers=workers, shuffle=False, sampler=val_sampler, collate_fn=collate_fn)
    #print("\n end of dataloader2 \n")

    # Optimization
    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = DistanceBasedLogisticLoss(2, ni*n*batch_size)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,patience=8)

    # Checkpoint paths
    #train_tag = make_train_tag(model, lr, batch_size)
    train_tag = make_train_tag(model.module, lr, batch_size) #uncomment if several gpus are used
    #train_tag = "_lr-0.0001_batch_size-1_time-2023-08-08 22:22:11.519313"
    bestval_path = os.path.join(weights_folder, train_tag , 'bestval.pth')
    last_path = os.path.join(weights_folder, train_tag , 'lastval.pth')

    os.makedirs(os.path.join(weights_folder, train_tag), exist_ok=True)

    # Load model from checkpoint
    min_val_loss = 1e10
    epoch = 0
    net_state = None
    opt_state = None
    if initial_model is not None:
        # If given load initial model
        print('Loading model form: {}'.format(initial_model))
        state = torch.load(initial_model, map_location='cpu')
        if denoiser == "Uformer":
            state = state["state_dict"]
        if denoiser == "DnCNN":
            state = state
        if denoiser == "Restormer":
            state = state["params"]
            #state = state['net']
            #Add "module." prefix from keys
            net_state = {}
            for key, value in state.items():
               new_key = "module." + key
               net_state[new_key] = value

        if net_state is None:
            net_state = state
        # Remove "module." prefix from keys
        #for key, value in state.items():
        #    if key.startswith('module.'):
        #        new_key = key[7:]  # Remove the "module." prefix
        #        net_state[new_key] = value
        #    else:
        #        net_state[key] = value
        #net_state = state
        #print(net_state)
        #print("Loaded state_dict:")
        #for param_tensor in net_state:
        #    print(param_tensor, "\t", net_state[param_tensor].size())

    elif not train_from_scratch and os.path.exists(last_path):
        print("check")
        print('Loading model form: {}'.format(last_path))
        state = torch.load(last_path, map_location='cpu')
        net_state = state['net']
        opt_state = state['opt']
        epoch = state['epoch']

    if not train_from_scratch and os.path.exists(bestval_path):
        state = torch.load(bestval_path, map_location='cpu')
        min_val_loss = state['val_loss']
    if net_state is not None:
        incomp_keys = model.load_state_dict(net_state, strict=False)
        print(incomp_keys)
    if opt_state is not None:
        for param_group in opt_state['param_groups']:
            param_group['lr'] = lr
        optimizer.load_state_dict(opt_state)

    # Initialize Tensorboard
    logdir = os.path.join(logs_folder, train_tag)

    # Tensorboard instance
    tb = SummaryWriter(log_dir=logdir)

    # Training-validation loop
    train_tot_it = 0
    val_tot_it = 0
    es_counter = 0
    init_epoch = epoch
    for epoch in range(init_epoch, epochs):

        # Training
        model.train()
        optimizer.zero_grad()
        train_loss = train_num = 0
        for batch_data in tqdm(train_dl, desc='Training epoch {}'.format(epoch), leave=False, total=len(train_dl)):
            # Fetch data
            batch_img, batch_label = batch_data


            # Forward pass
            #print("\n loading data to gpu...\n")
            batch_loss = batch_forward(model, device, criterion, batch_img, batch_label)
            #print("\n end loading data to gpu...\n")
            # Backpropagation
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Statistics
            batch_num = len(batch_label)
            train_num += batch_num
            train_tot_it += batch_num
            train_loss += batch_loss.item() * batch_num

            # Iteration logging
            tb.add_scalar('train/it-loss', batch_loss.item(), train_tot_it)

        print('\nTraining loss epoch {}: {:.4f}'.format(epoch, train_loss / train_num))

        # Validation
        model.eval()
        val_loss = val_num = 0
        for batch_data in tqdm(val_dl, desc='Validating epoch {}'.format(epoch), leave=False, total=len(val_dl)):
            # Fetch data
            batch_img, batch_label = batch_data

            with torch.no_grad():
                # Forward pass
                batch_loss = batch_forward(model, device, criterion, batch_img, batch_label)

            # Statistics
            batch_num = len(batch_label)
            val_num += batch_num
            val_tot_it += batch_num
            val_loss += batch_loss.item() * batch_num

            # Iteration logging
            tb.add_scalar('validation/it-loss', batch_loss.item(), val_tot_it)

        print('\nValidation loss epoch {}: {:.4f}'.format(epoch, val_loss / val_num))

        # Logging
        train_loss /= train_num
        val_loss /= val_num
        tb.add_scalar('train/epoch-loss', train_loss, epoch)
        tb.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        tb.add_scalar('validation/epoch-loss', val_loss, epoch)
        tb.flush()

        # Learning rate scheduling
        lr_scheduler.step(val_loss)

        # save last model
        save_model(model, optimizer, train_loss, val_loss, batch_size, epoch, last_path)
        #torch.save(model.state_dict(), last_path)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            save_model(model, optimizer, train_loss, val_loss, batch_size, epoch, bestval_path)
            #torch.save(model.state_dict(), bestval_path)
            es_counter = 0
        else:
            es_counter += 1

        if optimizer.param_groups[0]['lr'] <= min_lr:
            print('Reached minimum learning rate. Stopping.')
            break
        # check the early stopping: not to be confused with the condition to reduce the learning rate.
        elif es_counter == es_patience:
            print('Early stopping patience reached. Stopping.')
            break

    # Needed to flush out last events
    tb.close()

    print('Training completed! Bye!')


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
