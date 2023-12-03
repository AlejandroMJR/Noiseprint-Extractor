import torch
import os
import pandas as pd
from torchvision.io import read_image
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader, BatchSampler
import numpy as np
import torch as tc
import torchvision.transforms as transforms
import time
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
random.seed(123)
torch.manual_seed(123)

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


class CustomBatchSampler(BatchSampler):
    def __init__(self, img_labels, batch_size, img_frac, num_classes, img_per_class):
        #super(CustomBatchSampler, self).__init__()
        self.classes_idx = [[] for _ in range(len(img_labels.model.unique()))]
        for i in range(len(img_labels)):
            for j, klass in enumerate(img_labels.model.unique()):
                if img_labels["model"].values[i] == klass:
                    self.classes_idx[j].append(img_labels["id"].values[i])

        self.total_num_classes = len(self.classes_idx)
        self.num_samples_per_class = [len(self.classes_idx[i]) for i in range(len(self.classes_idx))]
        self.batch_size = batch_size
        self.img_frac = img_frac
        self.num_classes = num_classes
        self.img_per_class = img_per_class
        self.img_per_batch = num_classes * img_per_class * batch_size
        self.num_batch_per_epoch = len(img_labels) // self.img_per_batch

    def __iter__(self):
        for _ in range(len(self)):
            # randomly select 25 classes
            classes = random.sample(range(self.total_num_classes), self.num_classes)
            # for each class, randomly select 4 samples
            batch = []
            for c in classes:
                samples = random.sample(range(self.num_samples_per_class[c]), self.img_per_class*self.batch_size)
                batch += [self.classes_idx[c][s] for s in samples]
            #print(len(batch))
            yield batch

    def __len__(self):
        return self.num_batch_per_epoch//self.img_frac


class MyDataset(Dataset):
    def __init__(self,  img_labels, patch_size, num_channels, num_classes, img_per_class):
        #self.root_dir = root_dir
        self.num_classes = num_classes
        self.img_per_class = img_per_class
        #self.img_labels = pd.read_csv(os.path.join(root_dir, "VisionNaturalDataset.csv"), sep="|")
        self.img_labels = img_labels
        self.classes_idx = [[] for _ in range(len(self.img_labels.model.unique()))]
        for i in range(len(self.img_labels)):
            for j, klass in enumerate(self.img_labels.model.unique()):
                if self.img_labels["model"].values[i] == klass:
                    self.classes_idx[j].append(self.img_labels["id"].values[i])
        #self.num_classes = len(self.classes_idx)
        self.num_samples_per_class = [len(self.classes_idx[i]) for i in range(len(self.classes_idx))]
        self.patch_size = patch_size
        self.num_channels = num_channels

    def __getitem__(self, index):
        #print("getting batch samples... \n")
        #print("num_images:", len(index))
        batch_size = len(index)//(self.img_per_class*self.num_classes)
        start = time.process_time()
        indeces = np.array(index)
        indeces = indeces.reshape(-1, self.img_per_class)
        #print(index)
        # get the indices of the samples in the current batch
        batch_images = []
        for group in range(len(indeces)):
            #print("check", group)
            imgs = self.load_image_group(indeces[group])

            w, h = imgs[0].size
            for im in range(1, len(imgs)):
                w_dumb, h_dumb = imgs[im].size
                if h_dumb < h:
                    w = h_dumb
                if w_dumb < w:
                    h = w_dumb

            x1 = random.sample(range(w - self.patch_size[0]), 2)
            y1 = random.sample(range(h - self.patch_size[1]), 2)
            patches1a = self.extract_patch(imgs, x1[0], y1[0])
            for i in range(len(patches1a)):
                batch_images.append(patches1a[i])
            patches1b = self.extract_patch(imgs, x1[1], y1[1])
            for i in range(len(patches1b)):
                batch_images.append(patches1b[i])
        batch_labels = []
        for i in range(self.num_classes*2*batch_size):
            row = np.zeros(2*self.img_per_class*self.num_classes*batch_size)
            for j in range(self.img_per_class):
                row[self.img_per_class*i+j] = 1
            for k in range(self.img_per_class):
                batch_labels.append(row)
        #print(batch_labels)
        #batch_images = np.asarray(batch_images)
        batch_labels = np.asarray(batch_labels)
        #print(batch_images[10][1,10,10])
        #print(time.process_time() - start)
        return batch_images, batch_labels

    def __len__(self):
        # return the total number of batches
        return len(self.img_labels)

    def load_image_group(self, idxs):
        # load the image from the dataset given index
        images = []
        for i in idxs:
            #img_path = os.path.join(self.root_dir, self.img_labels["probe"].values[i])
            #img_path = os.path.join(self.root_dir, self.img_labels.loc[self.img_labels["Unnamed: 0"] == i, "probe"].values[0])
            img_path = os.path.join(self.img_labels.loc[self.img_labels["id"] == i, "probe"].values[0])
            if self.num_channels == 1:
                im = Image.open(img_path).convert("L")
                #print("check1")
            if self.num_channels == 3:
                im = Image.open(img_path).convert("RGB")
                #print("check3")
            #tr = transforms.ToTensor()
            #im = tr(im)
            #im = im.to(torch.float)
            #if self.transform:
            #    im = self.transform(im)
            images.append(im)
        #print("image", images[0].size())
        return images

    def extract_patch(self, imgs, x, y):
        # extract a patch from the image
        #channels, height, width = img.size()
        #x = random.sample(range(width - self.patch_size[2]), 2)
        #y = random.sample(range(height - self.patch_size[1]), 2)
        #print(img.size())
        patches = []
        #count = 0
        for i in imgs:
            #patch = i[:, y:y + self.patch_size[2], x:x + self.patch_size[1]]
            patch = i.crop((x, y, x + self.patch_size[0], y + self.patch_size[1]))
            #patch = patch.to(torch.float)
            #print(i.size())
            # apply the transformation if provided
            #if self.transform:
            #    patch = self.transform(patch)
            if patch.mode == "L":
                #print("check")
                patch = np.expand_dims(patch, axis=2)
            patch = torch.from_numpy(np.array(patch)).permute(2, 0, 1).float().div(255.)
            patches.append(patch)
            #print(patches[count].size())
            #count += 1
        #patch2 = img[:, y[1]:y[1] + self.patch_size[1], x[1]:x[1] + self.patch_size[2]]
        #print("patch", patches[0][0, 30, 30])
        return patches

if __name__ == '__main__':

    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("val.csv")

    n = 25
    ni = 4
    ps = 48
    num_channels = 1
    workers = 4
    # load sampler
    train_sampler = CustomBatchSampler(train_df, 1, 1, n, ni)
    val_sampler = CustomBatchSampler(val_df, 1, 1, n, ni)


    train_ds = MyDataset(img_labels=train_df, patch_size=(ps, ps), num_channels=num_channels, num_classes=n,
                         img_per_class=ni)
    val_ds = MyDataset(img_labels=val_df, patch_size=(ps, ps), num_channels=num_channels, num_classes=n,
                       img_per_class=ni)
    train_dl = DataLoader(dataset=train_ds, batch_size=None, num_workers=workers, shuffle=False, sampler=train_sampler,
                          collate_fn=collate_fn)
    val_dl = DataLoader(dataset=val_ds, batch_size=None, num_workers=workers, shuffle=False, sampler=val_sampler,
                        collate_fn=collate_fn)

    for batch_data in tqdm(train_dl, desc='Training epoch {}'.format(0), leave=False, total=len(train_dl)):
        # Fetch data
        batch_img, batch_label = batch_data
        print("end")


