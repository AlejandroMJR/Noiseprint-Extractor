import torch
import numpy as np
from utilityRead import imread2f
import math
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from albumentations import ImageCompression, Compose

device = 'cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu'
device

jpeg = Compose([ImageCompression(quality_lower=90, quality_upper=90, always_apply=True, p=1)])


def genNoiseprint(img, modelname, returnnumpy):
    if modelname == "DnCNN":
        from model import DnCNN
        #img = Image.open(img).convert("L")
        #img = np.array(img) / 255.

        model = DnCNN()
        model.to(device)
        # model.load_state_dict(torch.load("model_zoo/restormer.pth", map_location=lambda storage, loc: storage.cuda(0))["net"])
        inc = model.load_state_dict(
        torch.load("model_zoo/bestval_DnCNN.pth", map_location=lambda storage, loc: storage.cuda(0))["net"])
        model.eval()
        print(inc)

        #img = img.convert("L")
        image = np.expand_dims(img, axis=2)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        image = image.to(device)
        input_tensor = image
        with torch.no_grad():
            output = model(input_tensor.unsqueeze(0))

    if modelname == "Uformer":
        from model import Uformer

        #img = Image.open(img_path).convert("RGB")
        #img = np.array(img) / 255.

        ps = 128
        model = Uformer(img_size=ps, embed_dim=32, win_size=8, token_projection='linear', token_mlp='leff',
                        depths=[1, 2, 8, 8, 2, 8, 8, 2, 1], modulator=True, dd_in=3)
        model.to(device)
        state = torch.load("model_zoo/bestvalUformer.pth", map_location=lambda storage, loc: storage.cuda(0))
        state = state['net']
        net_state = {}
        for key, value in state.items():
            if key.startswith('module.'):
                new_key = key[7:]  # Remove the "module." prefix
                net_state[new_key] = value
            else:
                net_state[key] = value
        inc = model.load_state_dict(net_state, strict=False)
        model.eval()
        print(inc)

        input_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
        input_tensor = input_tensor.to(device)
        # input_tensor = preprocess_image(image)
        input_tensor = input_tensor.unsqueeze(0)

        _, _, h, w = input_tensor.shape


        # Set patch size and overlap
        patch_size = int(ps * 2)
        overlap = int(patch_size / 2)
        input_tensor, mask = expand2square(input_tensor, factor=patch_size)

        # Calculate number of patches in each dimension
        _, _, width, height = input_tensor.shape
        num_patches_h = (height - overlap) // (patch_size - overlap)
        num_patches_w = (width - overlap) // (patch_size - overlap)

        # Initialize empty array to store predictions
        predictions = torch.zeros((3, height, width))
        Overmask = torch.zeros((3, height, width))
        predictions = predictions.to(device)
        Overmask = Overmask.to(device)
        # Iterate over patches
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                # Calculate patch coordinates
                start_h = i * (patch_size - overlap)
                end_h = start_h + patch_size
                start_w = j * (patch_size - overlap)
                end_w = start_w + patch_size

                # Extract patch
                patch = input_tensor[:, :, start_h:end_h, start_w:end_w]

                #print(patch.shape)
                with torch.no_grad():
                    output = model(patch)

                # Merge patch predictions into the final image
                predictions[:, start_h:end_h, start_w:end_w] += output.squeeze()
                Overmask[:, start_h:end_h, start_w:end_w] += 1

        # Average the accumulated predictions by the mask
        predictions /= Overmask
        predictions = torch.masked_select(predictions, mask.bool()).reshape(3, h, w)
        #transform = transforms.Grayscale()
        #predGray = transform(predictions)
        #predGray = predGray.permute(1,2,0)
        predictions = predictions[1, :, :]
        output = predictions.unsqueeze(0)
        #output = predGray.unsqueeze(0)

    if modelname == "Restormer":
        print("generating noiseprint\n")
        from model import Restormer

        #img = Image.open(img_path).convert("L")
        #img = np.array(img) / 255.

        ps = 128
        model = Restormer(inp_channels=1, out_channels=1)
        model.to(device)
        # model.load_state_dict(torch.load("model_zoo/restormer.pth", map_location=lambda storage, loc: storage.cuda(0))["net"])
        state = torch.load("model_zoo/bestvalRestormer.pth", map_location=lambda storage, loc: storage.cuda(0))
        state = state['net']
        net_state = {}
        for key, value in state.items():
            if key.startswith('module.'):
                new_key = key[7:]  # Remove the "module." prefix
                net_state[new_key] = value
            else:
                net_state[key] = value
        imp = model.load_state_dict(net_state, strict=False)
        print(imp)
        model.eval()

        input_tensor = np.expand_dims(img, axis=2)
        input_tensor = torch.from_numpy(np.array(input_tensor)).permute(2, 0, 1).float()
        input_tensor = input_tensor.to(device)
        # input_tensor = preprocess_image(image)
        input_tensor = input_tensor.unsqueeze(0)

        _, _, h, w = input_tensor.shape


        # Set patch size and overlap
        patch_size = int(ps * 4)
        overlap = int(patch_size / 2)
        input_tensor, mask = expand2square1(input_tensor, factor=patch_size)

        # Calculate number of patches in each dimension
        _, _, width, height = input_tensor.shape
        num_patches_h = (height - overlap) // (patch_size - overlap)
        num_patches_w = (width - overlap) // (patch_size - overlap)

        # Initialize empty array to store predictions
        predictions = torch.zeros((1, height, width))
        Overmask = torch.zeros((1, height, width))
        predictions = predictions.to(device)
        Overmask = Overmask.to(device)
        # Iterate over patches
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                # Calculate patch coordinates
                start_h = i * (patch_size - overlap)
                end_h = start_h + patch_size
                start_w = j * (patch_size - overlap)
                end_w = start_w + patch_size

                # Extract patch
                patch = input_tensor[:, :, start_h:end_h, start_w:end_w]

                #print(patch.shape)
                with torch.no_grad():
                    output = model(patch)

                # Merge patch predictions into the final image
                predictions[:, start_h:end_h, start_w:end_w] += output.squeeze()
                Overmask[:, start_h:end_h, start_w:end_w] += 1

        # Average the accumulated predictions by the mask
        predictions /= Overmask
        predictions = torch.masked_select(predictions, mask.bool()).reshape(1, h, w)
        output = predictions.unsqueeze(0)
    print("noiseprint generated\n")
    if returnnumpy:
        return output.squeeze().detach().cpu().numpy()
    else:
        return output


def genNoiseprintFromFile(img_path, modelname, returnnumpy):
    if modelname == "DnCNN":
        from model import DnCNN
        img = Image.open(img_path).convert("L")
        img = np.array(img)
        img = jpeg(image=img)["image"]
        img = img / 255.
        model = DnCNN()
        model.to(device)
        inc = model.load_state_dict(
        #torch.load("weigths_new/net-DnCNN_lr-0.0001_batch_size-1_time-2023-08-03 18:02:43.397957/bestval.pth", map_location=lambda storage, loc: storage.cuda(0))["net"])
        #torch.load("/nas/home/ajaramillo/projects/Try/weigths_new/net-DnCNN_lr-0.0001_batch_size-1_time-2023-08-03 18:02:43.397957/bestval.pth", map_location=lambda storage, loc: storage.cuda(0))["net"])
        torch.load("model_zoo/bestval_DnCNN.pth", map_location=lambda storage, loc: storage.cuda(0))["net"])
        model.eval()
        print(inc)

        #img = img.convert("L")
        image = np.expand_dims(img, axis=2)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        image = image.to(device)
        input_tensor = image
        with torch.no_grad():
            output = model(input_tensor.unsqueeze(0))

    if modelname == "Uformer":
        from model import Uformer

        img = Image.open(img_path).convert("RGB")
        img = np.array(img) / 255.

        ps = 128
        model = Uformer(img_size=ps, embed_dim=32, win_size=8, token_projection='linear', token_mlp='leff',
                        depths=[1, 2, 8, 8, 2, 8, 8, 2, 1], modulator=True, dd_in=3)
        model.to(device)
        state = torch.load("model_zoo/bestvalUformer.pth", map_location=lambda storage, loc: storage.cuda(0))
        #state = torch.load("weigths_new/net-Uformer_lr-0.0001_batch_size-1_time-2023-08-05 16:07:14.728888/bestval.pth", map_location=lambda storage, loc: storage.cuda(0))
        state = state['net']
        net_state = {}
        for key, value in state.items():
            if key.startswith('module.'):
                new_key = key[7:]  # Remove the "module." prefix
                net_state[new_key] = value
            else:
                net_state[key] = value
        inc = model.load_state_dict(net_state, strict=False)
        model.eval()
        print(inc)

        input_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
        input_tensor = input_tensor.to(device)
        # input_tensor = preprocess_image(image)
        input_tensor = input_tensor.unsqueeze(0)

        _, _, h, w = input_tensor.shape


        # Set patch size and overlap
        patch_size = int(ps * 2)
        overlap = int(patch_size / 2)
        input_tensor, mask = expand2square(input_tensor, factor=patch_size)

        # Calculate number of patches in each dimension
        _, _, width, height = input_tensor.shape
        num_patches_h = (height - overlap) // (patch_size - overlap)
        num_patches_w = (width - overlap) // (patch_size - overlap)

        # Initialize empty array to store predictions
        predictions = torch.zeros((3, height, width))
        Overmask = torch.zeros((3, height, width))
        predictions = predictions.to(device)
        Overmask = Overmask.to(device)
        # Iterate over patches
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                # Calculate patch coordinates
                start_h = i * (patch_size - overlap)
                end_h = start_h + patch_size
                start_w = j * (patch_size - overlap)
                end_w = start_w + patch_size

                # Extract patch
                patch = input_tensor[:, :, start_h:end_h, start_w:end_w]

                #print(patch.shape)
                with torch.no_grad():
                    output = model(patch)

                # Merge patch predictions into the final image
                predictions[:, start_h:end_h, start_w:end_w] += output.squeeze()
                Overmask[:, start_h:end_h, start_w:end_w] += 1

        # Average the accumulated predictions by the mask
        predictions /= Overmask
        predictions = torch.masked_select(predictions, mask.bool()).reshape(3, h, w)
        #transform = transforms.Grayscale()
        #predGray = transform(predictions)
        #predGray = predGray.permute(1,2,0)
        predictions = predictions[1, :, :]
        output = predictions.unsqueeze(0)
        #output = predGray.unsqueeze(0)

    if modelname == "Restormer":
        print("generating noiseprint\n")
        from model import Restormer

        img = Image.open(img_path).convert("L")
        w, h = img.size
        img = img.crop((5, 8, w, h))
        img = np.array(img) / 255.

        ps = 48
        model = Restormer(inp_channels=1, out_channels=1)
        model.to(device)
        # model.load_state_dict(torch.load("model_zoo/restormer.pth", map_location=lambda storage, loc: storage.cuda(0))["net"])
        state = torch.load("model_zoo/bestvalRestormer.pth", map_location=lambda storage, loc: storage.cuda(0))
        state = state['net']
        net_state = {}
        for key, value in state.items():
            if key.startswith('module.'):
                new_key = key[7:]  # Remove the "module." prefix
                net_state[new_key] = value
            else:
                net_state[key] = value
        imp = model.load_state_dict(net_state, strict=False)
        print(imp)
        model.eval()

        input_tensor = np.expand_dims(img, axis=2)
        input_tensor = torch.from_numpy(np.array(input_tensor)).permute(2, 0, 1).float()
        input_tensor = input_tensor.to(device)
        # input_tensor = preprocess_image(image)
        input_tensor = input_tensor.unsqueeze(0)

        _, _, h, w = input_tensor.shape


        # Set patch size and overlap
        patch_size = int(ps * 4)
        overlap = int(patch_size / 2)
        input_tensor, mask = expand2square1(input_tensor, factor=patch_size)

        # Calculate number of patches in each dimension
        _, _, width, height = input_tensor.shape
        num_patches_h = (height - overlap) // (patch_size - overlap)
        num_patches_w = (width - overlap) // (patch_size - overlap)

        # Initialize empty array to store predictions
        predictions = torch.zeros((1, height, width))
        Overmask = torch.zeros((1, height, width))
        predictions = predictions.to(device)
        Overmask = Overmask.to(device)
        # Iterate over patches
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                # Calculate patch coordinates
                start_h = i * (patch_size - overlap)
                end_h = start_h + patch_size
                start_w = j * (patch_size - overlap)
                end_w = start_w + patch_size

                # Extract patch
                patch = input_tensor[:, :, start_h:end_h, start_w:end_w]

                #print(patch.shape)
                with torch.no_grad():
                    output = model(patch)

                # Merge patch predictions into the final image
                predictions[:, start_h:end_h, start_w:end_w] += output.squeeze()
                Overmask[:, start_h:end_h, start_w:end_w] += 1

        # Average the accumulated predictions by the mask
        predictions /= Overmask
        predictions = torch.masked_select(predictions, mask.bool()).reshape(1, h, w)
        output = predictions.unsqueeze(0)
    print("noiseprint generated\n")
    if returnnumpy:
        return output.squeeze().detach().cpu().numpy()
    else:
        return output


def expand2square(timg, factor=16.0):
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h, w) / float(factor)) * factor)

    img = torch.zeros(1, 3, X, X).type_as(timg)  # 3, h,w
    mask = torch.zeros(1, 1, X, X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)] = timg
    mask[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)].fill_(1)

    return img, mask

def expand2square1(timg, factor=16.0):
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h, w) / float(factor)) * factor)

    img = torch.zeros(1, 1, X, X).type_as(timg)  # 3, h,w
    mask = torch.zeros(1, 1, X, X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)] = timg
    mask[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)].fill_(1)

    return img, mask

def postprocess_predictions(predictions):
    postprocess = transforms.Compose([
        transforms.ToPILImage()  # Convert tensor to PIL image
    ])
    output_image = postprocess(predictions)
    return output_image


if __name__ == '__main__':
    imgfilename = "/nas/home/ajaramillo/projects/datasets/dso-dsi/DSO-1/splicing-04.png"
    #imgfilename = "/nas/home/ajaramillo/projects/datasets/dalle/fake/013-2.png"
    #img, mode = imread2f(imgfilename, channel=1)
    res = genNoiseprintFromFile(imgfilename, "DnCNN", False)
    #plt.imshow(res), plt.colorbar(), plt.clim(2.9,3.1), plt.show()
    #plt.imshow(res), plt.colorbar(), plt.clim(-2.56, -2.5), plt.show()
    #plt.imshow(res), plt.colorbar(), plt.clim(-0.005,0.01), plt.show()
    output_image = postprocess_predictions(res.squeeze().cpu())
    output_image.save('/nas/home/ajaramillo/projects/datasets/dalle/noiseprints/dso-04-jpeg95-Noiseprint-D.png')
    #np.save("testR", res)
    print("check")

