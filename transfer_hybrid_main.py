import torch
from torch import nn
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import time
from pathlib import Path
from openvino.runtime import Core
import os
import sys

def train_valid(_device, _model, _dataset, _update, _freezing=1, _valid_size=0.2, _epoch=5):
    print("Exp: ", _device, _model, _dataset, _valid_size, _epoch)

    if _device == 'ipex':
        import intel_extension_for_pytorch as ipex
        #ipex.enable_onednn_fusion(True)  #oneDNN graph fusion is enabled by default in ipex
        device = torch.device('cpu')
    elif _device == 'gpu':
        print(torch.cuda.is_available())
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    LR = 0.001
    DOWNLOAD = True
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if _dataset == 'cifar10':
        DATA = 'datasets/cifar10/'

        train_dataset = torchvision.datasets.CIFAR10(
                root=DATA,
                train=True,
                transform=transform,
                download=DOWNLOAD,
        )

        valid_dataset = torchvision.datasets.CIFAR10(
                root=DATA,
                train=True,
                transform=transform,
                download=DOWNLOAD,
        )

    elif _dataset == 'svhn':
        DATA = 'datasets/SVHN/'

        train_dataset = torchvision.datasets.SVHN(
                root=DATA,
                split='train',
                transform=transform,
                download=DOWNLOAD,
        )
        valid_dataset = torchvision.datasets.SVHN(
                root=DATA,
                split='train',
                transform=transform,
                download=DOWNLOAD,
        )
    
    if _model == 'res_18':
        model = torchvision.models.resnet18(pretrained=True)
    elif _model == 'res_50':
        model = torchvision.models.resnet50(pretrained=True)
    elif _model == 'mob_v2':
        model = torchvision.models.mobilenet_v2(pretrained=True)
    elif _model == 'mob_v3':
        model = torchvision.models.mobilenet_v3_large(pretrained=True)
    elif _model == 'shuffle_v2':
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)

    # model freezing
    if _freezing == 0:
        pass
    else:
        try: #resnet50 case
            layers = []
            ct = 0
            child_len = len(list(model.children()))
            freeze_num = child_len - _freezing
            for child in model.children():
                ct += 1
                if ct <= freeze_num:
                    for param in child.parameters():
                        param.requires_grad = False
                    layers.append(child)
            layers.append(nn.Flatten(1))
            frozen_net = nn.Sequential(*layers)
            num_ftrs = model.fc.in_features
            model = nn.Linear(num_ftrs, 10)

        except: #mobv2 case
            layers = []
            ct = 0
            child_list = list(model.children())
            child_len = len(child_list)
            freeze_num = child_len - _freezing
            for child in model.children():
                ct += 1
                if ct <= freeze_num:
                    for param in child.parameters():
                        param.requires_grad = False
                    layers.append(child)
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            layers.append(nn.Flatten(1))
            frozen_net = nn.Sequential(*layers)
            
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, 10)
            model = model.classifier   
    
    frozen_net.eval()
    # make dummy data
    batch_size = 128
    x = torch.rand(batch_size, 3, 224, 224, requires_grad=False).to(device)
    
    # onnx convert
    onnx_path = Path("mbv2_cifar10_16.onnx")
    if not onnx_path.exists() or _update==True:
        torch.onnx.export(frozen_net, x, onnx_path, export_params=True,do_constant_folding=False,
                        opset_version=11,input_names=['input'], output_names=['output'],)  
        print(f"ONNX model exported to {onnx_path}.")      
    else:
        print(f"ONNX model {onnx_path} already exists.")       
    
    ir_path = Path("mbv2_cifar10_16.xml")
    if not ir_path.exists() or _update==True:
        mo_command = f"""mo
                 --input_model "{onnx_path}"
                 --input_shape "[128,3, 224, 224]"
                 --data_type FP16
                 --output_dir "./"
                 """
        mo_command = " ".join(mo_command.split())
        os.system(mo_command)
        print("Exporting ONNX model to IR... This may take a few minutes.")
    else:
        print(f"IR model {ir_path} already exists.")
    
    ie = Core()
    model_ir = ie.read_model(model=ir_path)
    compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU")
    request_ir = compiled_model_ir.create_infer_request()
    input_layer_ir = next(iter(compiled_model_ir.inputs))
    output_layer_ir = next(iter(compiled_model_ir.outputs))

    #setting dataloader
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(_valid_size * num_train))

    random_seed = 3884
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            sampler=train_sampler,
            batch_size=128,
            drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset,
            sampler=valid_sampler,
            batch_size=128,
            drop_last=True
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = LR, momentum=0.9)

    model.to(device)
    model.train()
    
    if _device == 'ipex':
        model, optimizer = ipex.optimize(model, optimizer=optimizer)

    #train and valid
    valid_correct_history = []
    for epoch in range(_epoch):
        valid_correct = 0
        valid_total = 0

        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # data = data.to(device=device, memory_format=torch.channels_last)
            data = data.to(device=device)
            target = target.to(device=device)

            optimizer.zero_grad()
            res_ir = torch.from_numpy(np.squeeze(np.array(list(request_ir.infer(inputs=[data]).values())))).to(device)
            output = model(res_ir) #liner layer training
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            print("Train", _device, _model, _dataset ,batch_idx, epoch)

        model.eval()
        with torch.no_grad():
            for valid_batch_idx, (valid_data, valid_target) in enumerate(valid_loader):
                # valid_data = valid_data.to(device=device, memory_format=torch.channels_last)
                valid_data = valid_data.to(device=device)
                valid_target = valid_target.to(device=device)
                res_ir = torch.from_numpy(np.squeeze(np.array(list(request_ir.infer(inputs=[valid_data]).values())))).to(device)
                valid_output = model(res_ir)

                _, valid_preds = torch.max(valid_output, 1)
                valid_correct += torch.sum(valid_preds == valid_target.data)
                valid_total += len(valid_data)
                print("Val", _device, _model, _dataset ,valid_batch_idx, epoch)
                
        val_epoch_acc = valid_correct.float() / valid_total
        valid_correct_history.append(val_epoch_acc)

    return valid_correct_history

device = 'cpu'
model = 'mob_v2'
dataset = 'cifar10'
freeze_num = 1
epochs = 2

torch.cuda.empty_cache()
start_time = time.time()
result = train_valid(device,model,dataset,_update=True,_freezing=freeze_num,_epoch=epochs)
end_time = time.time()
torch.cuda.empty_cache()
print(result)
print(end_time - start_time)