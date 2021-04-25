import torch
import torch.nn as nn
import numpy as np 
import torch.optim as optim
from torch.utils.data import DataLoader
import shutil
from os.path import exists, join, basename, dirname
from torchsummary import summary


from model.cnn_geometric_model import GeometricMatching_model
from model.loss import TransformedGridLoss
from data.synth_dataset import SynthDataset
from geotnf.transformation import SynthPairTnf
from data.normalization import NormalizeImageDict





def save_checkpoint(state, is_best,file):
    model_dir = dirname(file)
    model_fn = basename(file)
    if model_dir != '' and not exists(model_dir):
        makedirs(model_dir)
    torch.save(state,file)
    if is_best:
        shutil.copyfile(file,join(model_dir,'best_'+model_fn))



def train_model(epoch, model, loss_fn, optimizer,
                dataloader, pair_generation_tnf,
                scheduler=False):
    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        tnf_batch = pair_generation_tnf(batch)
        theta = model(tnf_batch)
        loss = loss_fn(theta,tnf_batch['theta_GT'])
        loss.backward()
        optimizer.step()
        if scheduler:
        	scheduler.step()

        train_loss += loss.data.cpu().numpy().item()
    train_loss /= len(dataloader)
    #print('Train set: epoch: ',epoch' avg loss: {:,4f}'.format(train_loss))
    print('Train Epoch: {} \Mean Error: {:.6f}'.format(epoch, train_loss))


    return train_loss



def validate_model(model, loss_fn,
                   dataloader, pair_generation_tnf,
                   epoch):
    model.eval()
    val_loss = 0
    for batch_idx,batch in enumerate(dataloader):
        tnf_batch = pair_generation_tnf(batch)
        theta = model(tnf_batch)
        loss = loss_fn(theta,tnf_batch['theta_GT'])
        val_loss += loss.data.cpu().numpy().item()
    val_loss /= len(dataloader)
    #print('Val set: epoch: ',epoch,' avg loss: {:,4f}'.format(val_loss))
    print('Val Epoch: {} \Mean Error: {:.6f}'.format(epoch, val_loss))
    print('')
    return val_loss






def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')


    geometric_model = 'affine'
    feature_extraction_cnn = 'resnet101'
    matching_type='correlation'

    cnn_output_dim = 6
    num_epochs = 30
    #dataset_path = './data/'
    #dataval_path = './data/'
    dataset_path = '/media/vipl/DATA/Dataset/MSCOCO2017/train2017/'
    dataval_path = '/media/vipl/DATA/Dataset/MSCOCO2017/val2017/'

    print('Creating model...')
    model = GeometricMatching_model(use_cuda=use_cuda,
                         output_dim=cnn_output_dim,
                         feature_extraction_cnn=feature_extraction_cnn,
                         matching_type=matching_type)

    if torch.cuda.device_count() >1:
        model = nn.DataParallel(model)
        print("multi GPU mode")

    print('Creating loss...')
    loss = TransformedGridLoss(use_cuda=use_cuda, geometric_model= geometric_model)

    print('Init dataset')
    dataset = SynthDataset(geometric_model=geometric_model,
                           dataset_image_path=dataset_path,
                           transform = NormalizeImageDict(['image']))
    dataset_val = SynthDataset(geometric_model=geometric_model,
                           dataset_image_path=dataval_path,
                           transform = NormalizeImageDict(['image']))
    #Initialize DataLoader
    dataloader_train = DataLoader(dataset,batch_size=80,shuffle=True, num_workers=4)
    dataloader_val = DataLoader(dataset_val,batch_size=80,shuffle=True, num_workers=4)

    #Set Tnf pair generation func
    pair_generation_tnf = SynthPairTnf(geometric_model=geometric_model, use_cuda=use_cuda)

    

    #optimizer and scheduler
    optimizer = optim.Adam(model.module.FeatureRegression.parameters(),lr=0.001)
    #optimizer = optim.Adam(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
    best_val_loss = float("inf")

    #Checkpoint
    #Tensor Board

    #setup checkpoint
    checkpoint_path = 'GeometricMatching_model.pth'

    for epoch in range(1,num_epochs+1):
        _ = train_model(epoch, model, loss, optimizer,
                        dataloader_train, pair_generation_tnf,
                        scheduler=scheduler)

        val_loss = validate_model(model, loss,
                                  dataloader_val, pair_generation_tnf,
                                  epoch)

        #remember best loss
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss,best_val_loss)
        save_checkpoint({ 'epoch': epoch + 1,
                          'state_dict': model.module.state_dict(),
                          'best_val_loss': val_loss,
                          'optimizer': optimizer.state_dict()}, is_best,checkpoint_path)




if __name__=='__main__':
    main()
