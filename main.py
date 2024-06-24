import torchvision
import torch
import tqdm
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import neptune.new as neptune
import argparse
import sys
import configparser
import matplotlib.pyplot as plt

import Losses.Cross_Entropy
import Losses.Supervised_Contrastive_Loss
import Losses.Temperature_Normalized_Contrastive_Loss
import Losses.Contrastive_Loss
import Models.Classifier
import Models.Encoder

import Losses


def load_and_transform_data():
    # Load Data
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    return train_loader, test_loader
    
    
def init_encoder(device):
    
    if params['loss'] == 'Supervised_Contrastive_Loss':
        if params['learnable_t'] == 'yes':
            loss_function_for_encoder = Losses.Supervised_Contrastive_Loss.Supervised_Contrastive_Loss(True, params['temperature'])
        else:
            loss_function_for_encoder = Losses.Supervised_Contrastive_Loss.Supervised_Contrastive_Loss(False, params['temperature'])
    
    if params['loss'] == 'Temperature_Normalized_Contrastive_Loss':
        loss_function_for_encoder = Losses.Temperature_Normalized_Contrastive_Loss.Temperature_Normalized_Contrastive_Loss(params['temperature'])
        
    if params['loss'] == 'Contrastive_Loss':
        loss_function_for_encoder = Losses.Contrastive_Loss.Contrastive_Loss()
        
    if params['loss'] == 'Cross_Entropy':
        loss_function_for_encoder = Losses.Cross_Entropy.Cross_Entropy()
        
    encoder_F = Models.Encoder.Encoder(params['loss'], params['visual']).to(device)
    
    if params['loss'] == 'Supervised_Contrastive_Loss':
        if params['learnable_t'] == 'yes':
            optimizer_for_encoder = torch.optim.Adam(list(encoder_F.parameters()) + list(loss_function_for_encoder.parameters()), lr=params['lr'])
        else:
            optimizer_for_encoder = torch.optim.Adam(list(encoder_F.parameters()), lr=params['lr'])
    else:
        optimizer_for_encoder = torch.optim.Adam(list(encoder_F.parameters()), lr=params['lr'])
        
    scheduler_of_encoder = torch.optim.lr_scheduler.StepLR(optimizer_for_encoder,
                                                step_size=params['step_size'],
                                                gamma=params['gamma'])
    
    return loss_function_for_encoder, encoder_F, optimizer_for_encoder, scheduler_of_encoder


def init_classifier(device):
    
    loss_function_for_classifier = Losses.Cross_Entropy.Cross_Entropy()
    
    classifier_G = Models.Classifier.Classifier('cross_entropy').to(device)
    optimizer_for_classifier = torch.optim.Adam(classifier_G.parameters(), lr=params['lr'])
    scheduler_for_classifier = torch.optim.lr_scheduler.StepLR(optimizer_for_classifier,
                                                step_size=params['step_size'],
                                                gamma=params['gamma'])
    
    return loss_function_for_classifier, classifier_G, optimizer_for_classifier, scheduler_for_classifier


def train_and_test_encoder(device, 
                  train_loader,
                  test_loader, 
                  loss_function_for_encoder,
                  encoder_F, 
                  optimizer_for_encoder, 
                  scheduler_of_encoder):
    # Train
    for epoch in tqdm.tqdm(range(1, 1 + params['epochs_encoder'])):
        encoder_F.train()
        
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer_for_encoder.zero_grad()
            output = encoder_F(data)
            loss = loss_function_for_encoder(output, target)
            loss.backward()

            # if torch.isnan(encoder_F.conv1.weight.grad).any():
            #     print('nan in conv1')
                
            optimizer_for_encoder.step()
            epoch_loss += loss.item()
            

        run['train/lr'].log(optimizer_for_encoder.param_groups[0]['lr'])
        run['train/epoch_loss'].log(epoch_loss / len(train_loader))
        if params['loss'] == 'Supervised_Contrastive_Loss':
            if params['learnable_t'] == 'yes':
                run['train/tau'].log(loss_function_for_encoder.temperature.item())
        scheduler_of_encoder.step()
        
        # Test
        encoder_F.eval()
        with torch.no_grad():
            epoch_loss = 0.0
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = encoder_F(data)
                loss = loss_function_for_encoder(output, target)
                epoch_loss += loss.item()
            run['test/epoch_loss'].log(epoch_loss / len(test_loader))
            
        output_knn = output.detach().cpu().numpy()
        target_knn = target.detach().cpu().numpy()
        
        for num_neighbors in [1, 10, 100, 1000]:
            knn = KNeighborsClassifier(n_neighbors=num_neighbors)
            run[f'test/knn_{num_neighbors}'].log(sklearn.model_selection.cross_val_score(knn, output_knn, target_knn, cv=5).mean())
        
        if params['visual'] == 'yes':   
            visualize_encoded_data(output_knn, target_knn)
        
    return encoder_F


def visualize_encoded_data(output_knn, target_knn):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    fig = plt.figure(figsize=(16, 10))
    plt.scatter(output_knn[:, 0], output_knn[:, 1], c=target_knn.tolist(), cmap='tab10')
    for i in range(10):
        plt.scatter(output_knn[target_knn==i,0].mean(), output_knn[target_knn==i,1].mean(), c=colors[i], marker="D", s=160)
    run['test/visualization'].append(fig)
    plt.close(fig)


def train_and_test_classifier(device, 
                  train_loader,
                  test_loader, 
                  loss_function_for_classifier,
                  classifier_G,
                  encoder_F, 
                  optimizer_for_classifier, 
                  scheduler_for_classifier):
    
    for epoch in tqdm.tqdm(range(1, 1 + params['epochs_classifier'])): 
        classifier_G.train()
        
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer_for_classifier.zero_grad()
            output = classifier_G(encoder_F(data))
            loss = loss_function_for_classifier(output, target)
            loss.backward()
                
            optimizer_for_classifier.step()
            epoch_loss += loss.item()
        
        run['train/cross_entropy_loss'].log(epoch_loss / len(train_loader))
            
         # Test
        classifier_G.eval()
        correct = 0
        with torch.no_grad():
            epoch_loss = 0.0
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = classifier_G(encoder_F(data))
                loss = loss_function_for_classifier(output, target)
                epoch_loss += loss.item()
                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum().item()
            run['test/cross_entropy_loss'].log(epoch_loss)
            run['test/cross_entropy_score'].log(correct / 10000)  


def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader, test_loader = load_and_transform_data()
    
    loss_function_for_encoder, encoder_F, optimizer_for_encoder, scheduler_of_encoder = init_encoder(device)
    
    loss_function_for_classifier, classifier_G, optimizer_for_classifier, scheduler_for_classifier = init_classifier(device)

    encoder_F = train_and_test_encoder(device, 
                              train_loader, 
                              test_loader,
                              loss_function_for_encoder, 
                              encoder_F, 
                              optimizer_for_encoder, 
                              scheduler_of_encoder)
    
    if params['visual'] != 'yes':
        classifier_G = train_and_test_classifier(device, 
                    train_loader,
                    test_loader, 
                    loss_function_for_classifier,
                    classifier_G,
                    encoder_F, 
                    optimizer_for_classifier, 
                    scheduler_for_classifier)
    
    
    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train a model on FashionMNIST.')
        
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--step_size', type=int, default=100, help='Step size for learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=1.0, help='Gamma for learning rate scheduler')
    parser.add_argument('--epochs_encoder', type=int, help='Number of epochs for encoder')
    parser.add_argument('--epochs_classifier', type=int, help='Number of epochs for classifier')
    parser.add_argument('--loss', type=str, help='Loss function')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--mode', type=str, default='debug', help='Mode: debug or async')
    parser.add_argument('--visual', type=str, default='no', help='Visualize encoded data in R2')
    parser.add_argument('--learnable_t', type=str, default='no', help='Define if temperature is a hyperparameter or not')
    parser.add_argument('--temperature', type=float, default=0.1, help='Initial value of temperature parameter')
    args = parser.parse_args()
    
    params = vars(args)
    
    print(params)
    
    cfg = configparser.ConfigParser()
    cfg.read('neptune_config.cfg')
    project_name = cfg.get('PROJECT', 'neptune_project', raw='')
    project_token = cfg.get('TOKEN', 'neptune_token', raw='')

    run = neptune.init_run(project=project_name,
                            api_token=project_token,
                            mode=params['mode'])
    
    run['parameters'] = params
    run['command'] = 'python ' + ' '.join(sys.argv)

    main()
    run.stop()