import torch
from torch import nn
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import ConcatDataset
from model import LSTM
from dataset import MusicDataset, split_train_validation
from preprocess import PreProcessor
import matplotlib.pyplot as plt
from melody_generation import MelodyGenerator


def kfold_hyperparam_experiment(exp_dir, k_folds=5, epoches=10, **hyperparams):
    # pre processing the dataset
    dataset = hyperparams.get('dataset') if hyperparams.get('dataset') else 'dataset/deutschl/test'
    seq_len = hyperparams.get('seq_len') if hyperparams.get('seq_len') else 64
    pre = PreProcessor(dataset, exp_dir, seq_len=seq_len)
    songs = pre.process()

    # Configuration options
    loss_function = hyperparams.get('loss') if hyperparams.get('loss') else nn.CrossEntropyLoss()

    # For fold results
    results = []

    # Set fixed random number seed
    torch.manual_seed(42)

    kfold = KFold(n_splits=k_folds, shuffle=True)

    train_ids, validation_ids = split_train_validation(np.arange(len(songs) - pre.seq_len))

    # Prepare MNIST dataset by concatenating Train/Test part; we split later.
    dataset_train_part = MusicDataset(train_ids, songs, pre.mapping_file, seq_len=seq_len)

    dataset_test_part = MusicDataset(validation_ids, songs, pre.mapping_file,seq_len=seq_len)
    dataset = ConcatDataset([dataset_train_part, dataset_test_part])
    with tqdm(total=k_folds) as progress_bar:
        progress_bar.set_description(desc='kfold iterations')
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            progress_bar.update(1)

            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            # Define data loaders for training and testing data in this fold
            trainloader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=test_subsampler)

            # Init the neural network
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            hidden_size = hyperparams.get('hidden_size') if hyperparams.get('hidden_size') else 256
            model = LSTM(pre.outputs, hidden_size, pre.outputs, pre.mapping_file, device=device)
            model.to(device)

            # Initialize optimizer
            lr = hyperparams.get('lr') if hyperparams.get('lr') else 1e-4
            optim_name = hyperparams.get('optimizer')
            if optim_name == 'ADAM':
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            elif optim_name == 'SGD':
                optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            elif optim_name == 'RMSprop':
                optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Run the training loop for defined number of epochs
            for epoch in range(0, epoches):
                progress_bar.set_description(desc=f"Fold [{fold + 1}/{k_folds}] Epoch [{epoch + 1}/{epoches}]")
                # Set current loss value
                current_loss = 0.0

                # Iterate over the DataLoader for training data
                for i, data in enumerate(trainloader, 0):

                    # Get inputs
                    inputs, targets = data

                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    # Zero the gradients
                    optimizer.zero_grad()

                    # Perform forward pass
                    outputs = model(inputs)

                    # Compute loss
                    loss = loss_function(outputs, targets)

                    # Perform backward pass
                    loss.backward()

                    # Perform optimization
                    optimizer.step()

                    # Print statistics
                    current_loss += loss.item()

            # Evaluation for this fold
            correct, total = 0, 0
            with torch.no_grad():

                # Iterate over the test data and generate predictions
                for i, data in enumerate(testloader, 0):
                    # Get inputs
                    inputs, targets = data
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    # Generate outputs
                    outputs = model(inputs)

                    # Set total and correct
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

                if not os.path.exists(exp_dir):
                    os.mkdir(exp_dir)
                torch.save(model.state_dict(), os.path.join(exp_dir, 'model.pt'))
                gen = MelodyGenerator(exp_dir, hidden_size=hidden_size)
                song = gen.generate_melody(500, seq_len, 0.85)
                song_path = os.path.join(exp_dir, f"fold_{fold}.mid")
                gen.save_melody(song, file_name=song_path)
                # Print accuracy
                progress_bar.set_postfix(loss=current_loss, acc=100.0 * correct / total)

                results.append(100.0 * (correct / total))

        # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for i, value in enumerate(results, 1):
        print(f'Fold {i}: {value} %')
        sum += value
    print(f'Average: {sum / len(results)} %')
    return results


def hidden_size_exp(kfolds=5, output="experiment_results", show=True, **kwargs):
    dataset = kwargs.get('dataset') if kwargs.get('dataset') else 'dataset/deutschl/test'
    exp_dir = os.path.join(output, "hidden_size_exp")
    if not os.path.exists(output):
        os.mkdir(output)
    hidden_sizes = kwargs.get('hidden_sizes') if kwargs.get('hidden_sizes') else [16, 32, 64, 128, 256, 512, 1024]
    kfold_results = {}
    for size in hidden_sizes:
        print(f"------------ testing hidden size of {size} ----------")
        size_dir = os.path.join(exp_dir, f"size_{size}")
        result = kfold_hyperparam_experiment(k_folds=kfolds, epoches=5, hidden_size=size, dataset=dataset, exp_dir=size_dir)
        kfold_results[size] = result

    X = [fold for fold in range(1, kfolds + 1)]
    plt.figure()
    for size in hidden_sizes:
        plt.plot(X, kfold_results[size], label=f'size: {size}')

    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("kfolds")
    plt.ylabel("accuracy")
    plt.title("experimenting with hidden layer size")

    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()

    # To load the display window
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    plt.savefig(os.path.join(exp_dir, "hidden_size_graph.png"))
    if show:
        plt.show()


def optimizers_exp(kfolds=5, output="experiment_results", show=True, **kwargs):
    dataset = kwargs.get('dataset') if kwargs.get('dataset') else 'dataset/deutschl/test'
    exp_dir = os.path.join(output, "optimiser_exp")
    if not os.path.exists(output):
        os.mkdir(output)
    optimizers = ['ADAM', 'SGD', 'RMSprop']
    kfold_results = {}

    for opt in optimizers:
        print(f"------------ testing accuracy of {opt} ----------")
        opt_dir = os.path.join(exp_dir, f"opt_{opt}")
        result = kfold_hyperparam_experiment(k_folds=kfolds, epoches=5, optimizer=opt, dataset=dataset, exp_dir=opt_dir)
        kfold_results[opt] = result

    X = [fold for fold in range(1, kfolds + 1)]
    plt.figure()
    for opt in optimizers:
        plt.plot(X, kfold_results[opt], label=f'optimiser: {opt}')

    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("kfolds")
    plt.ylabel("accuracy")
    plt.title("experimenting with different optimisers")

    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()

    # To load the display window
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    plt.savefig(os.path.join(exp_dir, "optimiser_exp_graph.png"))
    if show:
        plt.show()


def seq_len_exp(kfolds=5, output="experiment_results", show=True, **kwargs):
    dataset = kwargs.get('dataset') if kwargs.get('dataset') else 'dataset/deutschl/test'
    exp_dir = os.path.join(output, "seq_len_exp")
    if not os.path.exists(output):
        os.mkdir(output)
    seq_lens = kwargs.get('seq_lens') if kwargs.get('seq_lens') else [8, 16, 32, 64, 128]
    kfold_results = {}
    for seq_len in seq_lens:
        print(f"------------ testing seq_len of {seq_len} ----------")
        seq_len_dir = os.path.join(exp_dir, f"len_{seq_len}")
        result = kfold_hyperparam_experiment(k_folds=kfolds, epoches=5, seq_len=seq_len, dataset=dataset, exp_dir=seq_len_dir)
        kfold_results[seq_len] = result

    X = [fold for fold in range(1, kfolds + 1)]
    plt.figure()
    for seq_len in seq_lens:
        plt.plot(X, kfold_results[seq_len], label=f'sequence len: {seq_len}')

    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("kfolds")
    plt.ylabel("accuracy")
    plt.title("experimenting with sequence length")

    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()

    # To load the display window
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    plt.savefig(os.path.join(exp_dir, "seq_len_exp_graph.png"))
    if show:
        plt.show()
