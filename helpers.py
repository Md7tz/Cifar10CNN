import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm


def get_train_val_data_loaders(batch_size, valid_size, transforms, num_workers):
    """
    Returns the training and validation data loaders.
    """
    # Download the training and validation datasets
    trainval_data = datasets.CIFAR10(
        "data", train=True, download=True, transform=transforms
    )

    # Compute how many items we will reserve for the validation set
    n_tot = len(trainval_data)
    split = int(np.floor(valid_size * n_tot))

    # Compute the indices for the training set and for the validation set
    torch.manual_seed(42)
    shuffled_indices = torch.randperm(n_tot)
    train_idx, valid_idx = shuffled_indices[split:], shuffled_indices[:split]

    # Define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(indices=train_idx)
    valid_sampler = SubsetRandomSampler(indices=valid_idx)

    # Prepare dataloaders
    train_loader = DataLoader(
        trainval_data,
        batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        trainval_data,
        batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
    )

    return train_loader, valid_loader


def get_test_data_loader(batch_size, transforms, num_workers):
    """
    Returns the test data loader.
    """
    test_data = datasets.CIFAR10(
        "data", train=False, transform=transforms, download=True
    )
    test_loader = DataLoader(
        test_data, batch_size, shuffle=False, num_workers=num_workers
    )

    return test_loader


def train_one_epoch(train_dataloader, model, optimizer, criterion):
    """
    Performs one epoch of training
    """

    # Move model to GPU if available
    if torch.cuda.is_available():
        model.cuda()

    # Set the model to training mode
    # (so all layers that behave differently between training and evaluation,
    # like batchnorm and dropout, will select their training behavior)
    model.train()

    # Loop over training data
    train_loss = 0.0

    for batch_idx, (data, target) in tqdm(
        enumerate(train_dataloader),
        "Training",
        total=len(train_dataloader),
        leave=True,
        ncols=80,
    ):
        # move data to GPU if available
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        # 1. clear the gradients of all optimized variables
        optimizer.zero_grad()

        # 2. forward pass: compute predicted outputs by passing the input to the model
        output = model(data)

        # 3. calculate the loss
        loss = criterion(output, target)

        # 4. backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # 5. perform a single optimization step (parameter update)
        optimizer.step()

        # update average training loss (running avg)
        train_loss += (1 / (batch_idx + 1)) * (loss.data.item() - train_loss)

    return train_loss


def valid_one_epoch(valid_dataloader, model, criterion):
    """
    Performs one epoch of validation
    """

    # During validation we don't need to accumulate gradients
    with torch.no_grad():
        # set the model to evaluation mode
        # (so all layers that behave differently between training and evaluation,
        # like batchnorm and dropout, will select their evaluation behavior)
        model.eval()

        # If the GPU is available, move the model to the GPU
        if torch.cuda.is_available():
            model.cuda()

        valid_loss = 0.0

        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader),
            "Validating",
            total=len(valid_dataloader),
            leave=True,
            ncols=80,
        ):
            # Move data to GPU if available
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

        # 1. forward pass
        output = model(data)

        # 2. calculate the loss
        loss = criterion(output, target)

        valid_loss += (1 / (batch_idx + 1)) * (loss.data.item() - valid_loss)

        return valid_loss


def optimize(
    data_loaders,
    model,
    optimizer,
    criterion,
    n_epochs,
    save_path,
    interactive_tracking=False,
):
    # initialize tracker for minimum validation loss
    if interactive_tracking:
        liveloss = PlotLosses()
    else:
        liveloss = None

    # Loop over the epochs and keep track of the minimum of the validation loss
    valid_loss_min = None
    logs = {}

    for epoch in range(1, n_epochs):
        train_loss = train_one_epoch(data_loaders["train"], model, optimizer, criterion)
        valid_loss = valid_one_epoch(data_loaders["valid"], model, optimizer, criterion)

        # print training/validation statistics
        print(
            "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss {:.6f}".format(
                epoch, train_loss, valid_loss
            )
        )

        # If the validation loss increases by 1% save the model
        if (
            valid_loss_min is None
            or (valid_loss_min - valid_loss) / valid_loss_min > 0.01
        ):
            print(f"New minimum validation loss: {valid_loss:.6f}. Saving model ...")

            # Save the weights to save_path
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss

        # Log the losses and the current learning rate
        if interactive_tracking:
            logs["loss"] = train_loss
            logs["val_loss"] = valid_loss

            liveloss.update(logs)
            liveloss.send()


def one_epoch_test(test_dataloader, model, criterion):
    # monitor test loss and accuracy
    test_loss = 0.0
    correct = 0.0
    total = 0.0

    # Disable gradient accumulation
    with torch.no_grad():
        # set the model to evaluation mode
        model.eval()

        # if the GPU is available, move the model to the GPU
        if torch.cuda.is_available():
            model.cuda()

        # Loop over test dataset
        # We also accumulate predictions and targets so we can return them
        preds = []
        actuals = []

        for batch_idx, (data, target) in tqdm(
            enumerate(test_dataloader),
            "Testing",
            total=len(test_dataloader),
            leave=True,
            ncols=80,
        ):
            # move data to GPU if available
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # 1. forward pass
            logits = model(data)

            # 2. calculate loss
            loss = criterion(
                logits, target
            ).detach()  # detached from the current graph.

            # update average test loss
            test_loss += test_loss + (
                (1 / batch_idx + 1) * (loss.data.item() - test_loss)
            )

            # convert logits to predicted class
            # Note: the predicted class is the index of the max of the logits
            _, pred = logits.data.max(1, keepdim=True)

            # compare predictions to true label
            correct += torch.sum(pred == target).item()
            total += data.size(0)

            preds.extend(pred.data.cpu().numpy().squeeze())
            actuals.extend(target.data.view_as(pred).cpu().numpy().squeeze())

    print("Test Loss: {:.6f}\n".format(test_loss))

    print("Test accuracy: %2d%% (%2d/%2d)" % (100.0 * correct / total, correct, total))

    return test_loss, preds, actuals
