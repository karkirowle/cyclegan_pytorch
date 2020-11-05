import numpy as np
from torch.utils.data import Dataset

from nnmnkwii.datasets import FileDataSource, FileSourceDataset, PaddedFileSourceDataset
import sys
sys.path.insert(1, '/home/boomkin/repos/cyclegan_pytorch/')

from data_utils import MFCCSource, ArticulatorySource, NanamiDataset, pad_collate
from modules import Modern_DBLSTM_1
import torch


def worker_init_fn(worker_id):
    manual_seed = 0
    # After creating the workers, each worker has an independent seed that is initialized to the curent random seed + the id of the worker
    np.random.seed(manual_seed + worker_id)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 20
    # Load MNGU0 features
    train_x = FileSourceDataset(MFCCSource("trainfiles.txt"))
    val_x = FileSourceDataset(MFCCSource("validationfiles.txt"))
    test_x = FileSourceDataset(MFCCSource("testfiles.txt"))

    train_y = FileSourceDataset(ArticulatorySource("trainfiles.txt"))
    val_y = FileSourceDataset(ArticulatorySource("validationfiles.txt"))
    test_y = FileSourceDataset(ArticulatorySource("testfiles.txt"))

    dataset = NanamiDataset(train_x, train_y)
    dataset_val = NanamiDataset(val_x, val_y)
    dataset_test = NanamiDataset(test_x, test_y)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=4, collate_fn=pad_collate)

    val_loader = torch.utils.data.DataLoader(dataset_val,
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=4, collate_fn=pad_collate)

    test_loader = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=4, collate_fn=pad_collate)

    num_epochs = 6
    input_size = 24
    hidden_size = 150
    hidden_size_2 = 150
    num_classes = 12
    learning_rate = 0.001
    model = Modern_DBLSTM_1(input_size, hidden_size, hidden_size_2, num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model

    epoch_log = np.zeros((num_epochs, 3))

    for epoch in range(num_epochs):
        total_loss = 0
        print("Epoch ", epoch + 1)
        for i, sample in enumerate(train_loader):
            xx_pad, yy_pad, _, _, mask = sample

            inputs = xx_pad.to(device)

            targets = yy_pad.to(device)

            mask = mask.to(device)

            outputs = model(inputs)

            loss = torch.sum(((outputs - targets) * mask) ** 2.0) / torch.sum(mask).item()

            if targets.shape[0] == batch_size:
                total_loss += loss.item()
            else:
                total_loss += loss.item() * (targets.shape[0] / batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss = np.sqrt(total_loss / len(train_loader))
        print('Epoch [{}/{}], Train RMSE: {:.4f} cm'.format(epoch + 1, num_epochs, total_loss))

        torch.cuda.empty_cache()

        with torch.no_grad():
            total_loss = 0
            for i, sample in enumerate(val_loader):
                xx_pad, yy_pad, _, _, mask = sample
                inputs = xx_pad.to(device)
                targets = yy_pad.to(device)

                mask = mask.to(device)

                outputs = model(inputs)

                loss = torch.sum(((outputs - targets) * mask) ** 2.0) / torch.sum(mask).item()

                # Weigh differently the last smaller batch
                if targets.shape[0] == batch_size:
                    total_loss += loss.item()
                else:
                    total_loss += loss.item() * (targets.shape[0] / batch_size)

            total_loss = np.sqrt(total_loss / len(val_loader))

            print('Epoch [{}/{}], Validation RMSE: {:.4f} cm'.format(epoch + 1, num_epochs, total_loss))

        torch.cuda.empty_cache()

        with torch.no_grad():
            total_loss = 0
            for i, sample in enumerate(test_loader):
                xx_pad, yy_pad, _, _, mask = sample
                inputs = xx_pad.to(device)
                targets = yy_pad.to(device)

                mask = mask.to(device)

                outputs = model(inputs)

                loss = torch.sum(((outputs - targets) * mask) ** 2.0) / torch.sum(mask).item()
                if targets.shape[0] == batch_size:
                    total_loss += loss.item()
                else:
                    total_loss += loss.item() * (targets.shape[0] / batch_size)

            total_loss = np.sqrt(total_loss / len(test_loader))

            epoch_log[epoch, 2] = total_loss
            print('Epoch [{}/{}], Test RMSE: {:.4f} cm'.format(epoch + 1, num_epochs, total_loss))

        torch.cuda.empty_cache()

    best_id = np.argmin(epoch_log[:i, 1])
    return model


if __name__ == '__main__':

    model = train()

    torch.save(model,"sarticulatory_model_pz.pt")