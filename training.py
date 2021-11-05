import os
from shutil import copyfile
import torch
from tqdm import tqdm


class Trainer:
    """
    helper class to manage training
    """
    def __init__(self, model, optimizer, criterion,
                 check_point_every=None, save=True, seq_len=64, device='cpu', output='trained_models'):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.check_point_every = check_point_every
        self.save = save
        self.seq_len = seq_len
        self.input_size = self.model.input_size
        self.device = device
        self.output = output

        self.model.to(device)

    def fit(self, train_loader, validation_loader, epoches=50):
        """
        fits the model's parameters
        :param train_loader: data loader with training samples
        :param validation_loader: data loader with validation samples
        :param epoches: number of epoches to run
        """
        loss_values = []
        validation_values = []

        with tqdm(total=epoches) as progress_bar:
            for epoch in range(epoches):
                progress_bar.set_description(desc=f"Epoch [{epoch}/{epoches}]")

                # set the model to train mode
                self.model.train()
                running_loss = 0.0

                # getting sequences and labels from the train loader
                for i, (sequences, labels) in enumerate(train_loader):
                    sequences = sequences.reshape(-1, self.seq_len, self.input_size).to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    outputs = self.model(sequences)

                    loss = self.criterion(outputs, labels.long())

                    # backward (derivatives)
                    loss.backward()

                    # optimizing the parameters
                    self.optimizer.step()

                    running_loss = +loss.item() * sequences.size(0)

                    progress_bar.set_description(
                        desc=f"Epoch [{epoch}/{epoches}] Step [{i}/{len(train_loader)}]")

                loss_values.append(running_loss / len(train_loader.dataset))

                # setting the model to evaluation mode
                self.model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for i, (sequences, labels) in enumerate(validation_loader):
                        sequences = sequences.reshape(-1, self.seq_len, self.input_size).to(self.device)
                        labels = labels.to(self.device)
                        outputs = self.model(sequences)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                validation_accuracy = 1.0 * correct / total
                progress_bar.set_postfix(loss=running_loss / len(train_loader.dataset),
                                         acc=validation_accuracy * 100)

                validation_values.append(validation_accuracy)

                if self.check_point_every is not None:
                    check_point_dir = os.path.join(self.output, 'checkpoints')
                    if not os.path.exists(check_point_dir):
                        os.mkdir(check_point_dir)
                    if epoch % self.check_point_every == 0:
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": self.model.state_dict(),
                                "optimiser_state_dict": self.optimizer.state_dict(),
                                "loss_values": loss_values,
                                "validation_values": validation_values,
                            },
                            os.path.join(check_point_dir, f"epoch-{epoch}.tar"),
                        )
                progress_bar.update(1)

        # saving the state_dict and the mapping.json file of the model
        torch.save(self.model.state_dict(), os.path.join(self.output, 'model.pt'))
        if self.model.mapping_path != os.path.join(self.output, 'mapping.json'):
            copyfile(self.model.mapping_path, os.path.join(self.output, 'mapping.json'))
