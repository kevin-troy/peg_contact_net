import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def load_data(filename="data_train_large.csv"):
    data = np.loadtxt(filename, delimiter=",")
    x = data[:,:3]
    y = data[:,3:]
    return torch.Tensor(x), torch.Tensor(y)


class ContactDataset(torch.utils.data.Dataset):
    def __init__(self, filename="data_train_small.csv"):
        self.x, self.y = load_data(filename)
        self.file = filename

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]


class ContactNet(nn.Module):
    def __init__(self, input_dim=3, fc_sizes=[10,50,50,10,10], peg_width=None, peg_height=None):
        super(ContactNet, self).__init__()
        self.peg_height = peg_height
        self.peg_width = peg_width
        self.fc0 = nn.Linear(input_dim, fc_sizes[0])
        self.fc1 = nn.Linear(fc_sizes[0], fc_sizes[1])
        self.fc2 = nn.Linear(fc_sizes[1], fc_sizes[2])
        self.fc3 = nn.Linear(fc_sizes[2], fc_sizes[3])
        self.fc4 = nn.Linear(fc_sizes[3], fc_sizes[4])
        self.out_x = nn.Linear(fc_sizes[4], 2)
        self.out_y = nn.Linear(fc_sizes[4], 2)
        self.out_theta = nn.Linear(fc_sizes[4], 1)

    def forward(self, x):
        # Primary FC layers
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        # Split into two clipped outputs. Use tanh to scale final outputs to be within the peg size bounds
        # if peg dims are given a priori
        if self.peg_height == None or self.peg_width == None:
            y_out = self.out_y(x)
            x_out = self.out_x(x)
        else:
            y_out = self.peg_height/2*F.tanh(self.out_y(x))
            x_out = self.peg_width/2*F.tanh(self.out_x(x))

        theta_out = 2*torch.pi*F.tanh(self.out_theta(x))

        return torch.concat([x_out, y_out, theta_out], axis=1)


def train_epoch(epoch_id):
    loss_sum = 0.
    for idx, data in enumerate(training_loader):
        x, y = data
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        y_hat = contact_net(x)
        loss = f_loss(y_hat, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item()
    return loss_sum


if __name__ == "__main__":
    save = True
    use_small = False
    use_clip = True
    epochs = 100
    width = 0.75
    height = 1.5
    save_file = "contact_net_v1"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Cuda version =", torch.version.cuda)
    print("Training with device =", device)

    if use_clip:
        contact_net = ContactNet(peg_width=0.75, peg_height=1.5)
        save_file+="_clipped"
    else:
        contact_net = ContactNet()
        save_file += "_unclipped"
    contact_net.to(device)

    f_loss = nn.MSELoss()
    opt = torch.optim.Adam(contact_net.parameters(), lr=1e-3)

    if use_small:
        #
        batch_size = 100
        val_batch_size = 10
        train_data = ContactDataset(filename="data_train_small.csv")
        validation_data = ContactDataset(filename="data_val_small.csv")
        save_file+="_small"
    else:
        # 100k v 1k
        batch_size = 1000
        val_batch_size = 100
        train_data = ContactDataset(filename="data_train.csv")
        validation_data = ContactDataset(filename="data_val.csv")
    training_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=val_batch_size, shuffle=True)

    # Train loop
    losses = []
    v_losses = []
    for epoch in range(epochs):
        contact_net.train(True)
        loss = train_epoch(epoch)
        losses += [loss]

        with torch.no_grad():
            v_loss_sum = 0.
            for idx, val_data in enumerate(validation_loader):
                x_v, y_v = val_data
                x_v, y_v = x_v.to(device), y_v.to(device)
                y_hat_v = contact_net(x_v)
                v_loss = f_loss(y_hat_v, y_v)
                v_loss_sum += v_loss.item()
            v_losses += [v_loss_sum]
        if epoch % 100 == 0:
            print("Epoch ", str(epoch), " complete.")
            print("\tTrain loss = ", losses[-1])
            print("\tValid. loss = ", v_losses[-1])
    print("Done training!")
    print("\tTrain loss = ", losses[-1])
    print("\tValid. loss = ", v_losses[-1])

    # Save
    if save:
        import pandas as pd
        torch.save(contact_net.state_dict(), "./models/"+save_file+".pt")
        df = pd.DataFrame(data=None, columns=["train_loss", "val_loss"])
        df["train_loss"] = losses
        df["val_loss"] = v_losses
        df.to_csv("./results/"+save_file+".csv")

    # Plot
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(losses, label="train")
    plt.plot(v_losses, '--', label="val")
    plt.title("losses")
    plt.legend()
    plt.subplot(3,1,2)
    plt.plot(losses)
    plt.title("train loss")
    plt.subplot(3,1,3)
    plt.plot(v_losses)
    plt.title("val loss")
    plt.show()
