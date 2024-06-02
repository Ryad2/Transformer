import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from ..utils import label_to_onehot

## MS2


class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes, hidden_d=(128, 64, 32, 28), dropout=0.25):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)
        
        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        
        # layers of the network: 4 hidden layers + 1 output layer
        self.fc1 = nn.Linear(input_size, hidden_d[0])
        self.bn1 = nn.BatchNorm1d(hidden_d[0])
        self.fc2 = nn.Linear(hidden_d[0], hidden_d[1])
        self.bn2 = nn.BatchNorm1d(hidden_d[1])
        self.fc3 = nn.Linear(hidden_d[1], hidden_d[2])
        self.bn3 = nn.BatchNorm1d(hidden_d[2])
        self.fc4 = nn.Linear(hidden_d[2], hidden_d[3])
        self.bn4 = nn.BatchNorm1d(hidden_d[3])
        self.fc5 = nn.Linear(hidden_d[3], n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        # dropout is applied after each layer
        pred = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        pred = F.relu(self.bn2(self.fc2(pred)))
        x = self.dropout(x)
        pred = F.relu(self.bn3(self.fc3(pred)))
        x = self.dropout(x)
        pred = F.relu(self.bn4(self.fc4(pred)))
        pred = self.fc5(pred)

        return pred


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes, filters=(32, 64, 128), dropout=0.25):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        
        # first layer: 2D convolution + batch normalization
        self.conv2d1 = nn.Conv2d(input_channels, filters[0], 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters[0])
        # adjust the number of channels to match the next layer
        self.adjust1 = nn.Conv2d(filters[0], filters[1], 1)

        # first block: 2D convolution + batch normalization with residual connections
        self.conv2d2_1 = nn.Conv2d(filters[0], filters[1], 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(filters[1])
        # adjust the number of channels to match the next layer
        self.adjust2 = nn.Conv2d(filters[1], filters[2], 1)
        self.conv2d2_2 = nn.Conv2d(filters[1], filters[1], 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(filters[1])

        # second block: 2D convolution + batch normalization with residual connections
        self.conv2d3_1 = nn.Conv2d(filters[1], filters[2], 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(filters[2])
        # adjust the number of channels to match the next layer
        self.adjust3 = nn.Conv2d(filters[1], filters[2], 1)
        self.conv2d3_2 = nn.Conv2d(filters[2], filters[2], 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(filters[2])

        # fully connected layers
        self.fc1 = nn.Linear(3 * 3 * filters[2], 128)
        self.fc2 = nn.Linear(128, n_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        # first convolution layer
        pred = F.relu(self.bn1(self.conv2d1(x)))
        res = F.max_pool2d(pred, 2)

        # first block
        pred = F.relu(self.bn2_1(self.conv2d2_1(res)))
        pred = self.dropout(pred)
        res = F.relu(pred + self.adjust1(res))
        pred = F.relu(self.bn2_2(self.conv2d2_2(res)))
        pred = self.dropout(pred)
        res = F.max_pool2d(F.relu(pred + res), 2)

        # second block
        pred = F.relu(self.bn3_1(self.conv2d3_1(res)))
        pred = self.dropout(pred)
        res = F.relu(pred + self.adjust2(res))
        pred = F.relu(self.bn3_2(self.conv2d3_2(res)))
        pred = self.dropout(pred)
        res = F.max_pool2d(F.relu(pred + res), 2)

        # fully connected layers
        pred = res.reshape((res.shape[0], -1))
        pred = F.relu(self.fc1(pred))
        pred = self.fc2(pred)

        return pred


def patchify(images, n_patches):
    # we assuume that h = w
    n, c, h, _ = images.shape

    patch_size = h // n_patches
    # unfold the image into patches
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # reshape the patches to have the same number of dimensions as the original image
    patches = patches.contiguous().view(n, c, n_patches ** 2, patch_size, patch_size)
    # permute the dimensions to have the patches as the first dimension
    patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(n, n_patches ** 2, -1)

    return patches

class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"
        self.d_head = d // n_heads

        # linear mappings for Q, K, V
        self.q_mappings = nn.Linear(d, d)
        self.k_mappings = nn.Linear(d, d)
        self.v_mappings = nn.Linear(d, d)

        # softmax for attention scores
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        batch_size, seq_length, d = sequences.size()

        Q = self.q_mappings(sequences).view(batch_size, seq_length, self.n_heads, self.d_head)
        K = self.k_mappings(sequences).view(batch_size, seq_length, self.n_heads, self.d_head)
        V = self.v_mappings(sequences).view(batch_size, seq_length, self.n_heads, self.d_head)

        # want shape (batch_size, n_heads, seq_length, d_head)
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # compute attention scores
        attention_scores = Q @ K.transpose(-2, -1) / (self.d_head ** 0.5)
        attention_probs = self.softmax(attention_scores)
        attention_output = attention_probs @ V

        # want shape (batch_size, seq_length, n_heads, d_head)
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        attention_output = attention_output.view(batch_size, seq_length, d)

        return attention_output

class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        # vit block: layer norm -> multi-head self-attention -> layer norm -> mlp
        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        # residual connections
        x = x + self.mhsa(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MyViT(nn.Module):
    """
    A Transformer-based neural network
    """

    def __init__(self, chw, n_patches, n_blocks, hidden_d, n_heads, out_d):
        """
        Initialize the network.
        
        """
        super().__init__()

        self.chw = chw
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        assert chw[1] % n_patches == 0 and chw[2] % n_patches == 0, "Input shape must be divisible by number of patches"
        self.patch_size = (chw[1] // n_patches, chw[2] // n_patches)

        self.input_d = chw[0] * self.patch_size[0] * self.patch_size[1]
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
        # learnable positional embeddings
        self.positional_embeddings = nn.Parameter(torch.rand(n_patches ** 2 + 1, hidden_d))
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])
        self.mlp = nn.Linear(self.hidden_d, out_d)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        n, _, _, _ = x.shape

        # patchify the input
        patches = patchify(x, self.n_patches)
        # linear mapping of the patches
        tokens = self.linear_mapper(patches)
        # add the class token
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
        # add positional embeddings
        preds = tokens + self.positional_embeddings[:tokens.size(1)].unsqueeze(0)

        # pass through the transformer blocks
        for block in self.blocks:
            preds = block(preds)

        # take the class token
        preds = preds[:, 0]
        preds = self.mlp(preds)
        
        return preds
    

class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size, device='cpu'):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size
        self.device = torch.device(device)
        # move the model to the device
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader, ep)

    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        avg_loss = 0
        n_losses = 0
        self.model.train()
        for batch in dataloader:
            x, y = batch
            # move the data to the device
            x, y = x.to(self.device), y.to(self.device)

            logits = self.model(x)
            loss = self.criterion(logits, y)

            avg_loss += loss.item()
            n_losses += 1

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        print(f"Epoch {ep} - Loss: {avg_loss / n_losses}")

    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation, 
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        self.model.eval()
        pred_labels = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                pred_labels.append(predicted)
        
        pred_labels = torch.cat(pred_labels)

        return pred_labels
    
    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        # First, prepare data for pytorch
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(), 
                                      torch.from_numpy(label_to_onehot(training_labels)))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.cpu().numpy()