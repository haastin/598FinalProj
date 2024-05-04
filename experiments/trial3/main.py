import numpy as np
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, Precision, Recall, F1Score
from tqdm import tqdm

class MemmapDataset(Dataset):
    def __init__(self, memmap_path, shape, label_memmap_path, label_shape, transform=None):
        """
        Args:
            memmap_path (string): Path to the memmap file.
            shape (tuple): Shape of the memmap array (num_images, height, width, channels).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = np.memmap(memmap_path, dtype='uint8', mode='r', shape=shape)
        self.labels = np.memmap(label_memmap_path, dtype='int64', mode='r', shape=label_shape)
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        # Select the single image
        image = self.images[idx]
        label = self.labels[idx]

        # Convert image from numpy array to PIL Image for compatibility with torchvision transforms
        image = transforms.functional.to_pil_image(image)

        if self.transform:
            image = self.transform(image)

        return image, label
    
def train_loop(dataloader, model, loss_fn, optimizer, writer):
    
    batch_size = len(dataloader)
    size = len(dataloader)

    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    loop = tqdm(dataloader, total=size,leave=True)
    
    #batch is an index, X is tensor of dim BxHxWxC and y is BxNum_classes
    for batch, (X, y) in enumerate(dataloader):

        X = X.cuda()
        y = y.cuda()

        # Compute prediction and loss
        pred = model(X)

        loss = loss_fn(pred, y)

        # -- Backpropagation --

        #this step calculates the change made to each weight as a result of the optimizer
        loss.backward()
        #this step adds or subtracts that amount to each weight's current value
        optimizer.step()
        #this step resets the variables that hold the amount to change the weights by to 0, as they accumulate with each backprop run
        optimizer.zero_grad()

        writer.add_scalar('Loss/train', loss.item(), epoch_num * len(dataloader) + batch)
        
        loop.set_description(f"Epoch {epoch_num+1}")
        loop.set_postfix(loss=loss.item())
        loop.update()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * batch_size + len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, writer):

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    #the base stats we derive our evaluation metrics from
    test_loss, correct = 0, 0

    num_classes = 62  # Adjust based on your dataset classes
    precision = Precision(task='multiclass',num_classes=num_classes, average=None)
    recall = Recall(task='multiclass',num_classes=num_classes, average=None)
    f1 = F1Score(task='multiclass',num_classes=num_classes, average=None)

    test_bar = tqdm(dataloader, desc='Testing', total=num_batches, leave=True)

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():

        #X is a tensor of dimension BxWxHxC
        #y is a tensor of dimension B (each entry is the correct label for that sample)
        for X, y in dataloader:

            X = X.cuda()
            y = y.cuda()

            #pred is a tensor w/ dimensions == Batch_size x Num_classes
            #thus, for each sample in the batch, the logit value for each output class is the output
            pred = model(X) 
            
            # -- explaining how the loss func in PyTorch API works below --

            #loss_fn is a variable that contains the loss function we have chosen
            #pred and y are tensors in BxNum_classes format
            #the loss func therefore computes loss from comparison between input to expected output vals
            #there is a "reduction" parameter in the original loss function we chose (in the main() code below)
            #which determines if or how the losses for each sample in the batch are  combined. you can do sum, 
            #mean or nothing. we don't specify, and mean is the default value
            #thus, this loss_fn outputs a scalar which is the mean of each sample loss in the batch 
            #.item() just converts a tensor val to standard Python number
                #the reason we need .item() is because the same loss function is used for both train and test, and
                #in train it needs to be
            loss = loss_fn(pred, y).item() 
            test_loss += loss

            # -- explaining how to find num of correctly predicted samples in the batch --

            #.argmax(<input dim>) returns the index of the max element in dimension <input dim>, 
            #in this case dim=1, which corresponds to the dimension to search to find the max val
            #output of pred.argmax == y is a 1-D (dim == # of batches) boolean tensor of values 
            #either False or True
            #.type(float) converts those values to either 1.0 or 0.0
            #.sum() combines the output vals for every sample in the batch
            #correct thus holds the num of correct samples in a batch
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            test_bar.set_postfix({'Loss': loss})
            test_bar.update()

            # Update metrics
            y=y.cpu()
            pred=pred.cpu()

            precision.update(pred, y)
            recall.update(pred, y)
            f1.update(pred, y)

    #average loss per batch
    test_loss /= num_batches

    #percentage of correctly labeled input images
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    writer.add_scalar('Loss/test', test_loss, epoch_num)
    writer.add_scalar('Accuracy/test', 100*correct, epoch_num)

    for i in range(num_classes):
        writer.add_scalar(f'Precision/class_{i}', precision.compute()[i], epoch_num)
        writer.add_scalar(f'Recall/class_{i}', recall.compute()[i], epoch_num)
        writer.add_scalar(f'F1-Score/class_{i}', f1.compute()[i], epoch_num)

#PyTorch loads the binary data from the memmapped file for us, so we don't need this, but keep it here for backup
#images = np.memmap("/scratch/ccui17/images_val.memmap", dtype="uint8", mode="r", shape=(53041, 224, 224, 3))

epoch_num = 0

if __name__ == "__main__":

    device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
    )
    
    #we are using a memmap file for this project becasue our dataset of images is too big to fit
    #entirely within RAM, so I/O time of reading the image from disk can be sped up by writing them to
    #a memmapped file (idk how but it does)
    train_memmap_path = '/scratch/ccui17/images_train.memmap'
    train_labels_memmap_path = '/scratch/ccui17/labels_train.memmap'
    test_memmap_path = '/scratch/ccui17/images_val.memmap'
    test_labels_memmap_path = '/scratch/ccui17/labels_val.memmap'

    #mean and std dev values used for normalization of the input image pixel values
    #in best practices, these are the mean and std dev of the input dataset or pretrained dataset,
    #but this model is not pretrained and we haven't found the mean and std dev of our pixel vals, so
    #using these vals for simplicity
    #the .memmap files we read the images from to create the dataset already has the images cropped,
    #resized to 256,256. just need to normalize and transform to Tensor here
    transform = transforms.Compose([  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
    ])

    learning_rate = 1e-5
    epochs = 20
    batch_size = 640
    train_dataset_len = 363572
    test_dataset_len = 53041
    data_portions = [0.25, 0.5, 0.75, 1.0]  # Portions of data to use for training

    #shape[0] == num images in the file, [1] and [2] are HxW of the images, [3] is # channels in the pics
    test_shape = (test_dataset_len, 224, 224, 3) 
    test_labels_shape=(test_dataset_len,)
    test_dataset = MemmapDataset(test_memmap_path, test_shape, test_labels_memmap_path,test_labels_shape,transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    

    for i, portion in enumerate(data_portions):

        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)

        #modify ResNet last layer to be custom num of classes
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 62)  # Change the last layer to 63 classes

        model = model.cuda()
        
        #the loss func determines what the model should optimize for, because it determines the penalties applied to a wrong answer
        loss_fn = nn.CrossEntropyLoss()
        
        #the optimizer determines how the amount of loss calculated will be used to adjust the weights of the model
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        #PyTorch Dataset object created here based our custom implementation by reimplementing derived class
        train_shape = ((i+1)*train_dataset_len//4, 224, 224, 3) 
        train_labels_shape = (train_dataset_len,)
        train_dataset = MemmapDataset(train_memmap_path, train_shape, train_labels_memmap_path,train_labels_shape,transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

        # Initialize TensorBoard writer
        writer = SummaryWriter(f'runs/resnet18_experiment{(i+1)*25}_logs')

        for t in range(epochs):
            epoch_num = t
            print(f"Epoch {t+1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer, writer)
            test_loop(test_dataloader, model, loss_fn, writer)

        print(f"Done training for experiment {(i+1)*25}")

        torch.save(model.state_dict(), f'resnet18_experiment{(i+1)*25}_resultweights.pth')
        writer.close()



