#An exploration of tensors, pytorch and friends.
#This code explores the creation and loading
#of sample test and training datasets using Pytorch (AKA  "Torch")
#
#Step 1: Creating a small toy dataset
import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

X_train = torch.tensor([
           [-1.2, 3.1],
           [-0.9, 2.9],
           [-0.5, 2.6],
           [2.3, -1.1],
           [2.7, -1.5]
       ])
y_train = torch.tensor([0, 0, 0, 1, 1])
X_test = torch.tensor([
           [-0.8, 2.8],
           [2.6, -1.6],
       ])
y_test = torch.tensor([0, 1])

#A function to compute the prediction accuracy
def compute_accuracy(model, dataloader):
    model = model.eval()
    correct = 0.0
    total_examples = 0
    for idx, (features, labels) in enumerate(dataloader):
        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)
    return (correct / total_examples).item()

#define a neural network class
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs): #Coding the number of inputs and outputs as variables allows us to reuse the same code for datasets with different numbers of features and classes
        super().__init__()
        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30), #The Linear layer takes the number of input and output nodes as arguments.
            torch.nn.ReLU(), #Nonlinear activation functions are placed between the hidden layers.
            # 2nd hidden layer
            torch.nn.Linear(30, 20), #The number of output nodes of one hidden layer has to match the number of inputs of the next layer.
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(20, num_outputs),
        )
    def forward(self, x):
        logits = self.layers(x)
        return logits #The outputs of the last layer are called logits.

#Defining a custom Dataset class
class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y
    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y
    def __len__(self):
        return self.labels.shape[0]

#define training dataset
train_ds = ToyDataset(X_train, y_train)
#define test dataset
test_ds = ToyDataset(X_test, y_test)

print("Length of training dataset: ", len(train_ds))

#Instantiating data loaders

torch.manual_seed(123)
train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    drop_last=True
)
test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

#iterate over the training dataset
for idx, (x, y) in enumerate(train_loader):
    print(f"Training Batch {idx+1}:", x, y)

for idx, (x, y) in enumerate(test_loader):
    print(f"Testing Batch {idx+1}:", x, y)

#Now we try Neural network training in PyTorch
torch.manual_seed(123)

model = NeuralNetwork(num_inputs=2, num_outputs=2)

#print the structure of the model
print(model)

optimizer = torch.optim.SGD(
    model.parameters(), lr=0.5 #The dataset has two features and two classes.
)
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        logits = model(features)
        loss = F.cross_entropy(logits, labels)
        #WTF is "Gradient Accumulation" ?? Sets the gradients from the previous round to 0 to prevent unintended gradient accumulation
        optimizer.zero_grad()
        loss.backward()  #Computes the gradients of the loss given the model parameters
        optimizer.step() #The optimizer uses the gradients to update the model parameters.
        ### LOGGING
        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train Loss: {loss:.2f}")
    model.eval() 
    # Insert optional model evaluation code after the above

#After we have trained the model, we can use it to make predictions:
model.eval()
with torch.no_grad():
    outputs = model(X_train)
print(outputs)

#use PyTorch’s softmax function To obtain the class membership probabilities ... Why?
#wtf is going on here?
#What's a "class membership probabilities" ?
torch.set_printoptions(sci_mode=False)
probas = torch.softmax(outputs, dim=1)
print(probas)

#We can then apply the function to the training set:
print(compute_accuracy(model, train_loader))

#Similarly, we can apply the function to the test set:
print(compute_accuracy(model, test_loader))

# Now that we’ve trained our model, let’s see how to save it so we can reuse it later. 
# Here’s the recommended way how we can save and load models in PyTorch to disk:
torch.save(model.state_dict(), "model.pth")

#Once we saved the model, we can restore it from disk:
model = NeuralNetwork(2, 2)
model.load_state_dict(torch.load("model.pth"))