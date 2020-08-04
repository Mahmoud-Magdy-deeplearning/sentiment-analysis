import torch
import torch.optim as optim
import torch.nn as nn
from torchtext import data
from torchtext import datasets


# our tokenizer
TEXT = data.Field(tokenize = 'spacy') 
LABEL = data.LabelField(dtype = torch.float)

# large movie reviews dataset
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL) 

//limit the size of vocab for one hot vector (vocabulary dict) to 25000 , adding <unk> & <pad> tokens  
TEXT.build_vocab(train_data, max_size = 25_000)
LABEL.build_vocab(train_data)

# working on GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# prepare data for iteration usig backiterator
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE,
    device = device)
    
# model building
class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):

        #text = [sent len, batch size]
        embedded = self.embedding(text)
        
        #embedded = [sent len, batch size, emb dim]
        output, hidden = self.rnn(embedded)
        
        #output = [sent len, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
        
        assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        
        return self.fc(hidden.squeeze(0))
        
        
        
# create instance of RNN
model = RNN(len(TEXT.vocab), 300, 256, 1)

# optimizing the model function
optimizer = optim.SGD(model.parameters(), lr=1e-3)

#binary cross entropy function (cost function for binary O/P)
criterion = nn.BCEWithLogitsLoss()

#make calculation on GPU
model = model.to(device)
criterion = criterion.to(device)

# calculate accuracy
def binary_accuracy(preds, y):
   
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

# train the model function
def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
                
        predictions = model(batch.text).squeeze(1)
        
        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
    

#evaluate model function
def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)



best_valid_loss = float('inf')

#train and evaluate the model using IMDB dataset
for epoch in range(10):

    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')

#for get best parameters we get from learning
model.load_state_dict(torch.load('tut1-model.pt'))
test_loss, test_acc = evaluate(model, test_iterator, criterion)

print('best valid loss is'+ best_valid_loss + 'test set loss score is'+ test loss)
    
