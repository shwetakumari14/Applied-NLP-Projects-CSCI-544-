import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

with open('./vocab_dictionary.pickle', 'rb') as fr1:
    Word_to_Index = pickle.load(fr1)
with open('./ner_dictionary.pickle', 'rb') as fr2:
    Index_to_Word = pickle.load(fr2)
with open('./int_vocab_dictionary.pickle', 'rb') as fr3:
    NER_to_Index = pickle.load(fr3)
with open('./int_ner_dictionary.pickle', 'rb') as fr4:
    Index_to_Ner = pickle.load(fr4)
with open('./loader_train.pickle', 'rb') as fr5:
    loader_train = pickle.load(fr5)
with open('./loader_dev.pickle', 'rb') as fr6:
    loader_dev = pickle.load(fr6)
with open('./loader_test.pickle', 'rb') as fr7:
    loader_test = pickle.load(fr7)

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class BLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, first_output_dim, output_dim, num_layers, bidirectional, drop_out): 
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.blstm = nn.LSTM(embedding_dim, hidden_dim, num_layers = num_layers, bidirectional = bidirectional, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim * 2, first_output_dim)
        self.dropout = nn.Dropout(drop_out)
        self.activation = nn.ELU()
        self.fc2 = nn.Linear(first_output_dim, output_dim)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        outputs, (hidden, cell) = self.blstm(embedded)
        outputs = self.dropout(outputs)
        outputs = self.activation(self.fc1(outputs))
        predictions = self.fc2(outputs)
        return predictions
    
def categoricalAccuracy(preds, y, tag_pad_idx, text, predict_table):
    tot = 0
    correct = 0
    max_preds = preds.argmax(dim = 1, keepdim = True) 
    for predict, real, word in zip(max_preds, y, text):
        if real.item() == tag_pad_idx:
            continue
        else:
            predict_table.append((word.item(), predict.item(), real.item()))
            if real.item() == predict.item():
                correct += 1
            tot += 1
    return tot, correct, predict_table

def categoricalEvaluate(preds, text, predictTable):

    max_preds = preds.argmax(dim = 1, keepdim = True)
    for predict, word in zip(max_preds, text):
        if word == 0:
            continue
        else:
            predictTable.append((word, predict[0]))

    return predictTable

tag_pad_idx=-100
criterion = nn.CrossEntropyLoss(ignore_index= -100)
def evaluateModelDev(model, dataloader, predict_table):

    epoch_loss = 0
    epoch_acc = 0
    epoch_tot = 0
    model.eval()

    with torch.no_grad():

        for text, tags in dataloader:
            tags = tags.to(device)
            text = text.to(device)
            predictions = model(text)

            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)
            
            loss = criterion(predictions, tags)

            tot, correct, predict_table = categoricalAccuracy(predictions, tags, tag_pad_idx, text.view(-1), predict_table)

            epoch_loss += loss.item()
            epoch_acc += correct
            epoch_tot +=tot

    return epoch_loss / len(dataloader), epoch_acc / epoch_tot, predict_table

def evaluateModelTest(model, loader, predictTable):

    model.eval()

    with torch.no_grad():

        for text in loader:
            text = text.to(device)
            predictions = model(text)
            predictions = predictions.view(-1, predictions.shape[-1])

            predictTable = categoricalEvaluate(predictions, text.view(-1), predictTable)

    return predictTable

def generateDevFile():
    checkpoint = torch.load('checkpoint.pth', map_location=torch.device('cpu'))
    evaluate_predict_table2 = []
    model = BLSTM(checkpoint['INPUT_DIM'], 
                checkpoint['EMBEDDING_DIM'], 
                checkpoint['HIDDEN_DIM'], 
                checkpoint['FIRST_OUTPUT_DIM'],
                checkpoint['OUTPUT_DIM'], 
                checkpoint['N_LAYERS'], 
                checkpoint['BIDIRECTIONAL'], 
                checkpoint['DROPOUT'])
    
    model.to(device)

    if is_cuda:
        model.load_state_dict(torch.load('./blstm1.pt'))
    else:
        model.load_state_dict(checkpoint['state_dict'])
    prediction_table = evaluateModelDev(model, loader_dev, evaluate_predict_table2)

    term = [int(x[0]) for x in evaluate_predict_table2]
    y_pred = [int(x[1]) for x in evaluate_predict_table2]

    i=0
    newfile = open('./dev1.out', "w")
    with open('./data/dev', "r") as train:
        for line in train:
            if not line.split():
                newfile.write('\n')
                continue
            index, word_type = line.split(" ")[0], line.split(" ")[1].strip('\n')
            for_tag = Index_to_Ner[y_pred[i]]
            newfile.write(str(index)+' '+str(word_type)+' '+for_tag+'\n')
            i += 1
    newfile.close()

def generateTestFile():
    checkpoint = torch.load('checkpoint.pth', map_location=torch.device('cpu'))
    evaluate_predict_table2 = []
    model = BLSTM(checkpoint['INPUT_DIM'], 
                checkpoint['EMBEDDING_DIM'], 
                checkpoint['HIDDEN_DIM'], 
                checkpoint['FIRST_OUTPUT_DIM'],
                checkpoint['OUTPUT_DIM'], 
                checkpoint['N_LAYERS'], 
                checkpoint['BIDIRECTIONAL'], 
                checkpoint['DROPOUT'])
    
    model.to(device)

    if is_cuda:
        model.load_state_dict(torch.load('./blstm1.pt'))
    else:
        model.load_state_dict(checkpoint['state_dict'])

    prediction_table = evaluateModelTest(model, loader_test, evaluate_predict_table2)

    term = [int(x[0]) for x in evaluate_predict_table2]
    y_pred = [int(x[1]) for x in evaluate_predict_table2]

    i=0
    newfile = open('./test1.out', "w")
    with open('./data/test', "r") as test:
        for line in test:
            if not line.split():
                newfile.write('\n')
                continue
            index, word_type = line.split(" ")[0], line.split(" ")[1].strip('\n')
            for_tag = Index_to_Ner[y_pred[i]]
            newfile.write(str(index)+' '+str(word_type)+' '+for_tag+'\n')
            i += 1
    newfile.close()

generateDevFile()
generateTestFile()