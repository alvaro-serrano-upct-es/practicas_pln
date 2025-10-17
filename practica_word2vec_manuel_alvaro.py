import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

EMBED_DIMENSION = 300 
EMBED_MAX_NORM = 1 
BATCH_SIZE = 420
TOLERANCIA = 0.001 # 0e-8

with open("dataset_word2vec.txt", 'r', encoding='utf-8') as archivo:
    contenido = archivo.readlines()
data = []
ventana = 5
vocabulario = dict()
vocab_ind = dict()
ind_vocab = []
vocab_size = 0
for linea in contenido:
    limpio = linea.rstrip("\n")
    palabras = limpio.split(" ")
    for i in range(len(palabras)):
        objetivo = palabras[i]
        if objetivo not in vocabulario:
            vocabulario[objetivo] = 1
            vocab_ind[objetivo] = vocab_size
            ind_vocab.append(objetivo)
            vocab_size += 1
        else:
            vocabulario[objetivo] += 1
        for j in range(1, ventana + 1):
            if i > j-1:
                data.append((objetivo, palabras[i-j]))
            if i < len(palabras)-j:
                data.append((objetivo, palabras[i+j]))
    
# print(len(data))
# print(vocab_size)
# print(list(filter(lambda x: 20 > vocabulario[x] > 5, vocabulario)))
# print(ind_vocab[100])
# print(vocab_ind['las'])

class SkipGram_Model(nn.Module):
    """
    Implementation of Skip-Gram model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, vocab_size: int):
        super(SkipGram_Model, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = self.linear(x)
        return x
    
skipgram = SkipGram_Model(vocab_size=vocab_size)
optimizer = optim.Adam(skipgram.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

loss_ = []
epoch = 0
epoch_loss = 10

# for i in range(200):
while epoch < 2000 and epoch_loss > TOLERANCIA:

    running_loss = []

    for i in range(0, len(data), BATCH_SIZE):
            
        batch = data[i: i + BATCH_SIZE] 
        
        centros = [vocab_ind[centro] for centro, _ in batch]
        contextos = [vocab_ind[contexto] for _, contexto in batch]

        center_var = torch.tensor(centros, dtype=torch.long)
        context_var = torch.tensor(contextos, dtype=torch.long)
        
        # Forward
        scores = skipgram(center_var)

        loss = loss_function(scores, context_var)

        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss.append(loss.item())
    
    epoch_loss = np.mean(running_loss)
    loss_.append(epoch_loss)
    # print(f"Epoch {epoch}, Loss: {epoch_loss}")
    epoch += 1