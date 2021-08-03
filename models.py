import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import PruningModulePT


r"""
The model is composed of the embeddingbag layer and the linear layer.
nn.EmbeddingBag computes the mean of 'bags' of embeddings. The text
entries here have different lengths. nn.EmbeddingBag requires no
padding because the lengths of sentences are saved in offsets.
Therefore, this method is much faster than the original one
with TorchText Iterator and Batch.
Additionally, since it accumulates the average across the embeddings on the fly,
nn.EmbeddingBag can enhance the performance and memory efficiency
to process a sequence of tensors.
"""


class TextSentiment(PruningModulePT):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        linear = nn.Linear
        self.fc_in = linear(embed_dim, hidden_dim)
        self.fc_hidden = linear(hidden_dim, hidden_dim)
        self.fc_out = linear(hidden_dim, num_class)

    def forward(self, text, text_lengths):
        r"""
        Arguments:
            text: 1-D tensor representing a bag of text tensors
            offsets: a list of offsets to delimit the 1-D text tensor
                into the individual sequences.
        """
        x = self.embedding(text)
        x = torch.sum(x, dim=1)
        x = F.tanh(self.fc_in(x))
        x = F.tanh(self.fc_hidden(x))
        x = self.fc_out(x)
        return x


class LSTMSentiment(PruningModulePT):

    # define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout):

        # Constructor
        super().__init__()
        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # lstm layer
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)
        # dense layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        # activation function
        # self.act = nn.Sigmoid()
        self.act = nn.Softmax()

    def forward(self, text, text_lengths):

        # text = [batch size,sent_length]
        embedded = self.embedding(text)
        # embedded = [batch size, sent_len, emb dim]
        # packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # hidden = [batch size, num layers * num directions,hid dim]
        # cell = [batch size, num layers * num directions,hid dim]
        # concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # hidden = [batch size, hid dim * num directions]
        dense_outputs = self.fc(hidden)
        # Final activation function
        outputs = self.act(dense_outputs)

        return dense_outputs
