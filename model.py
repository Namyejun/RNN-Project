import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BaseModel(nn.Module):
    '''
    input_size -> text vocab size
    '''
    def __init__(self, input_size, output_size, embedding_dim, hidden_dim, num_layers, batch_first):
        super(BaseModel, self).__init__()

        self.num_layers = num_layers
        self.batch_first = batch_first   
        self.hidden_dim = hidden_dim

        """
        TODO: Implement your own model. You can change the model architecture.
        """
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=batch_first)
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_dim, output_size)

    # the size of x in forward is (seq_length, batch_size) if batch_first=False
    def forward(self, input_text, input_lengths):
        batch_size = input_text.size(0) if self.batch_first else input_text.size(1)

        #h_0: (num_layers * num_directions, batch_size, hidden_size)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        text_emb = self.embedding(input_text)
        packed_input = pack_padded_sequence(text_emb, 
                                            input_lengths.cpu(), 
                                            batch_first=True, 
                                            enforce_sorted=False)
        packed_output, _ = self.rnn(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first = True)
        out_forward = output[range(len(output)), input_lengths - 1, :self.hidden_dim]
        out_reverse = output[:, 0, self.hidden_dim:]
        out_reduced = torch.cat((out_forward, out_reverse), 1) #[range(len(output)), input_lengths - 1 + 0 , self.hidden_dim]
        text_fea = self.drop(out_reduced)

        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)

        return text_out