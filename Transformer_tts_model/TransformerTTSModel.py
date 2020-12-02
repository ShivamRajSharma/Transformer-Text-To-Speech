import torch 
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dims, heads):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.embed_dims = embed_dims
        self.depth = embed_dims//heads
        
        self.query = nn.Linear(self.depth, self.depth)
        self.key = nn.Linear(self.depth, self.depth)
        self.value = nn.Linear(self.depth, self.depth)

        self.fc_out = nn.Linear(self.depth*self.heads*2, self.embed_dims)
    
    def forward(self, query, key, value, mask):
        batch, q_len, k_len, v_len = query.shape[0], query.shape[1], key.shape[1], value.shape[1]

        query = query.reshape(batch, q_len, self.heads, self.depth)
        key = key.reshape(batch, k_len, self.heads, self.depth)
        value = value.reshape(batch, v_len, self.heads, self.depth)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        energy = torch.einsum('bqhd, bkhd -> bhqk', [query, key])
        
        if mask is not None:
            energy.masked_fill(mask==0, float("-1e20"))

        energy =  torch.softmax((energy/((self.depth**1/2))), dim=-1)

        out = torch.einsum('bhqv, bvhd -> bqhd', [energy, value])

        out = out.reshape(batch, q_len, self.heads*self.depth)
        query = query.reshape(batch, q_len, self.heads*self.depth)

        out = torch.cat([query, out], dim=-1)
        out = self.fc_out(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dims, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.hidden_dims = hidden_dims
        self.heads = heads
        self.multihead_attention = SelfAttention(hidden_dims, heads)
        self.feed_forward = nn.Sequential(
            nn.Conv1d(hidden_dims, hidden_dims*forward_expansion, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dims*forward_expansion, hidden_dims, kernel_size=1)
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dims)
        self.layer_norm2 = nn.LayerNorm(hidden_dims)
    
    def forward(self, query, key, value, mask):
        attention_out = self.multihead_attention(query, key, value, mask)
        add = self.dropout(self.layer_norm1(attention_out + query))
        ffn_in = add.transpose(1, 2)
        ffn_out = self.feed_forward(ffn_in)
        ffn_out = ffn_out.transpose(1, 2)
        out = self.dropout(self.layer_norm2(ffn_out + add)) 
        return out


class EncoderPreNet(nn.Module):
    def __init__(self, embed_dims, hidden_dims, dropout):
        super(EncoderPreNet, self).__init__()
        self.conv1 = nn.Conv1d(
            embed_dims,
            hidden_dims,
            kernel_size=5, 
            padding=2
        )

        self.conv2 = nn.Conv1d(
            hidden_dims,
            hidden_dims,
            kernel_size=5, 
            padding=2
        )

        self.conv3 = nn.Conv1d(
            hidden_dims,
            hidden_dims,
            kernel_size=5, 
            padding=2
        )

        self.batch_norm1 = nn.BatchNorm1d(hidden_dims)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dims)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dims)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.fc_out = nn.Linear(hidden_dims, hidden_dims)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.dropout1(torch.relu(self.batch_norm1(self.conv1(x))))
        x = self.dropout2(torch.relu(self.batch_norm2(self.conv2(x))))
        x = self.dropout3(torch.relu(self.batch_norm3(self.conv3(x))))
        x = x.transpose(1, 2)
        x = self.fc_out(x)
        return x
    
class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dims,
        hidden_dims,
        max_len,
        heads,
        forward_expansion,
        num_layers, 
        dropout
    ):
        super(Encoder, self).__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dims)
        self.positional_embed = nn.Parameter(torch.zeros(1, max_len, hidden_dims))
        self.prenet = EncoderPreNet(embed_dims, hidden_dims, dropout)
        self.dropout = nn.Dropout(dropout)
        self.attention_layers = nn.Sequential(
            *[
                TransformerBlock(
                    hidden_dims, 
                    heads, 
                    dropout, 
                    forward_expansion
                )
                for _ in range(num_layers)
            ]
        )
    
    def forward(self, x, mask=None):
        seq_len = x.shape[1]
        token_embed = self.token_embed(x)
        positional_embed = self.positional_embed[:, :seq_len, :]
        x = self.prenet(token_embed)
        x += positional_embed
        x = self.dropout(x)
        for layer in self.attention_layers:
            x = layer(x, x, x, mask)
        return x


class DecoderPreNet(nn.Module):
    def __init__(self, mel_dims, hidden_dims, dropout):
        super(DecoderPreNet, self).__init__()
        self.fc_out = nn.Sequential(
            nn.Linear(mel_dims, hidden_dims),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):

        return self.fc_out(x)


class PostNet(nn.Module):
    def __init__(self, mel_dims, hidden_dims, dropout):
        #causal padding -> padding = (kernel_size - 1) x dilation
        #kernel_size = 5 -> padding = 4
        #Exclude the last padding_size output as we want only left padded output
        super(PostNet, self).__init__()
        self.conv1 = nn.Conv1d(mel_dims, hidden_dims, kernel_size=5, padding=4)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dims)
        self.dropout1 = nn.Dropout(dropout)
        self.conv_list = nn.Sequential(
            *[
                nn.Conv1d(hidden_dims, hidden_dims, kernel_size=5, padding=4)
                for _ in range(3)
            ]
        )

        self.batch_norm_list = nn.Sequential(
            *[
                nn.BatchNorm1d(hidden_dims)
                for _ in range(3)
            ]
        )
        
        self.dropout_list = nn.Sequential(
            *[
                nn.Dropout(dropout)
                for _ in range(3)
            ]
        )

        self.conv5 = nn.Conv1d(hidden_dims, mel_dims, kernel_size=5, padding=4)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.dropout1(torch.tanh(self.batch_norm1(self.conv1(x)[:, :, :-4])))
        for dropout, batchnorm, conv in zip(self.dropout_list, self.batch_norm_list, self.conv_list):
            x = dropout(torch.tanh(batchnorm(conv(x)[:, :, :-4])))
        out = self.conv5(x)[:, :, :-4]
        out = out.transpose(1, 2)
        return out


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dims,
        heads,
        forward_expansion,
        dropout
    ):
        super(DecoderBlock, self).__init__()
        self.causal_masked_attention = SelfAttention(embed_dims, heads)
        self.attention_layer = TransformerBlock(
            embed_dims, 
            heads, 
            dropout, 
            forward_expansion
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dims)
    
    def forward(self, query, key, value, src_mask, causal_mask):
        causal_masked_attention = self.causal_masked_attention(query, query, query, causal_mask)
        query = self.dropout(self.layer_norm(causal_masked_attention + query))
        out = self.attention_layer(query, key, value, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        mel_dims,
        hidden_dims,
        heads,
        max_len,
        num_layers,
        forward_expansion,
        dropout
    ):
        super(Decoder, self).__init__()
        self.positional_embed = nn.Parameter(torch.zeros(1, max_len, hidden_dims))
        self.prenet = DecoderPreNet(mel_dims, hidden_dims, dropout)
        self.attention_layers = nn.Sequential(
            *[
                DecoderBlock(
                    hidden_dims, 
                    heads, 
                    forward_expansion, 
                    dropout
                )
                for _  in range(num_layers)
            ]
        )
        self.mel_linear = nn.Linear(hidden_dims, mel_dims)
        self.stop_linear = nn.Linear(hidden_dims, 1)
        self.postnet = PostNet(mel_dims, hidden_dims, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, mel, encoder_output, src_mask, casual_mask):
        seq_len = mel.shape[1]
        prenet_out = self.prenet(mel)
        x = self.dropout(prenet_out + self.positional_embed[:, :seq_len, :])

        for layer in self.attention_layers:
            x = layer(x, encoder_output, encoder_output, src_mask, casual_mask)

        stop_linear = self.stop_linear(x)

        mel_linear = self.mel_linear(x)

        postnet = self.postnet(mel_linear)

        out = postnet + mel_linear

        return out, mel_linear, stop_linear


class TransformerTTS(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dims,
        hidden_dims, 
        heads,
        forward_expansion,
        num_layers,
        dropout,
        mel_dims,
        max_len,
        pad_idx
    ):
        super(TransformerTTS, self).__init__()
        self.encoder = Encoder(
            vocab_size,
            embed_dims,
            hidden_dims,
            max_len,
            heads,
            forward_expansion,
            num_layers,
            dropout
        )
        
        self.decoder = Decoder(
            mel_dims,
            hidden_dims,
            heads,
            max_len,
            num_layers,
            forward_expansion,
            dropout
        )

        self.pad_idx = pad_idx

    def target_mask(self, mel, mel_mask):
        seq_len = mel.shape[1]
        pad_mask = (mel_mask != self.pad_idx).unsqueeze(1).unsqueeze(3)
        causal_mask = torch.tril(torch.ones((1, seq_len, seq_len))).unsqueeze(1)
        return pad_mask, causal_mask
    
    def input_mask(self, x):
        mask = (x != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return mask
    
    def forward(self, text_idx, mel, mel_mask):
        input_pad_mask = self.input_mask(text_idx)
        target_pad_mask, causal_mask = self.target_mask(mel, mel_mask)
        encoder_out = self.encoder(text_idx, input_pad_mask)
        mel_postout, mel_linear, stop_linear = self.decoder(mel, encoder_out, target_pad_mask, causal_mask)
        return mel_postout, mel_linear, stop_linear


if __name__ == "__main__":
    a = torch.randint(0, 30, (4, 60))
    mel = torch.randn(4, 128, 80)
    mask = torch.ones((4, 128))
    model = TransformerTTS(
        vocab_size=30,
        embed_dims=512,
        hidden_dims=256, 
        heads=4,
        forward_expansion=4,
        num_layers=6,
        dropout=0.1,
        mel_dims=80,
        max_len=512,
        pad_idx=0
    )
    x, y, z = model(a, mel, mask)
    print(x.shape, y.shape, z.shape)