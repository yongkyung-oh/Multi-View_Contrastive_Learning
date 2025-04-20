import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * 
                             (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, hidden_dim]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class InteractionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(InteractionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, ht, hd, hf):
        N, L, D = ht.size()
        h = torch.stack([ht, hd, hf], dim=2)  # [N, L, 3, D]
        h = h.permute(0, 2, 1, 3).contiguous().view(N * 3, L, D)  # [N*3, L, D]
    
        # Apply attention across the time dimension
        attn_output, _ = self.multihead_attn(h, h, h)
        output = self.norm(h + attn_output)
        output = output.view(N, 3, L, D).permute(0, 2, 1, 3)  # [N, L, 3, D]
    
        ht_i, hd_i, hf_i = output[:, :, 0, :], output[:, :, 1, :], output[:, :, 2, :]
        return ht_i, hd_i, hf_i

        
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        ## initial encoding        
        self.positional_encoding = PositionalEncoding(args.num_embedding, args.dropout)

        self.input_layer_t = nn.Linear(args.num_feature, args.num_embedding)
        self.encoder_layers_t = nn.TransformerEncoderLayer(d_model=args.num_embedding, dim_feedforward=args.num_hidden, nhead=args.num_head, dropout=args.dropout, batch_first=True)
        self.transformer_encoder_t = nn.TransformerEncoder(self.encoder_layers_t, args.num_layers)        

        self.input_layer_d = nn.Linear(args.num_feature, args.num_embedding)
        self.encoder_layers_d = nn.TransformerEncoderLayer(d_model=args.num_embedding, dim_feedforward=args.num_hidden, nhead=args.num_head, dropout=args.dropout, batch_first=True)
        self.transformer_encoder_d = nn.TransformerEncoder(self.encoder_layers_d, args.num_layers)

        self.input_layer_f = nn.Linear(args.num_feature, args.num_embedding)
        self.encoder_layers_f = nn.TransformerEncoderLayer(d_model=args.num_embedding, dim_feedforward=args.num_hidden, nhead=args.num_head, dropout=args.dropout, batch_first=True)
        self.transformer_encoder_f = nn.TransformerEncoder(self.encoder_layers_f, args.num_layers)

        ## interaction
        self.interaction_layer = InteractionLayer(args.num_embedding, args.num_head)
        
        ## output
        self.output_layer_t = nn.Sequential(
            nn.Linear(args.num_embedding*2, args.num_hidden),
            nn.LayerNorm(args.num_hidden), nn.ReLU(), nn.Dropout(args.dropout),
            nn.Linear(args.num_hidden, args.num_hidden)
        )
        self.output_layer_d = nn.Sequential(
            nn.Linear(args.num_embedding*2, args.num_hidden),
            nn.LayerNorm(args.num_hidden), nn.ReLU(), nn.Dropout(args.dropout),
            nn.Linear(args.num_hidden, args.num_hidden)
        )
        self.output_layer_f = nn.Sequential(
            nn.Linear(args.num_embedding*2, args.num_hidden),
            nn.LayerNorm(args.num_hidden), nn.ReLU(), nn.Dropout(args.dropout),
            nn.Linear(args.num_hidden, args.num_hidden)
        )
    
    def forward(self, xt, dx, xf):
        xt, dx, xf = torch.nan_to_num(xt), torch.nan_to_num(dx), torch.nan_to_num(xf)

        # input -> hidden
        ht = self.input_layer_t(xt)
        ht = self.positional_encoding(ht)
        ht = self.transformer_encoder_t(ht)
        
        hd = self.input_layer_d(dx)
        hd = self.positional_encoding(hd)
        hd = self.transformer_encoder_d(hd)

        hf = self.input_layer_f(xf)
        hf = self.positional_encoding(hf)
        hf = self.transformer_encoder_f(hf)

        # interaction
        ht_i, hd_i, hf_i = self.interaction_layer(ht, hd, hf)
        
        # output layers
        zt = self.output_layer_t(torch.cat([ht.mean(dim=1), ht_i.mean(dim=1)], dim=-1))
        zd = self.output_layer_d(torch.cat([hd.mean(dim=1), hd_i.mean(dim=1)], dim=-1))
        zf = self.output_layer_f(torch.cat([hf.mean(dim=1), hf_i.mean(dim=1)], dim=-1))

        return ht, hd, hf, zt, zd, zf


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        q = self.query(x)  # (batch_size, 3, hidden_dim)
        k = self.key(x)    # (batch_size, 3, hidden_dim)
        v = self.value(x)  # (batch_size, 3, hidden_dim)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)  # (batch_size, 3, 3)
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, 3, 3)
        output = torch.matmul(attention_weights, v)  # (batch_size, 3, hidden_dim)
        return output, attention_weights
        

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.args = args
        if self.args.feature == 'latent':
            if args.loss_type == 'ALL':
                self.self_attention = SelfAttention(args.num_hidden)

        elif self.args.feature == 'hidden':
            ## interaction
            self.interaction_layer = InteractionLayer(args.num_embedding, args.num_head)
            
            ## output
            self.output_layer_t = nn.Sequential(
                nn.Linear(args.num_embedding*2, args.num_hidden),
                nn.LayerNorm(args.num_hidden), nn.ReLU(), nn.Dropout(args.dropout),
                nn.Linear(args.num_hidden, args.num_hidden)
            )
            self.output_layer_d = nn.Sequential(
                nn.Linear(args.num_embedding*2, args.num_hidden),
                nn.LayerNorm(args.num_hidden), nn.ReLU(), nn.Dropout(args.dropout),
                nn.Linear(args.num_hidden, args.num_hidden)
            )
            self.output_layer_f = nn.Sequential(
                nn.Linear(args.num_embedding*2, args.num_hidden),
                nn.LayerNorm(args.num_hidden), nn.ReLU(), nn.Dropout(args.dropout),
                nn.Linear(args.num_hidden, args.num_hidden)
            )

        self.fc = nn.Linear(len(args.loss_type)*args.num_hidden, args.num_target)
        
        self.fc.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, xt, dx, xf):
        xt, dx, xf = torch.nan_to_num(xt), torch.nan_to_num(dx), torch.nan_to_num(xf)

        if self.args.feature == 'latent':
            zt, zd, zf = xt, dx, xf

            if self.args.loss_type == 'ALL':
                stacked_emb = torch.stack([zt, zd, zf], dim=1) # [batch_size, 3, hidden_dim]
                emb = self.self_attention(stacked_emb)[0] + stacked_emb # [batch_size, 3, hidden_dim]
                zt, zd, zf = emb[:, 0, :], emb[:, 1, :], emb[:, 2, :]
        
        elif self.args.feature == 'hidden':
            ht, hd, hf = xt, dx, xf
    
            # interaction
            if self.args.loss_type == 'ALL':
                ht_i, hd_i, hf_i = self.interaction_layer(ht, hd, hf)
            else:
                ht_i, hd_i, hf_i = ht, hd, hf
            
            # output layers
            zt = self.output_layer_t(torch.cat([ht.mean(dim=1), ht_i.mean(dim=1)], dim=-1))
            zd = self.output_layer_d(torch.cat([hd.mean(dim=1), hd_i.mean(dim=1)], dim=-1))
            zf = self.output_layer_f(torch.cat([hf.mean(dim=1), hf_i.mean(dim=1)], dim=-1))
        
        if self.args.loss_type == 'ALL':
            emb = torch.cat([zt, zd, zf], dim=-1)
        else:
            emb_list = []
            # append embeddings based on the selected loss type
            if ('T' in self.args.loss_type):
                emb_list.append(zt)
            if ('D' in self.args.loss_type):
                emb_list.append(zd)
            if ('F' in self.args.loss_type):
                emb_list.append(zf)
            
            emb = torch.cat(emb_list, dim=-1)
            
        emb = emb.reshape(emb.shape[0], -1)
        return self.fc(emb)
