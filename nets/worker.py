import torch

class Encoder(torch.nn.Module):

    def __init__(self, node_dim, embed_dim=512, n_heads=8, n_layers=6, feed_forward_hidden_dim=2048):
        super(Encoder, self).__init__()

        self.init_embed = torch.nn.Linear(node_dim, embed_dim)
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=feed_forward_hidden_dim, batch_first=True),
            num_layers=n_layers
        )


    def forward(self, x):
        h = self.encoder(self.init_embed(x))
        return (h, torch.mean(h, dim=1))



#class Decoder(torch.nn.Module)

src = torch.rand((10, 100, 2))
m = Encoder(2)
l = m(src)
print(l[0].shape)
print(l[1].shape)
