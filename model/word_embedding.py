import torch.nn as nn
import torch


class WordEmbedding(nn.Module):
    def __init__(self, vocabSize, embeddingDim):
        super().__init__()
        self.inBeded = nn.Embedding(vocabSize,embeddingDim)
        self.outEmbed = nn.Linear(embeddingDim, vocabSize)
    
    def forward(self, center):
        emb = self.inBeded(center)
        out = self.outEmbed(emb)
        return out
    
    @torch.no_grad()
    def get_embeddings(self):
        return self.inBeded.weight.detach().cpu()