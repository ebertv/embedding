import torch
from torch import nn

class ContrastiveLearner(nn.Module):

    def __init__(self, audio_encoder, symbolic_encoder, temperature=0.1):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.symbolic_encoder = symbolic_encoder
        self.temperature = temperature
        self.proj_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, symbolic, positives, negatives):
        symbolic_embedding = self.symbolic_encoder(symbolic)
        positive_embeddings = []
        for positive in positives:
            positive_embeddings.append(self.audio_encoder(positive))
        negative_embeddings = []
        for negative in negatives:
            negative_embeddings.append(self.audio_encoder(negative))

        positive_embeddings = torch.stack(positive_embeddings)
        negative_embeddings = torch.stack(negative_embeddings)

        return symbolic_embedding, positive_embeddings, negative_embeddings
    
    def contrastive_loss(self, symbolic_embedding, positive_embeddings, negative_embeddings):
        positive_embeddings = positive_embeddings.unbind()
        negative_embeddings = negative_embeddings.unbind()

        for positive in positive_embeddings:
            positive = self.proj_head(positive)
        for negative in negative_embeddings:
            negative = self.proj_head(negative)

        positive_similarities = []
        negative_similarities = []
        for positive in positive_embeddings:
            positive_similarities.append(self.sigma(symbolic_embedding, positive))
        for negative in negative_embeddings:
            negative_similarities.append(self.sigma(symbolic_embedding, negative))

        positive_similarities = torch.stack(positive_similarities)
        negative_similarities = torch.stack(negative_similarities)

        loss = 0
        for i in range(len(positive_embeddings)):
            loss += torch.log(positive_similarities[i] / torch.sum(negative_similarities))

        frac = 1 / len(positive_embeddings)
        return -frac * loss
        
        
    def cosine_similarity(self, x, y):
        return torch.nn.functional.cosine_similarity(x, y, dim=-1)
    
    def sigma(self, x, y):
        return torch.exp(self.cosine_similarity(x, y) / self.temperature)
    
    def train(self, symbolic, positives, negatives):
        symbolic_embedding, positive_embeddings, negative_embeddings = self.forward(symbolic, positives, negatives)
        loss = self.contrastive_loss(symbolic_embedding, positive_embeddings, negative_embeddings)
        return loss
    

# start with basic audio feature: Chroma and basic symbolic feature: Piano Roll
class ChromaAudioEncoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            #input is 1X12
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
        )

    def forward(self, x):
        x = self.linear(x)
        return x.view(x.size(0), -1)
    
class PianoRollSymbolicEncoder(nn.Module):
        
        def __init__(self):
            super().__init__()
            self.linear = nn.Sequential(
                #input is 1X128
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
            )
    
        def forward(self, x):
            x = self.linear(x)
            return x.view(x.size(0), -1)


