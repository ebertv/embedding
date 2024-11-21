import torch
from torch import nn

# class ContrastiveLearner(nn.Module):

#     def __init__(self, audio_encoder, symbolic_encoder, temperature=0.1):
#         super().__init__()
#         self.audio_encoder = audio_encoder
#         self.symbolic_encoder = symbolic_encoder
#         self.temperature = temperature
#         self.proj_head = nn.Sequential(
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512)
#         )

#     def forward(self, audio, symbolic):
#         audio_emb = self.audio_encoder(audio)
#         symbolic_emb = self.symbolic_encoder(symbolic)
#         audio_emb = self.proj_head(audio_emb)
#         symbolic_emb = self.proj_head(symbolic_emb)
#         audio_emb = nn.functional.normalize(audio_emb, dim=1)
#         symbolic_emb = nn.functional.normalize(symbolic_emb, dim=1)
#         return audio_emb, symbolic_emb
    

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


