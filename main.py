import os
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import piano_roll as pr
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import contrastive_learner as cl

class SingleNoteDataset(Dataset):

    def __init__(self, note_dir, silence_dir, audio_embedding, symbolic_embedding, num_positives=10):
        self.note_dir = note_dir
        self.silence_dir = silence_dir

        self.note_files = os.listdir(note_dir)
        self.silence_files = os.listdir(silence_dir)
        self.note_files = [file for file in self.note_files if file.endswith('.aiff')]
        self.silence_files = [file for file in self.silence_files if file.endswith('.aiff')]
        self.note_files = [os.path.join(note_dir, file) for file in self.note_files]
        self.silence_files = [os.path.join(silence_dir, file) for file in self.silence_files]

        self.audio_embedding = audio_embedding
        self.symbolic_embedding = symbolic_embedding

        self.num_positives = num_positives

    def __len__(self):
        return 89
    
    def __getitem__(self, idx):
        name = self.get_name(idx)
        symbolic = self.get_symbolic(idx)
        positive_samples = []
        negative_samples = []
        if idx != 88:
            for i in range(len(self.note_files)):
                file = self.note_files[i]
                if name == file.split('.')[2]:
                    positive_audio = self.get_audio(file)
                    for i in range(positive_audio.shape[1]):
                        positive_samples.append(positive_audio[:, i])
                else:
                    if self.audio_embedding == 'chroma':
                        #for chroma, we don't want octaves to be negative samples (or positive samples)
                        note_name = name[:-1]
                        if note_name == file.split('.')[2][:-1]:
                            # print(f'Not adding {self.note_files[i]}')
                            continue
                    negative_audio = self.get_audio(file)
                    random_sample = np.random.randint(0, negative_audio.shape[1])
                    negative_samples.append(negative_audio[:, random_sample])
            
            random_positives = np.random.choice(len(positive_samples), self.num_positives)
            positive_samples = [positive_samples[i] for i in random_positives]


            random_silence = np.random.randint(0, len(self.silence_files))
            file = self.silence_files[random_silence]
            silence_audio = self.get_audio(file)
            for i in range(silence_audio.shape[1]):
                silence_sample = silence_audio[:, i]
            negative_samples.append(silence_sample)

            

        else:
            random_positives = np.random.choice(len(self.silence_files), self.num_positives)
            for i in range(len(self.silence_files)):
                if i in random_positives:
                    file = self.silence_files[i]
                    positive_audio = self.get_audio(file)
                    random_sample = np.random.randint(0, positive_audio.shape[1])
                    positive_samples.append(positive_audio[:, random_sample])
            for i in range(len(self.note_files)):
                file = self.note_files[i]
                negative_audio = self.get_audio(file)
                random_sample = np.random.randint(0, negative_audio.shape[1])
                negative_samples.append(negative_audio[:, random_sample])

        return name, symbolic, positive_samples, negative_samples
    
    def get_name(self, idx):
        if idx == 88:
            return 'Silence'
        else:
            name, octave = pr.get_note_name(idx+21)
            return f'{name}{octave}'
    
    def get_audio(self, file):
        y, sr = librosa.load(file)
        if self.audio_embedding == 'chroma':
            spec = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=441, n_fft=128)
            return torch.tensor(spec, dtype=torch.float32)
    
    def get_symbolic(self, idx):
        if idx == 88:
            return torch.zeros((128), dtype=torch.float32)
        else:
            piano_roll = torch.zeros((128), dtype=torch.float32)
            piano_roll[idx+21] = 1
            return piano_roll
    

def plot_embeddings(symbolic_embeddings, audio_embeddings, symbolic_names, audio_names, title):
    tsne = TSNE(n_components=2)
    symbolic_embeddings = np.array([embedding.detach().numpy() for embedding in symbolic_embeddings])
    audio_embeddings = np.array([embedding.detach().numpy() for embedding in audio_embeddings])
    all_embeddings = np.concatenate((symbolic_embeddings, audio_embeddings), axis=0)
    all_embeddings = all_embeddings.reshape((all_embeddings.shape[0], -1))
    all_embeddings = tsne.fit_transform(all_embeddings)

    fig, ax = plt.subplots(figsize=(10, 10))

    all_names = symbolic_names + audio_names

    unique_names = list(set(all_names))
    #make colormap so each name gets a unique color
    cmap = plt.get_cmap()
    colors = [unique_names.index(name) for name in all_names]
    #evenly space colors
    colors = [color/len(unique_names) for color in colors]
    colors = [cmap(color) for color in colors]
    

    for i, name in enumerate(all_names):
        color = colors[i]
        ax.scatter(all_embeddings[i, 0], all_embeddings[i, 1], color=color)
        #label only the symbolic embeddings
        if i < len(symbolic_embeddings):
            ax.annotate(name, (all_embeddings[i, 0], all_embeddings[i, 1]), fontsize=8)


    plt.title(title)
    plt.savefig(f'{title}.png')


#start with single notes
note_dir = fr'/gscratch/ark/ebertv/Alignment/Single Piano Notes (Trimmed)/'
silence_dir = fr'/gscratch/ark/ebertv/Alignment/Silence/'

chroma_pianoroll_dataset = SingleNoteDataset(note_dir, silence_dir, 'chroma', 'piano_roll', num_positives=10)
dataloader = DataLoader(chroma_pianoroll_dataset, batch_size=1, shuffle=True)

audio_encoder = cl.ChromaAudioEncoder()
symbolic_encoder = cl.PianoRollSymbolicEncoder()

learner = cl.ContrastiveLearner(audio_encoder, symbolic_encoder, 0.1)
optimizer = torch.optim.Adam(learner.parameters(), lr=0.1)

symbolic_embeddings = []
symbolic_names = []

audio_embeddings = []
audio_names = []
for name, symbolic, positive_samples, negative_samples in tqdm(dataloader):
    name = name[0]

    symbolic_embedding = symbolic_encoder(symbolic)
    symbolic_embeddings.append(symbolic_embedding)
    symbolic_names.append(name)


    for positive in positive_samples:
        positive_embedding = audio_encoder(positive)
        audio_embeddings.append(positive_embedding)
        audio_names.append(name)
    for negative in negative_samples:
        negative_embedding = audio_encoder(negative)
        audio_embeddings.append(negative_embedding)
        audio_names.append(name)

plot_embeddings(symbolic_embeddings, audio_embeddings, symbolic_names, audio_names, 'Before_Training')


for epoch in range(10):
    print(f'Epoch {epoch}')
    for name, symbolic, positive_samples, negative_samples in dataloader:
        loss = learner.train(symbolic, positive_samples, negative_samples)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'\tLoss: {loss.item()} after {name[0]}                     \r', end='', flush=True)
    print(f'\tLoss: {loss.item()}                      ', flush=True)
     
    plot_embeddings(symbolic_embeddings, audio_embeddings, symbolic_names, audio_names, f'After_Epoch_{epoch}')
        
        











