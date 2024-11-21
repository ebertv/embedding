import os
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import piano_roll as pr
import numpy as np

import contrastive_learner as cl

class SingleNoteDataset(Dataset):

    def __init__(self, note_dir, silence_dir, audio_embedding, symbolic_embedding):
        self.note_dir = note_dir
        self.silence_dir = silence_dir

        self.note_files = os.listdir(note_dir)
        self.silence_files = os.listdir(silence_dir)
        self.note_files = [file for file in self.note_files if file.endswith('.aiff')]
        self.silence_files = [file for file in self.silence_files if file.endswith('.aiff')]

        self.audio_embedding = audio_embedding
        self.symbolic_embedding = symbolic_embedding

    def __len__(self):
        return len(self.note_files) + len(self.silence_files)
    
    def __getitem__(self, idx):
        if idx < len(self.note_files):
            file = self.note_files[idx]
            audio = self.get_audio(os.path.join(self.note_dir, file))
            symbolic = self.get_symbolic(file)
        else:
            file = self.silence_files[idx - len(self.note_files)]
            audio = self.get_audio(os.path.join(self.silence_dir, file))
            symbolic = self.get_symbolic(file)
        
        return audio, symbolic
    
    def get_name(self, idx):
        if idx < len(self.note_files):
            note = self.note_files[idx]
        else:
            note = self.silence_files[idx - len(self.note_files)]
        name = note.split('.')[2]
        dynamic = note.split('.')[1]
        name = f'{name}_{dynamic}'

        if idx >= len(self.note_files):
            name = 'Silence'

        return name
    
    def get_audio(self, file):
        y, sr = librosa.load(file)
        if self.audio_embedding == 'chroma':
            spec = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=441, n_fft=128)
            mid = spec.shape[1]//2
            return spec[:, mid]
    
    def get_symbolic(self, file):
        if 'Silence' in file:
            return np.zeros((128), dtype=np.double)
        
        note = file.split('.')[2]
        octave = int(note[-1])
        note = note[:-1]

        midi_note = pr.get_midi_note(note, octave)

        if self.symbolic_embedding == 'piano_roll':
            piano_roll = np.zeros((128), dtype=np.double)
            piano_roll[midi_note] = 1
            piano_roll = torch.tensor(piano_roll, dtype=torch.float32)
            return piano_roll
        
class SingleNoteDataloader(DataLoader):
    """TODO"""


#start with single notes
note_dir = fr'/gscratch/ark/ebertv/Alignment/Single Piano Notes (Trimmed)/'
silence_dir = fr'/gscratch/ark/ebertv/Alignment/Silence/'

chroma_pianoroll_dataset = SingleNoteDataset(note_dir, silence_dir, 'chroma', 'piano_roll')
dataloader = DataLoader(chroma_pianoroll_dataset, batch_size=1, shuffle=True)

audio_encoder = cl.ChromaAudioEncoder()
symbolic_encoder = cl.PianoRollSymbolicEncoder()

for i, (audio, symbolic) in enumerate(dataloader):
    audio_emb = audio_encoder(audio)
    symbolic_emb = symbolic_encoder(symbolic)

    print(audio_emb.shape, symbolic_emb.shape)
    exit()

    










