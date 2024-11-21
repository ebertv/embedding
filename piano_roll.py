import numpy as np
import pandas as pd
import music_classes as music

def get_midi_note(note, octave):
    if octave > 0 and octave < 8:
        return music.NOTES.index(note)+((octave-1)*12)+24
    elif octave == 0:
        if note == 'A':
            return 21
        elif note == 'Bb':
            return 22
        if note == 'B':
            return 23
        else:
            raise Exception(f'Invalid note: {note}{octave}')
    elif octave == 8:
        if note == 'C':
            return 108
        else:
            raise Exception(f'Invalid note: {note}{octave}')
    else:
        raise Exception(f'Invalid note: {note}{octave}')

def get_note_name(val):
    if val == 21:
        return 'A', 0
    elif val == 22:
        return 'Bb', 0
    elif val == 23:
        return 'B', 0
    elif val == 108:
        return 'C', 8
    else:
        return music.NOTES[(val-24)%12], int((val-24)/12)+1

def create_piano_roll(note, octave, total_duration, start_time=0, end_time=-1, sampling_rate=44100, samples_per_step=1050, index_with_time=True, start_index=0):
    if note != None and octave != None:
        midi_note = get_midi_note(note, octave)
    else:
        midi_note = None
    total_duration = snap_to_time(total_duration, sampling_rate, samples_per_step)*sampling_rate
    start_time = snap_to_time(start_time, sampling_rate, samples_per_step)*sampling_rate
    start_index = snap_to_time(start_index, sampling_rate, samples_per_step)*sampling_rate
    if end_time == -1:
        end_time = total_duration
    end_time = snap_to_time(end_time, sampling_rate, samples_per_step)*sampling_rate
    piano_roll = np.zeros((int(total_duration/samples_per_step), 88), dtype=np.double)
    if midi_note != None:
        piano_roll[int(start_time/samples_per_step):int(end_time/samples_per_step), midi_note-21] = np.double(1)
    if index_with_time:
        index = [(((i+1)*samples_per_step)+start_index)/sampling_rate for i in range(int(total_duration/samples_per_step))]
    else:
        index = [((i+1)*samples_per_step)+start_index for i in range(int(total_duration/samples_per_step))]
    return pd.DataFrame(piano_roll, columns=get_all_notes(), index=index)

def create_multi_note_piano_roll(notes, total_duration, start_time=0, end_time=-1, sampling_rate=44100, samples_per_step=1050, index_with_time=True, start_index=0):
    if notes != None:
        midi_notes = []
        for note in notes:
            octave = int(note[-1])
            note = note[:-1]
            midi_notes.append(get_midi_note(note, octave))
    else:
        midi_notes = None
    total_duration = snap_to_time(total_duration, sampling_rate, samples_per_step)*sampling_rate
    start_time = snap_to_time(start_time, sampling_rate, samples_per_step)*sampling_rate
    start_index = snap_to_time(start_index, sampling_rate, samples_per_step)*sampling_rate
    if end_time == -1:
        end_time = total_duration
    end_time = snap_to_time(end_time, sampling_rate, samples_per_step)*sampling_rate
    piano_roll = np.zeros((int(total_duration/samples_per_step), 88), dtype=np.double)
    if midi_notes != None:
        for midi_note in midi_notes:
            piano_roll[int(start_time/samples_per_step):int(end_time/samples_per_step), midi_note-21] = np.double(1)
    if index_with_time:
        index = [(((i+1)*samples_per_step)+start_index)/sampling_rate for i in range(int(total_duration/samples_per_step))]
    else:
        index = [((i+1)*samples_per_step)+start_index for i in range(int(total_duration/samples_per_step))]
    return pd.DataFrame(piano_roll, columns=get_all_notes(), index=index)

def snap_to_time(time, sampling_rate, samples_per_step):
    return (int((time*sampling_rate)/samples_per_step)*samples_per_step)/sampling_rate

def get_all_notes():
    notes = []
    for i in range(21, 109):
        notes.append(get_note_name(i)[0])
    for i in range(len(notes)):
        notes[i] += str(((i-3)//12)+1)
    return notes
        
def create_piano_roll_from_note(note, time_signature, tempo, sampling_rate=44100, samples_per_step=1050, index_with_time=True, start_index=0):
    note_duration = note.get_duration(time_signature, tempo)
    return create_piano_roll(note.name, note.octave, note_duration, sampling_rate=sampling_rate, samples_per_step=samples_per_step, index_with_time=index_with_time, start_index=start_index)
    
def create_piano_roll_from_measure(measure, sampling_rate=44100, samples_per_step=1050, index_with_time=True, measure_start_time=0):
    piano_roll = None
    start_time = measure_start_time
    for note in measure.notes:
        note_piano_roll = create_piano_roll_from_note(note, measure.time_signature, measure.tempo, sampling_rate=sampling_rate, samples_per_step=samples_per_step, index_with_time=index_with_time, start_index=start_time)
        if piano_roll is None:
            piano_roll = note_piano_roll
        else:
            piano_roll = pd.concat([piano_roll, note_piano_roll], axis=0)
        start_time += note.get_duration(measure.time_signature, measure.tempo)
    return piano_roll
    
def create_piano_roll_from_part(part, sampling_rate=44100, samples_per_step=1050, index_with_time=True, part_start_time=0):
    piano_roll = None
    start_time = part_start_time
    for measure in part.measures:
        measure_piano_roll = create_piano_roll_from_measure(measure, sampling_rate=sampling_rate, samples_per_step=samples_per_step, index_with_time=index_with_time, measure_start_time=start_time)
        if piano_roll is None:
            piano_roll = measure_piano_roll
        else:
            piano_roll = pd.concat([piano_roll, measure_piano_roll], axis=0)
        start_time += measure.get_duration()
    return piano_roll

def create_piano_roll_from_score(score, sampling_rate=44100, samples_per_step=1050, index_with_time=True):
    piano_roll = None
    start_time = 0
    for part in score.parts:
        part_piano_roll = create_piano_roll_from_part(part, sampling_rate=sampling_rate, samples_per_step=samples_per_step, index_with_time=index_with_time, part_start_time=start_time)
        if piano_roll is None:
            piano_roll = part_piano_roll
        else:
            piano_roll = pd.concat([piano_roll, part_piano_roll], axis=0)
        start_time += part.get_duration()
    return piano_roll

def get_note_names_from_piano_roll_segment(piano_roll):
    notes = []
    for i in range(88):
        if piano_roll[i] == 1:
            note, octave = get_note_name(i+21)
            notes.append(note+str(octave))
    return notes