#!/usr/bin/env python3
import sys
import warnings
import scipy.stats as st
import scipy.spatial as ss

import librosa
import numpy as np
import librosa.feature as lf
import os


class Song:
    def __init__(self, name):
        self.name = name
        self.features = dict()

    def add_feature(self, tp, feature):
        self.features[tp] = [feature]

    def add_stat(self, tp, mean):
        self.features[tp].append(mean)

    def get_features(self):
        return self.features


def norm(array, nc):
    for i in range(nc):
        lb = np.amin(array[:, i])
        ub = np.amax(array[:, i])
        array[:, i] = (array[:, i] - lb) / (ub - lb)
        if ub == lb:
            array[:, i] = 0


def lin_norm(arr):
    lb = np.amin(arr)
    ub = np.amax(arr)
    print(f'lb:{lb}\tub:{ub}')
    arr = (arr - lb) / (ub - lb)
    if ub == lb:
        arr = np.zeros(arr.shape)
    return arr


def Q2():
    # Read file
    filename = '../Features/top100_features.csv'
    top100 = np.genfromtxt(filename, delimiter=',')
    nl, nc = top100.shape

    # Trim NaN values
    top100 = top100[1:, 1:(nc - 1)]
    nl, nc = top100.shape

    # Normalize matrix to [0, 1]
    top100_norm = np.copy(top100)
    norm(top100_norm, nc)

    # Write values into file
    filename = '../Features/top100_features_normalized.csv'
    np.savetxt(filename, top100_norm, fmt="%lf", delimiter=',')

    # Verify file
    top100_norm_read = np.genfromtxt(filename, delimiter=',')
    print(np.allclose(top100_norm, top100_norm_read, atol=.1))

    # Get features from song of Q1
    filename = '../Dataset/Q1/MT0000040632.mp3'
    testSong = Song('MT0000040632.mp3')

    fname = "../Features/python_features.csv"

    with open(fname, 'w') as f:
        pass

    fapp = open(fname, 'a')

    # calculateFeatures(filename, testSong)
    calculateFeatures(filename, testSong)

    # Write header
    weights = {'mfcc': 13, 'spec_centroid': 1, 'spec_bandwidth': 1, 'spec_contrast': 7, 'spec_flatness': 1, 'rolloff': 1, 'rms': 1, 'f0': 1, 'zero_crossing_rate': 1}
    fapp.write('Name, ')
    st = ""
    for k, v in testSong.get_features().items():
        st += f'{k},'
        print(v[0].shape)
        for j in range(weights[k]*7-1):
            st += '-,'
    fapp.write(f'{st[:-1]}\n')

    # Get to all songs
    file_count = 0
    for i in range(1, 5):
        folder = f'../Dataset/Q{i}'
        files = os.listdir(folder)
        for file in files:
            fsplit = file.split('.')
            if fsplit[len(fsplit)-1] != 'mp3':
                continue
            file_count += 1

            outputSong = Song(file)
            file = f'{folder}/{file}'
            calculateFeatures(file, outputSong)

            fapp.write(f'{outputSong.name}')
            for k, v in outputSong.get_features().items():
                fapp.write(',')
                np.savetxt(fapp, v, fmt="%lf", delimiter=',', newline='')
                print(f'count: {file_count}\tdoing {k}')
            fapp.write('\n')
    fapp.close()


def calculateFeatures(filename, song):
    sampleRate = 22050
    useMono = True
    warnings.filterwarnings("ignore")
    F0_minFreq = 20  # minimum audible frequency
    F0_maxFreq = 11025  # nyquist/2
    mfcc_dim = 13
    spContrast_dim = 7  # (6+1) bands

    inFile = librosa.load(filename, sr=sampleRate, mono=useMono)[0]

    # Calculates mfcc
    mfcc = lf.mfcc(inFile, n_mfcc=mfcc_dim)
    nl, nc = mfcc.shape
    song.add_feature('mfcc', calc_stats(mfcc, nl))

    # Calculates for spectral centroid
    cent = lf.spectral_centroid(inFile)
    nl, nc = cent.shape
    song.add_feature('spec_centroid', calc_stats(cent, nl))

    # Calculates for spectral bandwidth
    bandwidth = lf.spectral_bandwidth(inFile)
    nl, nc = bandwidth.shape
    song.add_feature('spec_bandwidth', calc_stats(bandwidth, nl))

    # Calculates for spectral contrast
    contrast = lf.spectral_contrast(inFile, n_bands=spContrast_dim - 1)
    nl, nc = contrast.shape
    song.add_feature('spec_contrast', calc_stats(contrast, nl))

    # Calculates spectral flatness
    flatness = lf.spectral_flatness(inFile)
    nl, nc = flatness.shape
    song.add_feature('spec_flatness', calc_stats(flatness, nl))

    # Calculates spectral rolloff
    rolloff = lf.spectral_rolloff(inFile)
    nl, nc = rolloff.shape
    song.add_feature('rolloff', calc_stats(rolloff, nl))

    # Calculates rms
    rms = lf.rms(inFile)
    nl, nc = rms.shape
    song.add_feature('rms', calc_stats(rms, nl))

    # Calculates F0
    f0 = librosa.yin(inFile, F0_minFreq, F0_maxFreq).reshape(1, -1)
    nl = f0.shape[0]
    song.add_feature('f0', calc_stats(f0, nl))

    # Calculates zero crossing rate
    zcr = lf.zero_crossing_rate(inFile)
    nl, nc = zcr.shape
    song.add_feature('zero_crossing_rate', calc_stats(zcr, nl))


def calc_stats(arr, nl):
    total = np.zeros((nl, 7))
    print(f'total : {total.shape}')
    for i in range(nl):
        mean = np.mean(arr[i, :])
        sk = st.skew(arr[i, :])
        kurt = st.kurtosis(arr[i, :])
        std = np.std(arr[i, :])
        median = np.median(arr[i, :])
        am = np.amax(arr[i, :])
        ami = np.amin(arr[i, :])
        total[i, :] = np.array([mean, sk, kurt, std, median, am, ami])
    total = total.flatten()
    return lin_norm(total)


def euclidean_distance(a, b):
    return ss.distance.euclidean(a, b)


def euclidean_code(a, b):
    ab = a - b
    return np.sqrt(np.dot(ab.T, ab))


def cosine_distance(a, b):
    return ss.distance.cosine(a, b)


def cosine_code(a, b):
    return np.dot(a, b) / (np.align.norm(a) * np.align.norm(b))


def manhattan_distance(a, b):
    return ss.distance.cityblock(a, b)


def manhattan_code(a, b):
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))


def Q3():
    euclid_lib = np.zeros((900, 900))
    euclid_feat = np.zeros((900, 900))
    manhat_lib = np.zeros((900, 900))
    manhat_feat = np.zeros((900, 900))
    cosine_lib = np.zeros((900, 900))
    cosine_feat = np.zeros((900, 900))

    filename = '../Features/top100_features_normalized.csv'
    top100 = np.genfromtxt(filename, delimiter=',')
    top100 = top100

    filename = '../Features/python_features.csv'
    lib_feat = np.genfromtxt(filename, delimiter=',')
    nl, nc = lib_feat.shape
    lib_feat = lib_feat[1:, 1:(nc - 1)]

    for i in range(900):
        for j in range(i, 900):
            print(f'doing {i} {j}')
            euclid_lib[i, j] = euclidean_distance(top100[i], top100[j])
            euclid_lib[j, i] = euclid_lib[i, j]
            euclid_feat[i, j] = euclidean_distance(lib_feat[i], lib_feat[j])
            euclid_feat[j, i] = euclid_feat[i, j]
            manhat_lib[i, j] = manhattan_distance(top100[i], top100[j])
            manhat_lib[j, i] = manhat_lib[i, j]
            manhat_feat[i, j] = manhattan_distance(lib_feat[i], lib_feat[j])
            manhat_feat[j, i] = manhat_feat[i, j]
            cosine_lib[i, j] = cosine_distance(top100[i], top100[j])
            cosine_lib[j, i] = cosine_lib[i, j]
            cosine_feat[i, j] = cosine_distance(lib_feat[i], lib_feat[j])
            cosine_feat[j, i] = cosine_feat[i, j]

    kv = {'euclid_lib': euclid_lib, 'euclid_feat': euclid_feat, 'manhat_lib': manhat_lib,
          'manhat_feat': manhat_feat, 'cosine_lib': cosine_lib, 'cosine_feat': cosine_feat}

    for k, v in kv.items():
        filename = f'../Features/{k}.csv'
        np.savetxt(filename, v, fmt="%lf", delimiter=',')


def get_pos_file(features, file, tp):
    if tp == 0:
        file = file[:-4]
    if tp > 0:
        file = f'\"{file}\"'
    if file in features:
        return np.where(features == file)[0][0]
    return -1


def get_on_index(i, tp):
    if tp == 0:
        filename = '../Features/top100_features.csv'
    else:
        filename = "../Features/python_features.csv"
    features = np.genfromtxt(filename, delimiter=',', dtype='<U16')
    features = features[1:, 0]
    return features[i]


def Q3_2(tp):
    path = '../Queries'

    if tp == 0:
        filename = '../Features/top100_features.csv'
    else:
        filename = "../Features/python_features.csv"

    features = np.genfromtxt(filename, delimiter=',', dtype='<U16')
    names = features[1:, 0]

    for file in os.listdir(path):
        pos_in = get_pos_file(names, file, tp)

        to_read = '../Features/euclid_lib.csv'
        to_save = np.genfromtxt(to_read, delimiter=',')
        chosen_euclid = to_save[pos_in, :]
        sorted_euclid = np.sort(chosen_euclid)

        to_read = '../Features/euclid_feat.csv'
        to_save = np.genfromtxt(to_read, delimiter=',')
        chosen_efeat = to_save[pos_in, :]
        sorted_e_feat = np.sort(chosen_efeat)

        to_read = '../Features/manhat_lib.csv'
        to_save = np.genfromtxt(to_read, delimiter=',')
        chosen_manhat = to_save[pos_in, :]
        sorted_manhat = np.sort(chosen_manhat)

        to_read = '../Features/manhat_feat.csv'
        to_save = np.genfromtxt(to_read, delimiter=',')
        chosen_emanhat = to_save[pos_in, :]
        sorted_e_manhat = np.sort(chosen_emanhat)

        to_read = '../Features/cosine_lib.csv'
        to_save = np.genfromtxt(to_read, delimiter=',')
        chosen_cosine = to_save[pos_in, :]
        sorted_cosine = np.sort(chosen_cosine)

        to_read = '../Features/cosine_feat.csv'
        to_save = np.genfromtxt(to_read, delimiter=',')
        chosen_ecosine = to_save[pos_in, :]
        sorted_e_cosine = np.sort(chosen_ecosine)

        print(f'Music {file}:')

        print('euclid lib')
        for i in range(6):
            x = np.where(chosen_euclid == sorted_euclid[i])[0]
            print(f'{i + 1} Place: ', end='')
            for el in x:
                print(f'{get_on_index(el, 0)}', end=' ')
            print()

        print('\neuclid feat')
        for i in range(6):
            x = np.where(chosen_efeat == sorted_e_feat[i])[0]
            print(f'{i + 1} Place: ', end='')
            for el in x:
                print(f'{get_on_index(el, 0)}', end=' ')
            print()

        print('\nmanhattan lib')
        for i in range(6):
            x = np.where(chosen_manhat == sorted_manhat[i])[0]
            print(f'{i + 1} Place: ', end='')
            for el in x:
                print(f'{get_on_index(el, 0)}', end=' ')
            print()

        print('\nmanhattan feat')
        for i in range(6):
            x = np.where(chosen_emanhat == sorted_e_manhat[i])[0]
            print(f'{i + 1} Place: ', end='')
            for el in x:
                print(f'{get_on_index(el, 0)}', end=' ')
            print()

        print('\ncosine lib')
        for i in range(6):
            x = np.where(chosen_cosine == sorted_cosine[i])[0]
            print(f'{i + 1} Place: ', end='')
            for el in x:
                print(f'{get_on_index(el, 0)}', end=' ')
            print()

        print('\ncosine feat')
        for i in range(6):
            x = np.where(chosen_ecosine == sorted_e_cosine[i])[0]
            print(f'{i+1} Place: ', end='')
            for el in x:
                print(f'{get_on_index(el, 0)}', end=' ')
            print()


def metadata(id):
    metadataRawMatrix = np.genfromtxt('../Dataset/panda_dataset_taffc_metadata.csv', delimiter=',', dtype="str")
    metadata = metadataRawMatrix[1:, [1, 3, 9, 11]]

    metadataScores = np.zeros((1, 900))
    metadataScores[0, id] = -1
    for i in range(metadata.shape[0]):
        if id == i:
            continue
        score = 0
        for j in range(metadata.shape[1]):
            if j < 2:
                if metadata[id, j] == metadata[i, j]:
                    score = score + 1
            else:
                listA = metadata[id, j][1:-1].split('; ')
                listB = metadata[i, j][1:-1].split('; ')
                for e in listA:
                    for ee in listB:
                        if e == ee:
                            score = score + 1
        metadataScores[0, i] = score

    return metadataScores


def Q4():
    meta = np.genfromtxt('../Dataset/panda_dataset_taffc_metadata.csv', delimiter=',', dtype="str")
    meta = meta[1:, 0]
    for file in os.listdir('../Queries'):
        file = f'\"{file[:-4]}\"'
        print(f'Song: {file}')
        id = np.where(meta == file)[0][0]
        unsorted_meta = metadata(id)
        sorted_meta = -np.sort(-unsorted_meta)
        i = 0
        while i < 20:
            values = np.where(unsorted_meta == sorted_meta[i])[1]
            old_i = i
            i += values.shape[0]
            if i > 20:
                i = 20
            for j in range(i):
                print(f'{old_i+j+1:02d} Place: {meta[values[j]]}')
        print()


def Q4_2():
    to_write = np.zeros((900, 900))
    for i in range(900):
        print(f'Calculating for {i}...')
        to_write[i, :] = metadata(i)
    file = '../Dataset/metadata_weighted.csv'
    np.savetxt(file, to_write, fmt='%lf', delimiter=',')


def top_20(unsorted, sorteds):
    i = 0
    vv = np.zeros(20, dtype=np.int16)
    while i < 20:
        values = np.where(unsorted == sorteds[i])
        if len(values) == 2:
            values = values[1]
        else:
            values = values[0]
        old_i = i
        i += values.shape[0]
        if i > 20:
            i = 20
        for j in range(i - old_i):
            vv[old_i + j] = values[j]
    print("TOP 20: ", end="")
    [print(f'{v},'
           f' ', end="") for v in vv]
    print()
    return vv


def Q4_3_aux(file):

    file = f'\"{file}\"'

    meta = np.genfromtxt('../Dataset/panda_dataset_taffc_metadata.csv', delimiter=',', dtype="str")
    meta = meta[1:, 0]

    filename = '../Features/top100_features.csv'
    features = np.genfromtxt(filename, delimiter=',', dtype='<U16')
    names = features[1:, 0]

    pos_in = get_pos_file(names, file, -1)

    print(f"Pos: {pos_in}")

    to_read = '../Features/euclid_lib.csv'
    to_save = np.genfromtxt(to_read, delimiter=',')
    chosen_euclid = to_save[pos_in, :]
    sorted_euclid = np.sort(chosen_euclid)[1:]

    to_read = '../Features/euclid_feat.csv'
    to_save = np.genfromtxt(to_read, delimiter=',')
    chosen_efeat = to_save[pos_in, :]
    sorted_e_feat = np.sort(chosen_efeat)[1:]

    to_read = '../Features/manhat_lib.csv'
    to_save = np.genfromtxt(to_read, delimiter=',')
    chosen_manhat = to_save[pos_in, :]
    sorted_manhat = np.sort(chosen_manhat)[1:]

    to_read = '../Features/manhat_feat.csv'
    to_save = np.genfromtxt(to_read, delimiter=',')
    chosen_emanhat = to_save[pos_in, :]
    sorted_e_manhat = np.sort(chosen_emanhat)[1:]

    to_read = '../Features/cosine_lib.csv'
    to_save = np.genfromtxt(to_read, delimiter=',')
    chosen_cosine = to_save[pos_in, :]
    sorted_cosine = np.sort(chosen_cosine)[1:]

    to_read = '../Features/cosine_feat.csv'
    to_save = np.genfromtxt(to_read, delimiter=',')
    chosen_ecosine = to_save[pos_in, :]
    sorted_e_cosine = np.sort(chosen_ecosine)[1:]

    id = np.where(meta == file)[0][0]
    print(f"ID found: {id}")
    unsorted_meta = metadata(id)
    fix_meta = unsorted_meta[0]
    sorted_meta = -np.sort(-fix_meta)

    meta20 = top_20(fix_meta, sorted_meta)
    print("From metadata of song")

    x = top_20(chosen_euclid, sorted_euclid)
    print(f'Weight euclid: {meta20[np.in1d(meta20, x)].shape[0]/20}')

    x = top_20(chosen_efeat, sorted_e_feat)
    print(f'Weight lib_euclid: {meta20[np.in1d(meta20, x)].shape[0]/20}')

    x = top_20(chosen_manhat, sorted_manhat)
    print(f'Weight manhattan: {meta20[np.in1d(meta20, x)].shape[0]/20}')

    x = top_20(chosen_emanhat, sorted_e_manhat)
    print(f'Weight lib_manhattan: {meta20[np.in1d(meta20, x)].shape[0]/20}')

    x = top_20(chosen_cosine, sorted_cosine)
    print(f'Weight cosine: {meta20[np.in1d(meta20, x)].shape[0]/20}')

    x = top_20(chosen_ecosine, sorted_e_cosine)
    print(names[x])
    print(f'Weight lib_cosine: {meta20[np.in1d(meta20, x)].shape[0]/20}')


def Q4_3():
    path = '../Queries'
    for file in os.listdir(path):
        file = file[:-4]
        print(f'Song: {file}')
        Q4_3_aux(file)
        print()


def main():
    Q4_3()


if __name__ == '__main__':
    main()
