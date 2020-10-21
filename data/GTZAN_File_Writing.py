import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import os,sys,csv
import sklearn

genres = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
path = 'E:\\GTZAN\\'
header = 'filename rmse_mean rmse_std spectral_centroid_mean spectral_centroid_std spectral_bandwidth_mean spectral_bandwidth_mean rolloff zero_crossing_rate_mean zero_crossing_rate_std'
for i in range(1,21):
	header += f' mfcc{i}'
for i in range(1,13):
	header += f' chroma{i}'	
header = header + ' label'
header = header.split()

print(header)

genre_to_file = {}
for g in genres:
	genre_to_file[g] = []
	for filename in os.listdir(path+g):
		genre_to_file[g].append(path+g+'\\'+filename)

print(genre_to_file['classical'][3])
        
file = open('GTZAN_Mean_STD.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

for g in genres:
	for songname in genre_to_file[g]:
		y,sr = librosa.load(songname,mono=True,duration=30)
		rmse = librosa.feature.rms(y)
		centroid = librosa.feature.spectral_centroid(y,sr)
		bandwidth = librosa.feature.spectral_bandwidth(y,sr)
		roll = librosa.feature.spectral_rolloff(y,sr)
		zcr = librosa.feature.zero_crossing_rate(y)
		mfcc = librosa.feature.mfcc(y,sr)
		chroma = librosa.feature.chroma_stft(y,sr)
		print("Features Extracted")      
		to_append = f'{songname} {np.mean(rmse)} {np.std(rmse)} {np.mean(centroid)} {np.std(centroid)} {np.mean(bandwidth)} {np.std(bandwidth)} {np.mean(roll)} {np.mean(zcr)} {np.std(zcr)}'
		for e in mfcc:
			to_append += f' {np.mean(e)}'
		for e in chroma:
			to_append += f' {np.mean(e)}'
		to_append += f' {g}'
		file = open('GTZAN_Mean_STD.csv', 'a', newline='')
		with file:
			writer = csv.writer(file)
			writer.writerow(to_append.split())


