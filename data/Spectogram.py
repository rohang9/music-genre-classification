import os,sys
import numpy as np
import librosa 

path = 'E:\\GTZAN\\'
genres = 'blues classical country disco jazz hiphop metal reggae rock pop'.split()

genre_to_file = {}
for g in genres:
	genre_to_file[g] = []
	for fname in os.listdir(path+g+'/'):
		genre_to_file[g].append(fname)

# y,sr = librosa.load(path+'blues\\'+genre_to_file['blues'][2])
# print(librosa.feature.melspectrogram(y=y,sr=sr)[:,645:1290][None,:,:].shape)
# print(np.array([[1],[2]]).shape)

print('-----------------------Spectogram Loading------------------------')
for i,g in enumerate(genres):
	for j,fname in enumerate(genre_to_file[g]):
		x,sr = librosa.load(path+g+'/'+fname,mono=True,duration=30)
		mel = librosa.feature.melspectrogram(y=x,sr=sr)
		cent = librosa.feature.spectral_centroid(y=x,sr=sr)
		roll = librosa.feature.spectral_rolloff(y=x,sr=sr)
		bw = librosa.feature.spectral_bandwidth(y=x,sr=sr)
		zcr = librosa.feature.zero_crossing_rate(y=x)	
		spec1 = np.concatenate((mel[:,:430],cent[:,:430],roll[:,:430],bw[:,:430],zcr[:,:430]),axis=0)[None,:,:]
		spec2 = np.concatenate((mel[:,430:860],cent[:,430:860],roll[:,430:860],bw[:,430:860],zcr[:,430:860]),axis=0)[None,:,:]
		spec3 = np.concatenate((mel[:,860:1290],cent[:,860:1290],roll[:,860:1290],bw[:,860:1290],zcr[:,860:1290]),axis=0)[None,:,:]

		try:
			spec = np.concatenate((spec,spec1,spec2,spec3),axis=0)
			y = np.concatenate((y,np.array([[i],[i],[i]])),axis=0)
		except NameError:
			spec = np.concatenate((spec1,spec2,spec3),axis=0)
			y = np.array([[0],[0],[0]])
		if j%10==9:
			print(j+1,' percent completed')
	print('------------'+g+' completed---------')

print(np.where(y==0))

np.save('spectogram_x3.npy',spec)
np.save('label_x3.npy',y)
