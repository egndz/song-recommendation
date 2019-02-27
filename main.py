import os
import librosa
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import csv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time


def getDirectory():
	return os.getcwd()


def construct_feature_list(fname, y, sr):
	print("zero-crossing")
	zerocross = np.array(librosa.feature.zero_crossing_rate(y))
	print("spectral_centroid")
	spectral_centroid = np.array(librosa.feature.spectral_centroid(y, sr))
	print("spectral_bandwidth")
	spectral_bandwidth = np.array(librosa.feature.spectral_bandwidth(y, sr))
	print("spectral_rolloff")
	spectral_rolloff = np.array(librosa.feature.spectral_rolloff(y, sr))
	print("rmse")
	rmse = np.array(librosa.feature.rmse(y))
	feature_list = [zerocross, spectral_centroid, spectral_bandwidth, spectral_rolloff, rmse]

	features = [fname]

	model = KMeans(n_clusters=2, random_state=0)
	print("kmeans")
	# 20features
	for i in range(0, len(feature_list)):
		indexes = np.array([range(0, feature_list[i].size)])
		tmp = np.array(feature_list[i]).reshape(1, -1)
		new_feature = np.transpose(np.vstack((indexes, tmp)))
		model.fit(new_feature)
		labels = model.labels_
		c1 = new_feature[np.array(np.where(labels == 0)), 1]
		c2 = new_feature[np.array(np.where(labels == 1)), 1]
		features = np.append(features, np.average(c1))
		features = np.append(features, np.var(c1))
		features = np.append(features, np.average(c2))
		features = np.append(features, np.var(c2))

	# 2 features extracted from spectral_contrast as a mean and variance.
	# essentia library has another functions to extract the tempo!
	features = np.append(features, np.average(librosa.feature.spectral_contrast(y, sr)))
	features = np.append(features, np.var((librosa.feature.spectral_contrast(y, sr))))

	# 24 features extracted from mfcc features which are mean and variance for each bins decided as 12
	print("mfcc")
	number_of_mfcc = 12
	tmp = librosa.feature.mfcc(y, sr, n_mfcc=number_of_mfcc)
	tmpp = [None] * number_of_mfcc
	tmpp2 = [None] * number_of_mfcc

	for i in range(number_of_mfcc):
		tmpp[i] = np.average(tmp[i])
		tmpp2[i] = np.var(tmp[i])
	features = np.append(features, tmpp)
	features = np.append(features, tmpp2)

	# 24 features extracted from chroma features which are mean and variance for each bins decided as 12
	print("chroma")
	number_of_chroma = 12  # default
	tmp = librosa.feature.chroma_cqt(y, sr)
	tmpp = [None] * 12
	tmpp2 = [None] * 12
	for i in range(12):
		tmpp[i] = np.average(tmp[i])
		tmpp2[i] = np.var(tmp[i])
	features = np.append(features, tmpp)
	features = np.append(features, tmpp2)

	return features


def write_to_file(filename,song_path,y,sr):
	with open(filename,"a") as csvfile:
		writer = csv.writer(csvfile)
		feature = construct_feature_list(song_path.split(str(os.sep))[-1], y, sr)
		writer = csv.writer(csvfile)
		writer.writerow(feature)


def main():
	songs_path = str(getDirectory()) + str(os.sep) + "music"  + str(os.sep)
	paths = [os.path.join(songs_path,i) for i in os.listdir(songs_path) if not os.path.isdir(i)]

	beginning = "first_30.csv"
	end = "last_30.csv"

	for i,song_path in enumerate(paths):
		print(str(i) + ". Analyzing: " + str(song_path))
		y, sr = librosa.load(song_path)
		y_first_30_sec = y[:(sr*30)]
		y_last_30_sec = y[(len(y)-(sr*30)):]
		print("FIRST 30 SEC")
		write_to_file(beginning,song_path,y_first_30_sec,sr)
		print("LAST 30 SEC")
		write_to_file(end,song_path,y_last_30_sec,sr)



if __name__=="__main__":
	main()



