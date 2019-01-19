import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler



def read_csv(name):
	return pd.read_csv(name,index_col=0)

def main():
	filename="songs_features.csv"
	df = read_csv(filename)
	song_names = df.index
	current_directory = os.getcwd()
	song_directory = str(current_directory) + str(os.sep) + sys.argv[1]

	knn_value=5
	scaler = StandardScaler()
	df = scaler.fit_transform(df)
	distances = pairwise_distances(df)
	df = pd.DataFrame(distances, columns=song_names)
	df.index = df.columns
	recommendations = df[str(song_directory)].sort_values(ascending=True).reset_index()
	for i in range(knn_value+1):
		print(str(recommendations.iloc[i][0]).split(os.sep)[-1])
		print("     ")
		print(recommendations.iloc[i][1])
		print("\n")








if __name__ == "__main__":
	main()