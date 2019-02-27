import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler



def read_csv(name):
	return pd.read_csv(name,index_col=0)


def apply_scaling(df):
	scaler = StandardScaler()
	df = scaler.fit_transform(df)
	return pd.DataFrame(df)


def main():
	first_30="first_30.csv"
	last_30="last_30.csv"
	first = read_csv(first_30)
	last = read_csv(last_30)
	song_names = first.index

	current_directory = os.getcwd()
	song_directory = str(current_directory) + str(os.sep) + sys.argv[1]
	song_directory = song_directory.split(str(os.sep))[-1]

	first_30 = apply_scaling(first)
	first_30.index = first.index
	last_30 = apply_scaling(last)
	last_30.index = last.index

	first_30.loc[song_directory] = last_30.loc[song_directory]

	knn_value=5
	distances = pairwise_distances(first_30)
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