main.py extracts the acoustic features from music directory and save them as a songs_featues.csv

recommend.py make the recommendation by using k-nn algorithm with a default parameter Euclidean distance. distance can be change.

if you would like to change the music file, just replace the songs with new ones and run the main.py.you can change csv file name inside of the code as well or you can merge them

TO RUN

1. pip install -r requirements.txt
2. python main.py
3. python recommend.py
4. python recommend.py music/David_August_-_Epikur_-_Epikur_EP_\(Official_Video\)-Zvi4JYmRXzI.mp3  ---> change the songname.mp3 
