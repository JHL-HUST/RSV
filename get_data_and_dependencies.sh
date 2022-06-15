
cd ./data
wget https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz
tar -zxvf ag_news_csv.tgz
rm ag_news_csv.tgz
mv ag_news_csv ag_news
# SST-2
# Downloaded from https://gluebenchmark.com/tasks, https://github.com/CS287/HW1/tree/master/data

# Download GloVe embeddings for training 
cd ./vectors
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip

# Download counter-fitted embeddings and stanford-postagger for synonyn selector
wget https://raw.githubusercontent.com/nmrksic/counter-fitting/master/word_vectors/counter-fitted-vectors.txt.zip
unzip counter-fitted-vectors.txt.zip
rm counter-fitted-vectors.txt.zip

cd ../aux_files
wget https://nlp.stanford.edu/software/stanford-postagger-2018-10-16.zip
unzip stanford-postagger-2018-10-16.zip
rm stanford-postagger-2018-10-16.zip

# USE model for Textfooler attack
# cd ../../model/USE
# curl -L https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed