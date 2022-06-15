## Randomized Substitution and Vote (RS&V)

This repository contains code to reproduce results from the paper:

[Detecting Textual Adversarial Examples through Randomized Substitution and Vote](https://openreview.net/pdf?id=Hu_4s88iqgc) (UAI 2022)

[Xiaosen Wang](http://xiaosenwang.com/), Yifeng Xiong, Kun He

## Datesets and Dependencies

There are three datasets used in our experiments. Download and put the dataset into the directory `./data/ag_news`, `./data/imdb` and `./data/yahoo_answers`, respectively.

- [AG's News](https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz)
- [IMDB](https://s3.amazonaws.com/fast-ai-nlp/imdb.tgz)
- [Yahoo! Answers](https://s3.amazonaws.com/fast-ai-nlp/yahoo_answers_csv.tgz)

There are Three dependencies for this project. Download and put the files `glove.840B.300d.txt` and `counter-fitted-vectors.txt` into the directory `./data/vectors`, put the directory `stanford-postagger-2018-10-16/` into the directory `./data/aux_files`.

- [GloVe vecors](http://nlp.stanford.edu/data/glove.840B.300d.zip)
- [Counter fitted vectors](https://github.com/nmrksic/counter-fitting/blob/master/word_vectors/counter-fitted-vectors.txt.zip)
- [stanford-postagger-2018-10-16](https://nlp.stanford.edu/software/stanford-postagger-2018-10-16.zip)

You can run the `get_data_and_dependencies.sh` to get test data：
   ```shell
   bash get_data_and_dependencies.sh
   ```

## File Description

- `./model`: Detail code for model architecture.
- `./utils`: Helper functions for training models and processing data.
- `./adversary`: Files for attack methods.
- `./data`: Datasets and GloVe vectors.

- `cnn_classifier.py`, `bert_classifier.py`, `robert_classifier.py` : Training code for CNN, bert and RoBERTa.

- `cnn_attack.py`: Attacking CNN model.
- `bert_attack.py` Attacking BERT and RoBERTa model.

- `build_embs.py`: Generating the dictionary, embedding matrix and distance matrix.
- `synonym_selector.py`: Generating synonyms set.

- `detect_transfer.py`: Converting adversarial examples through Randomized Substitution.
- `detect_eval.py`: Vote and Detection.

## Experiments

1. Generating the dictionary, embedding matrix and distance matrix:

   ```shell
   python build_embs.py --data_dir ./data/ --task_name ag_news
   ```

2. Training and attacking the models:

   For CNN:

   ```shell
   python cnn_classifier.py --output_dir ./output/model_file/ag_news/cnn --data_dir ./data/ --task_name ag_news --max_seq_length 128 --do_train --do_eval --vGPU 0
   python cnn_attack.py  --output_dir ./output/model_file/ag_news/cnn  --data_dir ./data/ --attack textfooler --task_name ag_news --max_seq_length 128  --max_candidate 50 --save_to_file ./output/adv_example/ag_news_cnn_textfooler --vGPU 0
   ```

   For BERT:

   ```shell
   python bert_classifier.py  --output_dir ./output/model_file/ag_news/bert --bert_model bert-base-uncased  --data_dir ./data/  --task_name ag_news --max_seq_length 128  --do_train --do_eval  --vGPU 0
   python bert_attack.py --data_dir ./data/ --task_name ag_news --attack textfooler --output_dir ./output/model_file/ag_news/bert/ --attack_batch 1000 --save_to_file ./output/adv_example/ag_news --bert_model bert-base-uncased  --max_candidate 50 --max_seq_length 128 --vGPU 0
   ```

3. Evaluating the detection performance:

   ```shell
   python detect_transfer.py --task_name ag_news --data_dir ./data/  --votenum 25 --randomrate 0.6 --fixrate 0.02 --advfile ./output/adv_example/ag_news_cnn_textfooler.pkl --out_file ./output/transfer/ag_news_cnn_textfooler.pkl
   ```

   ```
   python detect_eval.py --task_name ag_news --data_dir ./data/  --max_seq_length 128  --modeltype cnn --output_dir ./output/model_file/ag_news/cnn --eval_file ./output/transfer/ag_news_cnn_textfooler.pkl
   ```

## Contact

Questions and suggestions can be sent to xswanghuster@gmail.com.