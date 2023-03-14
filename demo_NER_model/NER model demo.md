The code lines show how a NER model is trained on an annotated training text data by using Flair NLP framework.

````python

    import flair
    from flair.models import SequenceTagger
    from flair.trainers import ModelTrainer
    from flair.data import Corpus
    from flair.datasets import  ColumnCorpus
    from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings

    columns = {0: 'text', 1: 'ner'}
    tag_type = 'ner'
    corpus_path = "path to the training data"# the data must be formatted like in the folder    "https://github.com/sefeoglu/clinical_trials_hybrid_ai/tree/master/annotated_training_data_for_ner"
    corpus: Corpus = ColumnCorpus(
                        corpus_path,
                        column_format=columns,
                        train_file='train.txt',
                        test_file='test.txt',
                        dev_file='dev.txt',
                        tag_to_bioes=tag_type
                    )

    label_dict = corpus.make_tag_dictionary(tag_type)
    

    # 4. initialize embedding stack with Flair and GloVe
    embedding_types = [
            WordEmbeddings('glove'),
            FlairEmbeddings('mix-forward', chars_per_chunk=128),
            FlairEmbeddings('mix-backward', chars_per_chunk=128)
    ]

    embeddings = StackedEmbeddings(embeddings=embedding_types)

    # 5. initialize sequence tagger
    model = SequenceTagger(hidden_size=256,
                            embeddings=embeddings,
                            tag_dictionary=label_dict,
                            tag_type=tag_type,
                            dropout=0.5)


    trainer: ModelTrainer = ModelTrainer(model, corpus)
    print("==================NER Training=====================")
    out_path = "path to the trained NER model"
    trainer.train(out_path,
                    learning_rate=0.001, 
                    mini_batch_size=32,
                    max_epochs= 5,
                    embeddings_storage_mode='cpu',
                    shuffle=False,
                    write_weights=True)

````
The NER model trained above predicts mentions in a given text as shown below.

````python

from flair.models import SequenceTagger
from flair.data import Sentence



sample_sentence_en = '''Acute ischemic symptoms compatible with diagnosis of MI,
                    such as chest pain or anginal equivalent symptoms at rest or new onset exertional anginal equivalent symptoms.'''
                    
model_path = "path to the trained NER model like ner_model.pt)"

model = SequenceTagger.load(model_path)

sentence = Sentence(sample_sentence_en)

model.predict(sentence)

for entity in sentence.get_spans("ner"):
  print(entity)
````


Output prediction:
````
Span[0:3]: "Acute ischemic symptoms" → SIGN_OR_SYMPTOMS (0.9999)
Span[7:8]: "MI" → DISEASE (1.0)
Span[11:13]: "chest pain" → SIGN_OR_SYMPTOMS (0.9746)
Span[13:17]: "anginal equivalent symptoms" → SIGN_OR_SYMPTOMS (0.9997)
Span[22:26]: "exertional anginal equivalent symptoms" → SIGN_OR_SYMPTOMS (0.9675)

````
