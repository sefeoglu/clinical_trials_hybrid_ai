The sample usage:

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
Span[7:8]: "MI" → CORONAVIRUS (1.0)
Span[11:13]: "chest pain" → SIGN_OR_SYMPTOMS (0.9746)
Span[13:17]: "anginal equivalent symptoms" → SIGN_OR_SYMPTOMS (0.9997)
Span[22:26]: "exertional anginal equivalent symptoms" → SIGN_OR_SYMPTOMS (0.9675)

````
