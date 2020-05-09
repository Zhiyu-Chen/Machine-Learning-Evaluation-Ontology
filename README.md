# sci_data

Run phase 1 (sientific entities recognition using CRF-LSTM)

- Training data should be formatted as below:
token/tag token/tag token/tag ...
token/tag token/tag token/tag ...

- create a folder called data that contains the data file data.all
- run ./prepare to obtain training and testing folds
- run python3 prepare_sent.py --data data/train to prepare the vocabularies and training 
- to train CRF-LSTM run python3 train.py --data data2/train.csv --epochs 100 --CharToIdx data/train.char_to_idx --WordToIdx data/train.word_to_idx --TagToIdx data/train.tag_to_idx --model_file data/phase1_model --valid data/valid
- use parameters.py to set up the parameters of the model
- for prediction, run python3 predict_sent.py --CharToIdx data/train.char_to_idx --WordToIdx data/train.word_to_idx --TagToIdx data/train.tag_to_idx --checkpoint data/phase1_model.epoch100 --test data/test
- for evaluation, after setting the correct arguments in predict.py code, run python3 evaluate.py --test data/test


Run phase 2 (IR-based relation extaction)

- cd Relation_Extraction_IR
- set the correct path to the collection of papers in metadata.py
- download glove.6B.50d.txt pretrained embedding from http://nlp.stanford.edu/data/glove.6B.zip
- to train and validate the IR based model, run convknrm_pointwise.py --emsize X --device GPU-ID --nbins 5 --pretrained_embedding path_to/glove.6B.Xd.txt
X=50 when using glove.6B.50d.txt


Pipeline Demo

- cd PipelineDemo
- to prepare an input paper, run python3 InputDemo.py --input_file data/test --output_file data2/demo.input --paper_id 1
- to extract TDM triples from paper that are used to build the knowledge graph, run python3 extract_triples_from_paper.py --CharToIdx data/train.char_to_idx --WordToIdx data/train.word_to_idx --TagToIdx data/train.tag_to_idx --test data/demo.input --checkpoint data/phase1_model.epoch100 --pretrained_embedding path_to/glove.6B.Xd.txt --phase2_model Relation_Extraction_IR/stage2_model.pt


