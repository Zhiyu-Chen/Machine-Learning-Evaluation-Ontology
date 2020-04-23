from metadata import *
from reader import PaperReader
from owlready2 import *
import os

NLP_TDM_rs_path = os.path.join(NLP_TDM_path,'annotations','resultsAnnotation.tsv')


class SDO(object):
    def __init__(self,onto_fname):
        onto_path.append(onto_path)
        onto = get_ontology(os.path.join(onto_path,onto_fname))
        onto.load()

    def create_individual_from(self,result_file):
        preader = PaperReader()
        paper_result = preader.get_paper_result()
        for pid in paper_result:
            print(paper_result[pid])

    def create_relation_from(self,result_file):
        pass



if __name__ == '__main__':
    preader = PaperReader()
    paper_result = preader.get_paper_result()
    for pid in paper_result:
        print(paper_result[pid])
        exit()