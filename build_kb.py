from metadata import *
from reader import PaperReader
from owlready2 import *
import os
from fuzzywuzzy import fuzz
import numpy as np
import pickle

NLP_TDM_rs_path = os.path.join(NLP_TDM_path,'annotations','resultsAnnotation.tsv')


class SDO(object):
    def __init__(self,onto_fname=onto_fname):
        onto_path.append(onto_dir)
        print(os.path.join(onto_dir,onto_fname))
        self.onto = get_ontology(os.path.join(onto_dir,onto_fname))
        self.onto.load()

    def fuzzy_metric(self,metric):
        '''
        classify metric by fuzzy matching
        '''
        metric_prop_names = [cls.__name__ for cls in self.onto.TestOnMetric.__subclasses__()]
        scores = [fuzz.ratio(each,metric) for each in metric_prop_names]
        metric_idx = np.argmax(scores)
        return getattr(self.onto, metric_prop_names[metric_idx])

    def build_from(self,result_path=vote_rs_path,db_name = 'final_db.owl'):
        f = open(result_path,'r')
        result_idx = 0
        for line in f:
            paper,task,dataset,metric,score = line.split('$')
            print(paper)
            paper_o = sdo.onto.Paper(paper)
            task_o = self.onto.NLPTask(task)
            dataset_o = self.onto.NLPDataset(dataset)
            metric_o = self.fuzzy_metric(metric)
            # buld individual relations
            paper_o.testOnDataset = [dataset_o]
            # creatre result individuals
            result_name = "result" + str(result_idx)
            result_o = sdo.onto.Result(result_name, namespace=sdo.onto,
                                       onTask=[task_o],
                                       reportResultsFrom=paper_o,
                                       testOnDataset=[dataset_o],
                                       metric_o=score
                                       )
            result_idx += 1

        self.onto.save(file = os.path.join(onto_dir,'final_db'), format = "rdfxml")

    def build_demo(self):
        paper_title, paper_task, paper_dataset, paper_metric, paper_scores, paper_entities = pickle.load(
            open(paper_entity_path, 'rb'))
        f = open('./data/kb_input.txt','r')
        result_idx = 0
        for line in f:
            print(line)
            pid,results = line.split(' ',1)
            paper = paper_title[pid]
            results = results.split('$')
            for result in results:
                task, dataset, metric = result.split('#')
                paper_o = sdo.onto.Paper(paper)
                task_o = self.onto.NLPTask(task)
                dataset_o = self.onto.NLPDataset(dataset)
                metric_o = self.fuzzy_metric(metric)
                try:
                    score = paper_scores[pid][0][1]
                except:
                    score = -1
                #print(score)
                # buld individual relations
                paper_o.testOnDataset = [dataset_o]
                # creatre result individuals
                result_name = "result" + str(result_idx)
                result_o = sdo.onto.Result(result_name, namespace=sdo.onto,
                                           onTask=[task_o],
                                           reportResultsFrom=paper_o,
                                           testOnDataset=[dataset_o],
                                           metric_o=score
                                           )
                result_idx += 1
        self.onto.save(file=os.path.join(onto_dir, 'demo_db.owl'), format="rdfxml")





if __name__ == '__main__':
    sdo = SDO()
    #sdo.build_from(vote_rs_path)
    sdo.build_demo()


