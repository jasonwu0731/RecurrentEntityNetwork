
import numpy as np
import random
import json

from sklearn.feature_extraction.text import TfidfVectorizer

from dataset_walker import do_parse_cmdline

'''
def do_parse_cmdline():

    from optparse import OptionParser
    parser = OptionParser()

    parser.add_option("--input-task-file", dest="inputtaskfile",
                      default="../data/dataset-E2E-goal-oriented/dialog-task1API-kb1_atmosphere-distr0.5-trn10000.json",
                      help="filename of the task", metavar="FILE")

    parser.add_option("--output_result-file", dest="outputresultfile",
                      default="../data/dataset-E2E-goal-oriented/dialog-task1API-output-result-tfidf.json",
                      help="output file results", metavar="FILE")

    (options, args) = parser.parse_args()

    return options.inputtaskfile, options.outputresultfile
'''

def do_load_storie_tfidf(json_data):

    lst_docs = []
    vect = TfidfVectorizer()

    for story in json_data:
        lst_docs.append(" ".join(story['utterances']))

    lst_docs = vect.fit_transform(lst_docs)

    return vect, lst_docs


### The current dialog format
### [{dialog_id : " ", lst_candidate_id: [{candidate_id: " ", rank: " "}, ...]}]

if __name__ == '__main__':

    # Parsing command line
    #inputtaskfile, outputresultfile = do_parse_cmdline()
    data_dir = "../data/dataset-E2E-goal-oriented/"
    taskchosen, filename = do_parse_cmdline()

    inputtaskfile = data_dir + filename + '-kb1_atmosphere-distr0.5-trn10000' + '.json'
    outputresultfile = data_dir + filename + "-output-result-tfidf.json"

    fd = open(inputtaskfile, 'rb')
    json_data = json.load(fd)
    fd.close()

    vect, lst_docs = do_load_storie_tfidf(json_data)
    #print vect
    #print lst_docs

    lst_responses = []

    for idx_story, story in enumerate(json_data):
        dict_answer_current = {}
        dict_answer_current['dialog_id'] = story['dialog_id']

        lst_candidate_id   = []
        lst_candidate_utterance = []
        for cand in story['candidates']:
            lst_candidate_id.append(cand['candidate_id'])
            lst_candidate_utterance.append(cand['utterance'])
        lst_candidate_utterance = vect.transform(lst_candidate_utterance)

        idx_rank = np.asarray(np.argsort(-np.dot(lst_docs[idx_story],
                                      np.transpose(lst_candidate_utterance)).todense().ravel()))[0]
        lst_candidate_rank = []
        for it in range (0, len(lst_candidate_id)):
            lst_candidate_rank.append({"candidate_id": lst_candidate_id[idx_rank[it]], "rank": it+1})

        dict_answer_current['lst_candidate_id'] = lst_candidate_rank
        lst_responses.append(dict_answer_current)

    fd = open(outputresultfile, 'wb')
    json.dump(lst_responses, fd)
    fd.close()