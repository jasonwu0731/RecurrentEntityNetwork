import numpy as np
import json
import os
from sklearn.feature_extraction.text import CountVectorizer

from path_config import *

def do_statistics(lst_dialogs):
    dict_stats = {}
    lst_utterances = []
    lst_nb_utterances = []
    for dialog in lst_dialogs:
        for utterance in dialog['utterances']:
            lst_utterances.append(utterance)
        lst_nb_utterances.append(len(dialog['utterances'])*1.0)

    dict_stats['nb_dialog'] = len(lst_dialogs)
    dict_stats['nb_utterance_total'] = len(lst_utterances)
    dict_stats['nb_utterance_unic'] = len(list(set(lst_utterances)))
    dict_stats['nb_utterance_per_dialog'] = np.mean(lst_nb_utterances)

    cnt = CountVectorizer()
    cnt.fit(lst_utterances)
    dict_stats['voc_total_len'] = len(cnt.get_feature_names())
    dict_stats['voc_total'] = cnt.get_feature_names()

    cnt = CountVectorizer(stop_words="english")
    cnt.fit(lst_utterances)
    dict_stats['voc_non_stop_len'] = len(cnt.get_feature_names())
    dict_stats['voc_non_stop'] = cnt.get_feature_names()

    return dict_stats

def get_taskfile_db(record=False, taskchosen='',testset='', temp_path=''):
    dest = temp_path if record else DATASET_PATH
    if not record:
        endfix = "trn10000.json" 
    else:
        endfix = '-RDL.json'
    
    testing_ratio = 0.1
    names = os.listdir(dest)
    inputtaskfile = []
    for name in names:
        if name.endswith(endfix):
            if taskchosen in ['1','2','3','4','5']:
                if TASK_NAME[int(taskchosen)] in name:
                    inputtaskfile.append(dest+name)
            else:
                print "[Error] Please select task 1-5"
                exit(1)
    
    officialtestfile = []
    if testset in ['1','2','3','4','all']:
        print "[IMPORTANT] Testing on official testset", testset
        testset_list = [testset] if testset!='all' else ['1','2','3','4']
        for tset in testset_list:
            dest = TEST_DATASET_PATH+'tst' + tset + '/'
            endfix = '-RDL' if record else ''
            endfix += '.json'
            names = os.listdir(dest)
            for name in names:
                if name.endswith(endfix):
                    if 'task'+taskchosen in name:
                        officialtestfile.append(dest+name)
    elif testset == None:
        print "[Info] No Test on official testset..."
    else:
        print "[Error] Please choose testset from 1 to 4..."
        exit(1)
    return inputtaskfile, taskchosen, testing_ratio, officialtestfile

def _get_source_paths(source_dir):
    prefix = "dialog-task"
    endfix = "tst1000.json"
    names = os.listdir(source_dir)
    task1, task2, task3, task4, task5 = None, None, None, None, None
    for name in names:
        if name.startswith(prefix) and name.endswith(endfix):
            task = int(name[11])
            plain_name = name.replace('.json', '')
            if task == 1:
                task1 = source_dir+plain_name
            elif task == 2:
                task2 = source_dir+plain_name
            elif task == 3:
                task3 = source_dir+plain_name
            elif task == 4:
                task4 = source_dir+plain_name
            elif task == 5:
                task5 = source_dir+plain_name
    return [task1, task2, task3, task4, task5]

### The current dialog format
### [{dialog_id : " ",
#     utterances: [" "],
#     candidates: [{candidate_id: " " , utterance: ""}, ... ],
#     answer: {candidate_id: " " , utterance: ""} ]
if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-d", "--data", dest="data", help="Run train or test data")
    (options, args) = parser.parse_args()
    
    if options.data=='train':
        data_dir = DATASET_PATH
    elif options.data=='test':
        data_dir = TEST_DATASET_PATH
        data = []
        for tst in ['tst1/', 'tst2/', 'tst3/', 'tst4/']:
            t12345 = _get_source_paths(data_dir+tst)
            data.append(t12345)

    for d12345 in data:
        for d in d12345:
            inputtaskfile =  d + '.json'
            outputtaskfile = d + '.plain'

            outputstatistics = None if options.data=='test' else d+'-statistics.plain'
            outputtruth = None if options.data=='test' else d+'-truth.json'
            fd = open(inputtaskfile, 'rb')
            json_data = json.load(fd)
            fd.close()
            print "Sample one story from json file: \n", json_data[0]

            ### Print-out plain dialogs
            if (outputtaskfile != None):

                fd_out = open(outputtaskfile, 'wb')
                for story in json_data:

                    fd_out.write(str('dialog_id: ') + str(story['dialog_id']) + "\n")
                    fd_out.write(str("Utterances:\n"))
                    for utterance in story['utterances']:
                        fd_out.write(" * " + str(utterance) + "\n")

                    fd_out.write("Candidates:\n")
                    for cand in story['candidates']:
                        fd_out.write(" * " + str(cand['candidate_id']) + " - " + str(cand['utterance']) + "\n")

                    fd_out.write(str("Answer:\n"))
                    if (story.get('answer') != None):
                        fd_out.write(" * " + str(story['answer']['candidate_id'])
                                     + " - " + str(story['answer']['utterance']) + "\n")
                    else:
                        fd_out.write(str(None) + "\n")
                    fd_out.write("\n")
                fd_out.close()

            ## Print-out statistics
            if (outputstatistics != None):
                with open(outputstatistics, "wb") as fd_out:
                    dict_stats = do_statistics(lst_dialogs=json_data)
                    for key in dict_stats.keys():
                        fd_out.write(str(key) + ":\n")
                        fd_out.write(str(dict_stats[key]) + "\n\n")

            if (outputtruth != None):
                with open(outputtruth, 'wb') as fd_out:
                    lst_responses = []
                    for story in json_data:

                        dict_answer_current = {}
                        dict_answer_current['dialog_id'] = story['dialog_id']

                        lst_candidate_id = []
                        lst_candidate_id.append(story['answer']['candidate_id'])

                        lst_candidate_rank = []
                        for it in range (0, len(lst_candidate_id)):
                            lst_candidate_rank.append({"candidate_id": lst_candidate_id[it], "rank": it+1})

                        dict_answer_current['lst_candidate_id'] = lst_candidate_rank
                        lst_responses.append(dict_answer_current)

                    json.dump(lst_responses, fd_out)
                    fd_out.close()


