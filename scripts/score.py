
import json
import os
### The current dialog format
### [{dialog_id : " ", lst_candidate_id: [{candidate_id: " ", rank: " "}, ...]}]

def do_load_json_result(filename, nb_candidate):

    dict_result = {}
    with open(filename, 'rb') as fd:

        json_data = json.load(fd)

        if (type(json_data) != list):
            print "[Error] The result file should be a list ..."
            exit(1)

        for item in json_data:
            #print item
            if (item.get('dialog_id') == None):
                print "[Error] No dialog_id key founded ..."
                continue

            if (item.get('lst_candidate_id') == None):
                print "[Error] No lst_candidate_id key founded ..."
                print 'filename', filename
                exit(1)

            lst_candidate = [None] * nb_candidate
            for candidate in item['lst_candidate_id']:

                if (candidate.get('rank') == None):
                    print "[Error] one candidate has no rank key ..."
                    exit(1)
                if (candidate.get('candidate_id') == None):
                    print "[Error] one candidate has no candidate_id key ..."
                    exit(1)

                if (int(candidate["rank"]) <= nb_candidate):
                    lst_candidate[int(candidate["rank"]) - 1] = candidate['candidate_id']

            dict_result[item['dialog_id']] = lst_candidate


    return dict_result

def get_dialog_utter(path):
    ### The current dialog format
    ### [{dialog_id : " ",
    #     utterances: [" "],
    #     candidates: [{candidate_id: " " , utterance: ""}, ... ],
    #     answer: {candidate_id: " " , utterance: ""} ]
    fd = open(path, 'rb')
    json_data = json.load(fd)
    fd.close()
    dialog = {}
    cand = {}
    for story in json_data:
        for c in story['candidates']:
            cand[c['candidate_id']] = c['utterance']
        dialog[story['dialog_id']] = [story['utterances'], cand]
    return dialog

def do_compute_score(dict_result_truth, dict_result_test, precision_at, filepath=''):
    if filepath != '':
        options_testset = filepath.split('/')[-2]
        options_taskchosen = filepath.split('/')[-1][7:12]
        dialog = get_dialog_utter(filepath) 
        fw = open('MISTAKES-'+str(options_testset)+str(options_taskchosen)+'.txt','w')
    nb_true  = 0
    nb_testdata = 0
    for key in dict_result_truth.keys():
        if (dict_result_test.get(key) != None):
            nb_testdata += 1
            if (dict_result_truth[key][0] in dict_result_test[key][0:precision_at]):
                nb_true += 1
            elif precision_at==1 and filepath!='':
                utters = dialog.get(key)[0]
                cands = dialog.get(key)[1]
                fw.write('dialog_id: '+ str(key) + '\n')
                fw.write('\tTruth: '+ str(cands[dict_result_truth[key][0]]) + '\n')
                fw.write('\tPrediction: '+ str(cands[dict_result_test[key][0]]) + '\n')
                fw.write('\tUtterances: '+ str(utters) + '\n\n')             
                #print '[Wrong Prediction]', key
                #print '\t truth:', dict_result_truth[key][0], ', predict:', dict_result_test[key][0]
        else:
            print 'No found dialog_id:', key
    if nb_testdata == 0:
        nb_testdata = 1
    return nb_true*1.0 / nb_testdata

if __name__ == '__main__':

    # Parsing command line
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-d", "--data", dest="data", help="Run train or test data")
    parser.add_option("-t", "--task", dest="taskchosen", help=" ")
    parser.add_option("-T", "--testset", dest="testset", help=" ")
    (options, args) = parser.parse_args()

    if options.data=='train':
        data_dir = "../data/dataset-E2E-goal-oriented/"
    elif options.data=='test':
        names = os.listdir('ranking/')
        for name in names:
            if ('tst'+options.testset in name) and ('task'+options.taskchosen in name):
                inputfiletest = 'ranking/'+name
        inputfiletruth='../data/dataset-E2E-goal-oriented-test-v1.0/tst'+options.testset+'/gold_task'+ \
                        options.taskchosen+'_tst'+options.testset+'.json'
        dict_result_test   = do_load_json_result(inputfiletest, 11)
        dict_result_truth  = do_load_json_result(inputfiletruth, 1)
    else:
        print '[ERROR] train or test'
        exit(1)

    ### Accuracy - Precision @1
    print 'Number of truth:', len(dict_result_truth), 'Number of test:', len(dict_result_test) 
    print str("Precision @1: ") + str(do_compute_score(dict_result_truth, dict_result_test, 1))
    print str("Precision @2: ") + str(do_compute_score(dict_result_truth, dict_result_test, 2))
    print str("Precision @5: ") + str(do_compute_score(dict_result_truth, dict_result_test, 5))
