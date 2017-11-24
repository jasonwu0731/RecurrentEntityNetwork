from operator import itemgetter
import numpy as np
import json
import os
from tqdm import tqdm
import tensorflow as  tf

from data_utils import get_type_dict, load_task
from dataset_walker import _get_source_paths
from path_config import *

### [{dialog_id : " ",
#     utterances: [" "],
#     candidates: [{candidate_id: " " , utterance: ""}, ... ],
#     answer: {candidate_id: " " , utterance: ""} ]
def get_RDL_data(kb_path, data): 
    type_dict = get_type_dict(kb_path)
    rest_dict = get_restaurant_dict(kb_path)
    dataNew = []
    prog_bar = tqdm(data)
    for d in prog_bar:
        ul, q, gold_a, c = d['utter_list'], d['q'], d['a'], d['cand_list']
        utterances = []
        story = {}
        story = {'dialog_id': d['dialog_id']}
        bot_answers = []
        utter_list = []
        contextInfo = {} # initial
        for key in type_dict.keys():
            contextInfo[key] = []

        for i in range(len(ul)):
            utterNew = entity_extraction(type_dict, ul[i], contextInfo)
            utter_list.append( ' '.join(utterNew) )

        _q = entity_extraction(type_dict, q, contextInfo)
        utter_list.append(' '.join(_q))
        contextInfo['goldAnswer'] = gold_a
        contextInfo['dialog_id'] = story['dialog_id']
        a = entity_extraction(type_dict, gold_a, contextInfo, updateInfo=False)

        for i in range(len(c)):
            bot_answers.append({ 'candidate_id': c[i]['candidate_id'], 
                                    'utterance': ' '.join(entity_extraction(type_dict, c[i]['utterance'], 
                                    contextInfo, updateInfo=False)) } )
        story['utterances_untemplate'] = [' '.join(u) for u in ul] + [' '.join(q)]
        story["utterances"] = utter_list
        story["candidates"] = bot_answers
        story["answer"] = { 'story_id': story['dialog_id'], 'utterance': ' '.join(a) } 
        story['contextInfo'] = contextInfo
        dataNew.append(story)
    return dataNew

def store_template_json(filepath, kb_path, data):
    dataNew = get_RDL_data(kb_path, data)
    add_name = '-RDL.json'
    with open(filepath+add_name, 'w') as f:
        json.dump(dataNew, f)

def entity_extraction(type_dict, sentence, contextInfo, updateInfo=True):
    sentNew = [ x for x in sentence]
    updated = []
    for index, word in enumerate(sentNew):
        word = word.replace(',','')
        for type_name in type_dict:
            if word in type_dict[type_name] and type_name != 'R_rating': 
                if word in contextInfo[type_name]:
                    entity_index = contextInfo[type_name].index(word) + 1
                    sentNew[index] = "#"+ type_name + '_' + str(entity_index) +"#"
                else:
                    if updateInfo:
                        contextInfo[type_name].append(word)
                        sentNew[index] = "#"+ type_name + '_' + str(len(contextInfo[type_name])) +"#"
                    else:
                        sentNew[index] = 'UNK'
    return sentNew

def get_restaurant_dict(kb_path):
    rest_dict = {}
    fd = open(kb_path, 'rb')
    for line in fd:
        x = line.replace('\n','').split('\t')
        rest_name = x[0].split(' ')[1]
        entity = x[0].split(' ')[2]
        entity_value = x[1]
        if rest_name not in rest_dict.keys():
            rest_dict[rest_name] = {}
        if entity not in rest_dict[rest_name].keys():
            rest_dict[rest_name][entity] = entity_value
    return rest_dict

def fill_template(dialogInfo, template_sentence, story_utters, rest_dict, record=False):
    generate_sent = []
    if record:
        for word in template_sentence.split(' '):
            if '#' in word:
                template_type = word.replace('#','')
                template_type = template_type.replace('r', 'R', 1)
                template_idx = int(template_type.split('_')[-1])
                template_type = '_'.join(template_type.split('_')[:-1])
                if template_idx > len(dialogInfo[template_type]):
                    if len(dialogInfo[template_type]) == 0:
                        generate_sent.append('UNK')
                    else:
                        generate_sent.append(dialogInfo[template_type][-1])
                else:
                    generate_sent.append(dialogInfo[template_type][int(template_idx)-1])
            else:
                generate_sent.append(word)
    else:
        generate_sent = template_sentence.split(' ')
    return generate_sent

def compare_with_golden(idx_cand, preds_array, dialogInfoList, storys, show_error=False, record=False):
    compare_list = []
    if show_error:
        fname = 'TestError.txt'
        print "Dump Error Samples to "+fname
        f = open(fname,'w')

    for index, dialogInfo in enumerate(dialogInfoList):
        gold_a = dialogInfo['goldAnswer']
        story_utters = storys[index]
        pred_sent = fill_template(dialogInfo, idx_cand[preds_array[index]], story_utters, {}, record=record)
        if gold_a == pred_sent:
            compare_list.append(1)
        else:
            compare_list.append(0)
            if show_error:
                f.write( 'Dialog_id: ' + str(dialogInfo['dialog_id']) + '\n' )
                f.write( 'GOLD: ' + str(gold_a) + '\n' )
                f.write( 'PREDICT: ' + str(idx_cand[preds_array[index]]) +  str(pred_sent) + '\n' )
                f.write( 'Story_utters:' + str(story_utters) + '\n')
                f.write( '\n\n' )
    return sum(compare_list)*1.0/len(compare_list)

def generate_RDL_data(FLAGS, testset=None):
    if testset==None:
        kb_path = DATASET_PATH+"extendedkb1.txt"
        for i in range(len(TASK_NAME)):
            inputtaskfile = DATASET_PATH+TASK_NAME[i]+'-kb1_atmosphere-distr0.5-trn10000.json'
            print "Generating RDL data for ", inputtaskfile
            data, _, _, _, _, _, _, _, _ = load_task([inputtaskfile], FLAGS, testing_ratio=0.0)
            store_template_json(FLAGS.temp_path+'task'+str(i+1) , kb_path, data)
    else:
        data = []
        loop_dir = ['tst4/', 'tst3/', 'tst2/', 'tst1/'] if testset=='all' else ['tst{}/'.format(str(testset))]
        for tst in loop_dir:
            t12345 = _get_source_paths(TEST_DATASET_PATH+tst)
            data.append(t12345)
        for d12345 in data:
            for d in d12345:
                inputtaskfile =  d + '.json'
                outputtaskfile = d + '-RDL.json'
                kb_path = TEST_DATASET_PATH+'extendedkb1.txt' if ('kb1' in d) else TEST_DATASET_PATH+'extendedkb2.txt'
                print "Generating RDL data for ", d     
                data, _, _, _, _, _, _, _, _ = load_task([inputtaskfile], FLAGS, testing_ratio=0.0, testing=True)
                store_template_json(d.replace('.json', '') , kb_path, data)

        
