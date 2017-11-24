import data_utils as utils
from sklearn import metrics
from sklearn.model_selection import train_test_split
from models.ent_net import *
from dataset_walker import get_taskfile_db
from templatized import compare_with_golden, generate_RDL_data
from score import do_compute_score, do_load_json_result

from six.moves import range, reduce
import tensorflow as tf
import numpy as np
import time
import collections
import os
import json
from tqdm import tqdm
import pickle as pkl

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Task 
tf.flags.DEFINE_boolean("train", False, "training the model")
tf.flags.DEFINE_boolean("generateRDL", False, "generate RDL data")
tf.flags.DEFINE_string("task", '1', "tasks 1-5")
tf.flags.DEFINE_string("testset", None, "testset 1-4 or all")
tf.flags.DEFINE_boolean("record", False, "use RDL")
tf.flags.DEFINE_boolean("postprocess", False, "postprocess the result")
tf.flags.DEFINE_boolean("all_utter", False, "False for only use bot utterances ")
# Parameters
tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for SGD.")
tf.flags.DEFINE_float("anneal_rate", 10, "Number of epochs between halving the learnign rate.")
tf.flags.DEFINE_float("anneal_stop_epoch", 40, "Epoch number to end annealed lr schedule.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 1, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size for training.")
tf.flags.DEFINE_integer("blk", 5, "Number of bloks in the mem")
tf.flags.DEFINE_integer("epochs", 100, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 100, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 100, "Maximum size of memory.")
tf.flags.DEFINE_integer("random_state", 88, "Random state.")
# Methods Forced
tf.flags.DEFINE_boolean("speaker_info", True, "Add speaker information to embedding.")
tf.flags.DEFINE_boolean("time_info_sents", True, "Add time information for per-response.")
# Methods 
tf.flags.DEFINE_boolean("augment", False, "increase dataset based on origin one.")
tf.flags.DEFINE_boolean('rm_unk_sent', True, "Give unk sent lower ranking ")
# File Path 
tf.flags.DEFINE_string("model_path", 'entnet-train-models/', "Directory containing database")
tf.flags.DEFINE_string("log_path", '../entnet-log/', "Directory log")
tf.flags.DEFINE_string("temp_path", '../data/processed/', "Directory of preprocessed data")
FLAGS = tf.flags.FLAGS

# preprocess the data
if not os.path.exists(FLAGS.temp_path): os.mkdir(FLAGS.temp_path)
if FLAGS.generateRDL:
    generate_RDL_data(FLAGS, FLAGS.testset)
    print "Finish RDL data generation..."
    exit(1)

# please do not use the totality of the GPU memory
session_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)    
session_config.gpu_options.per_process_gpu_memory_fraction = 0.90

inputtaskfile, taskchosen, testing_ratio, officialtestfile = get_taskfile_db(record=FLAGS.record, 
                                taskchosen=FLAGS.task, testset=FLAGS.testset, temp_path=FLAGS.temp_path)

# create dircotory
if not os.path.exists(FLAGS.model_path): os.mkdir(FLAGS.model_path)
if not os.path.exists(FLAGS.model_path+'{}/'.format(taskchosen)): os.mkdir(FLAGS.model_path+'{}/'.format(taskchosen))
if not os.path.exists(FLAGS.log_path): os.mkdir(FLAGS.log_path)

if FLAGS.train:
    # load task data and contextInfo
    train, val, test, candidates, train_cand, val_cand, test_cand, cand_idx, idx_cand = utils.load_task(inputtaskfile, 
                                                    FLAGS, testing_ratio=testing_ratio, template=FLAGS.record, 
                                                    buildtestset=(officialtestfile==[]))
    # get the contextInfo data stored
    trainInfo = [d['a'].pop() for d in train]
    valInfo = [d['a'].pop() for d in val]
    testInfo = [d['a'].pop() for d in test] if (len(officialtestfile)==0) else []

    # get vocab and sentence information from data
    vocab, word_idx, max_story_size, mean_story_size, sentence_size, query_size = utils.data_information(train, candidates)
    vocab_size = len(word_idx) + 1 # +1 for nil word (0 for nil)
    sentence_size = max(query_size, sentence_size) + 5 # add some space for testing data
    memory_size = min(FLAGS.memory_size, max_story_size) 

    # vectorize data
    trainS, trainQ, trainA, trainID = utils.vectorize_data(train, word_idx, sentence_size, FLAGS.batch_size, memory_size, cand_idx)
    valS, valQ, valA, valID = utils.vectorize_data(val, word_idx, sentence_size, FLAGS.batch_size, memory_size, cand_idx)
    testS, testQ, testA, testID = utils.vectorize_data(test, word_idx, sentence_size, FLAGS.batch_size, memory_size, cand_idx)
    C, cand_idx, idx_cand = utils.vectorize_candidates(cand_idx, idx_cand, word_idx, sentence_size)

    # params
    n_train = np.array(trainS).shape[0]
    n_test = np.array(testS).shape[0]
    n_val =  np.array(valS).shape[0]
    tf.set_random_seed(FLAGS.random_state)
    batch_size = FLAGS.batch_size
    batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
    batches = [(start, end) for start, end in batches]

    print "input data example: ", train[5]['utter_list'][0]
    print "overall bot utterance candidates: ", len(cand_idx)
    print 'vocab_size', vocab_size
    print "Longest sentence length", sentence_size
    print "Longest story length", max_story_size
    print "Training Size", n_train
    print "Validation Size", n_val
    print "Testing Size", n_test

    f_param = open(FLAGS.model_path+'{}/param'.format(taskchosen),'w') 
    pkl.dump([cand_idx,word_idx,sentence_size, vocab_size, memory_size, C], f_param)
else:
    if os.path.exists(FLAGS.model_path+'{}/param'.format(taskchosen)):
        f_param = open(FLAGS.model_path+'{}/param'.format(taskchosen),'r') 
        cand_idx, word_idx, sentence_size, vocab_size, memory_size, C = pkl.load(f_param)
    else:
        print '[ERROR] No param is stored...'
        exit(1)

with tf.Session(config=session_config) as sess:
    model = EntityNetwork(vocab_size, sentence_size, memory_size, FLAGS.blk, FLAGS.embedding_size,
                    len(cand_idx),0.0, FLAGS.max_grad_norm)

    start_time = time.time()
    if FLAGS.train:
        log = open(FLAGS.log_path+'log-task-template-entnet-'+taskchosen+'.txt', 'w')

        saver = tf.train.Saver()
        best_acc_val = 0
        cnt = 0
        cnt_one = 0
        for t in range(1, FLAGS.epochs+1):
            # Stepped learning rate
            if t - 1 <= FLAGS.anneal_stop_epoch:
                anneal = 2.0 ** ((t - 1) // FLAGS.anneal_rate)
            else:
                anneal = 2.0 ** (FLAGS.anneal_stop_epoch // FLAGS.anneal_rate)
            lr = FLAGS.learning_rate / anneal
            # data shuffling
            np.random.shuffle(batches)

            train_labels, val_labels = trainA, valA
            total_cost,total_acc,index_bat = 0.0, 0.0, 0
            prog_bar = tqdm(batches)
            for start, end in prog_bar:
                index_bat += 1
                s = trainS[start:end]
                q = trainQ[start:end]
                a = trainA[start:end]
                cost_t,acc_t = model.batch_fit(s, q, a, lr)
                total_cost += cost_t
                total_acc += acc_t
                prog_bar.set_description('Acc: {:10.4f} Loss: {:10.4f}'.format(total_acc/index_bat,total_cost/index_bat))

            if t % FLAGS.evaluation_interval == 0:
                val_preds, val_preds_prob, val_loss = utils.batch_evaluate(model, valS, valQ, valA, C, n_val, batch_size)
                val_preds_ten, val_preds_ranking = utils.get_pred_10cands(val_cand, val_preds_prob, dialogID=valID, 
                                                                            rm_unk_sent=FLAGS.rm_unk_sent)
                val_acc = metrics.accuracy_score(val_preds, val_labels)
                val_acc_ten = metrics.accuracy_score(val_preds_ten, val_labels)
                val_sq = [ [d['utter_list'], d['q']] for d in val]
                val_acc_ten_slotfilling = compare_with_golden(idx_cand, val_preds_ten, valInfo, utils.combine_SQ(val_sq), 
                                            show_error=False, record=FLAGS.record)
                
                print '-----------------------'
                print 'Epoch', t, 'Validation Loss:', val_loss, 'Val Acc Model:', val_acc
                print 'Val Acc from Candidates (w/o, w/ Lexicalization):', val_acc_ten, val_acc_ten_slotfilling

            if(val_acc >= best_acc_val):
                best_acc_val = val_acc
                cnt = 0
                model.saver.save(model._sess, FLAGS.model_path + '{}/entity_model.ckpt'.format(taskchosen), 
                    global_step=t)
                print '[Saving the model at val acc %s...]'%(str(val_acc))
            else:
                cnt += 1
            print 'COUNT VALUE %d' % int(cnt)
            print '-----------------------'
            if val_acc == 1.0:
                cnt_one += 1
            if val_acc == 1.0 and cnt_one == 3:
                break
            if cnt>= 10:
                break
    
    # restore checkpoint
    ckpt = tf.train.latest_checkpoint(FLAGS.model_path+str(taskchosen))
    if ckpt:
        print '>> restoring checkpoint from', ckpt
        model.saver.restore(model._sess, ckpt)

    # Testing 
    if officialtestfile == []:
        try:
            test_labels = testA
        except:
            if FLAGS.testset!=None and FLAGS.record:
                print "[Error] RDL data no found..."
            else:
                print "[Error] No test data given..."
            exit(1)

        test_preds, test_preds_prob, _ = utils.batch_evaluate(model, testS, testQ, [], C, n_test, batch_size)
        test_preds_ten, test_preds_ranking = utils.get_pred_10cands(test_cand, test_preds_prob, dialogID=testID, 
                                                                    rm_unk_sent=FLAGS.rm_unk_sent)
        test_acc = metrics.accuracy_score(test_preds, test_labels)
        test_acc_ten = metrics.accuracy_score(test_preds_ten, test_labels)
        test_sq = [ [d['utter_list'], d['q']] for d in test]
        test_acc_ten_slotfilling = compare_with_golden(idx_cand, test_preds_ten, testInfo, utils.combine_SQ(test_sq),
                                            show_error=True, record=FLAGS.record)
        print "### Testing Accuracy Model:", test_acc
        print '### Testing Accuracy from Candidates (w/, w/o Lexicalization):', test_acc_ten, test_acc_ten_slotfilling
        print "### %s seconds per epoch" % str(float(time.time() - start_time)/FLAGS.epochs) 
        with open('ranking.json', 'w') as f:
            json.dump(test_preds_ranking, f)
        task_name = ["dialog-task1API", "dialog-task2REFINE", "dialog-task3OPTIONS", "dialog-task4INFOS", "dialog-task5FULL"]
        dict_result_test   = do_load_json_result('RANKING.json', 10)
        dict_result_truth  = do_load_json_result("../data/dataset-E2E-goal-oriented/"+task_name[int(taskchosen)-1] + '-truth.json', 1)
        print str("Precision @1: ") + str(do_compute_score(dict_result_truth, dict_result_test, 1))
        print str("Precision @2: ") + str(do_compute_score(dict_result_truth, dict_result_test, 2))
        print str("Precision @5: ") + str(do_compute_score(dict_result_truth, dict_result_test, 5))
    else:
        print "Start predicting official testfile..."
        for testfile in officialtestfile:
            FLAGS.augment = False
            test_o, _, _, _, test_cand_o, _, _, cand_idx_new, idx_cand_o = utils.load_task([testfile], FLAGS, 
                                                testing_ratio=0, template=True, testing=True, cand_idx_out=cand_idx)
            testInfo_o = [d['a'].pop() for d in test_o]
            testS_o, testQ_o, _, testID_o = utils.vectorize_data(test_o, word_idx, sentence_size, FLAGS.batch_size, memory_size, 
                                                                cand_idx_new)
            test_preds_o, test_preds_prob_o, _ = utils.batch_evaluate(model, testS_o, testQ_o, [], C, len(testQ_o), FLAGS.batch_size)
            if FLAGS.postprocess:
                test_preds_ten_o, test_preds_ranking_o = utils.get_pred_10cands(test_cand_o, test_preds_prob_o, 
                                                        dialogID=testID_o, rm_unk_sent=FLAGS.rm_unk_sent,
                                                        data=test_o, idx_cand=idx_cand_o, testInfo=testInfo_o)
            else:
                test_preds_ten_o, test_preds_ranking_o = utils.get_pred_10cands(test_cand_o, test_preds_prob_o, 
                                                        dialogID=testID_o, rm_unk_sent=FLAGS.rm_unk_sent)
            test_sq_o = [ [d['utter_list'], d['q']] for d in test_o]
            ranking_path = 'ranking_ent/'
            options_testset = testfile.split('/')[-2]
            options_taskchosen = testfile.split('/')[-1][7:12]
            testname_origin = testfile.split('/')[-1].replace('-RDL','').replace('.json','.answers.json')
            if not os.path.exists(ranking_path): os.mkdir(ranking_path)
            if not os.path.exists(ranking_path+options_testset): os.mkdir(ranking_path+options_testset)   
            name = ranking_path+options_testset+'/'+testname_origin
            print 'Dumping: ', name
            with open(name, 'w') as f:
                json.dump(test_preds_ranking_o, f)