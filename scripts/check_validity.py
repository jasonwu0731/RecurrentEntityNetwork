import os
import json

### The current dialog format
### [{dialog_id : " ", lst_candidate_id: [{candidate_id: " ", rank: " "}, ...]}]

def do_parse_cmdline():

    from optparse import OptionParser
    parser = OptionParser()

    parser.add_option("--testpath", dest="testpath",
                      help="directory of the task", metavar="FILE")

    (options, args) = parser.parse_args()



    return options.testpath

def _get_source_paths(source_dir):
    prefix = "dialog-task"
    endfix = "tst1000.answers.json"
    names = os.listdir(source_dir)
    task = []
    for name in names:
        if name.startswith(prefix) and name.endswith(endfix):
            task.append(source_dir+name)
    return task

if __name__ == '__main__':


    dir_path = do_parse_cmdline()

    data = []
    for tst in ['tst4/', 'tst3/', 'tst2/', 'tst1/']:
        t12345 = _get_source_paths(dir_path+tst)
        data += t12345
    
    if len(data) != 20:
        print "[Error] The result should have 20 files..."
        exit(1)

    for f in data:
        with open(f, 'rb') as fd:

            print f
            json_data = json.load(fd)

            if len(json_data) != 1000:
                print "[Error] The result len should be 1000 ..."
                exit(1)

            if (type(json_data) != list):
                print "[Error] The result file should be a list ..."
                exit(1)

            for item in json_data:
                if (item.get("dialog_id") == None):
                    print "[Error] No dialog_id key founded ..."
                    exit(1)
                if (item.get('lst_candidate_id') == None):
                    print "[Error] No lst_candidate_id key founded ..."
                    exit(1)

                for candidate in item['lst_candidate_id']:
                    if (candidate.get('rank') == None):
                        print "[Error] one candidate has no rank key ..."
                        exit(1)
                    if (candidate.get('candidate_id') == None):
                        print "[Error] one candidate has no id key ..."
                        exit(1)


    print str("[Success]: Valid format")