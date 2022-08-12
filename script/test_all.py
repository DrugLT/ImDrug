import time

import numpy as np
import pandas as pd
import os
import json
import subprocess
import signal

def get_log(path):
    """
        retrieve all .log files from training output
    """
    log_list = []
    g = os.walk(path)
    for path, dir_list, file_list in g:
        for file_name in file_list:
            log_list.append(os.path.join(path, file_name))
    return log_list

def log_to_json(log, json_path):
    """
        convert .log file to configs in .json
    """
    with open(log, 'r') as f:
        next(f)

        params = f.readline()
        idx = params.find('{')
        params = eval(params[idx:])

        assert isinstance(params, dict)

        exp_id = os.path.basename(log).split('.')[0]
        params['test']['exp_id'] = exp_id

        with open(json_path, 'w') as f_json:
            json.dump(params, f_json)

    return params
    
if __name__ == '__main__':
    # for testing trained models all at once
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

    root_path = '/apdcephfs/share_1364275/shenyuan/DrugLT/multi_standard_output/drugbank/logs'
    result_path = '/apdcephfs/share_1364275/shenyuan/DrugLT/drugbank_standard_result/'
    json_path = './drugbank_temp.json'

    log_list = get_log(root_path)

    for log in log_list:
        params = log_to_json(log, json_path)

        # print('loss: %s, seed: %d' % (params['loss']['type'], params['seed']))
        start_time = time.time()
        cmd = 'python /apdcephfs/share_1364275/shenyuan/DrugLT/DrugLT/script/test.py --config %s > %s%s.log 2>&1' \ 
            % (json_path, result_path, params['test']['exp_id'])
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
        retrun_code = p.wait()
        # time.sleep(300)
        # os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    