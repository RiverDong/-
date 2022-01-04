import pandas as pd

def get_hardneg_train_dataset(input_file, output_file, max_hard_negs = 15):
    data = dict()
    with open(input_file,'r') as f:
        header = f.readline().strip().split('\t')
        count = 0
        for line in f:
            count += 1
            dataline = line.strip().split('\t')
            datapoint = dict()
            for index,title in enumerate(header):
                datapoint[title] = dataline[index]
            process_datapoint(datapoint, data, max_hard_negs = max_hard_negs)


def process_datapoint(datapoint, data, max_hard_negs, ignore_negatives=False):
    QID = 'qid'
    PID = 'pid'
    PASSAGE = 'passage'
    QUERY = 'query'
    LABEL = 'label'
    HARD_NEGATIVES = 'hard_negatives'
    if datapoint[QID] in data:
        val = data[datapoint[QID]]
    else:
        val = dict()
        val[QUERY] = datapoint[QUERY]
        val[PASSAGE] = []
        val[HARD_NEGATIVES] = []

    if int(datapoint[LABEL]) == 1:
        val[PASSAGE].append(datapoint[PASSAGE])
    elif (int(datapoint[LABEL]) == 0) and (not ignore_negatives) and len(val[HARD_NEGATIVES]) < max_hard_negs:
        val[HARD_NEGATIVES].append(datapoint[PASSAGE])
    data[datapoint[QID]] = val


def process_datapoint_to_triplet(datapoint, data):
    QID = 'qid'
    PID = 'pid'
    POS_PID = 'pos_pid'
    NEG_PID = 'neg_pid'
    PASSAGE = 'passage'
    QUERY = 'query'
    LABEL = 'label'
    HARD_NEGATIVES = 'hard_negatives'
    if datapoint[QID] in data:
        val = data[datapoint[QID]]
    else:
        val = dict()
        val[QID] = datapoint[QID]
        val[QUERY] = datapoint[QUERY]
        val[PASSAGE] = []
        val[POS_PID] = []
        val[HARD_NEGATIVES] = []
        val[NEG_PID] = []

    if int(datapoint[LABEL]) == 1:
        val[PASSAGE].append(datapoint[PASSAGE])
        val[POS_PID].append(datapoint[PID])
    elif int(datapoint[LABEL]) == 0:
        val[HARD_NEGATIVES].append(datapoint[PASSAGE])
        val[NEG_PID].append(datapoint[PID])
    data[datapoint[QID]] = val


def get_triplet_train_dataset(input_file, output_file, true_text=True):
    data = dict()
    with open(input_file, 'r') as f:
        header = f.readline().strip().split('\t')
        count = 0
        for line in f:
            count += 1
            dataline = line.strip().split('\t')
            datapoint = dict()
            for index, title in enumerate(header):
                datapoint[title] = dataline[index]
            process_datapoint_to_triplet(datapoint, data)

    keys_to_del = []
    for key in data:
        if len(data[key]['passage']) == 0:
            keys_to_del.append(key)

    for key in keys_to_del:
        del data[key]

    output = []
    for key in data:
        for i, passage in enumerate(data[key]['passage']):
            for j, neg_passage in enumerate(data[key]['hard_negatives']):
                temp = dict()
                if true_text:
                    temp['query'] = data[key]['query']
                    temp['passage'] = passage
                    temp['har_neg'] = neg_passage
                else:
                    temp['qid'] = key
                    temp['posid'] = data[key]['pos_pid'][i]
                    temp['negid'] = data[key]['neg_pid'][j]
                output.append(temp)
    out_df = pd.DataFrame(output)
    out_df.to_csv(output_file, header=False, index=False, sep='\t')