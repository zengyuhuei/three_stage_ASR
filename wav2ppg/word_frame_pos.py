import codecs
import re 
import copy
import json
import argparse
def time2frame(time):
    if time % 30:
        return int(time / 30)+1
    else:
        return int(time / 30)
def to_dict_by_id(list_of_dict):
    _list_of_dict = copy.deepcopy(list_of_dict)
    dict_by_id = {}
    for _dict in _list_of_dict:
        _id = _dict['id']
        _dict.pop('id', None)
        if _id not in dict_by_id:
            dict_by_id[_id] = [_dict]
        else:
            dict_by_id[_id].append(_dict)
    return dict_by_id

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset')
    args = p.parse_args()
   
    MODES = ['test','train','dev']
    for mode in MODES:
        ctm_all = {}
        ctm_text = codecs.open(f"exp/tri3b/{args.dataset}_{mode}_ali/ctm", 'r', 'utf8')
        for l in ctm_text:
            ctm = {} 
            ctm_per_line = l.strip().split(" ")
            if ctm_per_line[4] == '<eps>':
                continue
            _id = ctm_per_line[0]
            ctm['start'] = time2frame(int(float(ctm_per_line[2])*1000))
            ctm['end'] = time2frame(int(round(float(ctm_per_line[2]) + float(ctm_per_line[3]) , 2)*1000)) 
            ctm['word'] = ctm_per_line[4].replace('<UNK>','UNK')
            if _id not in ctm_all:
                ctm_all[_id] = [ctm]
            else:
                ctm_all[_id].append(ctm)
        with open(f'data/{args.dataset}/{mode}_hires/word_with_frame_num.json', 'w', encoding='utf8') as f:
            f.write(json.dumps(ctm_all, ensure_ascii=False))
        
        
    
