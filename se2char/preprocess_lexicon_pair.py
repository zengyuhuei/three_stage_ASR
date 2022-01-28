import re
from torch.nn.modules import linear
from tqdm import tqdm
import pickle

def is_only_english_letter_or_whitespace(text):
    reg = re.compile(r"^[A-Za-z\s]+$")
    if reg.match(text):
        return True
    else:
        return False

def is_any_chinese_letter(text):
    reg = re.compile(r"[\u4e00-\u9fff]+")
    if reg.match(text):
        return True
    else:
        return False

def split_lexicon(lexicon):
    splited_lexicons = []
    # chinese lexicon will end with digits
    reg = re.compile(r"[0-9]+")
    start = 0
    for i in re.finditer('[0-9]', lexicon):
        splited_lexicons.append(lexicon[start:i.end()])
        start = i.end() + 1
    return splited_lexicons

def preprocess(additional_lines = []):
    word_lexicon_pair = {}
    lexicon_word_pair = {}
    with open("./lexicon.txt", "r") as f:
        lines = f.readlines()
        for line in tqdm(lines + additional_lines):
            if is_only_english_letter_or_whitespace(line) or not is_any_chinese_letter(line):
                continue
            else:
                # remove English words and English lexicon
                #line = ' '.join(re.sub('[A-Z]+', '', line).split())
                line = re.sub('[A-Z]+', '', line)
            line = line.split()
            word, lexicon = line[0], ' '.join(line[1:])
            word_lexicon_pair[word] = lexicon
            for word_lexicon in zip([w for w in word], split_lexicon(lexicon)):
                w, l = word_lexicon[0], word_lexicon[1]
                if l not in lexicon_word_pair:
                    lexicon_word_pair[l] = []
                if w not in lexicon_word_pair[l]:
                    lexicon_word_pair[l].append(w)
    with open('data_for_MLM/lexicon_word_pair.pkl', 'wb') as f:
        pickle.dump(lexicon_word_pair, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data_for_MLM/word_lexicon_pair.pkl', 'wb') as f:
        pickle.dump(word_lexicon_pair, f, protocol=pickle.HIGHEST_PROTOCOL)
if __name__ == "__main__":
    additional_lines = ['茿 zh u2',
                        '痹 b i4',
                        '僈 m an4',
                        '瑮 l i4',
                        '荖 l ao3',
                        '謢 l u1',
                        '体 t i3',
                        '扺 zh i3',
                        '漧 g an1',
                        '蜁 x uan2',
                        '洒 s a3',
                        '婓 f ei1',
                        '苳 d ong1',
                        '衪 y i2',
                        '扥 d un4',
                        '羡 x ian4',
                        '塭 w en1',
                        '鈙 q in2']
    preprocess(additional_lines)
