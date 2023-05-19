
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import nltk
import json

def nltk_sentence_bleu(hypotheses, references, order=4):
    refs = []
    count = 0
    total_score = 0.0
    cc = SmoothingFunction()
    for hyp, ref in zip(hypotheses, references):
        hyp = hyp.split()
        ref = ref.split()
        refs.append([ref])
        if len(hyp) < order:
            continue
        else:
            score = nltk.translate.bleu([ref], hyp, smoothing_function=cc.method4)
            total_score += score
            count += 1
    S_BLEU = total_score / count

    return S_BLEU


ref = []
with open('', 'r') as f:  # ref1
    for line in f.readlines():
        jsonstr = json.loads(line)
        ref.append(jsonstr['gt'])
    f.close()

# print(ref)

hyp = []
with open('', 'r') as ff:  # ref1
    for line in ff.readlines():
        jsonstr = json.loads(line)
        hyp.append(jsonstr['gt'])
    f.close()

s_bleu = nltk_sentence_bleu(hyp, ref)
print(s_bleu)