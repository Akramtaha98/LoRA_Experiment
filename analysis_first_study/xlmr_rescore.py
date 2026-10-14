"""
Independent NLI re-scoring (verification pass).

Re-scores the 8 real Config C conditions' logged generations with a
second, architecturally distinct multilingual NLI model
(XLM-RoBERTa-base, trained on SNLI+MNLI+ANLI+XNLI) and compares the
resulting entailment probabilities against the training-time
mDeBERTa-v3-base-mnli-xnli F_faith scores, to check whether the
composite-loss faithfulness signal is specific to the classifier used
at training time.

Run analysis/build_rescore_contexts.py first to produce
nli_rescore_contexts.json.
"""
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL = "symanto/xlm-roberta-base-snli-mnli-anli-xnli"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.eval()
ENT_IDX = 0  # config.json: id2label = {0: ENTAILMENT, 1: NEUTRAL, 2: CONTRADICTION}


def entail_prob(context, answer):
    if not answer or not answer.strip() or not context.strip():
        return 0.0
    enc = tok(context, answer, truncation="only_first", max_length=512, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return round(probs[ENT_IDX].item(), 4)


with open("nli_rescore_contexts.json", encoding="utf-8") as f:
    ctx_data = json.load(f)

# 8 real Config C conditions, generated answers pulled from the checkpoint
# logs in experiment_results/checkpoint_full_*.jsonl ("samples" field,
# first 5 logged examples per run == eval_indices[0:5]).
conditions = {
    ("qlora", "arabic"):  {"mean_f_faith_full30": 0.4462, "samples": [
        {"gold": "308", "generated": "ماريو أ جان"},
        {"gold": "136", "generated": "فريق بانثرز"},
        {"gold": "118", "generated": "فريقبانثرز"},
        {"gold": "أربعة", "generated": "ثلاثة"},
        {"gold": "كاوان شورت", "generated": "الفريق بأعلى عدد من الأين"},
    ]},
    ("qlora", "malay"):   {"mean_f_faith_full30": 0.2925, "samples": [
        {"gold": "Untuk menambah kelantangan, tingkatkan daya anda menekan kekunci", "generated": "Tingkatkan daya anda menekan kekunci"},
        {"gold": "Lebih kelajuan", "generated": "Purang kelajuan"},
        {"gold": "Supaya imej memenuhi seluruh skrin", "generated": "Pada imej pada TV telah dipotong bingkainya"},
        {"gold": "Sari kata terpotong sedikit", "generated": "Bingkai terpotong sedikit"},
        {"gold": "Bordentown", "generated": "Bantuan di bawah Kepimpinan serangan utama 2,400 askar akan membawa 700 milisi merentasi sungai Assunpink Creek"},
    ]},
    ("adalora", "arabic"): {"mean_f_faith_full30": 0.4371, "samples": [
        {"gold": "308", "generated": "<extra_id_0> الخصمية بهم على ينت"},
        {"gold": "136", "generated": "<extra_id_0> ).ال مرة) "},
        {"gold": "118", "generated": "svá و. -ازات.     <extra_id_48>  او ل  شف    keskshire <extra_id_38> <extra_id_7> <extra_id_45>"},
        {"gold": "أربعة", "generated": "<extra_id_0>  <extra_id_1> على ترتيابية ينتقل"},
        {"gold": "كاوان شورت", "generated": "<extra_id_0>اً يمين لبول "},
    ]},
    ("adalora", "malay"):  {"mean_f_faith_full30": 0.1372, "samples": [
        {"gold": "Untuk menambah kelantangan, tingkatkan daya anda menekan kekunci", "generated": "<extra_id_0>langan"},
        {"gold": "Lebih kelajuan", "generated": "<extra_id_0>. <extra_id_46>"},
        {"gold": "Supaya imej memenuhi seluruh skrin", "generated": "<extra_id_0> tepi <extra_id_1> ke apatepi luar  <extra_id_55>akan (3) <extra_id_1>kan  <extra_id_56>  <extra_id_56> a Bagi"},
        {"gold": "Sari kata terpotong sedikit", "generated": "<extra_id_0>"},
        {"gold": "Bordentown", "generated": "<extra_id_0> British <extra_id_1> British di British, <extra_id_2>. Trenton. sungairangan di British British  <extra_id_50> British?trenangkanrangan - <extra_id_4> Trenton di Trenton"},
    ]},
    ("dora", "arabic"):   {"mean_f_faith_full30": 0.4440, "samples": [
        {"gold": "308", "generated": "خم خم "},
        {"gold": "136", "generated": "أسبعة"},
        {"gold": "118", "generated": "88 عرقلة"},
        {"gold": "أربعة", "generated": "[3]"},
        {"gold": "كاوان شورت", "generated": "ثلاثة"},
    ]},
    ("dora", "malay"):    {"mean_f_faith_full30": 0.2468, "samples": [
        {"gold": "Untuk menambah kelantangan, tingkatkan daya anda menekan kekunci", "generated": ""},
        {"gold": "Lebih kelajuan", "generated": "Kurang"},
        {"gold": "Supaya imej memenuhi seluruh skrin", "generated": "Underscan"},
        {"gold": "Sari kata terpotong sedikit", "generated": "Bingkai terpotong"},
        {"gold": "Bordentown", "generated": "Between"},
    ]},
    ("vera", "arabic"):   {"mean_f_faith_full30": 0.4408, "samples": [
        {"gold": "308", "generated": "<extra_id_0>ين. ( <extra_id_1> كأس،5 و"},
        {"gold": "136", "generated": "<extra_id_0>. <extra_id_55> قاء و &#8221;n <extra_id_1> LENA. <extra_id_56> : لعب <extra_id_54> لوس.  <extra_id_45> <extra_id_16>.w🙴ؤ    <extra_id_4>  <extra_id_47> كثير როგორ"},
        {"gold": "118", "generated": "<extra_id_0>  <extra_id_24>рым   <extra_id_42>  <extra_id_56>  <extra_id_53> 114  <extra_id_39> <extra_id_23>, قبلفى <extra_id_56> <extra_id_14>  <extra_id_28> <extra_id_51> ԫի <extra_id_35> Arvبيك <extra_id_4>대회 aș  <extra_id_39> تامارتنتوارة <extra_id_4>"},
        {"gold": "أربعة", "generated": "<extra_id_0> عدد الهداف في "},
        {"gold": "كاوان شورت", "generated": "<extra_id_0>.   وسطما <extra_id_20> [45].....  <extra_id_37> pront <extra_id_48>  <extra_id_39>ęczقسطات"},
    ]},
    ("vera", "malay"):    {"mean_f_faith_full30": 0.0734, "samples": [
        {"gold": "Untuk menambah kelantangan, tingkatkan daya anda menekan kekunci", "generated": "<extra_id_0> /秀:osan <extra_id_42>a  <extra_id_55> a <extra_id_5> di: zaidi,"},
        {"gold": "Lebih kelajuan", "generated": "<extra_id_0>Jadi., Al <extra_id_1>.  <extra_id_52>- <extra_id_14>Tahuultau <extra_id_53>quiz <extra_id_25> lancaman, <extra_id_56> ndakan berikut 間形עועות <extra_id_52>: ke - <extra_id_13>"},
        {"gold": "Supaya imej memenuhi seluruh skrin", "generated": "<extra_id_0> but <extra_id_56>"},
        {"gold": "Sari kata terpotong sedikit", "generated": "<extra_id_0> format :"},
        {"gold": "Bordentown", "generated": "<extra_id_0>on <extra_id_19>"},
    ]},
}

results = {}
for (variant, lang), d in conditions.items():
    ctx_list = ctx_data[lang]
    xlmr_scores = []
    for i, samp in enumerate(d["samples"]):
        context = ctx_list[i]["context"]
        p = entail_prob(context, samp["generated"])
        xlmr_scores.append(p)
    mean_xlmr = round(sum(xlmr_scores) / len(xlmr_scores), 4)
    results[f"{variant}_{lang}"] = {
        "xlmr_mean_entailment_5ex": mean_xlmr,
        "xlmr_scores_5ex": xlmr_scores,
        "mdeberta_mean_f_faith_30ex": d["mean_f_faith_full30"],
    }
    print(f"{variant:8s} {lang:7s}  XLM-R(5ex)={mean_xlmr:.4f}   mDeBERTa(30ex)={d['mean_f_faith_full30']:.4f}   scores={xlmr_scores}")

with open("xlmr_rescore_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\nWrote xlmr_rescore_results.json")
