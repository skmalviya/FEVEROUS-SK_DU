# Modified Hu Nan's script to apply BM25 to retrieve top-n pages 
import argparse
import json
import os.path

from tqdm import tqdm
from baseline.drqa import retriever
from baseline.drqa.retriever import DocDB
from utils.annotation_processor import AnnotationProcessor
from utils.wiki_processor import WikiDataProcessor
from baseline.drqa.retriever.doc_db import DocDB
from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage
from unidecode import unidecode
import unicodedata
from urllib.parse import unquote
from cleantext import clean
import re
pt = re.compile(r"\[\[.*?\|(.*?)]]")


def clean_text(text):
    text = re.sub(pt, r"\1", text)
    text = unquote(text)
    text = unicodedata.normalize('NFD', text)
    text = clean(text.strip(),fix_unicode=True,               # fix various unicode errors
    to_ascii=False,                  # transliterate to closest ASCII representation
    lower=False,                     # lowercase text
    no_line_breaks=False,           # fully strip line breaks as opposed to only normalizing them
    no_urls=True,                  # replace all URLs with a special token
    no_emails=False,                # replace all email addresses with a special token
    no_phone_numbers=False,         # replace all phone numbers with a special token
    no_numbers=False,               # replace all numbers with a special token
    no_digits=False,                # replace all digits with a special token
    no_currency_symbols=False,      # replace all currency symbols with a special token
    no_punct=False,                 # remove punctuations
    replace_with_url="<URL>",
    replace_with_email="<EMAIL>",
    replace_with_phone_number="<PHONE>",
    replace_with_number="<NUMBER>",
    replace_with_digit="0",
    replace_with_currency_symbol="<CUR>",
    lang="en"                       # set to 'de' for German special handling
    )
    return text


def process(ranker, query, k=1):
    doc_names, doc_scores = ranker.closest_docs(unidecode(clean_text(query)), k)

    return zip(doc_names, doc_scores)


def ent_w_unicode(ents, document_titles, document_titles_unidecode):
    ents_out = []
    for e in ents:
        if e in document_titles_unidecode:
            indx = document_titles_unidecode.index(e)
            ents_out.append(document_titles[indx])

    return ents_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',type=str)
    parser.add_argument('--count',type=int, default=1)
    parser.add_argument('--db',type=str)
    parser.add_argument('--data_path',type=str)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument("--db_path", type=str, default="data/feverous_wikiv1.db")
    args = parser.parse_args()
    #print(args)
    k = args.count
    split = args.split
    ranker = retriever.get_class('bm25')(lucene_index_path=args.model)
    annotation_processor = AnnotationProcessor("{}/{}.jsonl".format(args.data_path, args.split))
    db = DocDB(args.db)
    document_titles = set(db.get_doc_ids())
    document_titles_unidecode = [unidecode(d) for d in document_titles]
    # alias_dict_path = os.path.join(args.data_path, "alias_page_title_dict.json")
    # if os.path.exists(alias_dict_path):
    #     alias_dict = json.load(open(alias_dict_path,"r", encoding="utf-8"))
    # else:
    #     import re
    #     alias_dict = {}
    #     pt = re.compile('\"(.*?)\"')
    #     db = FeverousDB(args.db_path)
    #     for dt in tqdm(document_titles):
    #         page_json = db.get_doc_json(dt)
    #         wiki_page = WikiPage(dt, page_json)
    #         for sent_id in range(3):
    #             rd_sent = wiki_page.page_items.get(f"sentence_{sent_id}", None)
    #             if rd_sent and ("redirect" in rd_sent.content):
    #                 alias_titles = pt.findall(rd_sent.content)
    #                 for at in alias_titles:
    #                     alias_dict[at] = dt
    #                     #print(at, dt)
    #
    # print("alias_page_title_dict length:", len(alias_dict))

    with open("{0}/{1}.pages.p{2}.bm25.jsonl".format(args.data_path, args.split, k), "w", encoding='utf8') as f2:
        annotations = [annotation for annotation in annotation_processor]
        for i, annotation in enumerate(tqdm(annotations)):
            js = {}
            js['id'] = annotation.get_id()
            js['claim'] = annotation.get_claim()
            if js['id'] == 6957:
                s = 'shri'
            groups = annotation.get_claim_entities()
            entities = [el[0] for el in groups]

            groups_unidecode = annotation.get_claim_entities_unidecode()
            entities_unidecode = [el[0] for el in groups_unidecode]

            entities = list(set(entities + entities_unidecode))
            # entities = [alias_dict.get(e, e) for e in entities]
            # entities = list(set(entities))

            # / MY RULES
            # new_entities = []
            # for i, grp in enumerate(groups):
            #     if 'the' in grp[0] and grp[0].index('the') == 0:
            #         new = ' '.join(grp[0][grp[0].index('the') + 3:].split())
            #         new_entities += new
            #     if ',' in grp[0]:
            #         new = ' '.join(grp[0].replace(',', '').split())
            #         new_entities += new
            # entities += new_entities
            entities1 = [ele for ele in entities if ele in document_titles]
            entities2 = ent_w_unicode(set(entities) - set(entities1), list(document_titles), document_titles_unidecode)
            entities = list(set(entities1 + entities2))
            if len(entities) < args.count:
                pages = list(process(ranker, annotation.get_claim(), k=args.count))

            pages = [ele for ele in pages if ele[0] not in entities]
            pages_names = [ele[0] for ele in pages]

            entity_matches = [(el, 2000) if el in pages_names else (el, 500) for el in entities]
            # pages = process(ranker,annotation.get_claim(),k=k)
            js["predicted_pages"] = entity_matches + pages[:(args.count - len(entity_matches))]
            f2.write(json.dumps(js, ensure_ascii=False) + "\n")