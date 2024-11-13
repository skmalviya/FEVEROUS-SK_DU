# FEVEROUS-SK_DU
Evidence Retrieval for Fact Verification using Multi-stage Reranking at EMNLP2024.

## Page Retrieval

1. TFIDF: Follow the UNIFEE github repo for using tfidf for page Retrieval.
2. BM25: Load the `document_entity_bm25_pyserieni_ir.py` into
```
PYTHONPATH=src python src/baseline/retriever/document_entity_bm25_pyserieni_ir.py
```
3. Ensebmle both Pages from TFIDF and BM25 and to select 100
```
PYTHONPATH=src python my_methods/bm25_doc_retriever/ensemble_retrieved_pages.py
```
4. Cross-Enoder Reranking (Follow Page Re-ranker from UNIFEE Github repo)
```
PYTHONPATH=src python src/re-ranker/page_reranker.py
```
5. HybRank Reranking
6. HLATR Reranking

## Sentence Retrieval
1. Cross-Encoder (Follow Page Re-ranker from UNIFEE Github repo)
```
PYTHONPATH=src python src/my_methods/roberta_sentence_selector/pred_sentence_scores.py
```
* Use the model weights from [Link](https://durhamuniversity-my.sharepoint.com/:f:/g/personal/qvlw18_durham_ac_uk/EtN2GI4Ai6BKq4VWrWgRlnkBMdpmZxvvm2rsvxegOMMByA?e=0Yofix)
2. HybRank
3. HLATR Reranking

## Table Retrieval
1. SEE-ST
```
PYTHONPATH=src python src/my_method/table_extraction/eval_rc_retriever.py\
    --model_load_path checkpoints/${ckpt_name}\
    --select_criterion row*col
    --max_tabs 3
```
* Use the model weights from [Link](https://durhamuniversity-my.sharepoint.com/:f:/g/personal/qvlw18_durham_ac_uk/Eot9LsZxXtFGnFvG0ToSYjQBl5bajHZz1Uv9Y8IFZaUPYg?e=gruhEB)
2. HybRank
3. HLATR Reranking

## Cell Retrieval
(Follow the SEE-ST Github Repo)
1. Combine Sentence and Table evidence
```
PYTHONPATH=src python src/my_method/cell_selection/combine_retrieved_sents_and_tabs.py\
    --output_file_name roberta_sent.rc_table.not_precomputed.p5.s5.t5\
    --max_tabs 5
```
2. Cell extraction
```
PYTHONPATH=src python src/my_methods/cell_selection/extract.py
```
* Use the model weights from [Link](https://durhamuniversity-my.sharepoint.com/:f:/g/personal/qvlw18_durham_ac_uk/Eot9LsZxXtFGnFvG0ToSYjQBl5bajHZz1Uv9Y8IFZaUPYg?e=bUhjrw)

3. Cell output file
```
PYTHONPATH=src python src/my_method/cell_selection/rewrite_result_file.py --split dev --input_file dev.seest_fusion_results.jsonl
```

## Verdict prediction
* Follow the DCUF github repo
