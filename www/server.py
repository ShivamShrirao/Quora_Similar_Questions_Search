#!/usr/bin/env python3

from flask import Flask, request, render_template, url_for
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import json
import pickle
import time

DATASET_DIR = "../quora-question-pairs-dataset/"

with open (DATASET_DIR + 'corpus_sentences.list', 'rb') as fp:
	corpus_sentences = pickle.load(fp)

corpus_embeddings = torch.load(DATASET_DIR+'corpus_embeddings.pt')

model_name = 'quora-distilbert-multilingual'
model = SentenceTransformer(model_name)

app = Flask(__name__)


@app.route('/')
def index():
	return render_template("index.html")

def search_question(inp_question, top_k=10):
	question_embedding = model.encode(inp_question, convert_to_tensor=True)
	hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
	return hits[0]

cache_hits = {}
# TODO: Limit cache_hits size.
def get_resp_dicts(query):
	res = []
	hits = cache_hits.get(query)
	if hits is None:
		hits = search_question(query)
		cache_hits[query] = hits
	for hit in hits:
		res.append({
				"value": corpus_sentences[hit['corpus_id']],
				"score": f"({hit['score']:.3f}) ",
				"url": url_for('about', qid=hit['corpus_id'])
				})
	return res

@app.route('/search')
def search():
	start_time = time.time()
	res = []
	query = None
	try:
		query = request.args['q']
		if query:
			res = get_resp_dicts(query)
	except KeyError:
		pass
	end_time = time.time()
	time_taken = end_time-start_time
	return render_template("search.html", qtype="Search", res=res, query=query, time_taken=f"{time_taken:.4f}")


@app.route('/about/<qid>')
def about(qid):
	return render_template("about.html", main_que=corpus_sentences[qid], similar=similar)


if __name__ == '__main__':
	app.run(debug=False)
#	app.run(host='0.0.0.0', port=31796, debug=False)
