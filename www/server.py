#!/usr/bin/env python3

from flask import Flask, request, render_template, url_for
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import json
import pickle

with open (DATASET_DIR + 'corpus_sentences.list', 'rb') as fp:
    corpus_sentences = pickle.load(fp)

corpus_embeddings = torch.load('../quora-question-pairs-dataset/corpus_embeddings.pt')

model_name = 'quora-distilbert-multilingual'
model = SentenceTransformer(model_name)

app = Flask(__name__)


@app.route('/')
def index():
	return render_template("index.html")

def get_resp_dicts(query):
	res = []
	hits = search_question(query)
	for hit in hits:
		res.append({
				"value": corpus_sentences[hit['corpus_id']],
				"score": f"({hit['score']:.3f}) ",
				"url": url_for('about', id=hit['corpus_id'])
				})
	return res

@app.route('/fetch')
def submit():
	query = request.args['term']
	res = []
	if query:
		res = get_resp_dicts(query)
	return json.dumps(res)

@app.route('/search')
def search():
	res = []
	query = None
	try:
		query = request.args['q']
		if query:
			res = get_resp_dicts(query)
	except KeyError:
		pass
	return render_template("search.html", qtype="Search", res=res, query=query)


@app.route('/about/<id>')
def about(id):
	ret = db.movies.find_one({"id": int(id)}, {"_id": 0})
	similar = []
	if ret is None:
		ret = {"title": "Not Found!"}
	else:
		top_results, cos_scores = get_recommendations(ret["idx"])
		for i in top_results:
			similar.append(db.movies.find_one({"idx": i.item()}, {"_id": 0}))
			similar[-1]["cos_score"] = cos_scores[i].item()
		similar[:15] = sorted(similar[:15], key=lambda e: e["vote_weight"]*e["cos_score"], reverse=True)
	return render_template("about.html", main_movie=ret, similar=similar)


def search_question(inp_question, top_k=10):
    start_time = time.time()
    question_embedding = model.encode(inp_question, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    end_time = time.time()
    hits = hits[0]

    print("Input question:", inp_question)
    print("Results (after {:.3f} seconds):".format(end_time-start_time))
    for hit in hits:
        print("\t{:.3f}\t{}".format(hit['score'], corpus_sentences[hit['corpus_id']]))

    return hits


if __name__ == '__main__':
	app.run(debug=False)
#	app.run(host='0.0.0.0', port=31796, debug=False)
