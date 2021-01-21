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

corpus_embeddings = torch.load(DATASET_DIR+'corpus_embeddings.pt')#.cuda()

model_name = 'quora-distilbert-multilingual'
model = SentenceTransformer(model_name)

with open ('question_answer.json', 'r') as fp:
	chatbot_qa = json.load(fp)

cb_questions = list(chatbot_qa.keys())
qa_embedding = model.encode(cb_questions, show_progress_bar=True, convert_to_tensor=True)#.cuda()

app = Flask(__name__)


@app.route('/')
def index():
	return render_template("index.html")

def search_question(inp_question, embedding_db):
	question_embedding = model.encode(inp_question, convert_to_tensor=True)#.cuda()
	hits = util.semantic_search(question_embedding, embedding_db)
	return hits[0]

cache_hits = {}
# TODO: Limit cache_hits size.
def get_resp_dicts(query):
	resp = []
	hits = cache_hits.get(query)
	if hits is None:
		hits = search_question(query, corpus_embeddings)
		cache_hits[query] = hits
	for hit in hits:
		resp.append({
				"value": corpus_sentences[hit['corpus_id']],
				"score": f"{hit['score']:.3f}",
				"url": url_for('view', qid=hit['corpus_id'])
				})
	return resp

@app.route('/search')
def search():
	start_time = time.time()
	resp = []
	query = request.args.get('q')
	if query:
		resp = get_resp_dicts(query)
	end_time = time.time()
	time_taken = end_time-start_time
	return render_template("search.html", resp=resp, query=query, time_taken=f"{time_taken:.4f}")


@app.route('/view/<qid>')
def view(qid, view_page=True, q_posted=False):
	start_time = time.time()
	try:
		query = corpus_sentences[int(qid)]
		resp = get_resp_dicts(query)
	except IndexError:
		query = ""
		resp = []
	end_time = time.time()
	time_taken = end_time-start_time
	return render_template("search.html", resp=resp[1:], query=query, time_taken=f"{time_taken:.4f}",
							view_page=view_page, q_posted=q_posted)	 # skip first resp as it is same as question in db.

@app.route('/q_posted/<qid>')
def q_posted(qid):
	return view(qid, q_posted=True)

@app.route('/post_question', methods=['POST', 'GET'])
def post_question():
	if request.method == 'POST':
		query = request.form.get('question')
		if query:
			corpus_sentences.append(query)
			return q_posted(len(corpus_sentences)-1)

	return render_template("post_questions.html")

	
@app.route('/chat')
def chat():
	return render_template("chatbot.html")


@app.route('/chatbot')
def chatbot():
	resp = {}
	query = request.args.get('q')
	if query:
		hits = search_question(query, qa_embedding)
		hit = hits[0]  # best response
		resp = {
				"answer": "Sorry, I did not get your question. Can you please be more specific?",
				"similar": cb_questions[hit['corpus_id']],
				"score": f"{hit['score']:.3f}",
				}
		if hit['score'] > 0.85:
			resp["answer"] = chatbot_qa[cb_questions[hit['corpus_id']]]

	return json.dumps(resp)


'''
'''

if __name__ == '__main__':
	app.run(debug=False)
#	app.run(host='0.0.0.0', port=31796, debug=False)