#!/usr/bin/env python3

from flask import Flask, request, render_template, url_for, redirect
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import json
import pickle
import time

DATASET_DIR = "../quora-question-pairs-dataset/"

start_time = time.time()
with open (DATASET_DIR + 'corpus_sentences.list', 'rb') as fp:
	corpus_sentences = pickle.load(fp)
print(f"[+] Loaded sentences list.\t({time.time()-start_time:.3f}s)")

start_time = time.time()
corpus_embeddings = torch.load(DATASET_DIR+'corpus_embeddings.pt')#.cuda()
print(f"[+] Loaded corpus_embeddings.\t({time.time()-start_time:.3f}s)")

start_time = time.time()
model_name = 'quora-distilbert-multilingual'
model = SentenceTransformer(model_name)
print(f"[+] Loaded model.\t\t({time.time()-start_time:.3f}s)")

start_time = time.time()
with open ('question_answer.json', 'r') as fp:
	chatbot_qa = json.load(fp)
print(f"[+] Loaded Question-Answers.\t({time.time()-start_time:.3f}s)")

start_time = time.time()
cb_questions = list(chatbot_qa.keys())
qa_embedding = model.encode(cb_questions, show_progress_bar=True, convert_to_tensor=True)#.cuda()
print(f"[+] Encoded Questions.\t\t({time.time()-start_time:.3f}s)")

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
def view(qid, view_page=True):
	messages = False
	if bool(request.args.get('posted')):
		messages = ["Question has been posted."]
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
							view_page=view_page, messages=messages)	 # skip first resp as it is same as question in db.

@app.route('/post_question', methods=['POST', 'GET'])
def post_question():
	if request.method == 'POST':
		query = request.form.get('question')
		if query:
			corpus_sentences.append(query)
			return redirect(url_for('view', qid=len(corpus_sentences)-1, posted=True))

	return render_template("post_question.html")

	
@app.route('/chat')
def chat():
	return render_template("chatbot.html")


@app.route('/chatbot', methods=['POST'])
def chatbot():
	resp = {}
	query = request.get_json().get('q')
	if query:
		hits = search_question(query, qa_embedding)
		hit = hits[0]  # best response
		resp = {
				"answer": "Sorry, I did not get your question. Can you please be more specific?",
				"similar": cb_questions[hit['corpus_id']],
				"score": f"{hit['score']:.3f}",
				}
		if hit['score'] > 0.80:
			resp["answer"] = chatbot_qa[cb_questions[hit['corpus_id']]]

	return json.dumps(resp)


'''
'''

if __name__ == '__main__':
	app.run(debug=False)
#	app.run(host='0.0.0.0', port=31796, debug=False)