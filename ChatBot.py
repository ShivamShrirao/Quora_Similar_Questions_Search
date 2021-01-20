from sentence_transformers import SentenceTransformer, util


def Answer_InformationRetrival(query):
	'''Model Based on Information Retrival'''
	
    model = SentenceTransformer('msmarco-distilroberta-base-v2')

	#Making the answer set
    answer_set = ["This is a forum to answer question related to programming",
                  "I am an assistant bot which can answer your questions about this forum",
                  "The special thing about this forum is that it uses NLP to prevent redundant questions.",
                  "We use BERT to compare and find questions that are similar.",
                  "Bidirectional Encoder Representations or BERT is a from Transformers is a Transformer-based machine learning technique for natural language processing pre-training developed by Google."
                  "If your question matches some which already exists on our forum, you will be shown the matching the questions. If your answer is still not available, you can decide and still post your question."

                 ]
    
    scores = []
    query_embedding = model.encode(query)
    
	#Calculating the similarity scores
    for p in answer_set:
        passage_embedding = model.encode(p)
        scores.append(util.pytorch_cos_sim(query_embedding, passage_embedding))
	
	
	#Returning the answers
    if max(scores)< 0.30:
        return "Sorry, I did not get your question. Can you be more specific?" , max(scores)
    
    return answer_set[scores.index(max(scores))] , max(scores)

def Answer_QuestionComparison(query):
    model = SentenceTransformer('msmarco-distilroberta-base-v2')

	#Making the answer set
    answer_set = {"What is the forum for?": "This is a forum to answer question related to programming",
                  "Hello":"Hello! How are you today?",
                  "Who are you?": "I am an assistant bot which can answer your questions about this forum",
                  "What is special about this forum?": "The special thing about this forum is that it uses NLP to prevent redundant questions, even if it is in a different language.",
                  "How does the site find similar question?": "We use BERT to compare and find questions that are similar.",
                  "What is BERT?": "Bidirectional Encoder Representations or BERT is a from Transformers is a Transformer-based machine learning technique for natural language processing pre-training developed by Google.",
                  "How can I post questions on this blog?" : "You can enter your question in the ask-question box. If a similar question already exist, we will show you the existing questions and promt you again.",
                  "What if my question already exist?": "If your question matches some which already exists on our forum, you will be shown the matching the questions. If your answer is still not available, you can decide and still post your question."
                  }
    
    scores = []
    query_embedding = model.encode(query)
    
	#Calculating the similarity scores

    for p in answer_set.keys():
        passage_embedding = model.encode(p)
        scores.append(util.pytorch_cos_sim(query_embedding, passage_embedding))
    
	#Returning the answers
    if max(scores)< 0.30:
        return "Sorry, I did not get your question. Can you be more specific?" , max(scores)
    
    
    return list(answer_set.values())[scores.index(max(scores))] , max(scores)


if __name__ == "__main__":	
	ans, score = Answer_InformationRetrival("What is the forum for?")

	print(ans)
	print("Score: ",score)