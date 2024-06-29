import torch
from sentence_transformers import SentenceTransformer, util
import pickle
import numpy as np
from tensorflow import keras
from flask import Flask, request, render_template
app = Flask(__name__)



embeddings = pickle.load(open('embeddings.pkl','rb'))
sentences = pickle.load(open('sentences.pkl','rb'))
rec_model = pickle.load(open('rec_model.pkl','rb'))

## recommendation system
def recommendation(input_paper):
    cosine_scores = util.cos_sim(embeddings,rec_model.encode(input_paper))
    top_similar_papers = torch.topk(cosine_scores, dim=0, k=5, sorted=True)
    papers_list = []
    for i in top_similar_papers.indices:
        papers_list.append(sentences[i.item()])

    return papers_list


## craete app 
@app.route('/')
def index():
    return render_template('index.html')

# route for recommendation
@app.route('/recommend', methods = ['POST'])
def recommend():
    if request.method == 'POST':
        input_paper = request.form['paper_title']
        recommended_papers = recommendation(input_paper)
        return render_template('index.html', recommended_papers= recommended_papers)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)