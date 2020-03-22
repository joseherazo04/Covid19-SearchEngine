from flask import Flask, render_template, request
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/search/results', methods=['GET','POST'])
def request_search():
    search_terms = request.form['input']
    items = []
    for i in range(1, 11):
        i = str(i)
        dict == {}
        # # you just don't have to quote the keys
        item = dict(title="2012-02-" + i, abstract=i, doi="here")
        items.append(item)
    
    return render_template('results.html', items=items, search_terms=search_terms)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
