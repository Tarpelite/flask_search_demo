from flask import Flask, request
from sentence_transformers import SentenceTransformer
import faiss
import json
import torch

app = Flask(__name__)

model=SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
if torch.cuda.is_available():
    model.cuda()

d = 384
batch_size = 32
top_k = 5
global_index = faiss.IndexFlatL2(d)
uuid_list = {}

@app.route("/add_index", methods=["POST"])
def add_index():
    """
    JSON Format:
    {
        "uuid1":"text1",
        "uuid2":"text2",
        ...
    }

    """
    print(request)
    data = request.get_json()
    print(data)
    n = global_index.ntotal
    k = n
    batch = []
    for uuid in data:
        uuid_list[k] = uuid
        k += 1
        if len(batch) % 32 == 0 and len(batch) != 0:
            embeddings = model.encode(batch)
            # print(embeddings.shape)
            global_index.add(embeddings)
            batch = []
        batch.append(data[uuid])
    print(batch)
    embeddings = model.encode(batch)
    global_index.add(embeddings)
    # batch = []
    return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 



@app.route("/query_text", methods=["POST"])
def query_text():
    """
    JSON Format:
    {
        "query_id1": "query_text1",
        "query_id2": "query_text2",
        ....
    }
    """
    data = request.get_json()
    all_keys = list(data.keys())
    all_text = [data[k] for k in data]
    embeddings = model.encode(all_text)
    D, I = global_index.search(embeddings, top_k)
    res = {}
    for i in range(len(I)):
        key = all_keys[i]
        selected = []
        for j in range(len(I[i])):
            uuid = uuid_list[I[i,j]]
            dist = D[i, j]
            selected.append({"uuid":uuid, "dist":"%.4f"%dist})
        res[key] = selected
    
    return json.dumps(res), 200, {'ContentType':'application/json'} 
            

@app.route("/")
def hello():
    return "Hello, SeeThoughts!"

app.run(host="0.0.0.0", port="8080")