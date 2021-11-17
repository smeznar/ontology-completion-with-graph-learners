datasets = ["anatomy", "ehdaa", "emotions", "foodon", "go", "marine", "scto"]
baselines = ["Adamic", "Jaccard", "Preferential", "SNoRe", "node2vec", "Spectral",
             "TransE", "RotatE", "GAT", "GIN", "GCN", "GAE", "MetaPath2vec"]
for b in baselines:
    for d in datasets:
        print("python link_prediction.py --method {} --dataset ../data/{}.json --out ../results/{}_{}.txt".format(b, d, d, b))