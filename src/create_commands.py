datasets = ["anatomy", "emotions", "marine", "scto", "ehdaa", "foodon", "LKN", "go"]
baselines = ["Adamic", "Jaccard", "Preferential", "SNoRe", "node2vec", "Spectral",
             "TransE", "RotatE", "GAT", "GIN", "GCN", "GAE", "metapath2vec"]
for d in datasets:
    for b in baselines:
        print("python link_prediction.py --method {} --dataset ../data/{}.json --out ../results/{}_{}.txt".format(b, d, d, b))