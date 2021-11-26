datasets = ["anatomy", "emotions", "marine", "scto", "ehdaa", "foodon", "go"]
baselines = ["Adamic", "Jaccard", "Preferential", "SNoRe", "node2vec", "Spectral",
             "TransE", "RotatE", "GAT", "GIN", "GCN", "GAE", "metapath2vec"]
for d in datasets:
    for b in baselines:
        print("sudo docker run -v $(pwd):/app --rm link-analysis src/link_prediction.py --method {} --dataset "
              "data/{}.json --format json --out results/{}_{}.txt".format(b, d, d, b))

for b in baselines:
    print(
        "sudo docker run -v $(pwd):/app --rm link-analysis src/link_prediction.py --method {} --dataset data/{}.txt "
        "--format txt --out results/{}_{}.txt".format(b, "LKN", "LKN", b))