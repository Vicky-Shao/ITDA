"""
This is a simple application for sentence embeddings: clustering

Sentences are mapped to sentence embeddings and then k-mean clustering is applied.
"""
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
style_name = 'shakespeare'
embedder = SentenceTransformer('all-MiniLM-L6-v2')

corpus = []
input_file = style_name + '/'+style_name + '_0.txt'
with open(input_file, 'r') as fr:
    for i, line in enumerate(fr):
        corpus.append(line)
corpus_embeddings = embedder.encode(corpus)

# Perform kmean clustering
num_clusters = 120
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(corpus[sentence_id])

output_file = style_name + '/'+style_name + '-kmeans200.txt'
with open(output_file, 'w') as fw:
    for i, cluster in enumerate(clustered_sentences):
        fw.write("Cluster "+str(i+1)+'\n')
        for j, line in enumerate(cluster):
            fw.write(line)
        fw.write('\n')

