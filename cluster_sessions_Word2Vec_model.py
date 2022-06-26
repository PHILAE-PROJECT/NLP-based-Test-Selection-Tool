''' 
 Created on : 15.06.2020
 @author    : Bahareh
 Last modify: 30.06.2021
 ---------------------------

'''
import os
from pathlib import Path
import json
import io_adapt
import logging         # Setting up the loggings to monitor gensim
import numpy as np
import multiprocessing
from gensim.models import Word2Vec
import re              # For preprocessing
import pandas as pd    # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency
import csv
from csv import reader, writer
import configurations
import sessions_abstraction
import spacy           # For preprocessing
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import pairwise_distances
import networkx as nx
import seaborn as sns

sns.set()
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                    datefmt='%H:%M:%S', level=logging.INFO)


def float_formatter(x): return "%.3f" % x


np.set_printoptions(formatter={'float_kind': float_formatter})


# *****************************************************************
# Import Sessions from Sessions.py
Sessions = np.load('Sessions.npy', allow_pickle=True).item()
cores = multiprocessing.cpu_count()


def Create_Word2vec_Model(SentencesString, word_freq):
    w2v_model = Word2Vec(min_count=1, window=3, vector_size=len(
        word_freq), sample=0, alpha=0.03, min_alpha=0.0007, negative=2, workers=cores-1)
    w2v_model.build_vocab(SentencesString)
    w2v_model.train(
        SentencesString, total_examples=w2v_model.corpus_count, epochs=10, report_delay=1)
    index2word_set = set(w2v_model.wv.index_to_key)
    return w2v_model


def avg_feature_vector(sen, model, num_features, index2word_set):
    words = sen
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec


def distance_func(s1_afv, s2_afv):

    d = spatial.distance.cosine(s1_afv, s2_afv)
    return d


def Create_affinity_matrix(SentencesString, w2v_model, Len_word_freq):

    num_of_rows = len(SentencesString)
    num_of_cols = len(SentencesString)
    affinity_matrix = np.full((num_of_rows, num_of_cols), np.nan)
    affinity_matrix = np.zeros((num_of_rows, num_of_cols))

    index2word_set = set(w2v_model.wv.index2word)
    for k in range(len(SentencesString)):
        s1_afv = avg_feature_vector(
            SentencesString[k], model=w2v_model.wv, num_features=Len_word_freq, index2word_set=index2word_set)
        for j in range(len(SentencesString)):

            s2_afv = avg_feature_vector(
                SentencesString[j], model=w2v_model.wv, num_features=Len_word_freq, index2word_set=index2word_set)
            sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
            print(k, end="?=  "),
            print(j)
            print(sim)
            if (sim >= 0.989):
                # Assign 1 as value when two session are similar
                value = [1]
                position_row = k
                position_col = j
                position = [int(num_of_cols * position_row + position_col)]
                np.put(affinity_matrix, position, value)
                affinity_matrix[(k, j)] = 1

    np.save("affinity_matrix.npy", affinity_matrix)
    return (affinity_matrix)


def Find_Sessions_cluster_kmeans(n_clusters, X_2d):

    import matplotlib.pyplot as plt2
    kmeans = KMeans(n_clusters)
    kmeans.fit(X_2d)
    clusters = dict()
    for l in range(n_clusters):
        clientID = []
        OneCluster = np.where(kmeans.labels_ == l)

        for j in OneCluster[0]:

            clientID.append(j)
        clusters[l] = clientID

    return clusters


def Find_Sessions_cluster_spectral(n_clusters, X_2d, SentencesString, Len_word_freq, exist_affinity):
    # Nearest neighbors method for spectral
    #sc = SpectralClustering(n_clusters, affinity='nearest_neighbors', random_state=0)
    #sc_clustering = sc.fit(X_2d)
    # important - important - important
    # Create Affinity matrix for spectral
    # For the first time we have to create an affinity matrix and save it, so we can load it later
    # If it is the first time to run the spectral clustering we set exist_affinity parameter by value 0 otherwise  set 1 for saving time

    if (exist_affinity == 0):
        affinity_matrix = Create_affinity_matrix(
            SentencesString, w2v_model, Len_word_freq)
    if (exist_affinity == 1):
        affinity_matrix = np.load(
            'affinity_matrix.npy', allow_pickle=True).tolist()

    sc = SpectralClustering(
        n_clusters, affinity='precomputed', n_init=100, assign_labels='discretize')
    sc_clustering = sc.fit_predict(affinity_matrix)

    clusters_spectral = dict()
    for sl in range(n_clusters):
        clientIDSpectral = []
        OneClusterSpectral = np.where(sc_clustering == sl)

        for j in OneClusterSpectral[0]:

            clientIDSpectral.append(j)
        clusters_spectral[sl] = clientIDSpectral

    return clusters_spectral

# *************************** Create model wor2vec - cluster sessions - Tsne ******************************************


def Find_vectors_tsne(SentencesString, word_freq, w2v_model):
    index2word_set = set(w2v_model.wv.index_to_key)
    for w in word_freq:
        print(w)
        if w in w2v_model.wv:
            print(w2v_model.wv[w])
        print("^^^^^^^^^^^^^^^^^^^^^^^^^")
    X = []
    # Finding the average of word2vec vectors in each sentence
    for k in range(len(SentencesString)):
        avr_w3v_sent = avg_feature_vector(SentencesString[k], model=w2v_model.wv, num_features=len(
            word_freq), index2word_set=index2word_set)
        X.append(avr_w3v_sent)
    tsne = TSNE(n_components=2, perplexity=8.0, random_state=0)
    X_2d = tsne.fit_transform(X)  # project data in 2D
    y = range(len(X))
    target_ids = range(len(X))
    return X_2d


def Find_vectors_whitout_tsne(SentencesString, word_freq, w2v_model):
    index2word_set = set(w2v_model.wv.index2word)
    for w in word_freq:
        if w in w2v_model.wv:
            print(w2v_model.wv[w])
        print("^^^^^^^^^^^")
    X = []
    # Finding the average of word2vec vectors in each sentence
    for k in range(len(SentencesString)):
        print(k)
        print(SentencesString[k])
        avr_w3v_sent = avg_feature_vector(SentencesString[k], model=w2v_model.wv, num_features=len(
            word_freq), index2word_set=index2word_set)
        X.append(avr_w3v_sent)

    X = np.array(X)
    y = range(len(X))
    target_ids = range(len(X))
    return X


def find_client_maxlength(clusters):
    selected_clients = []
    for i, j in clusters.items():
        max = 0
        for elem in j:
            #print("cluster ",i," client ",elem," : ",Sessions[' client' +str(elem)])
            lengthofsession = len(Sessions[' client' + str(elem)])
            if (max < lengthofsession):
                max = lengthofsession
                client_max_length = elem
        # Selected client for each cluster based on length [ClientId for cluster 0,...]
        selected_clients.append(client_max_length)

    return selected_clients


def find_client_minlength(clusters):
    selected_clients = []
    for i, j in clusters.items():
        min = len(Sessions[' client' + str(j[0])])
        client_min_length = Sessions[' client' + str(j[0])]
        for elem in j:
            #print("cluster ",i," client ",elem," : ",Sessions[' client' +str(elem)])
            lengthofsession = len(Sessions[' client' + str(elem)])
            if (min > lengthofsession):
                min = lengthofsession
                client_min_length = elem
        # Selected client for each cluster based on length [ClientId for cluster 0,...]
        selected_clients.append(client_min_length)

    return selected_clients


def create_csv_from_traces(trace_list, typeofclustering):
    if (typeofclustering == 'k'):
        terminator_combination = open(
            dir_name + "reducted_client_kmeans.json", "w")
    elif(typeofclustering == 's'):
        terminator_combination = open(
            dir_name + "reducted_client_spectral.json", "w")
    for event in trace_list:
        terminator_combination.write(json.dumps(event) + "\n")
    terminator_combination.close()

    return 1


def find_all_traces_of_list_of_clients(list_clients):
    c = 0
    for i in list_clients:
        list_clients[c] = 'client' + str(i)
        c = c+1

    list_traces = []
    if not io_adapt.is_json(configurations.input_file):
        json_name = Path(configurations.input_file).stem + \
            ".AgilkiaTraces.json"
    else:
        json_name = configurations.input_file
    with open(json_name, 'r') as json_f:
        json_reader = json.load(json_f)
        for session in json_reader["traces"]:
            for event in session["events"]:
                if event["meta_data"]["sessionID"] in list_clients:
                    list_traces.append(event)
    return list_traces


def Create_result_Csv(n_clusters, numberofevent, selected_clients):
    resultCsv = writer(open(dir_name + "resultofclustering.csv", "w"),
                       delimiter=",", lineterminator='\n')
    line_count = 1
    resultCsv.writerow([n_clusters, numberofevent, selected_clients])
    line_count = line_count+1


def Create_Csv_clusters_members(n, clusters, Sentences, typeofclustering):
    json_cluster_labels = {}
    if (typeofclustering == 'k'):
        clusters_members = writer(open(
            dir_name + "Clusters_members_kmeans_k"+str(n)+".csv", "w"), delimiter=",", lineterminator='\n')
        output_name = "Clusters_members_kmeans_k"+str(n)+".csv"
    elif(typeofclustering == 's'):
        clusters_members = writer(open(
            dir_name + "Clusters_members_Spectral_k"+str(n)+".csv", "w"), delimiter=",", lineterminator='\n')
        output_name = "Clusters_members_Spectral_k"+str(n)+".csv"
    clusters_members.writerow(["#k", "label", "#Clients", "Sessions"])
    for labelcluster, sessioncluster in clusters.items():

        cluster_member_Session = []
        for j in sessioncluster:
            cluster_member_Session.append(Sentences[j])
            json_cluster_labels[j] = labelcluster

        clusters_members.writerow(
            [n, labelcluster, sessioncluster, cluster_member_Session])
    return json_cluster_labels.items(), output_name


Sentences = []
SentencesString = []
NewSentence = []
NewSentenceString = []
word_freq = defaultdict(int)

for sent in Sessions:
    NewSentence = Sessions[sent]
    Sentences.append(NewSentence)
    NewSentenceString = []
    for i in NewSentence:
        i = str(i)
        NewSentenceString.append(i)
        word_freq[i] += 1
    SentencesString.append(NewSentenceString)

json_in_name, json_out_name, csv_out_name, dir_name = io_adapt.get_outputs_names(
    configurations.input_file)

print("len of vocab :", len(word_freq))
Len_word_freq = len(word_freq)
os.system("pause")
SessionsString = dict()
SpecificSentenceString = []
for sent in Sessions:
    specificSentence = Sessions[sent]
    SpecificSentenceString = []
    for i in specificSentence:
        i = str(i)
        SpecificSentenceString.append(i)
    SessionsString[sent] = SpecificSentenceString

n_clusters = configurations.n_clusters
w2v_model = Create_Word2vec_Model(SentencesString, word_freq)
# ********
# The Data should be preprocess by using tsne or whitout using tsne
# So choose one of this function Find_vectors_tsne or Find_vectors_whitout_tsne
# ********
if (configurations.dimension_reduction == 1):
    data = Find_vectors_tsne(SentencesString, word_freq, w2v_model)
if (configurations.dimension_reduction == 0):
    data = Find_vectors_whitout_tsne(SentencesString, word_freq, w2v_model)
if(configurations.clustering_method == 'k'):
    clusters_kmeans = Find_Sessions_cluster_kmeans(n_clusters, data)
    print("Kmeans clusters are : ", clusters_kmeans)
    cluster_info, output_name = Create_Csv_clusters_members(
        n_clusters, clusters_kmeans, Sentences, 'k')
    if(configurations.selection_method == "max"):
        selected_clients_kmeans = find_client_maxlength(clusters_kmeans)
    if(configurations.selection_method == "min"):
        selected_clients_kmeans = find_client_minlength(clusters_kmeans)
    list_traces_selectedclientID_kmeans = find_all_traces_of_list_of_clients(
        selected_clients_kmeans)
    create_csv_from_traces(list_traces_selectedclientID_kmeans, 'k')

    with open(dir_name + "reducted_client_kmeans.json") as f:
        numberofevent_kmeans = sum(1 for line in f)
    Create_result_Csv(n_clusters, numberofevent_kmeans,
                      selected_clients_kmeans)

if(configurations.clustering_method == 's'):
    clusters_spectral = Find_Sessions_cluster_spectral(
        n_clusters, data, SentencesString, Len_word_freq, 0)
    # If it is first time to run the spectral clustering we set if exist_affinity by value 0 otherwise 1
    print("Spectral clusters are: ", clusters_spectral)
    cluster_info, output_name = Create_Csv_clusters_members(
        n_clusters, clusters_spectral, Sentences, 's')  # Sentences
    if(configurations.selection_method == "max"):
        selected_clients_spectral = find_client_maxlength(clusters_spectral)
    if(configurations.selection_method == "min"):
        selected_clients_spectral = find_client_minlength(clusters_spectral)
    list_traces_selectedclientID_spectral = find_all_traces_of_list_of_clients(
        selected_clients_spectral)
    create_csv_from_traces(list_traces_selectedclientID_spectral, 's')

    with open(dir_name + "reducted_client_spectral.json") as f2:
        numberofevent_spectral = sum(1 for line in f2)
        print(numberofevent_spectral)

    Create_result_Csv(n_clusters, numberofevent_spectral,
                      selected_clients_spectral)

io_adapt.make_output_json(configurations.input_file, output_name, cluster_info)

# if(os.path.exists(dir_name + "reducted_client_kmeans.json")):  # delete csv file for testing another K
#    os.remove(dir_name + "reducted_client_kmeans.json")
# if(os.path.exists(dir_name + "reducted_client_spectral.json")):  # delete csv file for testing another K
#    os.remove(dir_name + "reducted_client_spectral.json")
if(os.path.exists("Sessions.npy")):
    os.remove("Sessions.npy")
