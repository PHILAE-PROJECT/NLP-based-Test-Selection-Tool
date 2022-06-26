
# NLP-based Test Selection Tool

In software testing, a lot of effort is devoted to creating test suites that can detect faults in software. Since run an entire test suite is very time-consuming, it is desirable to remove ineffective test cases in the test suits and only keep a small number of test cases but with the same fault detection capability. This tool employs Natural Language Processing methods (e.g: Word2Vec) and provides test case selection from test suites while keeping the fault detection capability as the original test suite. 

This tool is part of the outputs of research published in NLPaSE 2020 - 1st Workshop on Natural Language Processing Advancements for Software Engineering. 

[Afshinpour, B., Groz, R., Amini, M., Ledru, Y., & Oriat, C. (2020). Reducing Regression Test Suites using the Word2Vec Natural Language Processing Tool. SEED/NLPaSE@APSEC.](http://ceur-ws.org/Vol-2799/Paper6_NLPaSE.pdf)

## Requirements

### Dependencies :

The code is written in python language. It is necessary to install these libraries :

```bash
Gensim
Sklearn
NumPy
```

### Agilkia Format :

In order to run properly, the input test suit file must respect the Agilkia format :
    - In the case of a CSV file, its rows must follow the following formatting :
        TraceID | Timestamp | sessionID | object | action | input | output
    For example : 
        3, 1584454656089, client1, scan1_1, unlock, [], 0
    - In the case of a Json file, it must also respect Agilkia formatting, for example an event might look like so :
          {
          "__class__": "Event",
          "__module__": "agilkia.json_traces",
          "action": "debloquer",
          "inputs": {},
          "outputs": {
            "Status": 0.0
          }


 

## Sessions_abstractions.py
Using the test suite file (which can be both a Json or a CSV file, as long as it respects Agilkia format) as input, it encodes every event in the CSV file to a vector. The generated vectors can be processed later by cluster_sessions_Word2Vec_model.py. It generates sessions.npy, in which, we have a sequence of vectors associated with each session. 

## configurations.py
This file has all of the parameters of the tool. 

•	Input_field_selection : The Sessions_abstractions.py file uses this parameter to extract fields in the input CSV/Json file. The fields should be indicated by their name (all available names are indicated in the file).  We keep this list of fields in the Input_field_selection variable. 

•	n_clusters : determines the number of clusters, in which, the sessions are clustered by the tool.
 
•	clustering_method : By choosing  “k” or “s” as the value in the clustering_method variable, it is possible to choose two different clustering methods. The value “K” means k-means clustering and “s” indicates the spectral clustering method. 

•	dimension_reduction : By setting the parameter to 1 we can enable t-SNE, dimension reduction before clustering data.

•	selection_method : After clustering, the tool chooses one representative session from each cluster. We can choose between “min” and “max” to make the tool to pick the shortest or the longest session from each cluster. 

## cluster_sessions_Word2Vec_model.py
This file creates a word2vec model and based on the Word2vec it finds a vector for each session. Then it clusters all the session vectors and selects a representative session from each cluster. It produces two CSV files as result. One of them is the results of the clustering method, in which, it saves the member of each cluster. The other one keeps the list of selected sessions which they are selected from each cluster as representative by using the selection method.

The script then puts the output in a custom directory based on the input file name, and produces both a json and a csv file :
    - The json file is formatted as follows : it contains all the traces of the input file, with an additional cluster_labels field corresponding to the cluster of each client, in ascending order.
    - The CSV file contains one line per cluster, formatted as such :
        | Nb of clusters | Cluster Label | List of clients in the cluster | List of lists of events corresponding to each client |

It also creates two extra files :
    - resultofclustering.csv : this file contains information about the representatives of each clusted, formatted as follows :
        | Nb of clusters | Total number of traces for the representatives | List of the name of the client that represents each cluster |
    - reducted_client_kmeans.json : this json file contains the list of events that correspond to each cluster representative

##csvToAgilkia.py
This script is an existing library used to convert CSV to Agilkia in case input is a CSV file. It's taken from another contribution of the Philae project.
