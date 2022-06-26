''' 
 Created on : 15.06.2020
 @author    : Bahareh Afshinpour
 Last modify: 02.06.2022
 ---------------------------
'''

input_file = "1026-steps-U.csv"

# Name of the fields that must be used as input of vectorization
# Available fields are (based on Agilikia format) :
# "timestamp", "sessionID", "object", "action", "input", "output"
Input_field_selection = ["action", "input", "output"]
# two different methods of clustering exist in this program.  K-means (k) and Spectral clustering (s). 
clustering_method = "k"
output_file = None
n_clusters = 7                  # the number of clusters
if (clustering_method == 'k'):
    output_file = "Clusters_members_kmeans_k"+str(n_clusters)+".csv"
elif(clustering_method == 's'):
    output_file = "Clusters_members_Spectral_k"+str(n_clusters)+".csv"
dimension_reduction = 1         # 1 means using t-SNE , 0 means without using t-SNE
selection_method = "max"        # max or min session selection method
