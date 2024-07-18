import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, NearestCentroid

def find_max_cosine(df_embeddings_2017, df_embeddings_2018, clustering, df_submission_labels):

    HYPERPARAM_k_num_neighbors = 10

    # This is the max possible value of cosine distance
    # set to worst case - used to find optimal

    HYPERPARAM_b_max_cosine = 2

    #setting up KNN for 2018
    neigh = NearestNeighbors(n_neighbors=HYPERPARAM_k_num_neighbors, metric = 'cosine', radius = 0.3)
    non_anchor_embedding_2018 = np.array(df_embeddings_2018.embedding.to_list())
    neigh.fit(non_anchor_embedding_2018)

    #setting up closest centroid for anchor group 2017
    anchor_embedding_2017 = np.array(df_embeddings_2017.embedding.tolist())
    anchor_agg_clusters_2017 = np.array(df_embeddings_2017.agg_cluster.tolist())
    clf = NearestCentroid()
    clf.fit(anchor_embedding_2017, anchor_agg_clusters_2017)


    #dictionary of all the agg clusters and the 10 KNN from 2018
    dict_agg_cluster_matches ={}
    for cluster_id in np.unique(clustering.labels_):

        distances, indices = neigh.kneighbors([clf.centroids_[cluster_id]])
        df_anchor_embedding = pd.concat([pd.DataFrame(data = distances.T,columns =['cos_dist']),pd.DataFrame(indices.T,columns=['indices'])],axis=1)

        #get all the specified cosine distance 2018 papers
        #tuple of (dataframe of 2018 matched papers, cosine distances)
        dict_agg_cluster_matches[cluster_id] = (
            df_embeddings_2018.iloc[df_anchor_embedding[df_anchor_embedding['cos_dist']<= HYPERPARAM_b_max_cosine].indices.to_list(), :],
            df_anchor_embedding[df_anchor_embedding['cos_dist']<= HYPERPARAM_b_max_cosine].cos_dist.to_list()
        )
        
    def lambda_get_2018_matches(row):
        #get embedding matches from 2018 papers
        #returning relevant information
        df_clustered_papers = dict_agg_cluster_matches[row.agg_cluster]
        lst_paper_titles = df_clustered_papers[0].title.tolist()
        lst_paper_ids = df_clustered_papers[0].paper_id.tolist()
        ls_paper_keywords = df_clustered_papers[0].keywords.values.tolist()
        ls_cos_distances = df_clustered_papers[1]

        return lst_paper_titles, ls_paper_keywords, lst_paper_ids, ls_cos_distances

    def get_num_knn_matches(row):
        return(len(row.titles_2018))

    df_embeddings_2017[['titles_2018','keywords_2018','id_2018','cos_dist_2018']]= df_embeddings_2017.apply(lambda x: lambda_get_2018_matches(x),axis=1, result_type ='expand')
    df_embeddings_2017['num_knn_matches'] = df_embeddings_2017.apply(lambda x: get_num_knn_matches(x),axis =1)

    assert df_embeddings_2017.shape[0] == df_submission_labels[df_submission_labels['conf_year']==2017].shape[0]
    
    
    df_num = pd.DataFrame(df_embeddings_2017["cos_dist_2018"].to_list())
    
    
    data = []
    found = False
    start = 0
    result = 0
    
    while found == False:

        for tuning_param_cos_dist in np.linspace(start, start+0.3 ,3000):
            input_row = {}
            sample_number_from_2017 = df_num[df_num<=tuning_param_cos_dist].any(axis=1).sum()

            input_row['sample_number_2017'] = sample_number_from_2017
            input_row['cosine_distance'] = tuning_param_cos_dist
            data.append(input_row)
        # Find the smallest cosine_distance for sample_number_2017 == all of the 2017 rows
        distances = [entry['cosine_distance'] for entry in data if entry['sample_number_2017'] == df_embeddings_2017.shape[0]]

        if distances:
            result = min(distances)
            found = True
        else:
            start = start + 0.3

        
    return(round(result, 4))