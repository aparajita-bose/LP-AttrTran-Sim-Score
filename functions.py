
import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import sklearn.metrics as metrics

from dataclasses import dataclass


def get_data(path="data.csv"):
    return pd.read_csv(path)


def prepare_data(
        df,
        COL_DATE="DATE",
        COL_SRC_NODE_ID = "SOURCE_ID",
        COL_SRC_ATTR = "SOURCE_SIC_CODE",
        COL_TARGET_NODE_ID = "TARGET_ID",
        COL_TARGET_ATTR = "TARGET_SIC_CODE"
    ):
    """
    Prepares training and testing from the Dataframe of Transactions
    and returns required variables as dict
    """

    # Taking TRAIN dataset as transactions from 2004 to 2012
    train = df[(df[COL_DATE] >= 20040000) & (df[COL_DATE] < 20130000)]
    # Taking TEST dataset as transactions from 2013 to 2015
    test = df[(df[COL_DATE] >= 20130000) & (df[COL_DATE] < 20160000)]

    # Creating TYPE_map[NODE_ID] = ATTR
    TYPE_lst = np.union1d(
        np.unique(train[COL_SRC_ATTR]),
        np.unique(train[COL_TARGET_ATTR]))
    source_lst = train.apply(lambda x: (x[COL_SRC_NODE_ID], x[COL_SRC_ATTR]), axis=1).tolist()
    target_lst = train.apply(lambda x: (x[COL_TARGET_NODE_ID], x[COL_TARGET_ATTR]), axis=1).tolist()
    TYPE_map = list(set(source_lst + target_lst))
    TYPE_map = {i[0]: i[1] for i in TYPE_map}

    # Removing duplicates edges from i (COL_SRC_NODE_ID) to j (COL_TARGET_NODE_ID)
    train = train[[COL_SRC_NODE_ID, COL_TARGET_NODE_ID]].drop_duplicates()
    test = test[[COL_SRC_NODE_ID, COL_TARGET_NODE_ID]].drop_duplicates()

    # get list of unique COL_SRC_NODE_ID + COL_TARGET_NODE_ID from TRAIN data
    node_lst = np.union1d(np.unique(train[COL_SRC_NODE_ID]),
                          np.unique(train[COL_TARGET_NODE_ID]))
    # node_lst = [1, 2, 3, 4]

    # Filter TEST to keep edges only if both COL_SRC_NODE_ID and COL_TARGET_NODE_ID is in TRAIN
    test = test[(test[COL_SRC_NODE_ID].isin(node_lst))
                & (test[COL_TARGET_NODE_ID].isin(node_lst))]

    # Converting TRAIN to list of (i, j)
    train_edge_lst = [(train.iloc[i, 0], train.iloc[i, 1])
                      for i in range(0, len(train))]
    # e.g.: train_edge_lst = [(1, 2), (1, 3), ..., (i,j)]

    # Converting TEST to list of  (i,j)
    test_edge_lst = [(test.iloc[i, 0], test.iloc[i, 1])
                     for i in range(0, len(test))]
    # e.g.: test_edge_lst = [(i,j), ...]

    # contains edges that newly appeared in test sets
    # i.e edges that appeared after 2013
    # this is POSITIVE
    edges_only_in_test = list(set(test_edge_lst)-set(train_edge_lst))

    G = nx.DiGraph()
    G.add_nodes_from(node_lst)
    G.add_edges_from(train_edge_lst)
    # G.add_edges_from(edges_only_in_test)
    G_adj = nx.adjacency_matrix(G)

    # Design Matrix
    X = {node: [0 if TYPE != TYPE_map[node] else 1 for TYPE in TYPE_lst]
         for node in node_lst}
    X = pd.DataFrame(X).T

    return {
        "G_adj": G_adj,
        "node_lst": node_lst,
        "edges_only_in_test": edges_only_in_test,
        "TYPE_map": TYPE_map,
        "X": X
    }


def get_transaction_similarity_matrix(graph, n, alpha):
    s_o = 0
    s_i = 0

    power = graph

    s_o += power @ power.T
    s_i += power.T @ power

    for i in range(1, n):

        power = power@graph

        s_o += (alpha**i) * (power @ power.T)
        s_i += (alpha**i) * (power.T @ power)

    return s_o, s_i


def get_attr_similarity_matrix(graph, X, m, alpha):
    beta_o = 0
    beta_i = 0

    power = graph.astype(float)

    beta_o += power @ X
    beta_i += power.T @ X

    for i in range(1, m):

        power = power @ graph

        beta_o += (alpha**i) * (power @ X)
        beta_i += (alpha**i) * (power.T @ X)

    B_O = cosine_matrix(beta_o)
    B_I = cosine_matrix(beta_i)

    return B_O, B_I


def cosine_matrix(mat):
    # calculate cosine distance matrix
    compatibility = squareform(pdist(mat, metric='cosine'))
    compatibility = np.nan_to_num(compatibility, nan=1)

    # cosine distance to cosine similarity
    compatibility = (compatibility*(-1))+1

    return compatibility


def compute_lp_ats_p_matrix(data, m, alpha, gamma):
    """
    Computes and returns LP-ATS Similarity Score Matrix

    Args:
        m (Integer): distance of neighbor (e.g., 1,2,3)
        alpha (Float): 0-1 float value - that decide influence of neighbor
        gamma (Float): 0-1 float value - tuning parameter
    """

    G_adj = data["G_adj"]
    X = data["X"]

    S_O, S_I = get_transaction_similarity_matrix(G_adj, m, alpha)
    B_O, B_I = get_attr_similarity_matrix(G_adj, X, m, alpha)

    C_O = (gamma * B_O) + ((1-gamma) * S_O)
    C_I = (gamma * B_I) + ((1-gamma) * S_I)

    p_matrix = C_O @ G_adj + G_adj @ C_I

    return p_matrix

@dataclass
class Result:
    roc_auc: float
    fpr: any
    tpr: any
def measure_result(data, p_matrix):
    """
    Computes and returns AUC score
    """
    node_lst = data["node_lst"]
    edges_only_in_test = data["edges_only_in_test"]

    # Creating y_scores (prediction score) by flattening p_matrix

    y_score = p_matrix.flatten()

    # Creating y_true by flattening TEST adjacency matrix

    g_test = nx.DiGraph()
    g_test.add_nodes_from(node_lst)
    g_test.add_edges_from(edges_only_in_test)
    adj_test = nx.adjacency_matrix(g_test)

    y_true = adj_test.todense().flatten()

    fpr, tpr, threshold = metrics.roc_curve(y_true, y_score)
    auc_score = metrics.auc(fpr, tpr)

    return Result(
        roc_auc=auc_score,
        fpr=fpr,
        tpr=tpr
    )