import numpy as np
import scipy.sparse as sp
import torch
import random
import torch.nn.functional as F


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_recall_curve, auc
from sklearn.preprocessing import label_binarize





#def encode_onehot(labels):
#    classes = set(labels)
#    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
#                    enumerate(classes)}
#    labels_onehot = np.array(list(map(classes_dict.get, labels)),
#                             dtype=np.int32)
#    return labels_onehot

def encode_onehot(labels):
    classes = set(labels)
    if len(classes) == 0:
        raise ValueError("Empty label array.")
    
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="/Users/adriankwan_mba/Downloads/pygcn-master/data/cora/", dataset="cora"):
#def load_data(path="/Users/adriankwan_mba/Downloads/pygcn-master/data/citeseer/", dataset="citeseer"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))


    train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), train_size=0.9, random_state=37)

    # Further split the training set into training and validation sets
    train_idx, val_idx = train_test_split(train_idx, train_size=9.444444e-01, random_state=37)
#    random.seed()
#    train_idx = np.random.choice(train_idx, size=int(len(train_idx) * 0.5), replace=False)
    # Convert indices to range objects
    idx_train = range(train_idx[0], train_idx[-1] + 1)
    idx_val = range(val_idx[0], val_idx[-1] + 1)
    idx_test = range(test_idx[0], test_idx[-1] + 1)

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def compute_auc_score(output, labels):
    preds = F.softmax(output, dim=1).detach().numpy()
    labels = labels.detach().numpy()

    # Compute AUC score
    auc_score = roc_auc_score(labels, preds, multi_class='ovr')
    return auc_score

    # AP score
def compute_ap_score(output, labels):
    # Convert the output predictions to probabilities
    probabilities = output.softmax(dim=1)
    probabilities = probabilities.detach().numpy()

    # Convert to one-hot encoded format
    labels_onehot = np.eye(labels.max() + 1)[labels]

    ap_score = average_precision_score(labels_onehot, probabilities, average='macro')
    return ap_score
    
    
    
#new calc
#def performance_scores(output, labels):
#    if len(labels.shape) == 1 or labels.shape[1] == 1:
#        # Binary classification
#        preds = output[:, 1].cpu().detach().numpy()
#        labels = labels.cpu().numpy()
#        auc_score = roc_auc_score(labels, preds)
#        ap_score = average_precision_score(labels, preds)
#    else:
#        preds = output.cpu().detach().numpy()
#        labels = labels.cpu().numpy()
#        auc_score = roc_auc_score(labels, preds, multi_class='ovr', average='macro')
#        ap_score = average_precision_score(labels, preds, average='macro')
#    return auc_score, ap_score
    
    
#def compute_auc_ap_score(output, labels):
#    # Apply softmax to the output to get class probabilities
#    probs = torch.softmax(output, dim=1)
#
#    # Convert the one-hot encoded labels to an array of class indices
#    labels = torch.argmax(labels, dim=1)
#
#    # Convert tensors to NumPy arrays
#    probs = probs.detach().cpu().numpy()
#    labels = labels.detach().cpu().numpy()
#
#    # Compute AUC score
#    auc_score = roc_auc_score(labels, probs, multi_class='ovr')
#
#    # Compute AP score
#    ap_score = average_precision_score(labels, probs, average='macro')
#
#    return auc_score, ap_score






#def compute_ap_score(output, labels):
#    num_classes = output.shape[1]
#    ap_scores = []
#
#    for class_idx in range(num_classes):
#        class_labels = np.where(labels == class_idx, 1, 0)
#        class_output = output[:, class_idx].detach().numpy()
#
#        precision, recall, _ = precision_recall_curve(class_labels, class_output)
#        ap = np.mean(precision)
#        ap_scores.append(ap)
#
#    mean_ap = np.mean(ap_scores)
#
#    return mean_ap
    
    
    #def compute_ap_score(output, labels):
#    # Convert the output predictions to probabilities
#    probabilities = output.softmax(dim=1).detach().numpy()
#
#    # Convert the labels to one-hot encoded format if necessary
#    if len(labels.shape) > 1 and labels.shape[1] > 1:
#        labels = labels.argmax(axis=1)
#
#    # Compute the average precision score
#    ap = average_precision_score(labels, probabilities, average='macro')
#
#    return ap
#
#def compute_ap_score(output, labels):
#    # Convert the labels to one-hot encoded format
#    labels_onehot = np.eye(labels.max() + 1)[labels]
#
#    # Convert the output tensor to a NumPy array
#    output_np = output.detach().numpy()
#
#    # Compute the average precision score for each class separately
#    ap_scores = []
#    for class_idx in range(labels_onehot.shape[1]):
#        ap = average_precision_score(labels_onehot[:, class_idx], output_np[:, class_idx])
#        ap_scores.append(ap)
#
#    # Compute the mean average precision (MAP) score
#    mean_ap = np.mean(ap_scores)
#
#    return mean_ap
    
    # Compute precision and recall curves
#    precision, recall, _ = precision_recall_curve(labels.ravel(), preds.ravel())
#
#    # Compute average precision score
    
#    num_classes = output.size(1)
#    ap_scores = []
#    for class_idx in range(num_classes):
#        class_labels = (labels == class_idx).astype(int)
#        class_preds = output[:, class_idx].detach().numpy()
#        ap_score = average_precision_score(class_labels, class_preds)
#        ap_scores.append(ap_score)
#
#    return auc_score, np.mean(ap_scores)

#
#def compute_average_precision(labels, preds):
#    if len(labels.shape) > 1:
#        num_classes = labels.shape[1]
#        average_precision = 0.0
#
#        for class_idx in range(num_classes):
#            ap_score = average_precision_score(labels[:, class_idx], preds[:, class_idx])
#            average_precision += ap_score
#
#        average_precision /= num_classes
#    else:
#        average_precision = average_precision_score(labels, preds)
#
#    return average_precision
#def compute_auc_ap(output, labels):
#    # Ensure both output and labels are numpy arrays
#    output = output.cpu().detach().numpy()
#    labels = labels.cpu().numpy()
#
#    # Flatten the arrays if necessary
#    if output.ndim > 1:
#        output = output[:, 1].flatten()
#    labels = labels.flatten()
#
#    # Sort predictions and labels in descending order of the output probabilities
#    sorted_indices = output.argsort()[::-1]
#    sorted_labels = labels[sorted_indices]
#
#    # Compute the True Positive (TP), False Positive (FP), and Positive (P) counts
#    tp_count = 0
#    fp_count = 0
#    p_count = sorted_labels.sum()
#
#    # Initialize variables for AUC and AP calculation
#    auc_score = 0.0
#    ap_score = 0.0
#
#    # Iterate over the sorted predictions and labels
#    for i, label in enumerate(sorted_labels):
#        if label == 1:
#            tp_count += 1
#        else:
#            fp_count += 1
#
#        # Update AUC score
#        auc_score += (fp_count / (p_count * fp_count)) if p_count > 0 else 0
#
#        # Update AP score if the current label is positive
#        if label == 1:
#            ap_score += tp_count / (i + 1)
#
#    # Normalize AUC and AP scores
#    auc_score /= p_count * (len(labels) - p_count) if p_count > 0 else 1
#    ap_score /= p_count if p_count > 0 else 1
#
#    return auc_score, ap_score


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
