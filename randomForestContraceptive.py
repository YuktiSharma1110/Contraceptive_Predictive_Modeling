import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def custom_accuracy_score(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    return correct_predictions / total_predictions

#Function for calculating precision , in a multi label scenario
def custom_precision_score(y_true, y_pred):
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    precision = 0

    for label in unique_classes:
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        if (tp + fp) > 0:
            label_precision = tp / (tp + fp)
        else:
            label_precision = 0
        precision += label_precision

    precision /= len(unique_classes)
    return precision

#Function for calculating recall , in a multi label scenario
def custom_recall_score(y_true, y_pred):
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    recall = 0

    for label in unique_classes:
        tp = np.sum((y_pred == label) & (y_true == label))
        fn = np.sum((y_pred != label) & (y_true == label))
        if (tp + fn) > 0:
            label_recall = tp / (tp + fn)
        else:
            label_recall = 0
        recall += label_recall

    # Averaging over all classes
    recall /= len(unique_classes)
    return recall

#Function for calculating f1 score , in a multi label scenario
def custom_f1_score(y_true, y_pred):
    precision = custom_precision_score(y_true, y_pred)
    recall = custom_recall_score(y_true, y_pred)
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    return f1

#Entropy Calculation for Decision Tree
def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


#Calculates the indices of the instances of dataset which fall in the different branches of teh dataset
def calculate_indexes(X_train, feature, value, phase):
    indexes = []
    mask = np.zeros(len(X_train), dtype=bool)
    if phase == 'left':
        for i in range(len(X_train)):
            if X_train[i, feature] <= value:
                indexes.append(i)
    else:
        for i in range(len(X_train)):
            if X_train[i, feature] > value:
                indexes.append(i)
    mask[indexes] = True
    return mask


#Function to calculate the best feature and split value using information gain
def helper(features, samples, X_train, y_train, parent_entropy, n_features_to_select):
    best_info_gain = -np.inf
    best_split = None
    # Select m features at random from the complete set of attributes
    selected_features = np.random.choice(features, n_features_to_select, replace=False)

    for feature in selected_features:  # Iterate only through the selected features
        unique_values = np.unique(X_train[:, feature])
        potential_splits = (unique_values[:-1] + unique_values[1:]) / 2
        for value in potential_splits:
            left_mask = calculate_indexes(X_train, feature, value, 'left')
            right_mask = calculate_indexes(X_train, feature, value, 'right')
            left_entropy = entropy(y_train[left_mask])
            right_entropy = entropy(y_train[right_mask])
            left_count = np.sum(left_mask)
            right_count = np.sum(right_mask)
            info_gain = parent_entropy - (left_count / samples * left_entropy + right_count / samples * right_entropy)

            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_split = {'feature': feature, 'value': value}
    return best_split


#Function to build the decision tree during the training phase of the model : Using Information Gain Evaluation
def recursive_tree(X_train, y_train, max_depth, depth=0):
    samples, features = X_train.shape
    # Calculate the number of features to select as the square root of total features
    n_features_to_select = int(np.sqrt(features))

    if depth >= max_depth or len(np.unique(y_train)) == 1 or samples < 2:
        return np.argmax(np.bincount(y_train))

    parent_entropy = entropy(y_train)
    best_split = helper(features, samples, X_train, y_train, parent_entropy, n_features_to_select)

    if best_split is None:
        return np.argmax(np.bincount(y_train))

    left_mask = calculate_indexes(X_train, best_split['feature'], best_split['value'], 'left')
    right_mask = calculate_indexes(X_train, best_split['feature'], best_split['value'], 'right')
    left_subtree = recursive_tree(X_train[left_mask], y_train[left_mask], max_depth, depth + 1)
    right_subtree = recursive_tree(X_train[right_mask], y_train[right_mask], max_depth, depth + 1)

    return {'feature': best_split['feature'], 'value': best_split['value'], 'left_subtree': left_subtree,
            'right_subtree': right_subtree}



#Function to make a prediction on the  unknown data, to be categorized to a label subsequently
def prediction(x, tree=None):
    tree = tree if tree is None else tree
    # print(tree)
    if isinstance(tree, (int, np.integer)):
        return tree
    elif x[tree['feature']] <= tree['value']:
        return prediction(x, tree['left_subtree'])
    else:
        return prediction(x, tree['right_subtree'])


def create_bootstrap_datasets(X_train, y_train, n_trees):
    bootstrap_datasets = []
    n_samples = X_train.shape[0]
    for _ in range(n_trees):
        bootstrap_indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
        X_bootstrap = X_train[bootstrap_indices]
        y_bootstrap = y_train[bootstrap_indices]
        bootstrap_datasets.append((X_bootstrap, y_bootstrap))
    return bootstrap_datasets


def majority_voting(X, trees):
    predictions = []
    for x in X:
        votes = [prediction(x, tree) for tree in trees]
        vote_count = {}

        # Manually count votes
        for vote in votes:
            if vote in vote_count:
                vote_count[vote] += 1
            else:
                vote_count[vote] = 1

        # Find the prediction with the maximum votes
        max_votes = 0
        final_prediction = None
        for vote, count in vote_count.items():
            if count > max_votes:
                max_votes = count
                final_prediction = vote

        predictions.append(final_prediction)
    return predictions


def custom_stratified_k_fold(X, y, k):
    # Convert to numpy arrays if not already, to facilitate easy indexing
    X = np.asarray(X)
    y = np.asarray(y)

    # Determine unique classes and their distribution
    classes, y_indices = np.unique(y, return_inverse=True)
    class_counts = np.bincount(y_indices)

    n_samples = len(y)
    n_classes = len(classes)

    # Initialize the folds as lists of indices
    folds = [[] for _ in range(k)]

    # For each class, distribute the samples across folds
    for cls_index, cls in enumerate(classes):
        # Find the indices of all samples belonging to the current class
        indices = np.where(y == cls)[0]
        np.random.shuffle(indices)  # Shuffle to ensure random distribution

        # Evenly distribute indices of the current class across folds
        n_samples_for_cls = len(indices)
        n_samples_per_fold = np.full(k, n_samples_for_cls // k, dtype=int)

        # Handle remainder by adding one more sample to some folds
        remainder = n_samples_for_cls % k
        n_samples_per_fold[:remainder] += 1

        # Distribute the samples across folds
        current_idx = 0
        for fold_index in range(k):
            start, stop = current_idx, current_idx + n_samples_per_fold[fold_index]
            folds[fold_index].extend(indices[start:stop])
            current_idx = stop

    # Shuffle indices within each fold to ensure random ordering
    for fold in folds:
        np.random.shuffle(fold)

    #Display the class distribution in each fold
    for i, fold in enumerate(folds):
        print(f"Fold {i+1}:")
        # Initialize a dictionary to count the occurrences of each class in the fold
        class_distribution = {cls: 0 for cls in classes}
        for index in fold:
            class_distribution[y[index]] += 1
        # Print the distribution
        for cls, count in class_distribution.items():
            print(f"  Class {cls}: {count} instances")

    return folds


def upload_dataset(ntrees_values):
    df = pd.read_csv('cmc.data')

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    folds = custom_stratified_k_fold(X, y, 10)
    metrics_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    for k in ntrees_values:
        fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        for fold_idx in range(len(folds)):
            test_index = folds[fold_idx]
            train_index = np.hstack([folds[i] for i in range(len(folds)) if i != fold_idx])

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            bootstrap_datasets = create_bootstrap_datasets(X_train, y_train, k)
            trees = []

            for X_bootstrap, y_bootstrap in bootstrap_datasets:
                tree = recursive_tree(X_bootstrap, y_bootstrap, 10)  # Adjust max_depth if necessary
                trees.append(tree)

            y_pred = majority_voting(X_test, trees)

            fold_metrics['accuracy'].append(custom_accuracy_score(y_test, y_pred))
            fold_metrics['precision'].append(custom_precision_score(y_test, y_pred))
            fold_metrics['recall'].append(custom_recall_score(y_test, y_pred))
            fold_metrics['f1'].append(custom_f1_score(y_test, y_pred))

        # Aggregate and print fold metrics
        for metric in fold_metrics:
            metrics_results[metric].append(np.mean(fold_metrics[metric]))
            print(f"ntrees: {k}, {metric.capitalize()}: {np.mean(fold_metrics[metric])}")

    return ntrees_values, metrics_results

#plotting the graph for accuracy
def plot_metrics(k_values, metrics_results, path_prefix):
    # Plotting each metric
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    for metric, title in zip(metrics, titles):
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, metrics_results[metric], marker='o', linestyle='-', label=metric)
        plt.title(f'{title} vs. Number of Trees')
        plt.xlabel('Number of Trees')
        plt.ylabel(title)
        plt.xticks(k_values)
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{path_prefix}_{metric}_vs_ntrees.png")
        #plt.show()


#Code for Contraceptive Dataset using Information Gain as Stopping Criterion

if __name__ == '__main__':

    ntrees_values = [1, 5, 10, 20, 30, 40, 50]

    k_values, metric_result = upload_dataset(ntrees_values)
    plot_metrics(k_values, metric_result, "RF_Contraceptive")



