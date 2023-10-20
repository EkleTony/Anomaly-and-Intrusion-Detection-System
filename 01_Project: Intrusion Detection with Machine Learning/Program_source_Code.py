import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load the training data
train_data = pd.read_csv('KDDTrain+.txt', header=None)  # Adjust the path as needed

# Load the test data
test_data = pd.read_csv('KDDTest+.txt', header=None)  # Adjust the path as needed


#test_data


# ===============loading the arff files to extract attribute names========
#=========================== Get Column names from arff files=============
#data = arff.loadarff('KDDTrain2+.arff')
#df = pd.DataFrame(data[0])
## print the data set
#print(df.head())
#print("========Attribute names ======== \n", df.columns)


# ## 2.0   Change df_train column names

column_mapping = {
    0: 'duration',
    1: 'protocol_type',
    2: 'service',
    3: 'flag',
    4: 'src_bytes',
    5: 'dst_bytes',
    6: 'land',
    7: 'wrong_fragment',
    8: 'urgent',
    9: 'hot',
    10: 'num_failed_logins',
    11: 'logged_in',
    12: 'num_compromised',
    13: 'root_shell',
    14: 'su_attempted',
    15: 'num_root',
    16: 'num_file_creations',
    17: 'num_shells',
    18: 'num_access_files',
    19: 'num_outbound_cmds',
    20: 'is_host_login',
    21: 'is_guest_login',
    22: 'count',
    23: 'srv_count',
    24: 'serror_rate',
    25: 'srv_serror_rate',
    26: 'rerror_rate',
    27: 'srv_rerror_rate',
    28: 'same_srv_rate',
    29: 'diff_srv_rate',
    30: 'srv_diff_host_rate',
    31: 'dst_host_count',
    32: 'dst_host_srv_count',
    33: 'dst_host_same_srv_rate',
    34: 'dst_host_diff_srv_rate',
    35: 'dst_host_same_src_port_rate',
    36: 'dst_host_srv_diff_host_rate',
    37: 'dst_host_serror_rate',
    38: 'dst_host_srv_serror_rate',
    39: 'dst_host_rerror_rate',
    40: 'dst_host_srv_rerror_rate',
    41: 'class'
}

# Rename the columns using the dictionary
train_data.rename(columns=column_mapping, inplace=True)
test_data.rename(columns=column_mapping, inplace=True)


# dropping the last column which is irrelevant
train_data = train_data.drop(columns=[42])
test_data = test_data.drop(columns=[42])
print("print training data: ", train_data.head())


# number of unique classes in train data
print("number of classes in train data: ",len(train_data["class"].unique()))
print("number of classes in test data: ",len(test_data["class"].unique()))
print('Samples of classes in train dataset: ', train_data["class"].unique())


# ## 3. =================visualization==============

# Calculate class counts and percentages
class_counts = train_data['class'].value_counts()
total_samples = len(train_data)
percentage_labels = [(count / total_samples) * 100 for count in class_counts]

# Select the top 15 classes with higher counts
top_15_class_counts = class_counts.head(15)

# Create a bar chart to visualize the class distribution
plt.figure(figsize=(20, 6))
ax = top_15_class_counts.plot(kind='bar')
plt.title('Top 15 Intrusion Detection classses in NSL-KDD', fontweight='bold', fontsize=14)
plt.xlabel('Cyber Attack Types', fontweight='bold', fontsize=14)
plt.ylabel('Number of Records', fontweight='bold', fontsize=14)
plt.xticks(rotation=0, fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

# Add percentage labels on top of the bars (make them bold)
for i, count in enumerate(top_15_class_counts):
    plt.text(i, count, f'{percentage_labels[i]:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Save the plot as an image (e.g., PNG)
#plt.savefig('01_Top_15_attacks.png', bbox_inches='tight')

#plt.show()


# ## 3================ Numerical Values vs Text Values
# Get the columns with text values
text_columns = train_data.select_dtypes(include=['object'])

# Print the text columns
#print("Columns with text values:")
#print(text_columns.columns.tolist())


# # Building the Decision Tree
print("===================================================")
print("====================Model Loading==================")
print("====================================================")
print("=================Training Accuracy loading==========")


def Decision_Tree_Learning(examples, attributes, parent_examples=None):
    if examples.empty:
        return majority_value(parent_examples)
    elif examples['class'].nunique() == 1:
        return examples['class'].iloc[0]
    elif len(attributes) == 0:
        return majority_value(examples)
    else:
        A = argmax_importance(attributes, examples)
        tree = {'attribute': A, 'branches': {}}
        attribute_values = examples[A]
        for vk in attribute_values.unique():
            exs = examples[examples[A] == vk]
            subtree = Decision_Tree_Learning(exs, [attr for attr in attributes if attr != A], examples)
            tree['branches'][f'{A}={vk}'] = subtree
        return tree

# Rest of your code remains unchanged
def majority_value(examples):
    # Calculate the most common class in the 'class' column of the examples DataFrame
    most_common_class = examples['class'].mode().iloc[0]
    return most_common_class


def argmax_importance(attributes, examples):
    best_attribute = None
    best_importance = -1

    for attribute in attributes:
        importance = IMPORTANCE(attribute, examples)
        if importance > best_importance:
            best_importance = importance
            best_attribute = attribute

    return best_attribute

def IMPORTANCE(attribute, examples):
    total_examples = len(examples)

    if total_examples == 0:
        return 0.0

    initial_entropy = ENTROPY(examples)

    attribute_values = examples[attribute].unique()

    attribute_entropy = 0.0
    for value in attribute_values:
        subset = examples[examples[attribute] == value]
        subset_weight = len(subset) / total_examples
        attribute_entropy += subset_weight * ENTROPY(subset)

    importance = initial_entropy - attribute_entropy

    return importance

def ENTROPY(examples):
    total_examples = len(examples)

    if total_examples == 0:
        return 0.0

    class_counts = examples['class'].value_counts()

    entropy = 0.0
    for class_count in class_counts:
        class_probability = class_count / total_examples
        entropy -= class_probability * np.log2(class_probability)

    return entropy


# =================================================================
# Load your CSV dataset as a pandas DataFrame
# Define a list of attributes (column names)
attributes = train_data.columns.tolist()
attributes.remove('class')  # Remove the target column from attributes

# Use the same data for both training and testing
data_for_both = train_data

# Load the test_data2 dataset for testing
test_data = test_data

# Build the decision tree
decision_tree = Decision_Tree_Learning(data_for_both, attributes)

# Create a prediction function
def predict(tree, example):
    while 'attribute' in tree:
        attribute = tree['attribute']
        value = example[attribute]
        branch_key = f'{attribute}={value}'
        if branch_key in tree['branches']:
            tree = tree['branches'][branch_key]
        else:
            return majority_value(example)
    return tree

# Create an accuracy function
def accuracy(tree, data):
    correct_predictions = 0
    total_predictions = len(data)
    for _, example in data.iterrows():
        prediction = predict(tree, example)
        if prediction == example['class']:
            correct_predictions += 1
    return correct_predictions / total_predictions

# Calculate accuracy on the training and testing sets (same dataset)
train_accuracy = accuracy(decision_tree, data_for_both)



print("======================== Training Accuracy ==================")
print("=============================================================")
print("\n")
print("Accuracy on training set:", train_accuracy)
print("\n")
print("======================Testing Accuracy Loading===================")


# ============================ TESTING DATA =======================================

def train_and_test_decision_tree(train_data, test_data):
    # Define a list of attributes (column names)
    attributes = train_data.columns.tolist()
    attributes.remove('class')  # Remove the target column from attributes

    # Build the decision tree on the training data
    decision_tree = Decision_Tree_Learning(train_data, attributes)

    # Create a prediction function
    def predict(tree, example):
        while 'attribute' in tree:
            attribute = tree['attribute']
            value = example[attribute]
            branch_key = f'{attribute}={value}'
            if branch_key in tree['branches']:
                tree = tree['branches'][branch_key]
            else:
                return majority_value(train_data)  # Use training data for plurality
        return tree

    # Create an accuracy function
    def accuracy(tree, data):
        correct_predictions = 0
        total_predictions = len(data)
        for _, example in data.iterrows():
            prediction = predict(tree, example)
            if prediction == example['class']:
                correct_predictions += 1
        return correct_predictions / total_predictions

    # Calculate accuracy on the test data
    test_accuracy = accuracy(decision_tree, test_data)

    return test_accuracy

# Load your CSV datasets as pandas DataFrames

train_data2 = train_data
test_data2 = test_data
# Call the train_and_test_decision_tree function with your datasets
accuracy_on_test_data = train_and_test_decision_tree(train_data2, test_data2)

print("======================== Testing Accuracy ==================")
print("=============================================================")
print("\n")
print("Accuracy on testing set:", accuracy_on_test_data)


# ======================== Plot Decision Tree ===================
# ================================================================


# ## Output of Decision Tree


def print_tree(tree, depth=0):
    if 'attribute' in tree:
        print("  " * depth + f"{tree['attribute']}:")
        for key, sub_tree in tree['branches'].items():
            print("  " * (depth + 1) + f"{key}")
            print_tree(sub_tree, depth + 2)
    else:
        print("  " * depth + f"Class: {tree}")
# Build the decision tree
#decision_tree = Decision_Tree_Learning(data_for_both, attributes)

# Print the decision tree using pre-order traversal
#print("=======================Decision Tree:=========================")
#print_tree(decision_tree)


# ## Output of Decision Tree program Saved in txt.

def save_tree_to_text(tree, depth=0, file_path="decision_tree.txt"):
    with open(file_path, 'w') as file:
        def write_tree_to_file(tree, depth=0):
            if 'attribute' in tree:
                file.write("  " * depth + f"{tree['attribute']}:\n")
                for key, sub_tree in tree['branches'].items():
                    file.write("  " * (depth + 1) + f"{key}\n")
                    write_tree_to_file(sub_tree, depth + 2)
            else:
                file.write("  " * depth + f"Class: {tree}\n")
        
        write_tree_to_file(tree, depth)

# Build the decision tree
#decision_tree = Decision_Tree_Learning(data_for_both, attributes)

# Specify the file path where you want to save the text file
#save_tree_to_text(decision_tree, file_path="decision_tree.txt")




