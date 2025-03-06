import csv
import random
import math
from collections import defaultdict
from collections import Counter

tree = None # My decision tree initalization

attribute_indices = { # Dict that will be used in lots of places in the code
    "age": 0,
    "job": 1,
    "marital": 2,
    "education": 3,
    "default": 4,
    "balance": 5,
    "housing": 6,
    "loan": 7,
    "contact": 8,
    "day": 9,
    "month": 10,
    "duration": 11,
    "campaign": 12,
    "pdays": 13,
    "previous": 14,
    "poutcome": 15,
}

file_path = "bank.csv" # Load dataset
examples = []
with open(file_path, mode='r') as file:
    reader = csv.reader(file)
    header = next(reader)  
    examples = [row for row in reader]

# More initalizations
training_data = None
testing_data = None
bucketed_training_data = None
bucketed_testing_data = None

# Compute the entropy of a group based on the target counts.
def entropy(target_counts):
    """
    
    
    
    """
    total = sum(target_counts.values())
    if total == 0:
        return 0  # If there are no examples, entropy is 0.
    
    entropy_value = 0
    for count in target_counts.values():
        if count > 0:
            probability = count / total
            entropy_value -= probability * math.log2(probability)
    return entropy_value

# Compute the conditional entropy of splitting the data by an attribute.
def conditional_entropy(examples, attribute_index, target_index):
    # Group examples by the attribute value
    groups = defaultdict(lambda: {"yes": 0, "no": 0})
    for row in examples:
        attribute_value = row[attribute_index]
        target_value = row[target_index]
        groups[attribute_value][target_value] += 1

    # Calculate weighted entropy
    total_examples = len(examples)
    conditional_entropy_value = 0
    for group_counts in groups.values():
        group_size = sum(group_counts.values())
        group_entropy = entropy(group_counts)
        conditional_entropy_value += (group_size / total_examples) * group_entropy
    
    return conditional_entropy_value

# Compute the information gain of splitting the data by an attribute.
def information_gain(examples, attribute_index, target_index):
    # Calculate entropy of the entire dataset
    total_counts = defaultdict(int)
    for row in examples:
        target_value = row[target_index]
        total_counts[target_value] += 1
    total_entropy = entropy(total_counts)
    
    # Calculate conditional entropy
    conditional_entropy_value = conditional_entropy(examples, attribute_index, target_index)
    
    # Information gain is the difference
    return total_entropy - conditional_entropy_value

# Select the best attribute for splitting based on information gain.
def best_attribute(data, attributes, target_index=-1):
   
    # Calculate entropy of the entire dataset
    target_counts = defaultdict(int)
    for row in data:
        target_counts[row[target_index]] += 1
    dataset_entropy = entropy(target_counts)

    #print(f"Entropy of the dataset: {dataset_entropy}")
    # Calculate information gain for each attribute
    best_attr = None
    max_information_gain = -1
    for attribute in attributes:
        attr_conditional_entropy = conditional_entropy(data, attribute, target_index)
        information_gain = dataset_entropy - attr_conditional_entropy

        #print(f"Attribute {attribute}: Conditional Entropy = {attr_conditional_entropy}, " f"Information Gain = {information_gain}")
        if information_gain > max_information_gain:
            max_information_gain = information_gain
            best_attr = attribute

    #print(f"Best attribute selected: {best_attr} with Information Gain: {max_information_gain}")
    return best_attr

# Calculate the Chi-Square statistic using the provided formula from the class
def chi_square_statistic(p, n, child_nodes):
    # Handle edge cases where the parent node is empty
    if p + n == 0:
        return 0  # No data, statistic is zero

    # Calculate expected proportions for "True" (p) and "False" (n) in the parent node
    p_hat = p / (p + n)
    n_hat = n / (p + n)

    chi_square = 0
    for child in child_nodes:
        p_i = child.get("p", 0)  # Observed "True" in the child node
        n_i = child.get("n", 0)  # Observed "False" in the child node

        # Avoid division by zero for child nodes
        if p_i > 0:
            chi_square += ((p_i - p_hat) ** 2) / (p_i ** 2)
        if n_i > 0:
            chi_square += ((n_i - n_hat) ** 2) / (n_i ** 2)

    return chi_square

def chi_square_critical_value(df):
    """
   - Retrieve the critical value for a given degrees of freedom from Chi squerd table that calculated before with scipy
   - Alpha is hard coded to 0.05
   - Took only 100 values with assumption that the splits will be much smaller then 100
    """
    critical_values = [
        3.841458820694124, 5.991464547107979, 7.814727903251179, 9.487729036781154,
        11.070497693516351, 12.591587243743977, 14.067140449340169, 15.50731305586545,
        16.918977604620448, 18.307038053275146, 19.67513757268249, 21.02606981748307,
        22.362032494826934, 23.684791304840576, 24.995790139728616, 26.29622760486423,
        27.58711163827534, 28.869299430392623, 30.14352720564616, 31.410432844230918,
        32.670573340917315, 33.92443847144381, 35.17246162690806, 36.41502850180731,
        37.65248413348277, 38.885138659830055, 40.113272069413625, 41.33713815142739,
        42.55696780429269, 43.77297182574219, 44.98534328036513, 46.19425952027847,
        47.39988391908093, 48.602367367294164, 49.80184956820181, 50.99846016571065,
        52.192319730102895, 53.383540622969356, 54.572227758941736, 55.75847927888702,
        56.94238714682408, 58.12403768086803, 59.30351202689981, 60.480886582336446,
        61.65623337627955, 62.829620411408165, 64.00111197221803, 65.17076890356982,
        66.3386488629688, 67.5048065495412, 68.66929391228578, 69.83216033984813,
        70.99345283378227, 72.15321616702309, 73.31149302908324, 74.46832415930936,
        75.62374846937608, 76.7778031560615, 77.93052380523042, 79.08194448784874,
        80.23209784876272, 81.3810151888991, 82.5287265414718, 83.67526074272097,
        84.82064549765667, 85.96490744123096, 87.10807219532191, 88.25016442187412,
        89.39120787250796, 90.53122543488065, 91.67023917605484, 92.80827038310771,
        93.94533960119225, 95.08146666924324, 96.21667075350383, 97.35097037903296,
        98.48438345934042, 99.61692732428385, 100.74861874635032, 101.87947396543588,
        103.00950871222618, 104.13873823027387, 105.26717729686034, 106.39484024272251,
        107.52174097071946, 108.6478929735076, 109.77330935028795, 110.89800282268448,
        112.02198574980785, 113.1452701425554, 114.26786767719355, 115.38978970826685,
        116.51104728087356, 117.63165114234555, 118.75161175336736, 119.87093929856714,
        120.98964369660958, 122.10773460981942, 123.2252214533618
    ]
    if 1 <= df <= 100:
        return critical_values[df - 1]  # Indexing is 0-based
    else:
        raise ValueError("Degrees of freedom must be between 1 and 100.")
    
# Prune the decision tree using Chi-Square pruning
def prune(tree, data, attribute_indices):
    
    # Reverse the mapping to get attribute names from indices
    index_to_name = {v: k for k, v in attribute_indices.items()}

    # If it's a leaf node, return it as-is
    if not isinstance(tree, dict):
        return tree

    attribute_index = tree["attribute"]  # The attribute index used in the tree
    branches = tree["branches"]

    # Split data based on branches
    child_data = {}
    for branch_value, subtree in branches.items():
        child_data[branch_value] = [
            row for row in data if row[attribute_index] == branch_value
        ]

    # Recursively prune subtrees and update branches
    for branch_value, subtree in branches.items():
        branches[branch_value] = prune(
            subtree, child_data[branch_value], attribute_indices
        )

    # After pruning subtrees, calculate Chi-Square for this node
    p = sum(row[-1] == "yes" for row in data)  # Total "yes" in this node
    n = sum(row[-1] == "no" for row in data)  # Total "no" in this node

    if p + n == 0:
        return "unknown"  # Handle edge case with no data

    child_nodes = []
    for branch_value, subset in child_data.items():
        p_i = sum(row[-1] == "yes" for row in subset)  # "yes" in child node
        n_i = sum(row[-1] == "no" for row in subset)  # "no" in child node
        child_nodes.append({"p": p_i, "n": n_i})

    # Degrees of freedom: number of branches - 1
    df = len(branches) - 1
    if df <= 0:
        return tree

    alpha = 0.05  # Assuming this because I didn't want to use python libraries ...
    critical_value = chi_square_critical_value(df) # In this funcation the alpha is 0.05 (Hard coded, very Hard coded)
    chi_square_value = chi_square_statistic(p, n, child_nodes)

    if chi_square_value < critical_value:
        majority_class = "yes" if p >= n else "no"
        return majority_class
    return tree

# Determines the most common target value (plurality) in the dataset.
def plurality_value(data):
    
    target_values = [row[-1] for row in data]
    most_common = Counter(target_values).most_common(1)[0][0]
    return most_common

# The main logic for tree buliding (the implentation of the algo from the class)
def decision_tree_learning(data, attributes, parent_data=None):
    if not data:  # No examples
        return plurality_value(parent_data)
    elif all(row[-1] == data[0][-1] for row in data):  # if all examples are in the same category
        return data[0][-1]
    elif not attributes:  # if there is no attributes to split so return
        return plurality_value(data)
    else:
        # selecting best attribute for split
        best_attr = best_attribute(data, attributes)
        #print(f"Best attribute selected: {best_attr}")
        tree = {"attribute": best_attr, "branches": {}}  # New Tree with the best_attr as root

        # Create subtrees for each value of the best attribute
        values = set(row[best_attr] for row in data)
        for value in values:
            #print(f"Splitting on attribute {best_attr} with value {value}")
            subset = [row for row in data if row[best_attr] == value]
            subtree = decision_tree_learning(subset,
            [attr for attr in attributes if attr != best_attr],
                data
            )
            tree["branches"][value] = subtree

        return tree
    
# Apply predefined buckets to the data for all attributes.
def apply_buckets(data, attribute_indices):

    bucketed_data = []
    
    for row in data:
        new_row = row[:]
        
        # Age Buckets
        age_index = attribute_indices.get("age", -1)
        if age_index >= 0:
            age = float(row[age_index])
            if 18 <= age <= 25:
                new_row[age_index] = "18-25"
            elif 26 <= age <= 35:
                new_row[age_index] = "26-35"
            elif 36 <= age <= 45:
                new_row[age_index] = "36-45"
            elif 46 <= age <= 55:
                new_row[age_index] = "46-55"
            elif 56 <= age <= 60:
                new_row[age_index] = "56-60"
            elif 61 <= age <= 63:
                new_row[age_index] = "61-63"
            elif 64 <= age <= 80:
                new_row[age_index] = "64-80"
            else:
                new_row[age_index] = "81+"

        # Job Buckets
        job_index = attribute_indices.get("job", -1)
        if job_index >= 0:
            job = row[job_index]
            if job in ["unknown"]:
                new_row[job_index] = "unknown"
            elif job in ["unemployed", "housemaid"]:
                new_row[job_index] = "unemployed & housemaid"
            elif job in ["technician", "student", "blue-collar"]:
                new_row[job_index] = "technician & student & blue-collar"
            elif job in ["management", "self-employed", "entrepreneur"]:
                new_row[job_index] = "management & self-employed & entrepreneur"
            elif job in ["services", "retired"]:
                new_row[job_index] = "services & retired"

        # Marital Buckets
        marital_index = attribute_indices.get("marital", -1)
        if marital_index >= 0:
            marital = row[marital_index]
            if marital in ["married"]:
                new_row[marital_index] = "married"
            elif marital in ["single"]:
                new_row[marital_index] = "single"
            elif marital in ["divorced"]:
                new_row[marital_index] = "divorced"

        # Education Buckets
        education_index = attribute_indices.get("education", -1)
        if education_index >= 0:
            education = row[education_index]
            if education in ["unknown"]:
                new_row[education_index] = "unknown"
            elif education in ["primary"]:
                new_row[education_index] = "primary"
            elif education in ["secondary", "tertiary"]:
                new_row[education_index] = "secondary & tertiary"

        # Default Buckets
        default_index = attribute_indices.get("default", -1)
        if default_index >= 0:
            default = row[default_index]
            new_row[default_index] = default  # "yes" or "no"

        # Balance Buckets
        balance_index = attribute_indices.get("balance", -1)
        if balance_index >= 0:
            balance = float(row[balance_index])
            if balance < 0:
                new_row[balance_index] = "<0"
            elif 0 <= balance <= 1000:
                new_row[balance_index] = "0-1000"
            elif 1001 <= balance <= 5000:
                new_row[balance_index] = "1001-5000"
            else:
                new_row[balance_index] = ">5000"

        # Housing, Loan Buckets
        for attr_name in ["housing", "loan"]:
            attr_index = attribute_indices.get(attr_name, -1)
            if attr_index >= 0:
                new_row[attr_index] = row[attr_index]  # "yes" or "no"

        # Contact Buckets
        contact_index = attribute_indices.get("contact", -1)
        if contact_index >= 0:
            contact = row[contact_index]
            if contact == "cellular":
                new_row[contact_index] = "cellular"
            elif contact == "telephone":
                new_row[contact_index] = "telephone"
            else:
                new_row[contact_index] = "unknown"

        # Day Buckets
        day_index = attribute_indices.get("day", -1)
        if day_index >= 0:
            day = int(row[day_index])
            if 1 <= day <= 7:
                new_row[day_index] = "1-7"
            elif 8 <= day <= 15:
                new_row[day_index] = "8-15"
            elif 16 <= day <= 23:
                new_row[day_index] = "16-23"
            elif 24 <= day <= 31:
                new_row[day_index] = "24-31"

        # Month Buckets
        month_index = attribute_indices.get("month", -1)
        if month_index >= 0:
            month = row[month_index]
            if month in ["jun", "jul", "aug"]:
                new_row[month_index] = "jun & jul & aug"
            elif month in ["may"]:
                new_row[month_index] = "may"
            elif month in ["oct", "dec", "mar", "sep"]:
                new_row[month_index] = "oct & dec & mar & sep"
            elif month in ["nov"]:
                new_row[month_index] = "nov"
            elif month in ["jan", "feb", "apr"]:
                new_row[month_index] = "jan & feb & apr"

        # Duration Buckets
        duration_index = attribute_indices.get("duration", -1)
        if duration_index >= 0:
            duration = float(row[duration_index])
            if 0 <= duration <= 50:
                new_row[duration_index] = "0-50"
            elif 51 <= duration <= 100:
                new_row[duration_index] = "51-100"
            elif 101 <= duration <= 150:
                new_row[duration_index] = "101-150"
            elif 151 <= duration <= 200:
                new_row[duration_index] = "151-200"
            elif 201 <= duration <= 300:
                new_row[duration_index] = "201-300"
            elif 301 <= duration <= 400:
                new_row[duration_index] = "301-400"
            elif 401 <= duration <= 500:
                new_row[duration_index] = "401-500"
            elif 501 <= duration <= 1000:
                new_row[duration_index] = "501-1000"
            else:
                new_row[duration_index] = "1001+"

        # Campaign Buckets
        campaign_index = attribute_indices.get("campaign", -1)
        if campaign_index >= 0:
            campaign = int(row[campaign_index])
            if campaign == 1:
                new_row[campaign_index] = "1"
            elif 2 <= campaign <= 3:
                new_row[campaign_index] = "2-3"
            else:
                new_row[campaign_index] = ">3"

        # Pdays Buckets
        pdays_index = attribute_indices.get("pdays", -1)
        if pdays_index >= 0:
            pdays = int(row[pdays_index])
            if pdays == -1:
                new_row[pdays_index] = "-1"
            elif 0 <= pdays <= 7:
                new_row[pdays_index] = "0-7"
            else:
                new_row[pdays_index] = ">7"

        # Previous Buckets
        previous_index = attribute_indices.get("previous", -1)
        if previous_index >= 0:
            previous = int(row[previous_index])
            if previous == 0:
                new_row[previous_index] = "0"
            elif previous == 1:
                new_row[previous_index] = "1"
            else:
                new_row[previous_index] = ">1"

        # Poutcome Buckets
        poutcome_index = attribute_indices.get("poutcome", -1)
        if poutcome_index >= 0:
            poutcome = row[poutcome_index]
            if poutcome in ["unknown"]:
                new_row[poutcome_index] = "unknown"
            elif poutcome in ["success"]:
                new_row[poutcome_index] = "success"
            elif poutcome in ["failure"]:
                new_row[poutcome_index] = "failure"
            else:
                new_row[poutcome_index] = "other"

        bucketed_data.append(new_row)

    return bucketed_data

def build_tree(ratio: float=1.0):
    global tree
    global examples
    global bucketed_training_data
    global testing_data
   
    random.shuffle(examples)
    split_index = int(len(examples)*ratio)
    training_data = examples[:split_index]
    testing_data = examples[split_index:] #updates the global attribute

    bucketed_training_data = apply_buckets(training_data,attribute_indices)

    #bucketed_testing_data = apply_buckets(testing_data,attribute_indices)
    #print(f"Total examples loaded: {len(examples)}") 
    #print(f"Training examples loaded: {len(training_data)}")
    #attributes = list(range(len(training_data[50]) - 1))  # alterntive is to hard code it to 16
    #print(attributes)
    #print(list(attribute_indices.values()))
    tree = decision_tree_learning(bucketed_training_data, list(attribute_indices.values()))
    #print('----------------------Start Pruning-------------------------')
    pruned_tree = prune(tree, bucketed_training_data, attribute_indices)
    training_error = tree_error(10)
    tree = pruned_tree
    #print(f"{pruned_tree} ")
    print_tree(tree)
    return tree, training_error

def tree_error(k):
    global tree
    global examples
    global testing_data
    global bucketed_testing_data

    fold_size = len(examples) // k  # Calculate fold size
    errors = []  
    bucketed_testing_data = apply_buckets(testing_data,attribute_indices)
    for i in range(k):
        # Split data into training and testing for the current fold
        start = i * fold_size
        end = start + fold_size
        testing_data = examples[start:end]
        #training_data = examples[:start] + examples[end:]

        # Evaluate the tree on the testing set
        correct = 0
        total = len(bucketed_testing_data)

        for test_example in bucketed_testing_data:
            predicted = classify(tree, test_example)
            actual = test_example[-1]  
            if predicted == actual:
                correct += 1

        error_rate = 1 - (correct / total)
        errors.append(error_rate)
        #print(f"Fold {i+1}/{k}: Error rate = {error_rate:.4f}") #debug

    # Calculate average error rate
    avg_error = sum(errors)/len(errors)
    print(f"Average error rate: {avg_error:.4f}")
    return avg_error

# Helper function to classify a single example using the tree
def classify(tree, example):
    if not isinstance(tree, dict):
       #print(f"im printing the leaf node from classifay func: {tree}")
        return tree
    attribute = tree["attribute"]
    value = example[attribute]
    if value in tree["branches"]:
        return classify(tree["branches"][value], example)
    else:
        return None  

def will_open_deposit(row_input):
    global tree

    # Ensure the tree has been built
    if tree is None:
        raise ValueError("The decision tree has not been built yet.")

    # Convert row_input values into corresponding buckets using apply_buckets logic
    bucketed_row = apply_buckets([row_input], attribute_indices)[0]

    # Classify the bucketed row using the tree
    prediction = classify(tree, bucketed_row)

    return 1 if prediction == 'yes' else 0

def parse_csv_row_to_array(row_str): # debug purpose
    
    return row_str.strip().split(",")

def print_tree(tree, depth=0, edge_label=''):
    global attribute_indices
    """
    Recursively print the custom tree structure in a readable format.
    
    :param tree: The tree to print (nested dictionary).
    :param depth: The current depth in the tree (used for indentation).
    :param edge_label: Label of the edge leading to the current node.
    """
    # Reverse the attribute_indices dictionary to map numbers to names
    attribute_indices_reversed = {v: k for k, v in attribute_indices.items()}
    indent = '-' * depth  # Indentation for current depth

    if isinstance(tree, str):  # Leaf node
        # Add color to leaf nodes based on their value
        if tree == "no":
            tree = f"\033[91m{tree}\033[0m"  # Red for "no"
        elif tree == "yes":
            tree = f"\033[92m{tree}\033[0m"  # Green for "yes"
        print(f"{indent}[{edge_label}] -> {tree}")
        return

    # Retrieve the current node's attribute
    attribute_number = tree.get('attribute')
    if attribute_number is None:
        print(f"{indent}[{edge_label}] -> ERROR: Missing 'attribute' key in node {tree}")
        return

    # Get the attribute name from attribute_indices_reversed
    attribute_name = attribute_indices_reversed.get(attribute_number, f"Unknown ({attribute_number})")

    # Print the current attribute
    if edge_label:
        print(f"{indent}[{edge_label}] -> Attribute: {attribute_name}")
    else:
        print(f"{indent}Attribute: {attribute_name}")

    # Recursively print each branch
    branches = tree.get('branches', {})
    for branch_label, subtree in branches.items():
        print_tree(subtree, depth + 1, edge_label=branch_label)




build_tree(0.6)
#print_tree(tree)
#tree_error(50)

print(will_open_deposit(parse_csv_row_to_array("58,management,married,tertiary,no,2143,yes,no,unknown,5,may,261,1,-1,0,unknown")))
print(will_open_deposit(parse_csv_row_to_array("44,technician,single,secondary,no,29,yes,no,unknown,5,may,151,1,-1,0,unknown")))
print(will_open_deposit(parse_csv_row_to_array("33,entrepreneur,married,secondary,no,2,yes,yes,unknown,5,may,76,1,-1,0,unknown")))
print(will_open_deposit(parse_csv_row_to_array("47,blue-collar,married,unknown,no,1506,yes,no,unknown,5,may,92,1,-1,0,unknown")))
print(will_open_deposit(parse_csv_row_to_array("33,unknown,single,unknown,no,1,no,no,unknown,5,may,198,1,-1,0,unknown")))
print(f"{will_open_deposit(parse_csv_row_to_array("59,admin.,married,secondary,no,2343,yes,no,unknown,5,may,1042,1,-1,0,unknown"))}")#yes
print(will_open_deposit(parse_csv_row_to_array("56,admin.,married,secondary,no,45,no,no,unknown,5,may,1467,1,-1,0,unknown")))#yes
print(will_open_deposit(parse_csv_row_to_array("41,technician,married,secondary,no,1270,yes,no,unknown,5,may,1389,1,-1,0,unknown")))#yes
print(will_open_deposit(parse_csv_row_to_array("54,admin.,married,tertiary,no,184,no,no,unknown,5,may,673,2,-1,0,unknown")))#yes
print(will_open_deposit(parse_csv_row_to_array("42,management,single,tertiary,no,0,yes,yes,unknown,5,may,562,2,-1,0,unknown")))#yes
print(will_open_deposit(parse_csv_row_to_array("56,management,married,tertiary,no,830,yes,yes,unknown,6,may,1201,1,-1,0,unknown")))#yes
print(will_open_deposit(parse_csv_row_to_array("60,retired,divorced,secondary,no,545,yes,no,unknown,6,may,1030,1,-1,0,unknown")))#yes
print(f"{will_open_deposit(parse_csv_row_to_array("28,blue-collar,single,secondary,no,-127,yes,no,cellular,4,jul,1044,3,-1,0,unknown"))}")#yes
