import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    month_to_num = {
        "Jan": 0,
        "Feb": 1,
        "Mar": 2,
        "Apr": 3,
        "May": 4,
        "June": 5,
        "Jul": 6,
        "Aug": 7,
        "Sep": 8,
        "Oct": 9,
        "Nov": 10,
        "Dec": 11
    }

    def convert_element(index, value):
        match index:
            case 0 | 2 | 4 | 11 | 12 | 13 | 14:
                return int(value)
            case 1 | 3 | 5 | 6 | 7 | 8 | 9:
                return float(value)
            case 10:
                return month_to_num[value]
            case 15:
                return 1 if "returning" in value.lower() else 0
            case 16:
                return 1 if value == "TRUE" else 0

    evidence = []
    labels = []
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        
        for i, row in enumerate(reader):
            if i == 0: continue
            label = 1 if row.pop() == "TRUE" else 0
            data = [convert_element(index, value) for index, value in enumerate(row)]
            labels.append(label)
            evidence.append(data)

    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    knc = KNeighborsClassifier(n_neighbors=1)
    knc.fit(evidence, labels)
    return knc


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    label_positive_amount = 0
    label_negative_amount = 0
    pred_positive_amount = 0
    pred_negative_amount = 0
    for index in range(0, len(predictions) - 1):
        label = labels[index]
        prediction = predictions[index]
        if label == 1:
            label_positive_amount += 1
            if prediction == 1:
                pred_positive_amount += 1
        if label == 0:
            label_negative_amount += 1
            if prediction == 0:
                pred_negative_amount += 1
    
    sensitivity = float(pred_positive_amount) / float(label_positive_amount)
    specificity = float(pred_negative_amount) / float(label_negative_amount)
    return (sensitivity, specificity)
    



if __name__ == "__main__":
    main()
