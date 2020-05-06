import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc


def compute_roc(y_true, y_score):
    """
    Computing the "Receiving Operating Characteristic curve" and area
    """
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_score)
    auroc = auc(false_positive_rate, true_positive_rate)
    return false_positive_rate, true_positive_rate, auroc


def plot_roc(y_true, y_score):
    """
    Ploting the Receiving Operating Characteristic curve
    """
    false_positive_rate, true_positive_rate, auroc = compute_roc(y_true, y_score)
    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.plot(false_positive_rate,
             true_positive_rate,
             color='darkorange',
             lw=2,
             label='ROC curve (area = {:.2f})'.format(auroc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver operating characteristic example', fontsize=15)
    plt.legend(loc="lower right", fontsize=14)
    plt.show()


def positive_negative_measurement(y_true, y_score):
    # Initialization
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # Calculating the model
    for i in range(len(y_score)):
        if y_true[i] == y_score[i] == 1:
            tp += 1
        if (y_score[i] == 1) and (y_true[i] != y_score[i]):
            fp += 1
        if y_true[i] == y_score[i] == 0:
            tn += 1
        if (y_score[i] == 0) and (y_true[i] != y_score[i]):
            fn += 1

    return tp, fp, tn, fn



