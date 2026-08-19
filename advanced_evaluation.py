# advanced_evaluation.py
"""
LeafSense AI - Advanced Evaluation Module
Measures inference latency, plots per-class Precision-Recall (PR) curves, classification report heatmap,
and confusion matrix.
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.preprocessing import LabelBinarizer

warnings.filterwarnings('ignore')

def measure_inference_time(model, test_images):
    """
    Measures and prints inference time over test dataset.
    """
    start_time = time.time()
    _ = model.predict(test_images, steps=len(test_images))
    end_time = time.time()
    inference_duration = end_time - start_time
    print(f"[INFO] Inference time for {len(test_images) * test_images.batch_size} items: {inference_duration:.2f} seconds")
    return inference_duration

def plot_precision_recall_curves(y_true_labels, pred_labels, class_names):
    """
    Plots Precision vs. Recall curves for each disease class.
    """
    lb = LabelBinarizer()
    lb.fit(class_names)

    y_true_onehot = lb.transform(y_true_labels)
    y_pred_onehot = lb.transform(pred_labels)

    if y_true_onehot.shape[1] == 1:
        y_true_onehot = np.hstack((1 - y_true_onehot, y_true_onehot))
        y_pred_onehot = np.hstack((1 - y_pred_onehot, y_pred_onehot))

    average_precision = {}
    n_classes = y_true_onehot.shape[1]

    plt.figure(figsize=(14, 10))

    for i in range(n_classes):
        cls_name = class_names[i] if i < len(class_names) else f"Class_{i}"
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_pred_onehot[:, i])
        avg_p = average_precision_score(y_true_onehot[:, i], y_pred_onehot[:, i])
        average_precision[i] = avg_p

        print(f"Class {cls_name}:")
        print(f"  Average Precision: {avg_p:.2f}\n")

        plt.plot(recall, precision, lw=2, label=f'{cls_name} (AP = {avg_p:0.2f})')

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.legend(loc="best", fontsize=10)
    plt.title("Precision vs. Recall Curve", fontsize=16, fontweight='bold')
    plt.grid(True)
    plt.show()

def plot_classification_report_heatmap(y_true, y_pred):
    """
    Computes classification report and renders a heatmap without support/accuracy noise.
    """
    # zero_division=0 suppresses UndefinedMetricWarning
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose()

    if 'accuracy' in df.index:
        df = df.drop(['accuracy'])
    if 'support' in df.columns:
        df = df.drop(['support'], axis=1)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap='YlOrBr', fmt='.2f', cbar=True)
    plt.title('Classification Report Heatmap', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()

def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(14, 10), text_size=10, norm=False, savefig=False):
    """
    Plots a labeled confusion matrix comparing predictions and ground truth.
    """
    plt.style.use('fivethirtyeight')
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    n_classes = cm.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.YlOrBr)
    fig.colorbar(cax)

    labels = classes if classes else np.arange(n_classes)

    ax.set(
        title="Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels
    )

    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    plt.xticks(rotation=45, fontsize=text_size, ha='right')
    plt.yticks(fontsize=text_size)

    threshold = (cm.max() + cm.min()) / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        val_str = f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)" if norm else f"{cm[i, j]}"
        plt.text(
            j, i, val_str,
            horizontalalignment="center",
            color="white" if cm[i, j] > threshold else "black",
            size=text_size
        )

    if savefig:
        fig.savefig("confusion_matrix.png")

    plt.tight_layout()
    plt.show()

def run_advanced_evaluation(model, test_images, test_df: pd.DataFrame):
    """
    Executes full advanced evaluation suite.
    """
    measure_inference_time(model, test_images)

    test_images.reset()
    pred_probabilities = model.predict(test_images, steps=len(test_images))
    pred_indices = np.argmax(pred_probabilities, axis=1)
    labels_map = {v: k for k, v in test_images.class_indices.items()}
    pred_labels = [labels_map[k] for k in pred_indices]

    y_test = list(test_df['Label']) if (test_df is not None and not test_df.empty) else [labels_map[i] for i in test_images.classes]
    class_names = list(test_images.class_indices.keys())

    print("[INFO] Computing Precision-Recall curves...")
    plot_precision_recall_curves(y_test, pred_labels, class_names)

    print("[INFO] Generating Classification Report Heatmap...")
    plot_classification_report_heatmap(y_test, pred_labels)

    print("[INFO] Rendering Confusion Matrix...")
    make_confusion_matrix(y_test, pred_labels, classes=class_names, norm=True)

# # advanced_evaluation.py
# """
# LeafSense AI - Advanced Evaluation Module
# Measures inference latency, plots per-class Precision-Recall (PR) curves, classification report heatmap,
# and confusion matrix.
# """
#
# import time
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import itertools
# from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
# from sklearn.preprocessing import LabelBinarizer
#
# def measure_inference_time(model, test_images):
#     """
#     Measures and prints inference time over test dataset.
#     """
#     start_time = time.time()
#     _ = model.predict(test_images, steps=len(test_images))
#     end_time = time.time()
#     inference_duration = end_time - start_time
#     print(f"[INFO] Inference time for {len(test_images) * test_images.batch_size} items: {inference_duration:.2f} seconds")
#     return inference_duration
#
# def plot_precision_recall_curves(y_true_labels, pred_labels, class_names):
#     """
#     Plots Precision vs. Recall curves for each disease class.
#     """
#     lb = LabelBinarizer()
#     lb.fit(class_names)
#
#     y_true_onehot = lb.transform(y_true_labels)
#     y_pred_onehot = lb.transform(pred_labels)
#
#     if y_true_onehot.shape[1] == 1:
#         y_true_onehot = np.hstack((1 - y_true_onehot, y_true_onehot))
#         y_pred_onehot = np.hstack((1 - y_pred_onehot, y_pred_onehot))
#
#     average_precision = {}
#     n_classes = y_true_onehot.shape[1]
#
#     plt.figure(figsize=(14, 10))
#
#     for i in range(n_classes):
#         cls_name = class_names[i] if i < len(class_names) else f"Class_{i}"
#         precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_pred_onehot[:, i])
#         avg_p = average_precision_score(y_true_onehot[:, i], y_pred_onehot[:, i])
#         average_precision[i] = avg_p
#
#         print(f"Class {cls_name}:")
#         print(f"  Average Precision: {avg_p:.2f}\n")
#
#         plt.plot(recall, precision, lw=2, label=f'{cls_name} (AP = {avg_p:0.2f})')
#
#     plt.xlabel("Recall", fontsize=12)
#     plt.ylabel("Precision", fontsize=12)
#     plt.legend(loc="best", fontsize=10)
#     plt.title("Precision vs. Recall Curve", fontsize=16, fontweight='bold')
#     plt.grid(True)
#     plt.show()
#
# def plot_classification_report_heatmap(y_true, y_pred):
#     """
#     Computes classification report and renders a heatmap without support/accuracy noise.
#     """
#     report = classification_report(y_true, y_pred, output_dict=True)
#     df = pd.DataFrame(report).transpose()
#
#     if 'accuracy' in df.index:
#         df = df.drop(['accuracy'])
#     if 'support' in df.columns:
#         df = df.drop(['support'], axis=1)
#
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(df, annot=True, cmap='YlOrBr', fmt='.2f', cbar=True)
#     plt.title('Classification Report Heatmap', fontsize=18, fontweight='bold')
#     plt.tight_layout()
#     plt.show()
#
# def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(14, 10), text_size=10, norm=False, savefig=False):
#     """
#     Plots a labeled confusion matrix comparing predictions and ground truth.
#     """
#     plt.style.use('fivethirtyeight')
#     cm = confusion_matrix(y_true, y_pred)
#     cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
#     n_classes = cm.shape[0]
#
#     fig, ax = plt.subplots(figsize=figsize)
#     cax = ax.matshow(cm, cmap=plt.cm.YlOrBr)
#     fig.colorbar(cax)
#
#     labels = classes if classes else np.arange(n_classes)
#
#     ax.set(
#         title="Confusion Matrix",
#         xlabel="Predicted label",
#         ylabel="True label",
#         xticks=np.arange(n_classes),
#         yticks=np.arange(n_classes),
#         xticklabels=labels,
#         yticklabels=labels
#     )
#
#     ax.xaxis.set_label_position("bottom")
#     ax.xaxis.tick_bottom()
#     plt.xticks(rotation=45, fontsize=text_size, ha='right')
#     plt.yticks(fontsize=text_size)
#
#     threshold = (cm.max() + cm.min()) / 2.
#
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         val_str = f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)" if norm else f"{cm[i, j]}"
#         plt.text(
#             j, i, val_str,
#             horizontalalignment="center",
#             color="white" if cm[i, j] > threshold else "black",
#             size=text_size
#         )
#
#     if savefig:
#         fig.savefig("confusion_matrix.png")
#
#     plt.tight_layout()
#     plt.show()
#
# def run_advanced_evaluation(model, test_images, test_df: pd.DataFrame):
#     """
#     Executes full advanced evaluation suite.
#     """
#     measure_inference_time(model, test_images)
#
#     test_images.reset()
#     pred_probabilities = model.predict(test_images, steps=len(test_images))
#     pred_indices = np.argmax(pred_probabilities, axis=1)
#     labels_map = {v: k for k, v in test_images.class_indices.items()}
#     pred_labels = [labels_map[k] for k in pred_indices]
#
#     y_test = list(test_df['Label']) if (test_df is not None and not test_df.empty) else [labels_map[i] for i in test_images.classes]
#     class_names = list(test_images.class_indices.keys())
#
#     print("[INFO] Computing Precision-Recall curves...")
#     plot_precision_recall_curves(y_test, pred_labels, class_names)
#
#     print("[INFO] Generating Classification Report Heatmap...")
#     plot_classification_report_heatmap(y_test, pred_labels)
#
#     print("[INFO] Rendering Confusion Matrix...")
#     make_confusion_matrix(y_test, pred_labels, classes=class_names, norm=True)

