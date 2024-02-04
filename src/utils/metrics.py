import torch


def dice_coefficient(preds, labels, smooth=1.0):
    preds = torch.sigmoid(preds)
    preds = preds.argmax(dim=0)
    labels = labels.float()
    intersection = (preds * labels).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + labels.sum() + smooth)
    return dice

def iou_score(preds, labels, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = preds.argmax(dim=0)
    labels = labels
    intersection = (preds & labels).sum()
    union = (preds | labels).sum()
    iou = (intersection + smooth) / (union + smooth)
    return iou

def accuracy(preds, labels):
    preds = torch.sigmoid(preds)
    preds = preds.argmax(dim=0)
    correct = (preds == labels).float().sum()
    return correct / labels.numel()

def precision(preds, labels, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = preds.argmax(dim=0)
    labels = labels.float()
    true_positives = (preds * labels).sum()
    predicted_positives = preds.sum()
    precision = true_positives / (predicted_positives + smooth)
    return precision

def recall(preds, labels, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = preds.argmax(dim=0)
    labels = labels.float()
    true_positives = (preds * labels).sum()
    possible_positives = labels.sum()
    recall = true_positives / (possible_positives + smooth)
    return recall

def compute_metrics(preds, labels):

    return {
        "accuracy": round(accuracy(preds, labels).item(), 4),
        "iou": round(iou_score(preds, labels).item(), 4),
        "dice": round(dice_coefficient(preds, labels).item(), 4),
        "precision": round(precision(preds, labels).item(), 4),
        "recall": round(recall(preds, labels).item(), 4)
    }
