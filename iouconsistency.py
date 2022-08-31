import numpy as np
import torch

def iou(box1, box2):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    box1 = torch.from_numpy(box1)
    box1 = box1.view(1,4)
    box2 = torch.from_numpy(box2)

    (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, 1), box2.chunk(4, 1)
    w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_


    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter

    # IoU
    iou = inter / union
    return iou.detach().numpy() if len(iou) > 0 else np.array([0])

def precision(f1, f2):
    pred = []
    for box in f1:
        ious = iou(box[1:], f2[:,1:])
        max_id = np.argmax(ious)
        pred.append(ious[max_id] > 0.5 and box[0] == f2[max_id,0])
    return float(sum(pred)/len(pred) if len(pred) > 0 else 1)

def recall(f1, f2):
    pred = []
    for box in f2:
        ious = iou(box[1:], f1[:,1:])
        max_id = np.argmax(ious)
        pred.append(ious[max_id] > 0.5 and box[0] == f1[max_id,0])
    return float(sum(pred)/len(pred) if len(pred) > 0 else 1)





def compute_iou(f1, f2):
    f2_0 = f2[f2[:,0]==0.][:, 1:]
    f2_1 = f2[f2[:,0]==1.][:, 1:]
    iou_h, iou_w = [], []
    for pred in f1:
        if pred[0] == 0:
            iou_h.append(np.max(iou(pred[1:], f2_0)))
        else:
            iou_w.append(np.max(iou(pred[1:], f2_1)))
    return float(sum(iou_h)/len(iou_h) if len(iou_h) > 0 else 1), float(sum(iou_w)/len(iou_w) if len(iou_w) > 0 else 1)

iou_h, iou_w, p, r = [], [], [], []
for i in range(497, 507):
#    print(i)
    f1 = np.loadtxt('Dataset/'+str(i).zfill(5)+'.txt')
    f2 = np.loadtxt('Comp3/'+str(i).zfill(5)+'.txt')
    f1 = np.atleast_2d(f1)
    f2 = np.atleast_2d(f2)
    ih, iw = compute_iou(f1, f2)
    iou_h.append(ih)
    iou_w.append(iw)
    r.append(recall(f1,f2))
    p.append(precision(f1, f2))
#    iou_h, iou_w = compute_iou(f2, f1)
#    print(iou_h)
#    print(iou_w)
#    print(precision(f2, f1))
#    print(recall(f2,f1))
iou_h = sum(iou_h) / len(iou_h)
iou_w = sum(iou_w) / len(iou_w)
r = sum(r) / len(r)
p = sum(p) / len(p)
print(iou_h)
print(iou_w)
print(r)
print(p)