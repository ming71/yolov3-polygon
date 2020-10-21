import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from shapely.geometry import Polygon
import shapely

def load_classes(path):
	"""
	Loads class labels at 'path'
	"""
	fp = open(path, "r")
	names = fp.read().split("\n")[:-1]
	return names

def model_info(model):  # Plots a line-by-line description of a PyTorch model
	nP = sum(x.numel() for x in model.parameters())  # number parameters
	nG = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
	print('\n%4s %70s %9s %12s %20s %12s %12s' % ('', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
	for i, (name, p) in enumerate(model.named_parameters()):
		name = name.replace('module_list.', '')
		print('%4g %70s %9s %12g %20s %12g %12g' % (
			i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
	print('\n%g layers, %g parameters, %g gradients' % (i + 1, nP, nG))

def plot_one_box(x, img, color=None, label=None, line_thickness=None):  # Plots one bounding box on image img
	tl = line_thickness or round(0.001 * max(img.shape[0:2])) + 1  # line thickness
	color = color or [random.randint(0, 255) for _ in range(3)]

	#c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
	#cv2.rectangle(img, c1, c2, color, thickness=tl)

	cv2.line(img, (int(x[0]), int(x[1])), (int(x[2]), int(x[3])), color, tl)
	cv2.line(img, (int(x[2]), int(x[3])), (int(x[4]), int(x[5])), color, tl)
	cv2.line(img, (int(x[4]), int(x[5])), (int(x[6]), int(x[7])), color, tl)
	cv2.line(img, (int(x[6]), int(x[7])), (int(x[0]), int(x[1])), color, tl)


	if label:
		tf = max(tl - 1, 1)  # font thickness
		t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
		c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
		cv2.rectangle(img, c1, c2, color, -1)  # filled
		cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def bbox_iou(box1, box2):
	"""
	Returns the IoU of two bounding boxes
	"""
	cuda = torch.cuda.is_available()
	device = torch.device('cuda:0' if cuda else 'cpu')

	nBox = box1.size()[0]

	iou = torch.zeros(nBox)
	for i in range(0, nBox):
		polygon1 = Polygon(box1[i,:].view(4,2)).convex_hull
		polygon2 = Polygon(box2[i,:].view(4,2)).convex_hull
		if polygon1.intersects(polygon2):
			try:
				inter_area = polygon1.intersection(polygon2).area
				union_area = polygon1.union(polygon2).area
				iou[i] =  inter_area / union_area
			except shapely.geos.TopologicalError:
				print('shapely.geos.TopologicalError occured, iou set to 0')
				iou[i] = 0

	return iou.to(device)

def reorganize_targets(t, nTb):
	t = t.float()
	tc = t[:, 0]
	t = t[:, 1:9].view(nTb, 4, 2)

	x = t[..., 0]
	y = t[..., 1]
	y_sorted, y_indices = torch.sort(y)
	
	x_sorted = torch.zeros(nTb, 4)
	for i in range(0, nTb):
		x_sorted[i] = x[i, y_indices[i]]


	x_sorted[:, :2], x_top_indices = torch.sort(x_sorted[:, :2])
	x_sorted[:, 2:4], x_bottom_indices = torch.sort(x_sorted[:, 2:4], descending=True)
	for i in range(0, nTb):
		y_sorted[i, :2] = y_sorted[i, :2][x_top_indices[i]]
		y_sorted[i, 2:4] = y_sorted[i, 2:4][x_bottom_indices[i]]

	t = torch.cat((x_sorted[:,0].unsqueeze(1), y_sorted[:,0].unsqueeze(1), x_sorted[:,1].unsqueeze(1), y_sorted[:,1].unsqueeze(1), 
					 x_sorted[:,2].unsqueeze(1), y_sorted[:,2].unsqueeze(1), x_sorted[:,3].unsqueeze(1), y_sorted[:,3].unsqueeze(1)), 1)

	return torch.cat((tc.unsqueeze(1), t), 1)


def build_targets(pred_boxes, pred_conf, pred_cls, target, anchor_wh, nA, nC, nG, requestPrecision):
	"""
	returns nT, nCorrect, tx, ty, tw, th, tconf, tcls
	"""
	nB = len(target)  # number of images in batch
	nT = [len(x) for x in target]  # torch.argmin(target[:, :, 4], 1)  # targets per batch
	t1_x = torch.zeros(nB, nA, nG, nG)  # batch size (4), number of anchors (3), number of grid points (13)
	t1_y = torch.zeros(nB, nA, nG, nG)
	t2_x = torch.zeros(nB, nA, nG, nG)
	t2_y = torch.zeros(nB, nA, nG, nG)
	t3_x = torch.zeros(nB, nA, nG, nG)
	t3_y = torch.zeros(nB, nA, nG, nG)
	t4_x = torch.zeros(nB, nA, nG, nG)
	t4_y = torch.zeros(nB, nA, nG, nG)	
	tconf = torch.BoolTensor(nB, nA, nG, nG).fill_(0)
	tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)  # nC = number of classes
	TP = torch.ByteTensor(nB, max(nT)).fill_(0)
	FP = torch.ByteTensor(nB, max(nT)).fill_(0)
	FN = torch.ByteTensor(nB, max(nT)).fill_(0)
	TC = torch.ShortTensor(nB, max(nT)).fill_(-1)  # target category
	cuda = torch.cuda.is_available()
	device = torch.device('cuda:0' if cuda else 'cpu')

	for b in range(nB):
		nTb = nT[b]  # number of targets per image
		if nTb == 0:
			continue
		t = target[b]
		FN[b, :nTb] = 1
		t = reorganize_targets(t, nTb)
		# Convert to position relative to box
		TC[b, :nTb] = t[:, 0].long()
		gp1_x, gp1_y = t[:, 1] * nG, t[:, 2] * nG
		gp2_x, gp2_y = t[:, 3] * nG, t[:, 4] * nG
		gp3_x, gp3_y = t[:, 5] * nG, t[:, 6] * nG
		gp4_x, gp4_y = t[:, 7] * nG, t[:, 8] * nG
		# Get grid box indices and prevent overflows (i.e. 13.01 on 13 anchors)
		gp1_i, gp1_j = torch.clamp(torch.round(gp1_x).long(), min=0, max=nG - 1), torch.clamp(torch.round(gp1_y).long(), min=0, max=nG - 1)
		gp2_i, gp2_j = torch.clamp(torch.round(gp2_x).long(), min=0, max=nG - 1), torch.clamp(torch.round(gp2_y).long(), min=0, max=nG - 1)
		gp3_i, gp3_j = torch.clamp(torch.round(gp3_x).long(), min=0, max=nG - 1), torch.clamp(torch.round(gp3_y).long(), min=0, max=nG - 1)
		gp4_i, gp4_j = torch.clamp(torch.round(gp4_x).long(), min=0, max=nG - 1), torch.clamp(torch.round(gp4_y).long(), min=0, max=nG - 1)

		box1 = t[:, 1:9] * nG

		# Get each target center
		gp_x = torch.cat((gp1_x.unsqueeze(1), gp2_x.unsqueeze(1), gp3_x.unsqueeze(1), gp4_x.unsqueeze(1)), 1).view(-1, 4)
		gp_y = torch.cat((gp1_y.unsqueeze(1), gp2_y.unsqueeze(1), gp3_y.unsqueeze(1), gp4_y.unsqueeze(1)), 1).view(-1, 4)

		gp_x_min = torch.min(gp_x, 1)[0]
		gp_x_max = torch.max(gp_x, 1)[0]
		gp_y_min = torch.min(gp_y, 1)[0]
		gp_y_max = torch.max(gp_y, 1)[0]

		gp_x_center = torch.mean(torch.cat((gp_x_min.unsqueeze(1), gp_x_max.unsqueeze(1)), 1), 1)
		gp_y_center = torch.mean(torch.cat((gp_y_min.unsqueeze(1), gp_y_max.unsqueeze(1)), 1), 1)
		# Set target center in a certain cell
		gp_x_center = torch.round(gp_x_center)
		gp_y_center = torch.round(gp_y_center)

		x_min = torch.clamp((gp_x_center.unsqueeze(1).repeat(1, nA).view(-1, nA, 1) - anchor_wh[:, 0].view(-1, nA, 1) / 2), min=0, max=nG-1)
		x_max = torch.clamp((gp_x_center.unsqueeze(1).repeat(1, nA).view(-1, nA, 1) + anchor_wh[:, 0].view(-1, nA, 1) / 2), min=0, max=nG-1)
		y_min = torch.clamp((gp_y_center.unsqueeze(1).repeat(1, nA).view(-1, nA, 1) - anchor_wh[:, 1].view(-1, nA, 1) / 2), min=0, max=nG-1)
		y_max = torch.clamp((gp_y_center.unsqueeze(1).repeat(1, nA).view(-1, nA, 1) + anchor_wh[:, 1].view(-1, nA, 1) / 2), min=0, max=nG-1)

		top_left = torch.cat((x_min.view(-1, 1), y_min.view(-1, 1)), 1)
		top_right = torch.cat((x_max.view(-1, 1), y_min.view(-1, 1)), 1)
		bottom_right = torch.cat((x_max.view(-1, 1), y_max.view(-1, 1)), 1)
		bottom_left = torch.cat((x_min.view(-1, 1), y_max.view(-1, 1)), 1)
		# Get bounding boxes
		box2 = torch.cat((top_left, top_right, bottom_right, bottom_left), 1).view(-1, nA, 8)

		iou_anch = torch.zeros(nA, nTb, 1)
		for i in range(0, nTb):
			for j in range(0, nA):
				polygon1 = Polygon(box1[i,:].view(4,2)).convex_hull
				polygon2 = Polygon(box2[i,j,:].view(4,2)).convex_hull
				if polygon1.intersects(polygon2):
					try:
						inter_area = polygon1.intersection(polygon2).area
						union_area = polygon1.union(polygon2).area
						iou_anch[j, i] =  inter_area / union_area
					except shapely.geos.TopologicalError:
						print('shapely.geos.TopologicalError occured, iou set to 0')

		iou_anch = iou_anch.squeeze(2)
		# Select best iou_pred and anchor
		iou_anch_best, a = iou_anch.max(0)  # best anchor [0-2] for each target
		
		# Select best unique target-anchor combinations
		if nTb > 1:
			iou_order = np.argsort(-iou_anch_best)  # best to worst

			# Unique anchor selection (slower but retains original order)
			u = torch.cat((gp1_i, gp1_j, gp2_i, gp2_j, gp3_i, gp3_j, gp4_i, gp4_j, a), 0).view(-1, nTb).cpu().numpy()
			_, first_unique = np.unique(u[:, iou_order], axis=1, return_index=True)  # first unique indices; each cell response to on target
			i = iou_order[first_unique]

			# best anchor must share significant commonality (iou) with target
			i = i[iou_anch_best[i] > 0.1]
			if len(i) == 0:
				continue

			a, gp1_i, gp1_j, gp2_i, gp2_j, gp3_i, gp3_j, gp4_i, gp4_j, t = \
			a[i], gp1_i[i], gp1_j[i], gp2_i[i], gp2_j[i], gp3_i[i], gp3_j[i], gp4_i[i], gp4_j[i], t[i]
			if len(t.shape) == 1:
				t = t.view(1, 5)
		else:
			if iou_anch_best < 0.1:
				continue
			i = 0

		tc = t[:, 0].long()
		gp1_x, gp1_y = t[:, 1] * nG, t[:, 2] * nG
		gp2_x, gp2_y = t[:, 3] * nG, t[:, 4] * nG
		gp3_x, gp3_y = t[:, 5] * nG, t[:, 6] * nG
		gp4_x, gp4_y = t[:, 7] * nG, t[:, 8] * nG

		# Get target center
		gp_x = torch.cat((gp1_x.unsqueeze(1), gp2_x.unsqueeze(1), gp3_x.unsqueeze(1), gp4_x.unsqueeze(1)), 1).view(-1, 4)
		gp_y = torch.cat((gp1_y.unsqueeze(1), gp2_y.unsqueeze(1), gp3_y.unsqueeze(1), gp4_y.unsqueeze(1)), 1).view(-1, 4)
		gp_x_min = torch.min(gp_x, 1)[0]
		gp_x_max = torch.max(gp_x, 1)[0]
		gp_y_min = torch.min(gp_y, 1)[0]
		gp_y_max = torch.max(gp_y, 1)[0]
		gp_x_center = torch.mean(torch.cat((gp_x_min.unsqueeze(1), gp_x_max.unsqueeze(1)), 1), 1)
		gp_y_center = torch.mean(torch.cat((gp_y_min.unsqueeze(1), gp_y_max.unsqueeze(1)), 1), 1)
		# Set target center in a certain cell
		gp_x_center = torch.round(gp_x_center).long()
		gp_y_center = torch.round(gp_y_center).long()

		# Coordinates
		t1_x[b, a, gp_y_center, gp_x_center] = gp1_x.cpu() - gp_x_center.float().cpu()
		t1_y[b, a, gp_y_center, gp_x_center] = gp1_y.cpu() - gp_y_center.float().cpu()
		t2_x[b, a, gp_y_center, gp_x_center] = gp2_x.cpu() - gp_x_center.float().cpu()
		t2_y[b, a, gp_y_center, gp_x_center] = gp2_y.cpu() - gp_y_center.float().cpu()
		t3_x[b, a, gp_y_center, gp_x_center] = gp3_x.cpu() - gp_x_center.float().cpu()
		t3_y[b, a, gp_y_center, gp_x_center] = gp3_y.cpu() - gp_y_center.float().cpu()
		t4_x[b, a, gp_y_center, gp_x_center] = gp4_x.cpu() - gp_x_center.float().cpu()
		t4_y[b, a, gp_y_center, gp_x_center] = gp4_y.cpu() - gp_y_center.float().cpu()

		# One-hot encoding of label
		tcls[b, a, gp_y_center, gp_x_center, tc] = 1
		tconf[b, a, gp_y_center, gp_x_center] = 1

		if requestPrecision:
			# predicted classes and confidence
			tb = torch.cat((gp1_x, gp1_y, gp2_x, gp2_y, gp3_x, gp3_y, gp4_x, gp4_y)).view(8, -1).t()
			pcls = torch.argmax(pred_cls[b, a, gp_y_center, gp_x_center], 1).cpu()
			pconf = torch.sigmoid(pred_conf[b, a, gp_y_center, gp_x_center].cpu()).cpu()
			iou_pred = bbox_iou(tb, pred_boxes[b, a, gp_y_center, gp_x_center].cpu()).cpu()
			TP[b, i] = ((pconf > 0.5) & (iou_pred > 0.5) & (pcls == tc.cpu())).byte()
			FP[b, i] = ((pconf > 0.5) & (TP[b, i] == 0)).byte()  # coordinates or class are wrong
			FN[b, i] = (pconf <= 0.5).byte()  # confidence score is too low (set to zero)
	return t1_x, t1_y, t2_x, t2_y, t3_x, t3_y, t4_x, t4_y, tconf, tcls, TP, FP, FN, TC


def bbox_iou_nms(box1, box2):
	cuda = torch.cuda.is_available()
	device = torch.device('cuda:0' if cuda else 'cpu')

	nBox = box2.size()[0]

	iou = torch.zeros(nBox)
	polygon1 = Polygon(box1.view(4,2)).convex_hull
	for i in range(0, nBox):
		polygon2 = Polygon(box2[i,:].view(4,2)).convex_hull
		if polygon1.intersects(polygon2):
			try:
				inter_area = polygon1.intersection(polygon2).area
				union_area = polygon1.union(polygon2).area
				iou[i] =  inter_area / union_area
			except shapely.geos.TopologicalError:
				print('shapely.geos.TopologicalError occured, iou set to 0')
				iou[i] = 0

	return iou.to(device)


def non_max_suppression(prediction, cls_thres=0.5, nms_thres=0.4):
	prediction = prediction.cpu()

	output = [None for _ in range(len(prediction))]

	for image_i, pred in enumerate(prediction):
		class_prob, class_pred = torch.max(F.softmax(pred[:, 9:], 1), 1)

		v = (class_prob > cls_thres).numpy()
		v = v.nonzero()

		pred = pred[v]
		class_prob = class_prob[v]
		class_pred = class_pred[v]

		# If none are remaining => process next image
		nP = pred.shape[0]
		if not nP:
			continue

		detections = torch.cat((pred[:, :9], class_prob.float().unsqueeze(1), class_pred.float().unsqueeze(1)), 1)
		# Iterate through all predicted classes
		unique_labels = detections[:, -1].cpu().unique()
		if prediction.is_cuda:
			unique_labels = unique_labels.cuda()

		for c in unique_labels:
			detections_class = detections[detections[:, -1] == c]
			# Sort through confidence in one class
			_, conf_sort_index = torch.sort(detections_class[:, 8], descending=True)
			detections_class = detections_class[conf_sort_index]

			max_detections = []

			while detections_class.shape[0]:
				# Get detection with highest confidence and save as max detection
				max_detections.append(detections_class[0].unsqueeze(0))
				# Stop if we're at the last detection
				if len(detections_class) == 1:
					break
				# Get the IOUs for all boxes with lower confidence
				ious = bbox_iou_nms(max_detections[-1].squeeze(0)[0:8], detections_class[1:][:, 0:8])

				# Remove detections with IoU >= NMS threshold
				detections_class = detections_class[1:][ious < nms_thres]

			if len(max_detections) > 0:
				max_detections = torch.cat(max_detections).data
				# Add max detections to outputs
				output[image_i] = max_detections if output[image_i] is None else torch.cat(
					(output[image_i], max_detections))

	return output