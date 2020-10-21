from collections import defaultdict

import torch.nn as nn

from utils.parse_config import *
from utils.utils import *

def create_modules(module_defs):
	"""
	Constructs module list of layer blocks from module configuration in module_defs
	"""
	hyperparams = module_defs.pop(0)
	output_filters = [int(hyperparams['channels'])]
	module_list = nn.ModuleList()
	for i, module_def in enumerate(module_defs):
		modules = nn.Sequential()

		if module_def['type'] == 'convolutional':
			bn = int(module_def['batch_normalize'])
			filters = int(module_def['filters'])
			kernel_size = int(module_def['size'])
			pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
			modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
														out_channels=filters,
														kernel_size=kernel_size,
														stride=int(module_def['stride']),
														padding=pad,
														bias=not bn))
			if bn:
				modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
			if module_def['activation'] == 'leaky':
				modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))

		elif module_def['type'] == 'upsample':
			upsample = nn.Upsample(scale_factor=int(module_def['stride']), mode='nearest')
			modules.add_module('upsample_%d' % i, upsample)

		elif module_def['type'] == 'route':
			layers = [int(x) for x in module_def['layers'].split(',')]
			filters = sum([output_filters[layer_i] for layer_i in layers])
			modules.add_module('route_%d' % i, EmptyLayer())

		elif module_def['type'] == 'shortcut':
			filters = output_filters[int(module_def['from'])]
			modules.add_module('shortcut_%d' % i, EmptyLayer())

		elif module_def['type'] == 'yolo':
			anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
			# Extract anchors
			anchors = [float(x) for x in module_def['anchors'].split(',')]
			anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
			anchors = [anchors[i] for i in anchor_idxs]
			num_classes = int(module_def['classes'])
			img_height = int(hyperparams['height'])
			# Define detection layer
			yolo_layer = YOLOLayer(anchors, num_classes, img_height, anchor_idxs)
			modules.add_module('yolo_%d' % i, yolo_layer)

		# Register module list and number of output filters
		module_list.append(modules)
		output_filters.append(filters)

	return hyperparams, module_list
class EmptyLayer(nn.Module):
	"""Placeholder for 'route' and 'shortcut' layers"""
	def __init__(self):
		super(EmptyLayer, self).__init__()

class YOLOLayer(nn.Module):

	def __init__(self, anchors, nC, img_dim, anchor_idxs):
		super(YOLOLayer, self).__init__()

		anchors = [(a_w, a_h) for a_w, a_h in anchors]  # (pixels)
		nA = len(anchors)

		self.anchors = anchors
		self.nA = nA  # number of anchors (3)
		self.nC = nC  # number of classes (1)
		self.bbox_attrs = 9 + nC
		self.img_dim = img_dim  # from hyperparams in cfg file, NOT from parser

		if anchor_idxs[0] == (nA * 2):  # 6
			stride = 32
		elif anchor_idxs[0] == nA:  # 3
			stride = 16
		else:
			stride = 8

		# Build anchor grids
		nG = int(self.img_dim / stride)  # number grid points
		self.grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).float()
		self.grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).float()
		self.scaled_anchors = torch.FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in anchors])
		self.anchor_w = self.scaled_anchors[:, 0:1].view((1, nA, 1, 1))
		self.anchor_h = self.scaled_anchors[:, 1:2].view((1, nA, 1, 1))
		#self.weights = class_weights()

	def forward(self, p, targets=None, requestPrecision=False):
		FT = torch.cuda.FloatTensor if p.is_cuda else torch.FloatTensor

		bs = p.shape[0]  # batch size
		nG = p.shape[2]  # number of grid points
		stride = self.img_dim / nG

		if p.is_cuda and not self.grid_x.is_cuda:
			self.grid_x, self.grid_y = self.grid_x.cuda(), self.grid_y.cuda()
			self.anchor_w, self.anchor_h = self.anchor_w.cuda(), self.anchor_h.cuda()
			#self.weights = self.weights.cuda()

		# p.view(1, 30, 13, 13) -- > (1, 3, 13, 13, 10)  # (bs, anchors, grid, grid, classes + xywh)
		p = p.view(bs, self.nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()  # prediction

		# Get outputs
		P1_x = p[..., 0]  # Point1 x
		P1_y = p[..., 1]  # Point1 y
		P2_x = p[..., 2]  # Point2 x
		P2_y = p[..., 3]  # Point2 y
		P3_x = p[..., 4]  # Point3 x
		P3_y = p[..., 5]  # Point3 y
		P4_x = p[..., 6]  # Point4 x
		P4_y = p[..., 7]  # Point4 y

		pred_boxes = FT(bs, self.nA, nG, nG, 8)
		pred_conf = p[..., 8]  # Conf
		pred_cls = p[..., 9:]  # Class

		# Training
		if targets is not None:
			MSELoss = nn.MSELoss()
			BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
			CrossEntropyLoss = nn.CrossEntropyLoss()
			SmoothL1Loss = nn.SmoothL1Loss()

			if requestPrecision:
				gx = self.grid_x[:, :, :nG, :nG]
				gy = self.grid_y[:, :, :nG, :nG]
				pred_boxes[..., 0] = P1_x.data + gx
				pred_boxes[..., 1] = P1_y.data + gy
				pred_boxes[..., 2] = P2_x.data + gx
				pred_boxes[..., 3] = P2_y.data + gy
				pred_boxes[..., 4] = P3_x.data + gx
				pred_boxes[..., 5] = P3_y.data + gy
				pred_boxes[..., 6] = P4_x.data + gx
				pred_boxes[..., 7] = P4_y.data + gy

			t1_x, t1_y, t2_x, t2_y, t3_x, t3_y, t4_x, t4_y, mask, tcls, TP, FP, FN, TC = \
				build_targets(pred_boxes, pred_conf, pred_cls, targets, self.scaled_anchors, self.nA, self.nC, nG,
							  requestPrecision)
			
			tcls = tcls[mask]
			if P1_x.is_cuda:
				t1_x, t1_y, t2_x, t2_y, t3_x, t3_y, t4_x, t4_y, mask, tcls = \
				t1_x.cuda(), t1_y.cuda(), t2_x.cuda(), t2_y.cuda(), t3_x.cuda(), t3_y.cuda(), t4_x.cuda(), t4_y.cuda(), mask.cuda(), tcls.cuda()

			# Compute losses
			nT = sum([len(x) for x in targets])	 	# Number of targets
			nM = mask.sum().float()					# Number of anchors (assigned to targets)
			nB = len(targets)						# Batch size
			k = nM / nB
			if nM > 0:
				lx1 = (k) * SmoothL1Loss(P1_x[mask], t1_x[mask]) / 8
				ly1 = (k) * SmoothL1Loss(P1_y[mask], t1_y[mask]) / 8
				lx2 = (k) * SmoothL1Loss(P2_x[mask], t2_x[mask]) / 8
				ly2 = (k) * SmoothL1Loss(P2_y[mask], t2_y[mask]) / 8
				lx3 = (k) * SmoothL1Loss(P3_x[mask], t3_x[mask]) / 8
				ly3 = (k) * SmoothL1Loss(P3_y[mask], t3_y[mask]) / 8
				lx4 = (k) * SmoothL1Loss(P4_x[mask], t4_x[mask]) / 8
				ly4 = (k) * SmoothL1Loss(P4_y[mask], t4_y[mask]) / 8

				lconf = (k * 10) * BCEWithLogitsLoss(pred_conf, mask.float())
				lcls = (k / self.nC) * CrossEntropyLoss(pred_cls[mask], torch.argmax(tcls, 1))
			else:
				lx1, ly1, lx2, ly2, lx3, ly3, lx4, ly4, lcls, lconf = \
				FT([0]).requires_grad_(True), FT([0]).requires_grad_(True), FT([0]).requires_grad_(True), FT([0]).requires_grad_(True), FT([0]).requires_grad_(True), \
					FT([0]).requires_grad_(True), FT([0]).requires_grad_(True), FT([0]).requires_grad_(True), FT([0]).requires_grad_(True), FT([0]).requires_grad_(True), 

			# Sum loss components
			loss = lx1 + ly1 + lx2 + ly2 + lx3 + ly3 + lx4 + ly4 + lconf + lcls

			# Sum False Positives from unassigned anchors
			i = torch.sigmoid(pred_conf[~mask]) > 0.5
			if i.sum() > 0:
				FP_classes = torch.argmax(pred_cls[~mask][i], 1)
				FPe = torch.bincount(FP_classes, minlength=self.nC).float().cpu()
			else:
				FPe = torch.zeros(self.nC)
			return loss, loss.item(), lconf.item(), lcls.item(), nT, TP, FP, FPe, FN, TC

		else:
			pred_boxes[..., 0] = P1_x + self.grid_x
			pred_boxes[..., 1] = P1_y + self.grid_y
			pred_boxes[..., 2] = P2_x + self.grid_x
			pred_boxes[..., 3] = P2_y + self.grid_y
			pred_boxes[..., 4] = P3_x + self.grid_x
			pred_boxes[..., 5] = P3_y + self.grid_y
			pred_boxes[..., 6] = P4_x + self.grid_x
			pred_boxes[..., 7] = P4_y + self.grid_y

			output = torch.cat((pred_boxes.view(bs, -1, 8) * stride,
								torch.sigmoid(pred_conf.view(bs, -1, 1)), pred_cls.view(bs, -1, self.nC)), -1)

			return output

class Darknet(nn.Module):
	"""YOLOv3 object detection model"""

	def __init__(self, cfg_path, img_size=416):
		super(Darknet, self).__init__()
		self.module_defs = parse_model_config(cfg_path)
		self.module_defs[0]['height'] = img_size
		self.hyperparams, self.module_list = create_modules(self.module_defs)
		self.img_size = img_size
		self.loss_names = ['loss', 'conf', 'cls', 'nT', 'TP', 'FP', 'FPe', 'FN', 'TC']

	def forward(self, x, targets=None, requestPrecision=False):
		is_training = targets is not None
		output = []
		self.losses = defaultdict(float)
		layer_outputs = []

		for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
			if module_def['type'] in ['convolutional', 'upsample']:
				x = module(x)
			elif module_def['type'] == 'route':
				layer_i = [int(x) for x in module_def['layers'].split(',')]
				x = torch.cat([layer_outputs[i] for i in layer_i], 1)
			elif module_def['type'] == 'shortcut':
				layer_i = int(module_def['from'])
				x = layer_outputs[-1] + layer_outputs[layer_i]
			elif module_def['type'] == 'yolo':
				# Train phase: get loss
				if is_training:
					# if module[0](x, targets, requestPrecision)[0].item()==0:
					x, *losses = module[0](x, targets, requestPrecision)
					for name, loss in zip(self.loss_names, losses):
						self.losses[name] += loss
				# Test phase: Get detections
				else:
					x = module(x)
				output.append(x)
			layer_outputs.append(x)

		if is_training:
			self.losses['nT'] /= 3
			self.losses['TC'] /= 3  # target category
			metrics = torch.zeros(3, len(self.losses['FPe']))  # TP, FP, FN

			ui = np.unique(self.losses['TC'])[1:]
			for i in ui:
				j = self.losses['TC'] == float(i)
				metrics[0, i] = (self.losses['TP'][j] > 0).sum().float()  # TP
				metrics[1, i] = (self.losses['FP'][j] > 0).sum().float()  # FP
				metrics[2, i] = (self.losses['FN'][j] == 3).sum().float()  # FN
			metrics[1] += self.losses['FPe']

			self.losses['TP'] = metrics[0].sum()
			self.losses['FP'] = metrics[1].sum()
			self.losses['FN'] = metrics[2].sum()
			self.losses['TC'] = 0
			self.losses['metrics'] = metrics
		return sum(output) if is_training else torch.cat(output, 1)

def load_weights(self, weights_path, cutoff=-1):
	# Parses and loads the weights stored in 'weights_path'
	# @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)

	if weights_path.endswith('darknet53.conv.74'):
		cutoff = 75

	# Open the weights file
	fp = open(weights_path, 'rb')
	header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

	# Needed to write header when saving weights
	self.header_info = header

	self.seen = header[3]
	weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
	fp.close()

	ptr = 0
	for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
		if module_def['type'] == 'convolutional':
			conv_layer = module[0]
			if module_def['batch_normalize']:
				# Load BN bias, weights, running mean and running variance
				bn_layer = module[1]
				num_b = bn_layer.bias.numel()  # Number of biases
				# Bias
				bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
				bn_layer.bias.data.copy_(bn_b)
				ptr += num_b
				# Weight
				bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
				bn_layer.weight.data.copy_(bn_w)
				ptr += num_b
				# Running Mean
				bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
				bn_layer.running_mean.data.copy_(bn_rm)
				ptr += num_b
				# Running Var
				bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
				bn_layer.running_var.data.copy_(bn_rv)
				ptr += num_b
			else:
				# Load conv. bias
				num_b = conv_layer.bias.numel()
				conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
				conv_layer.bias.data.copy_(conv_b)
				ptr += num_b
			# Load conv. weights
			num_w = conv_layer.weight.numel()
			conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
			conv_layer.weight.data.copy_(conv_w)
			ptr += num_w