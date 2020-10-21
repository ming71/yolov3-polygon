import argparse
import time

from models import *
from utils.datasets import *
from utils.utils import *
from utils.map import eval_mAP

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

parser = argparse.ArgumentParser()

parser.add_argument('-image_folder', type=str, default='/py/datasets/UCAS_AOD/Test', help='path to images')
# parser.add_argument('-image_folder', type=str, default='/py/YOLOv3-quadrangle/data/samples', help='path to images')
parser.add_argument('-output_folder', type=str, default='datasets/UCAS_AOD/detection-results', help='path to outputs')
parser.add_argument('-plot_flag', type=bool, default=False)
parser.add_argument('-txt_out', type=bool, default=True)

parser.add_argument('-cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
parser.add_argument('-weights_path', type=str, default='weights/best.pt', help='weight file path')
parser.add_argument('-class_path', type=str, default='data/ucas_aod.names', help='path to class label file')
parser.add_argument('-conf_thres', type=float, default=0.1, help='object confidence threshold')
parser.add_argument('-nms_thres', type=float, default=0.2, help='iou threshold for non-maximum suppression')
parser.add_argument('-batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('-img_size', type=int, default=608, help='size of each image dimension')
opt = parser.parse_args()
print(opt)


def detect(opt):
	os.system('rm -rf ' + opt.output_folder)
	os.makedirs(opt.output_folder, exist_ok=True)

	# Load model
	model = Darknet(opt.cfg, opt.img_size)

	weights_path = opt.weights_path
	if weights_path.endswith('.weights'):  # saved in darknet format
		load_weights(model, weights_path)
	else:  # endswith('.pt'), saved in pytorch format
		checkpoint = torch.load(weights_path, map_location='cpu')
		model.load_state_dict(checkpoint['model'])
		del checkpoint

	model.to(device).eval()

	# Set Dataloader
	classes = load_classes(opt.class_path)  # Extracts class labels from file
	dataloader = load_images(opt.image_folder, batch_size=opt.batch_size, img_size=opt.img_size)

	imgs = []  # Stores image paths
	img_detections = []  # Stores detections for each image index
	prev_time = time.time()
	for batch_i, (img_paths, img) in enumerate(dataloader):
		print(batch_i, img.shape, end=' ')

		# Get detections
		with torch.no_grad():
			chip = torch.from_numpy(img).unsqueeze(0).to(device)
			pred = model(chip)
			pred = pred[pred[:, :, 8] > opt.conf_thres]

			if len(pred) > 0:
				detections = non_max_suppression(pred.unsqueeze(0), 0.1, opt.nms_thres)
				img_detections.extend(detections)
				imgs.extend(img_paths)

		print('Batch %d... (Done %.3f s)' % (batch_i, time.time() - prev_time))
		prev_time = time.time()

	# Bounding-box colors
	color_list = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]

	if len(img_detections) == 0:
		return

	# Iterate through images and save plot of detections
	for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
		print("image %g: '%s'" % (img_i, path))

		img = cv2.imread(path)

		# The amount of padding that was added
		pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
		pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
		# Image height and width after padding is removed
		unpad_h = opt.img_size - pad_y
		unpad_w = opt.img_size - pad_x

		# Draw bounding boxes and labels of detections
		if detections is not None:
			unique_classes = detections[:, -1].cpu().unique()
			bbox_colors = random.sample(color_list, len(unique_classes))

			# write results to .txt file
			results_img_path = os.path.join(opt.output_folder, path.split('/')[-1])
			
			results_txt_path = results_img_path.replace('png', 'txt')
			if os.path.isfile(results_txt_path):
				os.remove(results_txt_path)

			for i in unique_classes:
				n = (detections[:, -1].cpu() == i).sum()

			for P1_x, P1_y, P2_x, P2_y, P3_x, P3_y, P4_x, P4_y, conf, cls_conf, cls_pred in detections:
				P1_y = max((((P1_y - pad_y // 2) / unpad_h) * img.shape[0]).round().item(), 0)
				P1_x = max((((P1_x - pad_x // 2) / unpad_w) * img.shape[1]).round().item(), 0)
				P2_y = max((((P2_y - pad_y // 2) / unpad_h) * img.shape[0]).round().item(), 0)
				P2_x = max((((P2_x - pad_x // 2) / unpad_w) * img.shape[1]).round().item(), 0)
				P3_y = max((((P3_y - pad_y // 2) / unpad_h) * img.shape[0]).round().item(), 0)
				P3_x = max((((P3_x - pad_x // 2) / unpad_w) * img.shape[1]).round().item(), 0)
				P4_y = max((((P4_y - pad_y // 2) / unpad_h) * img.shape[0]).round().item(), 0)
				P4_x = max((((P4_x - pad_x // 2) / unpad_w) * img.shape[1]).round().item(), 0)
				# write to file
				if opt.txt_out:
					with open(results_txt_path, 'a') as f:
						f.write(('%s %.2f %g %g %g %g %g %g %g %g  \n') % \
							(classes[int(cls_pred)], cls_conf * conf, P1_x, P1_y, P2_x, P2_y, P3_x, P3_y, P4_x, P4_y ))


if __name__ == '__main__':
	torch.cuda.empty_cache()
	detect(opt)
	mAP = eval_mAP('datasets/UCAS_AOD')
	print(mAP)
