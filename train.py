import argparse
import time

from models import *
from utils.datasets import *
from utils.utils import *
import torch.optim.lr_scheduler as lr_scheduler

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=1, help='size of each image batch')
parser.add_argument('--data_config_path', type=str, default='cfg/UCAS_AOD.data', help='data config file path')
parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
parser.add_argument('--img_size', type=int, default=608, help='size of each image dimension')
parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if cuda:
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True


def main(opt):
    os.makedirs('weights', exist_ok=True)

    # Configure run
    data_config = parse_data_config(opt.data_config_path)
    num_classes = int(data_config['classes'])
    train_path = data_config['train']

    # Initialize model
    model = Darknet(opt.cfg, opt.img_size)

    # Get dataloader
    dataloader = load_images_and_labels(train_path, batch_size=opt.batch_size, img_size=opt.img_size, augment=True)

    # Reload saved optimizer state
    start_epoch = 0
    best_loss = float('inf')
    if opt.resume:
        checkpoint = torch.load('weights/latest.pt', map_location='cpu')

        model.load_state_dict(checkpoint['model'])
        # if torch.cuda.device_count() > 1:
        #	print('Using ', torch.cuda.device_count(), ' GPUs')
        #	model = nn.DataParallel(model)
        model.to(device).train()

        # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
        # 							lr=1e-3, momentum=.9, weight_decay=5e-4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        start_epoch = checkpoint['epoch'] + 1
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_loss = checkpoint['best_loss']

        del checkpoint  # current, saved
    else:
        # Initialize model with darknet53 weights (optional)
        if not os.path.isfile('weights/darknet53.conv.74'):
            os.system('wget https://pjreddie.com/media/files/darknet53.conv.74 -P weights')
        model.load_weights('weights/darknet53.conv.74')

        model.to(device).train()

        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=.9, weight_decay=5e-4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # model_info(model)
    t0, t1 = time.time(), time.time()
    print('%11s' * 6 % (
        'Epoch', 'Batch', 'conf', 'cls', 'loss', 'time'))
    for epoch in range(opt.epochs):
        epoch += start_epoch

        # Update scheduler (manual)
        if epoch < 30:
            lr = 1e-3
        elif epoch < 60:
            lr = 5e-4
        elif epoch < 90:
            lr = 1e-4
        else:
            lr = 5e-5

        for g in optimizer.param_groups:
            g['lr'] = lr

        ui = -1
        rloss = defaultdict(float)  # running loss
        metrics = torch.zeros(3, num_classes)
        optimizer.zero_grad()

        for i, (imgs, targets) in enumerate(dataloader):

            if sum([len(x) for x in targets]) < 1:  # if no targets continue
                continue
            # SGD burn-in
            if (epoch == 0) & (i <= 1000):
                lr = 1e-4 * (i / 1000) ** 4
                for g in optimizer.param_groups:
                    g['lr'] = lr

            # Compute loss, compute gradient, update parameters
            loss = model(imgs.to(device), targets, requestPrecision=True)
            loss.backward()

            # accumulated_batches = 1  # accumulate gradient for 4 batches before stepping optimizer
            # if ((i+1) % accumulated_batches == 0) or (i == len(dataloader) - 1):
            optimizer.step()
            optimizer.zero_grad()

            # Compute running epoch-means of tracked metrics
            ui += 1
            metrics += model.losses['metrics']
            TP, FP, FN = metrics
            for key, val in model.losses.items():
                rloss[key] = (rloss[key] * ui + val) / (ui + 1)

            # Precision
            precision = TP / (TP + FP)
            k = (TP + FP) > 0
            if k.sum() > 0:
                mean_precision = precision[k].mean()

            # Recall
            recall = TP / (TP + FN)
            k = (TP + FN) > 0
            if k.sum() > 0:
                mean_recall = recall[k].mean()

            s = ('%11s%11s' + '%11.3g' * 4) % (
                '%g/%g' % (epoch, opt.epochs - 1), '%g/%g' % (i, len(dataloader) - 1), rloss['conf'], rloss['cls'],
                rloss['loss'], time.time() - t1)
            t1 = time.time()
            print(s)

        # Write epoch results
        with open('results.txt', 'a') as file:
            file.write(s + '\n')

        # Update best loss
        loss_per_target = rloss['loss']  # / rloss['nT']
        if loss_per_target < best_loss:
            best_loss = loss_per_target

        # Save latest checkpoint
        checkpoint = {'epoch': epoch,
                      'best_loss': best_loss,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, 'weights/latest.pt')

        # Save best checkpoint
        if best_loss == loss_per_target:
            os.system('cp weights/latest.pt weights/best.pt')

        # Save backup weights every 10 epochs
        if (epoch > 0) & (epoch % 10 == 0):
            os.system('cp weights/latest.pt weights/backup' + str(epoch) + '.pt')

    # Save final model
    dt = time.time() - t0
    print('Finished %g epochs in %.2fs (%.2fs/epoch)' % (epoch, dt, dt / (epoch + 1)))


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main(opt)
    torch.cuda.empty_cache()