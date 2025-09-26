import torch
from torchvision import transforms
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
sys.path.append('..')
from data_utils.augmentations import Augmentations
from model_components.freespace_seg_network import FreespaceSegNetwork


class FreespaceSegTrainer():
    def __init__(self, checkpoint_path='', learning_rate=0.0001):
        self.image = 0
        self.gt = 0
        self.image_tensor = 0
        self.gt_tensor = 0

        # Checking devices (GPU vs CPU)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device} for inference')

        self.model = FreespaceSegNetwork()
        if len(checkpoint_path) > 0:
            # Loading model with full pre-trained weights
            self.model.load_state_dict(
                torch.load(checkpoint_path, weights_only=True, map_location=self.device))
            print('Loading pre-trained model weights of FreespaceSeg')

        # Model to device
        self.model = self.model.to(self.device)

        # TensorBoard
        self.writer = SummaryWriter()

        # Learning rate and optimizer
        self.learning_rate = learning_rate
        self.optimizer = optim.AdamW(
            self.model.parameters(), self.learning_rate)

        # Loaders
        self.image_loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        self.gt_loader = transforms.Compose([
            transforms.ToTensor()])

        # Augmentations
        self.augTrain = Augmentations(is_train=True, data_type='BINARY_SEGMENTATION')
        self.augVal = Augmentations(is_train=False, data_type='BINARY_SEGMENTATION')

        # Loss function
        self.loss = nn.BCEWithLogitsLoss()

    # Logging Training Loss
    def log_loss(self, log_count, loss=None):
        if loss is None:
            loss = self.get_loss()
        print('Logging Training Loss', log_count, loss)
        self.writer.add_scalar("Loss/train", loss, (log_count))

    # Logging Validation mIoU Score
    def log_IoU(self, mIoU, log_count):
        print('Logging Validation')
        self.writer.add_scalar("Val/IoU", mIoU, (log_count))

    # Assign input variables
    def set_data(self, image, gt):
        self.image = image
        self.gt = gt

    # Image augmentations
    def apply_augmentations(self, is_train):
        aug = self.augTrain if is_train else self.augVal
        aug.setData(self.image, self.gt)
        self.image, self.gt = aug.applyTransformBinarySeg(
            image=self.image, ground_truth=self.gt)

    # Load Data
    def load_data(self):
        self.load_image_tensor()
        self.load_gt_tensor()

    # Run Model
    def run_model(self):
        self.prediction = self.model(self.image_tensor)
        self.calc_loss = self.loss(self.prediction, self.gt_tensor)

    # Loss Backward Pass
    def loss_backward(self):
        self.calc_loss.backward()

    # Get loss value
    def get_loss(self):
        return self.calc_loss.item()

    # Run Optimizer
    def run_optimizer(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    # Set train mode
    def set_train_mode(self):
        self.model = self.model.train()

    # Set evaluation mode
    def set_eval_mode(self):
        self.model = self.model.eval()

    # Save predicted visualization
    def save_visualization(self, log_count):
        print('Saving Visualization')

        # Get prediction
        prediction_vis = self.prediction.squeeze(0).cpu().detach()
        prediction_vis = prediction_vis.permute(1, 2, 0)
        prediction_vis = self.make_visualization(prediction_vis)

        # Get ground truth
        gt_vis = self.gt_tensor.squeeze(0).cpu().detach()
        gt_vis = gt_vis.permute(1, 2, 0)
        gt_vis = self.make_visualization(gt_vis)

        # Blending factor
        alpha = 0.5

        # Predicttion visualization
        prediction_vis = cv2.addWeighted(
            prediction_vis, alpha, self.image, 1 - alpha, 0)
        
        # Ground truth visualization
        gt_vis = cv2.addWeighted(
            gt_vis, alpha, self.image, 1 - alpha, 0)

        fig_img = plt.figure(figsize=(8, 4))
        plt.axis('off')
        plt.imshow(self.image)
        self.writer.add_figure(
            'Image/train', fig_img, global_step=(log_count))

        fig_pred = plt.figure(figsize=(8, 4))
        plt.axis('off')
        plt.imshow(prediction_vis)
        self.writer.add_figure(
            'Prediction/train', fig_pred, global_step=(log_count))

        fig_gt = plt.figure(figsize=(8, 4))
        plt.axis('off')
        plt.imshow(gt_vis)
        self.writer.add_figure(
            'Ground-Truth/train', fig_gt, global_step=(log_count))

    # Load Image as Tensor
    def load_image_tensor(self):
        image_tensor = self.image_loader(self.image)
        image_tensor = image_tensor.unsqueeze(0)
        self.image_tensor = image_tensor.to(self.device)

    # Load Ground Truth as Tensor
    def load_gt_tensor(self):
        gt_tensor = self.gt_loader(self.gt)
        gt_tensor = gt_tensor.unsqueeze(0)
        self.gt_tensor = gt_tensor.to(self.device)

    # Zero Gradient
    def zero_grad(self):
        self.optimizer.zero_grad()

    # Save Model
    def save_model(self, model_save_path):
        print('Saving model')
        torch.save(self.model.state_dict(), model_save_path)

    # Calculate IoU score for validation
    def calc_IoU_val(self):
        output_val = self.model(self.image_tensor)
        output_val = output_val.squeeze(0).cpu().detach()
        output_val = output_val.permute(1, 2, 0)
        output_val = output_val.numpy()
        output_val[output_val <= 0] = 0.0
        output_val[output_val > 0] = 1.0
        iou_score = self.IoU(output_val, self.gt)

        return iou_score

    # IoU calculation
    def IoU(self, output, label):
        intersection = np.logical_and(label, output)
        union = np.logical_or(label, output)
        iou_score = (np.sum(intersection) + 1) / float(np.sum(union) + 1)
        return iou_score

    # Run Validation and calculate metrics
    def validate(self, image, gt):

        # Set Data
        self.set_data(image, gt)

        # Augmenting Image
        self.apply_augmentations(is_train=False)

        # Converting to tensor and loading
        self.load_data()

        # Calculate IoU score
        iou_score = self.calc_IoU_val()

        return iou_score

    # Visualize predicted result
    def make_visualization(self, result):
        # Getting size of prediction
        shape = result.shape
        row = shape[0]
        col = shape[1]

        # Creating visualization image
        vis_predicted = np.zeros((row, col, 3), dtype="uint8")

        # Assigning background colour
        vis_predicted[:,:,0] = 61
        vis_predicted[:,:,1] = 93
        vis_predicted[:,:,2] = 255

        # Getting road labels
        road_lables = np.where(result > 0)

        # Assigning road colour
        vis_predicted[road_lables[0], road_lables[1], 0] = 0
        vis_predicted[road_lables[0], road_lables[1], 1] = 255
        vis_predicted[road_lables[0], road_lables[1], 2] = 220

        return vis_predicted

    def cleanup(self):
        self.writer.flush()
        self.writer.close()
        print('Finished Training')
