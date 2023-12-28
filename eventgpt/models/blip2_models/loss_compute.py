import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class LossComputer:
    def __init__(self):
        pass
    
    def smooth_l1_loss(self, pred_box, gt_box):
        # Implementation of smooth L1 loss
        return F.smooth_l1_loss(pred_box, gt_box)
    
    def giou_loss(self, pred_box, gt_box):
        # Convert boxes to [x1, y1, x2, y2] format
        pred_box = torch.tensor(pred_box) if not isinstance(pred_box, torch.Tensor) else pred_box
        gt_box = torch.tensor(gt_box) if not isinstance(gt_box, torch.Tensor) else gt_box
        
        # Extract coordinates
        x1_pred, y1_pred, x2_pred, y2_pred = pred_box
        x1_gt, y1_gt, x2_gt, y2_gt = gt_box
        
        # Calculate intersection area
        x_left = torch.max(x1_pred, x1_gt)
        y_top = torch.max(y1_pred, y1_gt)
        x_right = torch.min(x2_pred, x2_gt)
        y_bottom = torch.min(y2_pred, y2_gt)
        
        inter_area = torch.clamp(x_right - x_left, min=0) * torch.clamp(y_bottom - y_top, min=0)
        
        # Calculate union area
        area1 = (x2_pred - x1_pred) * (y2_pred - y1_pred)
        area2 = (x2_gt - x1_gt) * (y2_gt - y1_gt)
        union_area = area1 + area2 - inter_area
        
        # Calculate IoU
        iou = inter_area / union_area
        
        # Calculate diagonal squared distance (dIoU term)
        diag_dist = (x2_gt - x1_gt)**2 + (y2_gt - y1_gt)**2
        
        # Calculate enclosing box (cIoU term)
        enclose_x1 = torch.min(x1_pred, x1_gt)
        enclose_y1 = torch.min(y1_pred, y1_gt)
        enclose_x2 = torch.max(x2_pred, x2_gt)
        enclose_y2 = torch.max(y2_pred, y2_gt)
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
        
        # Calculate GIoU term
        giou_term = (enclose_area - union_area) / enclose_area
        
        # Compute GIoU loss
        giou_loss = 1 - (iou - giou_term)
        return giou_loss.mean()
    
    def diou_loss(self, pred_box, gt_box):
        # Convert boxes to [x1, y1, x2, y2] format
        pred_box = torch.tensor(pred_box) if not isinstance(pred_box, torch.Tensor) else pred_box
        gt_box = torch.tensor(gt_box) if not isinstance(gt_box, torch.Tensor) else gt_box
        
        # Extract coordinates
        x1_pred, y1_pred, x2_pred, y2_pred = pred_box
        x1_gt, y1_gt, x2_gt, y2_gt = gt_box
        
        # Calculate intersection area
        x_left = torch.max(x1_pred, x1_gt)
        y_top = torch.max(y1_pred, y1_gt)
        x_right = torch.min(x2_pred, x2_gt)
        y_bottom = torch.min(y2_pred, y2_gt)
        
        inter_area = torch.clamp(x_right - x_left, min=0) * torch.clamp(y_bottom - y_top, min=0)
        
        # Calculate union area
        area1 = (x2_pred - x1_pred) * (y2_pred - y1_pred)
        area2 = (x2_gt - x1_gt) * (y2_gt - y1_gt)
        union_area = area1 + area2 - inter_area
        
        # Calculate IoU
        iou = inter_area / union_area
        
        # Calculate diagonal squared distance (dIoU term)
        diag_dist = (x2_gt - x1_gt)**2 + (y2_gt - y1_gt)**2
        
        # Calculate enclosing box (cIoU term)
        enclose_x1 = torch.min(x1_pred, x1_gt)
        enclose_y1 = torch.min(y1_pred, y1_gt)
        enclose_x2 = torch.max(x2_pred, x2_gt)
        enclose_y2 = torch.max(y2_pred, y2_gt)
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
        
        # Calculate dIoU term
        diou_term = diag_dist / enclose_area
        
        # Compute dIoU loss
        diou_loss = 1 - (iou - diou_term)
        return diou_loss.mean()

    def ciou_loss(self, pred_box, gt_box):
        # Implementation of CIoU loss
        # Convert boxes to [x1, y1, x2, y2] format
        if not isinstance(pred_box, torch.Tensor):
            pred_box = torch.tensor(pred_box)
        
        if not isinstance(gt_box, torch.Tensor):
            gt_box = torch.tensor(gt_box)
        
        # Extract coordinates
        x1_pred, y1_pred, x2_pred, y2_pred = pred_box
        x1_gt, y1_gt, x2_gt, y2_gt = gt_box
        
        # Calculate intersection area
        x_left = torch.max(x1_pred, x1_gt)
        y_top = torch.max(y1_pred, y1_gt)
        x_right = torch.min(x2_pred, x2_gt)
        y_bottom = torch.min(y2_pred, y2_gt)
        
        inter_area = torch.clamp(x_right - x_left, min=0) * torch.clamp(y_bottom - y_top, min=0)
        
        # Calculate union area
        area1 = (x2_pred - x1_pred) * (y2_pred - y1_pred)
        area2 = (x2_gt - x1_gt) * (y2_gt - y1_gt)
        union_area = area1 + area2 - inter_area
        
        # Calculate IoU
        iou = inter_area / union_area
        
        # Calculate enclosing box (cIoU term)
        enclose_x1 = torch.min(x1_pred, x1_gt)
        enclose_y1 = torch.min(y1_pred, y1_gt)
        enclose_x2 = torch.max(x2_pred, x2_gt)
        enclose_y2 = torch.max(y2_pred, y2_gt)
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
        
        # Calculate diagonal squared distance (dIoU term)
        diag_dist = (x2_gt - x1_gt)**2 + (y2_gt - y1_gt)**2
        
        # Calculate cIoU term
        ciou_term = (enclose_area - union_area) / enclose_area
        
        # Compute cIoU loss
        ciou_loss = 1 - (iou - ciou_term + diag_dist)
        return ciou_loss.mean()

    def iou(self, box1, box2):
        # Implementation of IoU calculation
        # Convert boxes to [x1, y1, x2, y2] format
        if not isinstance(box1, torch.Tensor):
            box1 = torch.tensor(box1)
        
        if not isinstance(box2, torch.Tensor):
            box2 = torch.tensor(box2)
        
        # Extract coordinates
        x1_box1, y1_box1, x2_box1, y2_box1 = box1
        x1_box2, y1_box2, x2_box2, y2_box2 = box2
        
        # Calculate intersection area
        x_left = torch.max(x1_box1, x1_box2)
        y_top = torch.max(y1_box1, y1_box2)
        x_right = torch.min(x2_box1, x2_box2)
        y_bottom = torch.min(y2_box1, y2_box2)
        
        inter_area = torch.clamp(x_right - x_left, min=0) * torch.clamp(y_bottom - y_top, min=0)
        
        # Calculate union area
        area1 = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
        area2 = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)
        union_area = area1 + area2 - inter_area
        
        # Calculate IoU
        iou = inter_area / union_area
        return iou

    def linear_assign(self, iou_matrix):
        try:
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        except:
            print(f"metrix: {iou_matrix}")
            import pdb; pdb.set_trace()
        return row_ind, col_ind

    def compute_loss(self, pred_boxes, gt_boxes):
        """ Compute loss of predicted bounding boxes

        Parameters:
            pred_boxes: list of [xyxy] format or torch.Tensor[M, 4]
            gt_boxes: list of [xyxy] format or torch.Tensor[N, 4]
        
        Return: 
            average_loss: torch.Tensor
        """
        total_loss = 0
        
        device = pred_boxes[0].device  # Get the device of the tensors
        
        # Convert boxes to tensors and calculate IoU matrix
        iou_matrix = torch.zeros((len(pred_boxes), len(gt_boxes)), device=device)
        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                iou_matrix[i, j] = self.iou(pred_box, gt_box)

        # Perform linear assignment using IoU scores
        row_ind, col_ind = self.linear_assign(iou_matrix.detach().cpu().numpy())

        # Calculate losses based on matched pairs
        for i, j in zip(row_ind, col_ind):
            pred_box = pred_boxes[i]
            gt_box = gt_boxes[j]

            # Calculate smooth L1 loss
            smooth_l1 = self.smooth_l1_loss(pred_box, gt_box)

            # diou = self.diou_loss(pred_box, gt_box)
            # giou = self.giou_loss(pred_box, gt_box)
            # ciou = self.ciou_loss(pred_box, gt_box)

            # Combine losses as needed
            # combined_loss = smooth_l1 + giou
            combined_loss = smooth_l1 * 100

            total_loss += combined_loss

        average_loss = total_loss / len(row_ind) if len(row_ind) > 0 else 0
        return average_loss


if __name__ == "__main__":
    loss_computer = LossComputer()
    pred_boxes = [[10, 10, 50, 50], [20, 20, 40, 40], [30, 30, 60, 60]]
    gt_boxes = [[15, 15, 45, 45], [25, 25, 45, 45]]

    # pred_boxes = [[10, 10, 50, 50], [20, 20, 40, 40], [30, 30, 60, 60]]
    # gt_boxes = [[10, 10, 50, 50], [20, 20, 40, 40], [30, 30, 60, 60]]

    # convert list to tensor,并转为float32
    pred_boxes = torch.tensor(pred_boxes, dtype=torch.float32)
    gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)

    # define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pred_boxes = pred_boxes.to(device)
    gt_boxes = gt_boxes.to(device)

    loss = loss_computer.compute_loss(pred_boxes, gt_boxes)
    print("Total Loss:", loss.item())
