import torch
import torch.nn.functional as F

from models.repmono.layers import SSIM, get_smooth_loss

from math import exp


class RepMonoUnsupervisedLoss():

    def __init__(self):
        self.scales = [0, 1, 2]
        self.ssim = SSIM()

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def __call__(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """

        losses = {}
        total_loss = 0

        frame_ids = [0, -1, 1]

        for scale in self.scales:
            loss = 0
            reprojection_losses = []

            source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            identity_reprojection_losses = []
            for frame_id in frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))

            identity_reprojection_losses = torch.cat(
                identity_reprojection_losses, 1)

            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape) * 0.00001

            combined = torch.cat(
                (identity_reprojection_losses, reprojection_losses), dim=1)

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            outputs["identity_selection/{}".format(scale)] = (
                idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += 1e-3 * smooth_loss / (2**scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= len(self.scales)
        return total_loss


class DepthLoss():

    def __init__(self, alpha, beta, gamma, maxDepth=10.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.maxDepth = maxDepth

        self.L1_Loss = torch.nn.L1Loss()

    def __call__(self, output, depth):
        if self.beta == 0 and self.gamma == 0:
            valid_mask = depth > 0.0
            output = output[valid_mask]
            depth = depth[valid_mask]
            l_depth = self.L1_Loss(output, depth)
            loss = l_depth
        else:
            l_depth = self.L1_Loss(output, depth)
            l_ssim = torch.clamp(
                (1 - self.ssim(output, depth, self.maxDepth)) * 0.5, 0, 1)
            l_grad = self.gradient_loss(output, depth)

            loss = self.alpha * l_depth + self.beta * l_ssim + self.gamma * l_grad
        return loss

    def ssim(self,
             img1,
             img2,
             val_range,
             window_size=11,
             window=None,
             size_average=True,
             full=False):
        L = val_range

        padd = 0

        (_, channel, height, width) = img1.size()
        if window is None:
            real_size = min(window_size, height, width)
            window = self.create_window(real_size,
                                        channel=channel).to(img1.device)
            padd = window_size // 2

        mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padd,
                             groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padd,
                             groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padd,
                           groups=channel) - mu1_mu2

        C1 = (0.01 * L)**2
        C2 = (0.03 * L)**2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs

        return ret

    def gradient_loss(self, gen_frames, gt_frames, alpha=1):
        gen_dx, gen_dy = self.gradient(gen_frames)
        gt_dx, gt_dy = self.gradient(gt_frames)

        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        # condense into one tensor and avg
        grad_comb = grad_diff_x**alpha + grad_diff_y**alpha

        return torch.mean(grad_comb)

    def gradient(self, x):
        """
        idea from tf.image.image_gradients(image)
        https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
        """
        h_x = x.size()[-2]
        w_x = x.size()[-1]

        left = x
        right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

        dx, dy = right - left, bottom - top

        # dx will always have zeros in the last column, right-left
        # dy will always have zeros in the last row,    bottom-top
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        return dx, dy

    def create_window(self, window_size, channel=1):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(
            _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size,
                                   window_size).contiguous()
        return window

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([
            exp(-(x - window_size // 2)**2 / float(2 * sigma**2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
