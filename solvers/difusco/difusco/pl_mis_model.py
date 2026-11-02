"""Lightning module for training the DIFUSCO MIS model."""

import os

import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from co_datasets.mis_dataset import MISDataset
from utils.diffusion_schedulers import InferenceSchedule
from pl_meta_model import COMetaModel
from utils.mis_utils import mis_decode_np


class MISModel(COMetaModel):
  def __init__(self,
               param_args=None):
    super(MISModel, self).__init__(param_args=param_args, node_feature_only=True)

    data_label_dir = None
    if self.args.training_split_label_dir is not None:
      data_label_dir = os.path.join(self.args.storage_path, self.args.training_split_label_dir)

    self.train_dataset = MISDataset(
        data_file=os.path.join(self.args.storage_path, self.args.training_split),
        data_label_dir=data_label_dir,
    )

    self.test_dataset = MISDataset(
        data_file=os.path.join(self.args.storage_path, self.args.test_split),
    )

    self.validation_dataset = MISDataset(
        data_file=os.path.join(self.args.storage_path, self.args.validation_split),
    )

  def forward(self, x, t, edge_index, pisg_mode=False, pisg_perturb="uniform"):
    return self.model(
        x, t, edge_index=edge_index, pisg_mode=pisg_mode, pisg_perturb=pisg_perturb
    )

  @staticmethod
  def _normalize_guided_probabilities(prob):
    return prob / prob.sum(dim=-1, keepdim=True).clamp_min(1e-12)

  def _prepare_categorical_denoiser_input(self, xt, device):
    xt_model = xt.float().to(device)
    encoding = getattr(self.args, "mis_categorical_input_encoding", "training")
    if encoding == "training":
      xt_model = xt_model * 2 - 1
      xt_model = xt_model * (1.0 + 0.05 * torch.rand_like(xt_model))
    elif encoding != "binary":
      raise ValueError(f"Unsupported MIS categorical input encoding: {encoding}")
    return xt_model

  @staticmethod
  def _duplicate_pisg_timesteps(t, num_nodes, duplicated_num_nodes):
    if t.numel() == 1:
      return t
    if t.numel() == num_nodes:
      return torch.cat([t, t], dim=0)
    if t.numel() == duplicated_num_nodes:
      return t
    raise ValueError(
        f"Unexpected MIS PISG timestep shape {tuple(t.shape)} for num_nodes={num_nodes}"
    )

  def _pisg_forward(self, xt_model, t, device, edge_index=None, pisg_scale=1.0, pisg_perturb="uniform"):
    if edge_index is None:
      raise ValueError("edge_index is required for MIS PISG denoising")

    xt_model_dup = torch.cat([xt_model, xt_model], dim=0)
    t_dup = self._duplicate_pisg_timesteps(t, xt_model.shape[0], xt_model_dup.shape[0])
    edge_index_dup = self.duplicate_edge_index_copies(edge_index, xt_model.shape[0], 2)

    if pisg_perturb == "input-noise":
      noise_std = 0.5
      xt_model_dup[xt_model.shape[0]:] = xt_model_dup[xt_model.shape[0]:] + noise_std * torch.randn_like(
          xt_model_dup[xt_model.shape[0]:]
      )
      output = self.forward(
          xt_model_dup.float().to(device),
          t_dup.float().to(device),
          edge_index_dup.long().to(device),
          pisg_mode=False,
      )
    else:
      output = self.forward(
          xt_model_dup.float().to(device),
          t_dup.float().to(device),
          edge_index_dup.long().to(device),
          pisg_mode=True,
          pisg_perturb=pisg_perturb,
      )

    out_good, out_bad = output.chunk(2, dim=0)
    guided = out_good + pisg_scale * (out_good - out_bad)
    return guided, out_good, out_bad

  def categorical_training_step(self, batch, batch_idx):
    _, graph_data, point_indicator = batch
    t = np.random.randint(1, self.diffusion.T + 1, point_indicator.shape[0]).astype(int)
    node_labels = graph_data.x
    edge_index = graph_data.edge_index

    # Sample from diffusion
    node_labels_onehot = F.one_hot(node_labels.long(), num_classes=2).float()
    node_labels_onehot = node_labels_onehot.unsqueeze(1).unsqueeze(1)

    t = torch.from_numpy(t).long()
    t = t.repeat_interleave(point_indicator.reshape(-1).cpu(), dim=0).numpy()

    xt = self.diffusion.sample(node_labels_onehot, t)
    xt = xt * 2 - 1
    xt = xt * (1.0 + 0.05 * torch.rand_like(xt))

    t = torch.from_numpy(t).float()
    t = t.reshape(-1)
    xt = xt.reshape(-1)
    edge_index = edge_index.to(node_labels.device).reshape(2, -1)

    # Denoise
    x0_pred = self.forward(
        xt.float().to(node_labels.device),
        t.float().to(node_labels.device),
        edge_index,
    )

    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(x0_pred, node_labels)
    self.log("train/loss", loss)
    return loss

  def gaussian_training_step(self, batch, batch_idx):
    _, graph_data, point_indicator = batch
    t = np.random.randint(1, self.diffusion.T + 1, point_indicator.shape[0]).astype(int)
    node_labels = graph_data.x
    edge_index = graph_data.edge_index
    device = node_labels.device

    # Sample from diffusion
    node_labels = node_labels.float() * 2 - 1
    node_labels = node_labels * (1.0 + 0.05 * torch.rand_like(node_labels))
    node_labels = node_labels.unsqueeze(1).unsqueeze(1)

    t = torch.from_numpy(t).long()
    t = t.repeat_interleave(point_indicator.reshape(-1).cpu(), dim=0).numpy()
    xt, epsilon = self.diffusion.sample(node_labels, t)

    t = torch.from_numpy(t).float()
    t = t.reshape(-1)
    xt = xt.reshape(-1)
    edge_index = edge_index.to(device).reshape(2, -1)
    epsilon = epsilon.reshape(-1)

    # Denoise
    epsilon_pred = self.forward(
        xt.float().to(device),
        t.float().to(device),
        edge_index,
    )
    epsilon_pred = epsilon_pred.squeeze(1)

    # Compute loss
    loss = F.mse_loss(epsilon_pred, epsilon.float())
    self.log("train/loss", loss)
    return loss

  def training_step(self, batch, batch_idx):
    if self.diffusion_type == 'gaussian':
      return self.gaussian_training_step(batch, batch_idx)
    elif self.diffusion_type == 'categorical':
      return self.categorical_training_step(batch, batch_idx)

  def categorical_denoise_step(self, xt, t, device, edge_index=None, target_t=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
      xt_model = self._prepare_categorical_denoiser_input(xt, device)
      x0_pred = self.forward(
          xt_model,
          t.float().to(device),
          edge_index.long().to(device) if edge_index is not None else None,
      )
      x0_pred_prob = x0_pred.reshape((1, xt.shape[0], -1, 2)).softmax(dim=-1)
      xt = self.categorical_posterior(target_t, t, x0_pred_prob, xt)
      return xt

  def categorical_pisg_denoise_step(self, xt, t, device, edge_index=None, target_t=None,
                                    pisg_scale=1.0, pisg_perturb="uniform"):
    if pisg_scale == 0.0:
      return self.categorical_denoise_step(xt, t, device, edge_index=edge_index, target_t=target_t)

    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
      xt_model = self._prepare_categorical_denoiser_input(xt, device)
      guided, out_good, out_bad = self._pisg_forward(
          xt_model,
          t,
          device,
          edge_index=edge_index,
          pisg_scale=pisg_scale,
          pisg_perturb=pisg_perturb,
      )

      fusion_space = getattr(self.args, "pisg_fusion_space", "logit")
      if fusion_space == "logit":
        x0_pred_prob = guided.reshape((1, xt.shape[0], -1, 2)).softmax(dim=-1)
      elif fusion_space == "probability":
        prob_good = out_good.softmax(dim=-1)
        prob_bad = out_bad.softmax(dim=-1)
        guided_prob = (prob_good + pisg_scale * (prob_good - prob_bad)).clamp_min(0.0)
        guided_prob = self._normalize_guided_probabilities(guided_prob)
        x0_pred_prob = guided_prob.reshape((1, xt.shape[0], -1, 2))
      else:
        raise ValueError(f"Unsupported PISG fusion space: {fusion_space}")

      xt = self.categorical_posterior(target_t, t[:1], x0_pred_prob, xt)
      return xt

  def gaussian_denoise_step(self, xt, t, device, edge_index=None, target_t=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
      pred = self.forward(
          xt.float().to(device),
          t.float().to(device),
          edge_index.long().to(device) if edge_index is not None else None,
      )
      pred = pred.squeeze(1)
      xt = self.gaussian_posterior(target_t, t, pred, xt)
      return xt

  def gaussian_pisg_denoise_step(self, xt, t, device, edge_index=None, target_t=None,
                                 pisg_scale=1.0, pisg_perturb="uniform"):
    if pisg_scale == 0.0:
      return self.gaussian_denoise_step(xt, t, device, edge_index=edge_index, target_t=target_t)

    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
      guided, _, _ = self._pisg_forward(
          xt.float().to(device),
          t,
          device,
          edge_index=edge_index,
          pisg_scale=pisg_scale,
          pisg_perturb=pisg_perturb,
      )
      if guided.ndim > 1 and guided.shape[-1] == 1:
        guided = guided.squeeze(-1)
      xt = self.gaussian_posterior(target_t, t, guided, xt)
      return xt

  def test_step(self, batch, batch_idx, draw=False, split='test'):
    device = batch[-1].device

    real_batch_idx, graph_data, point_indicator = batch
    node_labels = graph_data.x
    edge_index = graph_data.edge_index

    stacked_predict_labels = []
    edge_index = edge_index.to(node_labels.device).reshape(2, -1)
    edge_index_np = edge_index.cpu().numpy()
    adj_mat = scipy.sparse.coo_matrix(
        (np.ones_like(edge_index_np[0]), (edge_index_np[0], edge_index_np[1])),
    )

    pisg_active_steps = None
    if getattr(self.args, "pisg", False):
      pisg_active_steps = self.parse_pisg_active_steps(
          getattr(self.args, "pisg_active_steps", "all"),
          self.args.inference_diffusion_steps,
      )

    for _ in range(self.args.sequential_sampling):
      xt = torch.randn_like(node_labels.float())
      if self.args.parallel_sampling > 1:
        xt = xt.repeat(self.args.parallel_sampling, 1, 1)
        xt = torch.randn_like(xt)

      if self.diffusion_type == 'gaussian':
        xt.requires_grad = True
      else:
        xt = (xt > 0).long()
      xt = xt.reshape(-1)

      if self.args.parallel_sampling > 1:
        edge_index = self.duplicate_edge_index(edge_index, node_labels.shape[0], device)

      batch_size = 1
      steps = self.args.inference_diffusion_steps
      time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule,
                                        T=self.diffusion.T, inference_T=steps)

      for i in range(steps):
        t1, t2 = time_schedule(i)
        t1 = np.array([t1 for _ in range(batch_size)]).astype(int)
        t2 = np.array([t2 for _ in range(batch_size)]).astype(int)
        current_pisg_scale = 0.0
        if getattr(self.args, "pisg", False):
          current_pisg_scale = self.compute_pisg_scale(
              step_idx=i,
              total_steps=steps,
              base_scale=getattr(self.args, "pisg_scale", 1.0),
              schedule=getattr(self.args, "pisg_scale_schedule", "constant"),
              active_steps=pisg_active_steps,
          )

        if self.diffusion_type == 'gaussian':
          if getattr(self.args, "pisg", False):
            xt = self.gaussian_pisg_denoise_step(
                xt,
                t1,
                device,
                edge_index=edge_index,
                target_t=t2,
                pisg_scale=current_pisg_scale,
                pisg_perturb=getattr(self.args, "pisg_perturb", "uniform"),
            )
          else:
            xt = self.gaussian_denoise_step(
                xt, t1, device, edge_index, target_t=t2)
        else:
          if getattr(self.args, "pisg", False):
            xt = self.categorical_pisg_denoise_step(
                xt,
                t1,
                device,
                edge_index=edge_index,
                target_t=t2,
                pisg_scale=current_pisg_scale,
                pisg_perturb=getattr(self.args, "pisg_perturb", "uniform"),
            )
          else:
            xt = self.categorical_denoise_step(
                xt, t1, device, edge_index, target_t=t2)

      if self.diffusion_type == 'gaussian':
        predict_labels = xt.float().cpu().detach().numpy() * 0.5 + 0.5
      else:
        predict_labels = xt.float().cpu().detach().numpy() + 1e-6
      stacked_predict_labels.append(predict_labels)

    predict_labels = np.concatenate(stacked_predict_labels, axis=0)
    all_sampling = self.args.sequential_sampling * self.args.parallel_sampling

    splitted_predict_labels = np.split(predict_labels, all_sampling)
    solved_solutions = [mis_decode_np(predict_labels, adj_mat) for predict_labels in splitted_predict_labels]
    solved_costs = [solved_solution.sum() for solved_solution in solved_solutions]
    best_solved_cost = np.max(solved_costs)

    gt_cost = node_labels.cpu().numpy().sum()
    metrics = {
        f"{split}/gt_cost": gt_cost,
    }
    for k, v in metrics.items():
      self.log(k, v, on_epoch=True, sync_dist=True)
    self.log(f"{split}/solved_cost", best_solved_cost, prog_bar=True, on_epoch=True, sync_dist=True)
    return metrics

  def validation_step(self, batch, batch_idx):
    return self.test_step(batch, batch_idx, split='val')
