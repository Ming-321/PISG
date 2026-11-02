"""Lightning module for training the DIFUSCO TSP model."""

import os

import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.utilities import rank_zero_info

from co_datasets.tsp_graph_dataset import TSPGraphDataset
from pl_meta_model import COMetaModel
from utils.diffusion_schedulers import InferenceSchedule
from utils.tsp_utils import TSPEvaluator, batched_two_opt_torch, merge_tours


class TSPModel(COMetaModel):
  def __init__(self,
               param_args=None):
    super(TSPModel, self).__init__(param_args=param_args, node_feature_only=False)

    self.train_dataset = TSPGraphDataset(
        data_file=os.path.join(self.args.storage_path, self.args.training_split),
        sparse_factor=self.args.sparse_factor,
    )

    self.test_dataset = TSPGraphDataset(
        data_file=os.path.join(self.args.storage_path, self.args.test_split),
        sparse_factor=self.args.sparse_factor,
    )

    self.validation_dataset = TSPGraphDataset(
        data_file=os.path.join(self.args.storage_path, self.args.validation_split),
        sparse_factor=self.args.sparse_factor,
    )

  def forward(self, x, adj, t, edge_index, pisg_mode=False, pisg_perturb="uniform"):
    return self.model(
        x, t, adj, edge_index, pisg_mode=pisg_mode, pisg_perturb=pisg_perturb
    )

  @staticmethod
  def _normalize_guided_probabilities(prob, sparse):
    if not sparse:
      denom = prob.sum(dim=1, keepdim=True).clamp_min(1e-12)
    else:
      denom = prob.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return prob / denom

  def _run_two_opt(self, np_points, tours, device):
    if self.args.disable_two_opt or self.args.two_opt_iterations <= 0:
      return np.array(tours).astype('int64'), 0.0
    solved_tours, iterations = batched_two_opt_torch(
        np_points.astype("float64"),
        np.array(tours).astype('int64'),
        max_iterations=self.args.two_opt_iterations,
        device=device,
    )
    return solved_tours, float(iterations)

  def _prepare_categorical_denoiser_input(self, xt, device):
    xt_model = xt.float().to(device)
    xt_model = xt_model * 2 - 1
    xt_model = xt_model * (1.0 + 0.05 * torch.rand_like(xt_model))
    return xt_model

  def categorical_x0_pred_prob(self, points, xt, t, device, edge_index=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
      xt_model = self._prepare_categorical_denoiser_input(xt, device)
      x0_pred = self.forward(
          points.float().to(device),
          xt_model,
          t.float().to(device),
          edge_index.long().to(device) if edge_index is not None else None,
      )

      if not self.sparse:
        return x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
      return x0_pred.reshape((1, points.shape[0], -1, 2)).softmax(dim=-1)

  def categorical_training_step(self, batch, batch_idx):
    edge_index = None
    if not self.sparse:
      _, points, adj_matrix, _ = batch
      t = np.random.randint(1, self.diffusion.T + 1, points.shape[0]).astype(int)
    else:
      _, graph_data, point_indicator, edge_indicator, _ = batch
      t = np.random.randint(1, self.diffusion.T + 1, point_indicator.shape[0]).astype(int)
      route_edge_flags = graph_data.edge_attr
      points = graph_data.x
      edge_index = graph_data.edge_index
      num_edges = edge_index.shape[1]
      batch_size = point_indicator.shape[0]
      adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))

    # Sample from diffusion
    adj_matrix_onehot = F.one_hot(adj_matrix.long(), num_classes=2).float()
    if self.sparse:
      adj_matrix_onehot = adj_matrix_onehot.unsqueeze(1)

    xt = self.diffusion.sample(adj_matrix_onehot, t)
    xt = self._prepare_categorical_denoiser_input(xt, adj_matrix.device)

    if self.sparse:
      t = torch.from_numpy(t).float()
      t = t.reshape(-1, 1).repeat(1, adj_matrix.shape[1]).reshape(-1)
      xt = xt.reshape(-1)
      adj_matrix = adj_matrix.reshape(-1)
      points = points.reshape(-1, 2)
      edge_index = edge_index.float().to(adj_matrix.device).reshape(2, -1)
    else:
      t = torch.from_numpy(t).float().view(adj_matrix.shape[0])

    # Denoise
    x0_pred = self.forward(
        points.float().to(adj_matrix.device),
        xt.float().to(adj_matrix.device),
        t.float().to(adj_matrix.device),
        edge_index,
    )

    # Compute loss
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(x0_pred, adj_matrix.long())
    self.log("train/loss", loss)
    return loss

  def gaussian_training_step(self, batch, batch_idx):
    if self.sparse:
      # TODO: Implement Gaussian diffusion with sparse graphs
      raise ValueError("DIFUSCO with sparse graphs are not supported for Gaussian diffusion")
    _, points, adj_matrix, _ = batch

    adj_matrix = adj_matrix * 2 - 1
    adj_matrix = adj_matrix * (1.0 + 0.05 * torch.rand_like(adj_matrix))
    # Sample from diffusion
    t = np.random.randint(1, self.diffusion.T + 1, adj_matrix.shape[0]).astype(int)
    xt, epsilon = self.diffusion.sample(adj_matrix, t)

    t = torch.from_numpy(t).float().view(adj_matrix.shape[0])
    # Denoise
    epsilon_pred = self.forward(
        points.float().to(adj_matrix.device),
        xt.float().to(adj_matrix.device),
        t.float().to(adj_matrix.device),
        None,
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

  def categorical_denoise_step(self, points, xt, t, device, edge_index=None, target_t=None):
    with torch.no_grad():
      t_tensor = torch.from_numpy(t).view(1)
      x0_pred_prob = self.categorical_x0_pred_prob(points, xt, t, device, edge_index=edge_index)
      xt = self.categorical_posterior(target_t, t_tensor, x0_pred_prob, xt)
      return xt

  def categorical_pisg_denoise_step(self, points, xt, t, device, edge_index=None, target_t=None,
                                    pisg_scale=1.0, pisg_perturb="uniform"):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
      xt_scale = self._prepare_categorical_denoiser_input(xt, device)

      points_dup = torch.cat([points, points], dim=0)
      xt_scale_dup = torch.cat([xt_scale, xt_scale], dim=0)

      if self.sparse:
        t_for_forward = t
        edge_index_dup = self.duplicate_edge_index_copies(edge_index, points.shape[0], 2)
      else:
        batch_size = points.shape[0]
        if t.numel() == 1:
          t = t.repeat(batch_size)
        elif t.numel() != batch_size:
          raise ValueError(
              f"Unexpected dense timestep shape: {tuple(t.shape)} for batch size {batch_size}"
          )
        t_for_forward = torch.cat([t, t], dim=0)
        edge_index_dup = edge_index

      if pisg_perturb == "input-noise":
        noise_std = 0.5
        half_size = points.shape[0] if not self.sparse else xt_scale.shape[0]
        xt_scale_dup[half_size:] = xt_scale_dup[half_size:] + noise_std * torch.randn_like(
            xt_scale_dup[half_size:]
        )
        output = self.forward(
            points_dup.float().to(device),
            xt_scale_dup.to(device),
            t_for_forward.float().to(device),
            edge_index_dup.long().to(device) if edge_index_dup is not None else None,
            pisg_mode=False,
        )
      else:
        output = self.forward(
            points_dup.float().to(device),
            xt_scale_dup.to(device),
            t_for_forward.float().to(device),
            edge_index_dup.long().to(device) if edge_index_dup is not None else None,
            pisg_mode=True,
            pisg_perturb=pisg_perturb,
        )

      out_good, out_bad = output.chunk(2, dim=0)
      fusion_space = getattr(self.args, "pisg_fusion_space", "logit")
      if fusion_space == "logit":
        guided = out_good + pisg_scale * (out_good - out_bad)
        if not self.sparse:
          x0_pred_prob = guided.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
        else:
          x0_pred_prob = guided.reshape((1, points.shape[0], -1, 2)).softmax(dim=-1)
      elif fusion_space == "probability":
        if not self.sparse:
          prob_good = out_good.softmax(dim=1)
          prob_bad = out_bad.softmax(dim=1)
        else:
          prob_good = out_good.softmax(dim=-1)
          prob_bad = out_bad.softmax(dim=-1)
        guided = (prob_good + pisg_scale * (prob_good - prob_bad)).clamp_min(0.0)
        guided = self._normalize_guided_probabilities(guided, self.sparse)
        if not self.sparse:
          x0_pred_prob = guided.permute((0, 2, 3, 1)).contiguous()
        else:
          x0_pred_prob = guided.reshape((1, points.shape[0], -1, 2))
      else:
        raise ValueError(f"Unsupported PISG fusion space: {fusion_space}")

      xt = self.categorical_posterior(target_t, t[:1], x0_pred_prob, xt)
      return xt

  def gaussian_denoise_step(self, points, xt, t, device, edge_index=None, target_t=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
      pred = self.forward(
          points.float().to(device),
          xt.float().to(device),
          t.float().to(device),
          edge_index.long().to(device) if edge_index is not None else None,
      )
      pred = pred.squeeze(1)
      xt = self.gaussian_posterior(target_t, t, pred, xt)
      return xt

  def test_step(self, batch, batch_idx, split='test'):
    edge_index = None
    np_edge_index = None
    device = batch[-1].device
    if not self.sparse:
      real_batch_idx, points, adj_matrix, gt_tour = batch
      np_points = points.cpu().numpy()[0]
      np_gt_tour = gt_tour.cpu().numpy()[0]
    else:
      real_batch_idx, graph_data, point_indicator, edge_indicator, gt_tour = batch
      route_edge_flags = graph_data.edge_attr
      points = graph_data.x
      edge_index = graph_data.edge_index
      num_edges = edge_index.shape[1]
      batch_size = point_indicator.shape[0]
      adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))
      points = points.reshape((-1, 2))
      edge_index = edge_index.reshape((2, -1))
      np_points = points.cpu().numpy()
      np_gt_tour = gt_tour.cpu().numpy().reshape(-1)
      np_edge_index = edge_index.cpu().numpy()

    stacked_tours = []
    ns, merge_iterations = 0, 0

    if getattr(self.args, "pisg", False) and self.diffusion_type != 'categorical':
      raise ValueError("PISG transfer currently only supports categorical diffusion")

    if self.args.parallel_sampling > 1:
      if not self.sparse:
        points = points.repeat(self.args.parallel_sampling, 1, 1)
      else:
        points = points.repeat(self.args.parallel_sampling, 1)
        edge_index = self.duplicate_edge_index(edge_index, np_points.shape[0], device)

    for _ in range(self.args.sequential_sampling):
      xt = torch.randn_like(adj_matrix.float())
      if self.args.parallel_sampling > 1:
        if not self.sparse:
          xt = xt.repeat(self.args.parallel_sampling, 1, 1)
        else:
          xt = xt.repeat(self.args.parallel_sampling, 1)
        xt = torch.randn_like(xt)

      if self.diffusion_type == 'gaussian':
        xt.requires_grad = True
      else:
        xt = (xt > 0).long()

      if self.sparse:
        xt = xt.reshape(-1)

      steps = self.args.inference_diffusion_steps
      time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule,
                                        T=self.diffusion.T, inference_T=steps)
      pisg_active_steps = None
      if getattr(self.args, "pisg", False):
        pisg_active_steps = self.parse_pisg_active_steps(
            getattr(self.args, "pisg_active_steps", "all"), steps
        )

      # Diffusion iterations
      for i in range(steps):
        t1, t2 = time_schedule(i)
        t1 = np.array([t1]).astype(int)
        t2 = np.array([t2]).astype(int)

        if self.diffusion_type == 'gaussian':
          xt = self.gaussian_denoise_step(
              points, xt, t1, device, edge_index, target_t=t2)
        else:
          if getattr(self.args, "pisg", False):
            current_pisg_scale = self.compute_pisg_scale(
                step_idx=i,
                total_steps=steps,
                base_scale=getattr(self.args, "pisg_scale", 1.0),
                schedule=getattr(self.args, "pisg_scale_schedule", "constant"),
                active_steps=pisg_active_steps,
            )
            xt = self.categorical_pisg_denoise_step(
                points,
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
                points, xt, t1, device, edge_index, target_t=t2)

      if self.diffusion_type == 'gaussian':
        adj_mat = xt.cpu().detach().numpy() * 0.5 + 0.5
      else:
        adj_mat = xt.float().cpu().detach().numpy() + 1e-6

      if self.args.save_numpy_heatmap:
        self.run_save_numpy_heatmap(adj_mat, np_points, real_batch_idx, split, np_edge_index=np_edge_index)

      tours, merge_iterations = merge_tours(
          adj_mat, np_points, np_edge_index,
          sparse_graph=self.sparse,
          parallel_sampling=self.args.parallel_sampling,
      )

      solved_tours, ns = self._run_two_opt(np_points, tours, device)
      stacked_tours.append(solved_tours)

    solved_tours = np.concatenate(stacked_tours, axis=0)

    tsp_solver = TSPEvaluator(np_points)
    gt_cost = tsp_solver.evaluate(np_gt_tour)

    total_sampling = self.args.parallel_sampling * self.args.sequential_sampling
    all_solved_costs = [tsp_solver.evaluate(solved_tours[i]) for i in range(total_sampling)]
    best_solved_cost = np.min(all_solved_costs)

    metrics = {
        f"{split}/gt_cost": gt_cost,
        f"{split}/2opt_iterations": float(ns),
        f"{split}/merge_iterations": merge_iterations,
    }
    for k, v in metrics.items():
      self.log(k, v, on_epoch=True, sync_dist=True)
    self.log(f"{split}/solved_cost", best_solved_cost, prog_bar=True, on_epoch=True, sync_dist=True)
    return metrics

  def run_save_numpy_heatmap(self, adj_mat, np_points, real_batch_idx, split, np_edge_index=None):
    if self.args.parallel_sampling > 1 or self.args.sequential_sampling > 1:
      raise NotImplementedError("Save numpy heatmap only support single sampling")
    exp_save_dir = os.path.join(self.logger.save_dir, self.logger.name, self.logger.version)
    heatmap_path = os.path.join(exp_save_dir, 'numpy_heatmap')
    rank_zero_info(f"Saving heatmap to {heatmap_path}")
    os.makedirs(heatmap_path, exist_ok=True)
    real_batch_idx = real_batch_idx.cpu().numpy().reshape(-1)[0]

    if self.sparse:
      if np_edge_index is None:
        raise ValueError("Sparse heatmap export requires the sparse edge index")
      dense_adj = scipy.sparse.coo_matrix(
          (adj_mat.astype(np.float32), (np_edge_index[0], np_edge_index[1])),
          shape=(np_points.shape[0], np_points.shape[0]),
      ).toarray()
      np.save(os.path.join(heatmap_path, f"{split}-heatmap-{real_batch_idx}.npy"), dense_adj)
    else:
      np.save(os.path.join(heatmap_path, f"{split}-heatmap-{real_batch_idx}.npy"), adj_mat)

    np.save(os.path.join(heatmap_path, f"{split}-points-{real_batch_idx}.npy"), np_points)

  def validation_step(self, batch, batch_idx):
    return self.test_step(batch, batch_idx, split='val')
