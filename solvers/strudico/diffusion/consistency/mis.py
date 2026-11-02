import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
from torch_sparse import SparseTensor
from diffusion.utils.diffusion_schedulers import InferenceSchedule
from diffusion.consistency.meta import MetaConsistency
from diffusion.utils.mis_utils import mis_decode_np

import time


class MISConsistency(MetaConsistency):
    def __init__(
            self,
            args,
            sigma_max=1000,
            sigma_min=0,
            weight_schedule="uniform",
            boundary_func='truncate'
    ):
        super(MISConsistency, self).__init__(
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            weight_schedule=weight_schedule,
            boundary_func=boundary_func)

        self.args = args

    def consistency_losses(self, model, batch):
        _, graph_data, point_indicator = batch  # point_indicator: B, 1, number of nodes in each batch
        node_labels = graph_data.x  # N*B
        edge_index = graph_data.edge_index  # 2, E*B

        x0 = F.one_hot(node_labels.long(), num_classes=2).float()

        device = node_labels.device
        t = torch.randint(1, model.diffusion.T + 1, [point_indicator.shape[0]]).to(device)
        t2 = (model.args.alpha * t).int().to(device)

        t = t.repeat_interleave(point_indicator.reshape(-1), dim=0)
        t2 = t2.repeat_interleave(point_indicator.reshape(-1), dim=0)

        x_t = model.diffusion.sample(x0.unsqueeze(1).unsqueeze(1), t.cpu().numpy())
        x_t2 = model.diffusion.sample(x0.unsqueeze(1).unsqueeze(1), t2.cpu().numpy())

        t = t.reshape(-1).float()
        t2 = t2.reshape(-1).float()
        x_t = x_t.reshape(-1).to(device)  # N
        x_t2 = x_t2.reshape(-1).to(device)
        edge_index = edge_index.to(device).reshape(2, -1)  # 2, E

        model_output, denoise = self.denoise(model, x_t, t, edge_index, x0)
        model_output2, denoise2 = self.denoise(model, x_t2, t2, edge_index, x0)

        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(denoise, node_labels) + loss_func(denoise2, node_labels)

        return loss

    def denoise(self, model, x_t, t, edge_index, x0):
        x_t = x_t * 2 - 1
        x_t = x_t * (1.0 + 0.05 * torch.rand_like(x_t))
        model_output = model(x_t, t, edge_index)

        c_skip, c_out = [
            self.append_dims(x, model_output.ndim)
            for x in self.get_scalings_for_boundary_condition(t)
        ]

        denoise = c_out * model_output + c_skip * x0

        return model_output, denoise

    def consistency_test_step(self, model, batch, batch_idx, split='test'):
        device = batch[-1].device
        t_start = time.time()

        real_batch_idx, graph_data, point_indicator = batch
        node_labels = graph_data.x
        edge_index = graph_data.edge_index

        torch.cuda.reset_peak_memory_stats(device)

        stacked_predict_labels = []
        edge_index = edge_index.to(node_labels.device).reshape(2, -1)
        edge_index_np = edge_index.cpu().numpy()
        adj_mat = scipy.sparse.coo_matrix(
            (np.ones_like(edge_index_np[0]), (edge_index_np[0], edge_index_np[1])),
        )

        save_intermediates = getattr(model.args, "save_intermediates", False)
        if save_intermediates and model.args.parallel_sampling > 1:
            import warnings
            warnings.warn(
                "save_intermediates with parallel_sampling > 1 produces "
                "folded tensors incompatible with analyze_branch_delta.py. "
                "Use parallel_sampling=1 for signal analysis.",
                stacklevel=2,
            )
        intermediate_steps = [] if save_intermediates else None
        seq_id = 0

        if model.args.parallel_sampling > 1:
            edge_index = model.duplicate_edge_index(model.args.parallel_sampling, edge_index, node_labels.shape[0], device)

        for _ in range(model.args.sequential_sampling):
            xt = torch.zeros_like(node_labels.float())
            if model.args.parallel_sampling > 1:
                xt = xt.repeat(model.args.parallel_sampling, 1, 1)
                xt = torch.zeros_like(xt)

            xt = (xt > 0).long().reshape(-1)

            steps = model.args.inference_diffusion_steps
            time_schedule = InferenceSchedule(
                inference_schedule=model.args.inference_schedule,
                T=model.diffusion.T,
                inference_T=steps,
            )

            pisg_active_steps = None
            if getattr(model.args, "pisg", False):
                pisg_active_steps = self._parse_active_steps(
                    getattr(model.args, "pisg_active_steps", "all"), steps
                )

            for i in range(steps):
                t1, t2 = time_schedule(i)
                t1 = torch.tensor([t1], device=device).float()

                branch_normal = branch_perturbed = None
                if getattr(model.args, "pisg", False):
                    current_scale = self.compute_pisg_scale(
                        i, steps,
                        getattr(model.args, "pisg_scale", 1.0),
                        schedule=getattr(model.args, "pisg_scale_schedule", "constant"),
                        active_steps=pisg_active_steps,
                    )
                    if current_scale == 0.0:
                        x0_pred = self._plain_denoise(model, xt, t1, device, edge_index)
                    else:
                        pisg_result = self.pisg_denoise_step(
                            model, xt, t1, device,
                            edge_index=edge_index,
                            pisg_scale=current_scale,
                            pisg_perturb=getattr(model.args, "pisg_perturb", "uniform"),
                            return_branches=(intermediate_steps is not None),
                        )
                        num_nodes = xt.reshape(-1).shape[0]
                        if isinstance(pisg_result, dict):
                            guided_raw = pisg_result["guided"]
                            branch_normal = pisg_result["normal"]
                            branch_perturbed = pisg_result["perturbed"]
                        else:
                            guided_raw = pisg_result
                        x0_pred = guided_raw.reshape(1, num_nodes, -1, 2)
                else:
                    x0_pred = self._plain_denoise(model, xt, t1, device, edge_index)

                if intermediate_steps is not None:
                    step_record = {
                        "step": i, "seq_id": seq_id, "t": t1.item(),
                        "x0_pred": x0_pred.detach().cpu(),
                    }
                    if branch_normal is not None:
                        step_record["x0_normal"] = branch_normal.softmax(dim=-1).detach().cpu()
                        step_record["x0_perturbed"] = branch_perturbed.softmax(dim=-1).detach().cpu()
                    intermediate_steps.append(step_record)


                if not t2.item == 0:
                    alpha = model.args.m1
                    mask = (x0_pred[..., 1] >= alpha).float()
                    x0 = torch.bernoulli(mask * x0_pred[..., 1].clamp(0, 1))
                    x0_onehot = F.one_hot(
                        x0.long(), num_classes=2
                    ).float()  # [B, N, N, 2]
                    Q_bar = (
                        torch.from_numpy(model.diffusion.Q_bar[t2])
                        .float()
                        .to(x0_onehot.device)
                    )
                    xt_prob = torch.matmul(x0_onehot, Q_bar)  # [B, N, N, 2]
                    xt = torch.bernoulli(xt_prob[..., 1].clamp(0, 1))

            predict_labels = x0_pred[..., 1].float().cpu().detach().numpy().reshape(-1) + 1e-6
            stacked_predict_labels.append(predict_labels)
            seq_id += 1

        predict_labels = np.concatenate(stacked_predict_labels, axis=0)

        all_sampling = model.args.sequential_sampling * model.args.parallel_sampling
        split_predict_labels = np.split(predict_labels, all_sampling)
        solved_solutions = [mis_decode_np(predict_labels, adj_mat, model.args.ep) for predict_labels in split_predict_labels]
        solved_costs = [solved_solution.sum() for solved_solution in solved_solutions]
        best_solved_cost = np.max(solved_costs)
        best_solved_id = np.argmax(solved_costs)

        gt_cost = node_labels.cpu().numpy().sum()
        gap = (gt_cost - best_solved_cost) / gt_cost * 100

        guided_gap, g_best_solved_cost = -1., -1.
        # Local Rewrite
        if model.args.rewrite:
            rewrite_alpha = model.args.m2 if model.args.m2 > 0 else alpha
            mask = (x0_pred[..., 1] >= rewrite_alpha).float().squeeze(-1)
            g_best_solution = solved_solutions[best_solved_id]
            g_best_solved_cost = best_solved_cost
            for _ in range(model.args.rewrite_steps):
                g_stacked_predict_labels = []
                g_x0 = torch.from_numpy(g_best_solution).unsqueeze(0).to(device)    # 1, b*n
                if model.args.parallel_sampling > 1:
                    g_x0 = g_x0.repeat(1, model.args.parallel_sampling)

                g_x0_onehot = F.one_hot(g_x0.long(), num_classes=2).float()

                steps_T = int(model.args.diffusion_steps * model.args.rewrite_ratio)

                Q_bar = torch.from_numpy(model.diffusion.Q_bar[steps_T]).float().to(g_x0_onehot.device)

                g_xt_prob = torch.matmul(g_x0_onehot, Q_bar)  # [B, N, 2]

                g_xt = torch.bernoulli(mask*g_xt_prob[..., 1].clamp(0, 1)).to(g_x0_onehot.device)  # [B, N]

                g_xt = g_xt.reshape(-1)
                t = torch.tensor([model.args.diffusion_steps * model.args.rewrite_ratio]).int()

                g_x0 = self.denoise_step(model, g_xt, t, device, edge_index)    # 1, b*n, 1

                mask = (g_x0 >= rewrite_alpha).float().squeeze(-1)
                g_predict_labels = g_x0.float().cpu().detach().numpy().reshape(-1) + 1e-6
                g_stacked_predict_labels.append(g_predict_labels)
                g_predict_labels = np.concatenate(g_stacked_predict_labels, axis=0)

                g_split_predict_labels = np.split(g_predict_labels, model.args.parallel_sampling)
                g_solved_solutions = [mis_decode_np(g_predict_labels, adj_mat, model.args.ep) for g_predict_labels in
                                      g_split_predict_labels]
                g_solved_costs = [g_solved_solution.sum() for g_solved_solution in g_solved_solutions]

                g_best_solved_cost_tmp, g_best_id = np.max(g_solved_costs), np.argmax(g_solved_costs)

                if g_best_solved_cost_tmp > g_best_solved_cost:
                    g_best_solved_cost = g_best_solved_cost_tmp
                    g_best_solution = g_solved_solutions[g_best_id]
                guided_gap = (gt_cost - g_best_solved_cost) / gt_cost * 100

        if model.args.rewrite:
            metrics = {
                f"{split}/rewrite_ratio": float(model.args.rewrite_ratio),
                f"{split}/gap": gap,
                f"{split}/guided_gap": guided_gap,
                f"{split}/gt_cost": gt_cost,
                f"{split}/solved_cost": best_solved_cost,
                f"{split}/guided_solved_cost": g_best_solved_cost,
                f"{split}/time": time.time() - t_start,
            }
        else:
            metrics = {
                f"{split}/gap": gap,
                f"{split}/gt_cost": gt_cost,
                f"{split}/solved_cost": best_solved_cost,
                f"{split}/time": time.time() - t_start,
            }

        if intermediate_steps is not None and len(intermediate_steps) > 0:
            import os, warnings
            output_dir = getattr(model.args, "intermediate_dir", "intermediates")
            os.makedirs(output_dir, exist_ok=True)
            if len(intermediate_steps) > 50:
                warnings.warn(
                    f"Saving {len(intermediate_steps)} intermediate steps.",
                    stacklevel=2,
                )
            rank = getattr(model, "global_rank", 0)
            num_nodes_save = node_labels.shape[0]
            torch.save({
                "metadata": {
                    "task": "mis",
                    "method": "strudico",
                    "num_nodes": num_nodes_save,
                    "num_steps": len(intermediate_steps),
                    "parallel_sampling": model.args.parallel_sampling,
                    "sequential_sampling": model.args.sequential_sampling,
                },
                "edge_index": edge_index.cpu() if edge_index is not None else None,
                "node_labels": node_labels.cpu(),
                "gt_cost": gt_cost,
                "steps": intermediate_steps,
            }, os.path.join(output_dir, f"intermediates_rank{rank}_batch{batch_idx}.pt"))

        solved_cost_key = f"{split}/solved_cost"
        for k, v in metrics.items():
            model.log(
                k,
                v,
                on_epoch=True,
                sync_dist=True,
                prog_bar=(k == solved_cost_key),
            )

        return metrics

    def denoise_step(self, model, xt, t, device, edge_index=None):
        with torch.no_grad():
            xt_scale = xt * 2 - 1
            xt_scale = xt_scale * (
                1.0 + 0.05 * torch.rand_like(xt.float(), device=device)
            )   # b*n (700)

            # [b*n, 2]
            x0_pred = model.forward(
                xt_scale.to(device),
                t.float().to(device),
                edge_index.long().to(device) if edge_index is not None else None,
            )

            x0_pred_prob = x0_pred.reshape((1, xt.shape[0], -1, 2)).softmax(dim=-1)

            return x0_pred_prob[..., 1]

    def _plain_denoise(self, model, xt, t, device, edge_index=None):
        xt_scale = xt.float() * 2 - 1
        xt_scale = xt_scale * (1.0 + 0.05 * torch.rand_like(xt_scale, device=device))
        xt_flat = xt_scale.reshape(-1)
        x0_pred = model(
            xt_flat,
            t.float().to(device),
            edge_index.long().to(device) if edge_index is not None else None,
        )
        num_nodes = xt_flat.shape[0]
        return x0_pred.reshape((1, num_nodes, -1, 2)).softmax(dim=-1)

    @staticmethod
    def _duplicate_sparse_edge_index(edge_index, num_nodes):
        if edge_index is None:
            return None
        edge_index_bad = edge_index.clone()
        edge_index_bad[0] += num_nodes
        edge_index_bad[1] += num_nodes
        return torch.cat([edge_index, edge_index_bad], dim=1)

    def compute_pisg_scale(self, step_idx, total_steps, base_scale,
                           schedule="constant", active_steps=None):
        if active_steps is not None and step_idx not in active_steps:
            return 0.0
        if total_steps <= 1:
            return 0.0
        if schedule == "constant":
            return base_scale
        if schedule == "linear":
            return base_scale * step_idx / (total_steps - 1)
        if schedule == "reverse-linear":
            return base_scale * (total_steps - 1 - step_idx) / (total_steps - 1)
        raise ValueError(f"Unsupported PISG scale schedule: {schedule}")

    def pisg_denoise_step(self, model, xt, t, device, edge_index=None,
                          pisg_scale=1.0, pisg_perturb="uniform",
                          return_branches=False):
        with torch.no_grad():
            xt_scale = xt.float() * 2 - 1
            xt_scale = xt_scale * (1.0 + 0.05 * torch.rand_like(xt_scale, device=device))
            xt_scale = xt_scale.reshape(-1)
            num_nodes = xt_scale.shape[0]
            xt_dup = torch.cat([xt_scale, xt_scale], dim=0)
            edge_index_dup = self._duplicate_sparse_edge_index(edge_index, num_nodes)

            output = model.model(
                xt_dup.to(device),
                t.float().to(device),
                edge_index=edge_index_dup.long().to(device),
                pisg_mode=True,
                pisg_perturb=pisg_perturb,
            )  # (2N, 2)

            out_good = output[:num_nodes]
            out_bad  = output[num_nodes:]

            fusion_space = getattr(model.args, "pisg_fusion_space", "probability")

            if return_branches:
                # guided: always probability space (MIS applies softmax/normalize here).
                # normal/perturbed: raw logits (caller converts via .softmax(dim=-1)).
                if fusion_space == "logit":
                    guided = (out_good + pisg_scale * (out_good - out_bad)).softmax(dim=-1)
                else:
                    prob_good = out_good.softmax(dim=-1)
                    prob_bad  = out_bad.softmax(dim=-1)
                    guided = (prob_good + pisg_scale * (prob_good - prob_bad)).clamp_min(0.0)
                    denom = guided.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    guided = guided / denom
                return {"guided": guided, "normal": out_good, "perturbed": out_bad}

            if fusion_space == "logit":
                guided_logit = out_good + pisg_scale * (out_good - out_bad)
                return guided_logit.softmax(dim=-1)

            prob_good = out_good.softmax(dim=-1)
            prob_bad  = out_bad.softmax(dim=-1)
            guided = (prob_good + pisg_scale * (prob_good - prob_bad)).clamp_min(0.0)
            denom  = guided.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            return guided / denom  # (N, 2)
