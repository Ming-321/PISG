import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusion.consistency.meta import MetaConsistency
from diffusion.utils.diffusion_schedulers import InferenceSchedule
from diffusion.utils.tsp_utils import TSPEvaluator, batched_two_opt_torch, merge_sub_tours

import time


class TSPConsistency(MetaConsistency):
    def __init__(
        self,
        args,
        sigma_max=1000,
        sigma_min=0,
        weight_schedule="uniform",
        boundary_func="truncate",
    ):
        super(TSPConsistency, self).__init__(
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            weight_schedule=weight_schedule,
            boundary_func=boundary_func,
        )
        self.args = args

    def consistency_losses(self, model, batch):
        edge_index = None
        if model.sparse:
            _, graph_data, point_indicator, edge_indicator, _ = batch
            route_edge_flags = graph_data.edge_attr
            # points: B*N, 2
            # edge_index: B*E, 2
            # adj_matrix: B, N*k
            points = graph_data.x
            edge_index = graph_data.edge_index
            num_edges = edge_index.shape[1]
            batch_size = point_indicator.shape[0]
            adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))
            t = torch.randint(
                1,
                model.diffusion.T + 1,
                [point_indicator.shape[0]],
                device=points.device,
            )
            t2 = (model.args.alpha * t).int()
        else:
            # points: B, N, 2
            # adj_matrix: B, N, N
            _, points, adj_matrix, _ = batch
            batch_size = points.shape[0]
            t = torch.randint(
                1, model.diffusion.T + 1, [points.shape[0]], device=points.device
            )
            t2 = (model.args.alpha * t).int()

        x0 = F.one_hot(
            adj_matrix.long(), num_classes=2
        ).float()  # B, N, N, 2 if not sparse else B, N*N, 2
        if model.sparse:
            x0 = x0.unsqueeze(1)  # B, 1, N*K, 2

        x_t = model.diffusion.sample(x0, t.cpu().numpy())
        x_t2 = model.diffusion.sample(x0, t2.cpu().numpy())

        if model.sparse:
            t = t.reshape(-1, 1).repeat(1, adj_matrix.shape[1]).reshape(-1)
            t2 = t2.reshape(-1, 1).repeat(1, adj_matrix.shape[1]).reshape(-1)
            x_t = x_t.reshape(-1)
            x_t2 = x_t2.reshape(-1)
            adj_matrix = adj_matrix.reshape(-1)
            points = points.reshape(-1, 2)
            edge_index = edge_index.float().to(adj_matrix.device).reshape(2, -1)

        model_output, denoise = self.denoise(model, points, x_t, t, edge_index, x0)
        model_output2, denoise2 = self.denoise(model, points, x_t2, t2, edge_index, x0)
        adj_matrix = adj_matrix.long()
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(denoise, adj_matrix) + loss_func(denoise2, adj_matrix)

        return loss

    def denoise(self, model, points, x_t, t, edge_index, x0):
        x_t = x_t * 2 - 1
        x_t = x_t * (1.0 + 0.05 * torch.rand_like(x_t))
        model_output = model(
            points, x_t, t, edge_index
        )  # B, 2, N, N if not sparse else B*N*K, 2; k is the sparse factor

        c_skip, c_out = [
            self.append_dims(x, model_output.ndim)
            for x in self.get_scalings_for_boundary_condition(t)
        ]  # B, 1, 1, 1 or B*N*K, 1

        if not model.sparse:
            x0 = x0.permute(0, 3, 1, 2)
        else:
            x0 = x0.reshape((-1, 2))
        denoise = c_out * model_output + c_skip * x0

        return model_output, denoise

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

    @staticmethod
    def _duplicate_sparse_edge_index(edge_index, num_nodes):
        if edge_index is None:
            return None
        edge_index_bad = edge_index.clone()
        edge_index_bad[0] += num_nodes
        edge_index_bad[1] += num_nodes
        return torch.cat([edge_index, edge_index_bad], dim=1)

    @staticmethod
    def _raw_logit_to_prob(raw, sparse):
        if not sparse:
            return raw.permute(0, 2, 3, 1).contiguous().softmax(-1)
        return raw.reshape((-1, 2)).softmax(dim=-1)

    @staticmethod
    def _normalize_guided_probabilities(prob):
        denom = prob.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return prob / denom

    def pisg_denoise_step(
        self,
        model,
        points,
        xt,
        t,
        device,
        edge_index=None,
        pisg_scale=1.0,
        pisg_perturb="uniform",
        sparse=False,
        return_branches=False,
    ):
        with torch.no_grad():
            xt = xt.float()
            xt_scale = xt * 2 - 1
            xt_scale = xt_scale * (1.0 + 0.05 * torch.rand_like(xt, device=device))

            points_dup = torch.cat([points, points], dim=0)
            xt_scale_dup = torch.cat([xt_scale, xt_scale], dim=0)

            if sparse:
                t_for_forward = t
            else:
                batch_size = points.shape[0]
                if t.numel() == 1:
                    t = t.repeat(batch_size)
                elif t.numel() != batch_size:
                    raise ValueError(
                        f"Unexpected dense timestep shape: {tuple(t.shape)} for batch size {batch_size}"
                    )
                t_for_forward = torch.cat([t, t], dim=0)

            if sparse and edge_index is not None:
                edge_index_dup = self._duplicate_sparse_edge_index(edge_index, points.shape[0])
            else:
                edge_index_dup = edge_index

            if pisg_perturb == "input-noise":
                half = points.shape[0] if not sparse else xt_scale.shape[0]
                xt_scale_dup[half:] = xt_scale_dup[half:] + 0.5 * torch.randn_like(
                    xt_scale_dup[half:]
                )
                output = model.model(
                    points_dup.float().to(device),
                    t_for_forward.float().to(device),
                    xt_scale_dup.to(device),
                    edge_index_dup.long().to(device) if edge_index_dup is not None else None,
                    pisg_mode=False,
                )
            else:
                output = model.model(
                    points_dup.float().to(device),
                    t_for_forward.float().to(device),
                    xt_scale_dup.to(device),
                    edge_index_dup.long().to(device) if edge_index_dup is not None else None,
                    pisg_mode=True,
                    pisg_perturb=pisg_perturb,
                )

            out_good, out_bad = output.chunk(2, dim=0)
            fusion_space = getattr(model.args, "pisg_fusion_space", "probability")

            if return_branches:
                # guided: logit space if fusion_space=="logit", else probability space.
                # normal/perturbed: always raw logits (caller converts via _raw_logit_to_prob).
                if fusion_space == "logit":
                    guided = out_good + pisg_scale * (out_good - out_bad)
                else:
                    prob_good = self._raw_logit_to_prob(out_good, sparse)
                    prob_bad = self._raw_logit_to_prob(out_bad, sparse)
                    guided = (prob_good + pisg_scale * (prob_good - prob_bad)).clamp_min(0.0)
                    guided = self._normalize_guided_probabilities(guided)
                return {"guided": guided, "normal": out_good, "perturbed": out_bad}

            if fusion_space == "logit":
                return out_good + pisg_scale * (out_good - out_bad)

            prob_good = self._raw_logit_to_prob(out_good, sparse)
            prob_bad = self._raw_logit_to_prob(out_bad, sparse)
            guided = (prob_good + pisg_scale * (prob_good - prob_bad)).clamp_min(0.0)
            return self._normalize_guided_probabilities(guided)

    def consistency_test_step(self, model, batch, batch_idx, split="test"):
        t_start = time.time()
        edge_index = None
        np_edge_index = None
        device = batch[-1].device
        original_edge_index = None

        torch.cuda.reset_peak_memory_stats(device)


        if not model.sparse:
            real_batch_idx, points, adj_matrix, gt_tour = batch
            np_points = points.cpu().numpy()[0]
            np_gt_tour = gt_tour.cpu().numpy()[0]
        else:
            real_batch_idx, graph_data, point_indicator, edge_indicator, gt_tour = batch
            route_edge_flags = graph_data.edge_attr
            points = graph_data.x
            edge_index = graph_data.edge_index
            original_edge_index = edge_index.clone()
            num_edges = edge_index.shape[1]
            batch_size = point_indicator.shape[0]
            adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))
            points = points.reshape((-1, 2))
            edge_index = edge_index.reshape((2, -1))
            np_points = points.cpu().numpy()
            np_gt_tour = gt_tour.cpu().numpy().reshape(-1)
            np_edge_index = edge_index.cpu().numpy()

        if model.args.parallel_sampling > 1:

            if not model.sparse:
                points = points.repeat(model.args.parallel_sampling, 1, 1)
            else:
                points = points.repeat(model.args.parallel_sampling, 1)
                edge_index = model.duplicate_edge_index(
                    model.args.parallel_sampling, edge_index, np_points.shape[0], device
                )

        # Initialize with original diffusion
        stacked_tours = []

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

        for _ in range(model.args.sequential_sampling):

            xt = torch.zeros_like(adj_matrix.float())
            if model.args.parallel_sampling > 1:
                if not model.sparse:
                    xt = xt.repeat(model.args.parallel_sampling, 1, 1)
                else:
                    xt = xt.repeat(model.args.parallel_sampling, 1)  # [B, E]
                xt = torch.zeros_like(xt)


            xt = (xt > 0).long()

            if model.sparse:
                xt = xt.reshape(-1)  # [E]

            # Diffusion iterations
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

            alpha = model.args.m1
            for i in range(steps):
                t1, t2 = time_schedule(i)
                t1 = torch.tensor([t1], device=device).float()

                if getattr(model.args, "pisg", False):
                    current_pisg_scale = self.compute_pisg_scale(
                        step_idx=i,
                        total_steps=steps,
                        base_scale=getattr(model.args, "pisg_scale", 1.0),
                        schedule=getattr(model.args, "pisg_scale_schedule", "constant"),
                        active_steps=pisg_active_steps,
                    )
                    branch_normal = branch_perturbed = None
                    if current_pisg_scale == 0.0:
                        x0_pred = self.denoise_step(
                            model, points, xt, t1, device, edge_index, return_prob=True
                        )
                    else:
                        pisg_result = self.pisg_denoise_step(
                            model,
                            points,
                            xt,
                            t1,
                            device,
                            edge_index=edge_index,
                            pisg_scale=current_pisg_scale,
                            pisg_perturb=getattr(model.args, "pisg_perturb", "uniform"),
                            sparse=model.sparse,
                            return_branches=(intermediate_steps is not None),
                        )
                        if isinstance(pisg_result, dict):
                            guided_raw = pisg_result["guided"]
                            branch_normal = pisg_result["normal"]
                            branch_perturbed = pisg_result["perturbed"]
                        else:
                            guided_raw = pisg_result
                        if getattr(model.args, "pisg_fusion_space", "probability") == "logit":
                            x0_pred = self._raw_logit_to_prob(guided_raw, model.sparse)
                        else:
                            x0_pred = guided_raw
                else:
                    branch_normal = branch_perturbed = None
                    x0_pred = self.denoise_step(
                        model, points, xt, t1, device, edge_index, return_prob=True
                    )

                if intermediate_steps is not None:
                    step_record = {
                        "step": i, "seq_id": seq_id, "t": t1.item(),
                        "x0_pred": x0_pred.detach().cpu(),
                    }
                    if branch_normal is not None:
                        step_record["x0_normal"] = self._raw_logit_to_prob(
                            branch_normal, model.sparse
                        ).detach().cpu()
                        step_record["x0_perturbed"] = self._raw_logit_to_prob(
                            branch_perturbed, model.sparse
                        ).detach().cpu()
                    intermediate_steps.append(step_record)

                if not t2 == 0:
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

                    xt = (torch.bernoulli(xt_prob[..., 1].clamp(0, 1)).bool() | xt.bool()).float()  # [B, N, N]

            adj_mat = x0_pred[..., 1].cpu().detach().numpy() + 1e-6  # [B, N, N]

            tours, merge_iterations = merge_sub_tours(
                adj_mat, np_points, np_edge_index,
                sparse_graph=model.sparse,
                parallel_sampling=model.args.parallel_sampling,
                alpha=float(alpha),
                seed=getattr(model.args, "stitch_seed", None),
            )
            # Refine using 2-opt
            solved_tours, ns = batched_two_opt_torch(
                np_points.astype("float64"),
                np.array(tours).astype("int64"),
                max_iterations=model.args.two_opt_iterations,
                device=device,
            )

            stacked_tours.append(solved_tours)
            seq_id += 1

        tsp_solver = TSPEvaluator(np_points)  # np_points: [N, 2] ndarray
        gt_cost = tsp_solver.evaluate(np_gt_tour)  # np_gt_tour: [N+1] ndarray

        solved_tours = np.concatenate(stacked_tours, axis=0)  # [B, N+1] ndarray

        all_solved_costs = [
            tsp_solver.evaluate(solved_tours[i]) for i in range(solved_tours.shape[0])
        ]
        best_solved_cost, best_id = np.min(all_solved_costs), np.argmin(
            all_solved_costs
        )
        gap = (best_solved_cost - gt_cost) / gt_cost * 100

        g_best_tour = solved_tours[best_id]  # [N+1] ndarray

        guided_gap, g_best_solved_cost = -1.0, -1.0
        # Local Rewrite
        if model.args.rewrite:
            rewrite_alpha = model.args.m2 if model.args.m2 > 0 else alpha
            mask = (x0_pred[..., 1] >= rewrite_alpha).float()
            g_best_solved_cost = best_solved_cost
            for _ in range(model.args.rewrite_steps):
                g_stacked_tours = []
                # optimal adjacent matrix
                g_x0 = model.tour2adj(
                    g_best_tour,
                    np_points,
                    model.sparse,
                    model.args.sparse_factor,
                    original_edge_index,
                )

                g_x0 = g_x0.unsqueeze(0).to(device)  # [1, N, N] or [1, N]
                if model.args.parallel_sampling > 1:
                    if not model.sparse:
                        g_x0 = g_x0.repeat(
                            model.args.parallel_sampling, 1, 1
                        )  # [1, N ,N] -> [B, N, N]
                    else:
                        g_x0 = g_x0.repeat(model.args.parallel_sampling, 1)

                if model.sparse:
                    g_x0 = g_x0.reshape(-1)

                g_x0_onehot = F.one_hot(
                    g_x0.long(), num_classes=2
                ).float()  # [B, N, N, 2]

                steps_T = int(model.args.diffusion_steps * model.args.rewrite_ratio)

                Q_bar = (
                    torch.from_numpy(model.diffusion.Q_bar[steps_T])
                    .float()
                    .to(g_x0_onehot.device)
                )
                g_xt_prob = torch.matmul(g_x0_onehot, Q_bar)  # [B, N, N, 2]

                t = torch.tensor(
                    [model.args.diffusion_steps * model.args.rewrite_ratio]
                ).int()

                # add noise for the steps_T samples, namely rewrite
                g_xt = torch.bernoulli(mask * g_xt_prob[..., 1].clamp(0, 1))  # [B, N, N]
                g_x0 = self.denoise_step(model, points, g_xt, t, device, edge_index)
                mask = (g_x0 >= rewrite_alpha).float()

                g_adj_mat = g_x0.float().cpu().detach().numpy() + 1e-6
                if model.args.save_numpy_heatmap:
                    model.run_save_numpy_heatmap(
                        g_adj_mat, np_points, real_batch_idx, split
                    )


                g_tours, g_merge_iterations = merge_sub_tours(
                    g_adj_mat, np_points, np_edge_index,
                    sparse_graph=model.sparse,
                    parallel_sampling=model.args.parallel_sampling,
                    alpha=float(rewrite_alpha),
                    seed=getattr(model.args, "stitch_seed", None),
                )

                # Refine using 2-opt
                g_solved_tours, g_ns = batched_two_opt_torch(
                    np_points.astype("float64"),
                    np.array(g_tours).astype("int64"),
                    max_iterations=model.args.two_opt_iterations,
                    device=device,
                )

                for g_tour in g_solved_tours:
                    g_stacked_tours.append(g_tour)

                g_solved_tours = np.array(g_stacked_tours)

                g_all_solved_costs = [
                    tsp_solver.evaluate(g_solved_tours[i])
                    for i in range(g_solved_tours.shape[0])
                ]
                g_best_solved_cost_tmp, g_best_id = np.min(
                    g_all_solved_costs
                ), np.argmin(g_all_solved_costs)

                if g_best_solved_cost_tmp < g_best_solved_cost:
                    g_best_solved_cost = g_best_solved_cost_tmp
                    g_best_tour = g_solved_tours[g_best_id]

                guided_gap = (g_best_solved_cost - gt_cost) / gt_cost * 100

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
            num_nodes = np_points.shape[1] if np_points.ndim == 3 else np_points.shape[0]
            torch.save({
                "metadata": {
                    "task": "tsp",
                    "method": "strudico",
                    "num_nodes": num_nodes,
                    "num_steps": len(intermediate_steps),
                    "parallel_sampling": model.args.parallel_sampling,
                    "sequential_sampling": model.args.sequential_sampling,
                },
                "edge_index": edge_index.cpu() if edge_index is not None else None,
                "points": torch.from_numpy(np_points),
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

    def denoise_step(self, model, points, xt, t, device, edge_index=None, return_prob=False):
        with torch.no_grad():
            xt = xt.float()  # b, n, n
            xt_scale = xt * 2 - 1
            xt_scale = xt_scale * (
                1.0 + 0.05 * torch.rand_like(xt.float(), device=device)
            )

            # [b, 2, n, n]
            x0_pred = model.forward(
                points.float().to(device),
                xt_scale.to(device),
                t.float().to(device),
                edge_index.long().to(device) if edge_index is not None else None,
            )
            temperature = getattr(model.args, "logit_temperature", 1.0)
            if temperature != 1.0:
                x0_pred = x0_pred / temperature

            if not model.sparse:
                x0_pred_prob = (
                    x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
                )
            else:
                x0_pred_prob = x0_pred.reshape((-1, 2)).softmax(dim=-1)

            if return_prob:
                return x0_pred_prob
            return x0_pred_prob[..., 1]
