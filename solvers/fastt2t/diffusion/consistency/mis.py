import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
from torch_sparse import SparseTensor
from diffusion.utils.diffusion_schedulers import InferenceSchedule
from diffusion.consistency.meta import MetaConsistency
from diffusion.utils.mis_utils import mis_decode_np


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

    def duplicate_edge_index(self, edge_index, num_nodes):
        if edge_index is None:
            return None
        edge_index_bad = edge_index.clone()
        edge_index_bad[0] += num_nodes
        edge_index_bad[1] += num_nodes
        return torch.cat([edge_index, edge_index_bad], dim=1)

    @staticmethod
    def _cross_entropy_from_probs(prob, target):
        target_onehot = F.one_hot(target.long(), num_classes=2).float()
        return -(target_onehot * prob.clamp_min(1e-12).log()).sum(dim=-1).mean()

    def _pisg_forward(
            self,
            model,
            xt_scale,
            t,
            device,
            edge_index=None,
            pisg_scale=1.0,
            pisg_perturb="uniform",
            return_branches=False,
    ):
        xt_scale_dup = torch.cat([xt_scale, xt_scale], dim=0)

        if t.numel() == 1:
            t_dup = t
        elif t.numel() == xt_scale.shape[0]:
            t_dup = torch.cat([t, t], dim=0)
        elif t.numel() == xt_scale_dup.shape[0]:
            t_dup = t
        else:
            raise ValueError(
                f"Unexpected MIS PISG timestep shape {tuple(t.shape)} for xt shape {tuple(xt_scale.shape)}"
            )

        edge_index_dup = self.duplicate_edge_index(edge_index, xt_scale.shape[0]) if edge_index is not None else None
        if pisg_perturb == "input-noise":
            noise_std = 0.5
            half_n = xt_scale.shape[0]
            xt_scale_dup[half_n:] = xt_scale_dup[half_n:] + noise_std * torch.randn_like(
                xt_scale_dup[half_n:]
            )
            output = model.model(
                xt_scale_dup.to(device),
                t_dup.float().to(device),
                edge_index=edge_index_dup.long().to(device) if edge_index_dup is not None else None,
                pisg_mode=False,
            )
        else:
            output = model.model(
                xt_scale_dup.to(device),
                t_dup.float().to(device),
                edge_index=edge_index_dup.long().to(device) if edge_index_dup is not None else None,
                pisg_mode=True,
                pisg_perturb=pisg_perturb,
            )

        out_good, out_bad = output.chunk(2, dim=0)
        fusion_space = getattr(model.args, "pisg_fusion_space", "logit") if hasattr(model, "args") else "logit"
        if fusion_space == "probability":
            raise NotImplementedError(
                "pisg_fusion_space=probability is not supported for MIS. "
                "Use pisg_fusion_space=logit (default)."
            )
        guided = out_good + pisg_scale * (out_good - out_bad)

        if return_branches:
            return {"guided": guided, "normal": out_good, "perturbed": out_bad}
        return guided

    def pisg_denoise_step(
            self,
            model,
            xt,
            t,
            device,
            edge_index=None,
            pisg_scale=1.0,
            pisg_perturb="uniform",
            return_branches=False,
    ):
        if edge_index is None:
            raise ValueError("edge_index is required for MIS PISG denoising")
        if pisg_scale == 0 and not return_branches:
            return self.denoise_step(model, xt, t, device, edge_index).reshape(-1)

        with torch.no_grad():
            xt = xt.float().reshape(-1)
            xt_scale = xt * 2 - 1
            xt_scale = xt_scale * (
                1.0 + 0.05 * torch.rand_like(xt, device=device)
            )
            guided = self._pisg_forward(
                model,
                xt_scale,
                t,
                device,
                edge_index=edge_index,
                pisg_scale=pisg_scale,
                pisg_perturb=pisg_perturb,
                return_branches=return_branches,
            )
            if return_branches:
                return guided
            fusion_space = getattr(model.args, "pisg_fusion_space", "logit") if hasattr(model, "args") else "logit"
            if fusion_space == "logit":
                return guided
            return guided[..., 1]

    def consistency_test_step(self, model, batch, batch_idx, split='test'):
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

        original_edge_index = edge_index.clone()
        if model.args.parallel_sampling > 1:
            edge_index = model.duplicate_edge_index(
                model.args.parallel_sampling,
                edge_index,
                node_labels.shape[0],
                device,
            )

        save_intermediates = getattr(model.args, "save_intermediates", False)
        intermediate_steps = [] if save_intermediates else None
        seq_id = 0

        for _ in range(model.args.sequential_sampling):
            xt = torch.randn_like(node_labels.float())
            if model.args.parallel_sampling > 1:
                xt = xt.repeat(model.args.parallel_sampling, 1, 1)
                xt = torch.randn_like(xt)

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
                branch_normal = None
                branch_perturbed = None
                if getattr(model.args, "pisg", False):
                    current_pisg_scale = self.compute_pisg_scale(
                        step_idx=i,
                        total_steps=steps,
                        base_scale=getattr(model.args, "pisg_scale", 1.0),
                        schedule=getattr(model.args, "pisg_scale_schedule", "constant"),
                        active_steps=pisg_active_steps,
                    )
                    if current_pisg_scale == 0 and intermediate_steps is None:
                        xt_scale = xt * 2 - 1
                        xt_scale = xt_scale * (
                            1.0 + 0.05 * torch.rand_like(xt.float(), device=device)
                        )
                        x0_pred = model(
                            xt_scale.reshape(-1),
                            t1.float().to(device),
                            edge_index.long().to(device) if edge_index is not None else None,
                        )
                        x0_pred = x0_pred.reshape((1, xt.shape[0], -1, 2)).softmax(dim=-1)
                    else:
                        pisg_out = self.pisg_denoise_step(
                            model,
                            xt,
                            t1,
                            device,
                            edge_index,
                            pisg_scale=current_pisg_scale,
                            pisg_perturb=getattr(model.args, "pisg_perturb", "uniform"),
                            return_branches=(intermediate_steps is not None),
                        )
                        if isinstance(pisg_out, dict):
                            branch_normal = pisg_out["normal"]
                            branch_perturbed = pisg_out["perturbed"]
                            pisg_out = pisg_out["guided"]
                        fusion_space = getattr(model.args, "pisg_fusion_space", "logit")
                        if fusion_space == "logit":
                            x0_pred = pisg_out.reshape((1, xt.shape[0], -1, 2)).softmax(dim=-1)
                        else:
                            x0_pred = torch.stack((1 - pisg_out, pisg_out), dim=-1)
                            x0_pred = x0_pred.reshape((1, xt.shape[0], -1, 2))
                else:
                    xt_scale = xt * 2 - 1
                    xt_scale = xt_scale * (
                        1.0 + 0.05 * torch.rand_like(xt.float(), device=device)
                    )
                    x0_pred = model(
                        xt_scale.reshape(-1),
                        t1.float().to(device),
                        edge_index.long().to(device) if edge_index is not None else None,
                    )
                    temperature = getattr(model.args, "logit_temperature", 1.0)
                    if temperature != 1.0:
                        x0_pred = x0_pred / temperature
                    x0_pred = x0_pred.reshape((1, xt.shape[0], -1, 2)).softmax(dim=-1)

                if intermediate_steps is not None:
                    step_record = {
                        "step": i,
                        "seq_id": seq_id,
                        "t": t1.item(),
                        "x0_pred": x0_pred.detach().cpu(),
                    }
                    if branch_normal is not None:
                        step_record["x0_normal"] = branch_normal.reshape(
                            (1, xt.shape[0], -1, 2)
                        ).softmax(dim=-1).detach().cpu()
                        step_record["x0_perturbed"] = branch_perturbed.reshape(
                            (1, xt.shape[0], -1, 2)
                        ).softmax(dim=-1).detach().cpu()
                    intermediate_steps.append(step_record)

                # predict_labels = x0_pred[..., 1].float().cpu().detach().numpy().reshape(-1) + 1e-6  # 770,
                # stacked_predict_labels.append(predict_labels)

                if not t2.item == 0:
                    x0 = torch.bernoulli(x0_pred[..., 1].clamp(0, 1))
                    x0_onehot = F.one_hot(
                        x0.long(), num_classes=2
                    ).float()  # [B, N, N, 2]
                    Q_bar = (
                        torch.from_numpy(model.diffusion.Q_bar[t2])
                        .float()
                        .to(x0_onehot.device)
                    )
                    xt_prob = torch.matmul(x0_onehot, Q_bar)  # [B, N, N, 2]
                    xt = torch.bernoulli(xt_prob[..., 1].clamp(0, 1))  # [B, N, N]

            predict_labels = x0_pred[..., 1].float().cpu().detach().numpy().reshape(-1) + 1e-6  # 770,
            stacked_predict_labels.append(predict_labels)
            seq_id += 1

        predict_labels = np.concatenate(stacked_predict_labels, axis=0)  # 770,
        # import time
        # b = time.time()

        all_sampling = model.args.sequential_sampling * model.args.parallel_sampling
        split_predict_labels = np.split(predict_labels, all_sampling)
        # print(len(split_predict_labels), split_predict_labels[0].shape)
        solved_solutions = [mis_decode_np(predict_labels, adj_mat) for predict_labels in split_predict_labels]
        solved_costs = [solved_solution.sum() for solved_solution in solved_solutions]
        best_solved_cost = np.max(solved_costs)
        best_solved_id = np.argmax(solved_costs)

        gt_cost = node_labels.cpu().numpy().sum()
        gap = (best_solved_cost - gt_cost) / gt_cost * 100

        guided_gap, g_best_solved_cost = -1., -1.
        # print(time.time() - b)
        # input()

        # Local Rewrite
        if model.args.rewrite:
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

                g_xt = torch.bernoulli(g_xt_prob[..., 1].clamp(0, 1)).to(g_x0_onehot.device)  # [B, N]

                g_xt = g_xt.reshape(-1)
                t = torch.tensor([model.args.diffusion_steps * model.args.rewrite_ratio]).int()

                if model.args.guided:
                    if getattr(model.args, "pisg_tg", False):
                        g_x0 = self.guided_pisg_denoise_step(
                            model,
                            g_xt_prob,
                            g_x0.reshape(-1),
                            t,
                            device,
                            edge_index,
                            pisg_scale=getattr(model.args, "pisg_scale", 1.0),
                            pisg_perturb=getattr(model.args, "pisg_perturb", "uniform"),
                        )
                    else:
                        g_x0 = self.guided_denoise_step(model, g_xt_prob, g_x0.reshape(-1), t, device, edge_index)
                else:
                    if getattr(model.args, "pisg_tg", False):
                        g_pisg = self.pisg_denoise_step(
                            model,
                            g_xt,
                            t,
                            device,
                            edge_index=edge_index,
                            pisg_scale=getattr(model.args, "pisg_scale", 1.0),
                            pisg_perturb=getattr(model.args, "pisg_perturb", "uniform"),
                        )
                        fusion_space = getattr(model.args, "pisg_fusion_space", "logit")
                        if fusion_space == "logit":
                            g_x0 = g_pisg.reshape((1, g_xt.shape[0], -1, 2)).softmax(dim=-1)[..., 1]
                        else:
                            g_x0 = g_pisg.reshape((1, g_xt.shape[0], -1))
                    else:
                        g_x0 = self.denoise_step(model, g_xt, t, device, edge_index)    # 1, b*n, 1

                g_predict_labels = g_x0.float().cpu().detach().numpy().reshape(-1) + 1e-6
                g_stacked_predict_labels.append(g_predict_labels)
                g_predict_labels = np.concatenate(g_stacked_predict_labels, axis=0)

                g_split_predict_labels = np.split(g_predict_labels, model.args.parallel_sampling * 2
                        if model.args.guided
                        else model.args.parallel_sampling)
                g_solved_solutions = [mis_decode_np(g_predict_labels, adj_mat) for g_predict_labels in
                                      g_split_predict_labels]
                g_solved_costs = [g_solved_solution.sum() for g_solved_solution in g_solved_solutions]

                g_best_solved_cost_tmp, g_best_id = np.max(g_solved_costs), np.argmax(g_best_solved_cost)

                if g_best_solved_cost_tmp > g_best_solved_cost:
                    g_best_solved_cost = g_best_solved_cost_tmp
                    g_best_solution = g_solved_solutions[g_best_id]
                guided_gap = (g_best_solved_cost - gt_cost) / gt_cost * 100

        if model.args.rewrite:
            metrics = {
                f"{split}/rewrite_ratio": float(model.args.rewrite_ratio),
                f"{split}/gap": gap,
                f"{split}/guided_gap": guided_gap,
                f"{split}/gt_cost": gt_cost,
                f"{split}/guided_solved_cost": g_best_solved_cost,
            }
        else:
            metrics = {
                f"{split}/gap": gap,
                f"{split}/gt_cost": gt_cost,
            }

        for k, v in metrics.items():
            model.log(k, v, on_epoch=True, sync_dist=True)
        model.log(f"{split}/solved_cost", best_solved_cost, prog_bar=True, on_epoch=True, sync_dist=True)

        if intermediate_steps is not None:
            import os
            import warnings

            output_dir = getattr(model.args, "intermediate_dir", "intermediates")
            os.makedirs(output_dir, exist_ok=True)
            original_num_nodes = node_labels.shape[0]

            if len(intermediate_steps) > 50:
                warnings.warn(
                    f"Saving {len(intermediate_steps)} intermediate steps. "
                    "This may use significant memory and disk space.",
                    stacklevel=2,
                )

            rank = getattr(model, "global_rank", 0)
            edge_index_cpu = original_edge_index.cpu()
            torch.save({
                "metadata": {
                    "task": "mis",
                    "sparse": True,
                    "num_nodes": original_num_nodes,
                    "num_steps": len(intermediate_steps),
                    "parallel_sampling": model.args.parallel_sampling,
                    "sequential_sampling": model.args.sequential_sampling,
                },
                "edge_index": edge_index_cpu,
                "node_labels": node_labels.cpu(),
                "gt_cost": gt_cost,
                "steps": intermediate_steps,
            }, os.path.join(output_dir, f"intermediates_rank{rank}_batch{batch_idx}.pt"))

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

            x0_pred_prob = x0_pred.reshape((1, xt.shape[0], -1, 2)).softmax(dim=-1)     # 1, b*n, 1, 2
            # x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)

            return x0_pred_prob[..., 1]

    def guided_denoise_step(self, model, xt_prob, x0, t, device, edge_index):
        torch.set_grad_enabled(True)

        xt_prob.requires_grad = True
        xt = xt_prob[..., 1].reshape(-1)    # b*n
        # xt.requires_grad = True

        with torch.inference_mode(False):
            xt_scale = xt * 2 - 1
            xt_scale = xt_scale * (1.0 + 0.05 * torch.rand_like(xt.float(), device=device))

            x0_pred = model(
                xt_scale,   # b*n
                t.float().to(device),   # 1
                edge_index.long().to(device) if edge_index is not None else None,   # 2, E(73742)
            )   # b*n, 2

            loss_func = nn.CrossEntropyLoss()

            bce_loss = loss_func(x0_pred, x0.long())

            p_theta = x0_pred.reshape((1, xt.shape[0], -1, 2)).softmax(dim=-1)

            num_nodes = xt.shape[0]
            adj_matrix = SparseTensor(
                row=edge_index[0],
                col=edge_index[1],
                value=torch.ones_like(edge_index[0].float()),
                sparse_sizes=(num_nodes, num_nodes),
            ).to_dense()
            adj_matrix.fill_diagonal_(0)
            pred_nodes = p_theta[..., 1].squeeze(0)

            f_mis = -pred_nodes.sum()
            g_mis = adj_matrix @ pred_nodes
            g_mis = (pred_nodes * g_mis).sum()
            cost_est = f_mis + 0.5 * g_mis

            # loss = 2 * bce_loss + 2 * cost_est    # for er
            # loss = 1 * bce_loss + 10 * cost_est    # for sat
            loss = self.args.c1 * bce_loss + self.args.c2 * cost_est
            loss.backward()

        torch.set_grad_enabled(False)
        assert xt_prob.grad is not None

        with torch.no_grad():
            # xt_grad = (
            #     torch.zeros(xt.shape, device=xt.device).unsqueeze(-1).repeat(1, 1, 2)   # b, n, 2
            # )
            # xt_grad[..., 1] = xt.grad
            # xt_grad[..., 0] = -xt.grad

            # scale = 0.5
            xt_prob.grad = xt_prob.grad.clamp(-1, 1)    # without this, exp(grad) explodes!
            p_phi = torch.exp(-xt_prob.grad)     # b, n, 2
            xt_prob_u = (xt_prob * p_phi) / torch.sum(
                (xt_prob * p_phi + 1e-6), dim=-1, keepdim=True
            )

            xt_u = torch.bernoulli(xt_prob_u[..., 1].clamp(0, 1))
            xt_u = xt_u.float().reshape(-1)  # b * n
            xt_scale_u = xt_u * 2 - 1
            xt_scale_u = xt_scale_u * (
                    1.0 + 0.05 * torch.rand_like(xt_u.float(), device=device)
            )

            x0_pred_u = model.forward(
                xt_scale_u.to(device),  # b*n
                t.float().to(device),   # 1
                edge_index.long().to(device) if edge_index is not None else None,   # 2, E
            )

            x0_pred_prob_u = x0_pred_u.reshape((1, xt.shape[0], -1, 2)).softmax(dim=-1)
            output = torch.cat((p_theta[..., 1], x0_pred_prob_u[..., 1]), dim=0)

            return output

    def guided_pisg_denoise_step(
            self,
            model,
            xt_prob,
            x0,
            t,
            device,
            edge_index,
            pisg_scale=1.0,
            pisg_perturb="uniform",
    ):
        torch.set_grad_enabled(True)

        xt_prob.requires_grad = True
        xt = xt_prob[..., 1].reshape(-1)

        with torch.inference_mode(False):
            xt_scale = xt * 2 - 1
            xt_scale = xt_scale * (1.0 + 0.05 * torch.rand_like(xt.float(), device=device))

            pisg_out = self._pisg_forward(
                model,
                xt_scale,
                t,
                device,
                edge_index=edge_index,
                pisg_scale=pisg_scale,
                pisg_perturb=pisg_perturb,
            )
            fusion_space = getattr(model.args, "pisg_fusion_space", "logit")
            loss_func = nn.CrossEntropyLoss()
            if fusion_space == "logit":
                x0_pred = pisg_out
                bce_loss = loss_func(x0_pred, x0.long())
                p_theta = x0_pred.reshape((1, xt.shape[0], -1, 2)).softmax(dim=-1)
            else:
                p_theta = pisg_out.reshape((1, xt.shape[0], -1, 2))
                bce_loss = self._cross_entropy_from_probs(p_theta, x0)

            num_nodes = xt.shape[0]
            adj_matrix = SparseTensor(
                row=edge_index[0],
                col=edge_index[1],
                value=torch.ones_like(edge_index[0].float()),
                sparse_sizes=(num_nodes, num_nodes),
            ).to_dense()
            adj_matrix.fill_diagonal_(0)
            pred_nodes = p_theta[..., 1].squeeze(0)

            f_mis = -pred_nodes.sum()
            g_mis = adj_matrix @ pred_nodes
            g_mis = (pred_nodes * g_mis).sum()
            cost_est = f_mis + 0.5 * g_mis

            loss = self.args.c1 * bce_loss + self.args.c2 * cost_est
            loss.backward()

        torch.set_grad_enabled(False)
        assert xt_prob.grad is not None

        with torch.no_grad():
            xt_prob.grad = xt_prob.grad.clamp(-1, 1)
            p_phi = torch.exp(-xt_prob.grad)
            xt_prob_u = (xt_prob * p_phi) / torch.sum(
                (xt_prob * p_phi + 1e-6), dim=-1, keepdim=True
            )

            xt_u = torch.bernoulli(xt_prob_u[..., 1].clamp(0, 1)).float().reshape(-1)
            xt_scale_u = xt_u * 2 - 1
            xt_scale_u = xt_scale_u * (
                    1.0 + 0.05 * torch.rand_like(xt_u.float(), device=device)
            )

            pisg_out_u = self._pisg_forward(
                model,
                xt_scale_u,
                t,
                device,
                edge_index=edge_index,
                pisg_scale=pisg_scale,
                pisg_perturb=pisg_perturb,
            )
            if fusion_space == "logit":
                x0_pred_prob_u = pisg_out_u.reshape((1, xt.shape[0], -1, 2)).softmax(dim=-1)
            else:
                x0_pred_prob_u = pisg_out_u.reshape((1, xt.shape[0], -1, 2))
            output = torch.cat((p_theta[..., 1], x0_pred_prob_u[..., 1]), dim=0)

            return output
