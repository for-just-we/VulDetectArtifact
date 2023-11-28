import torch
from torch import Tensor
import torch.nn as nn

from graph.explainers.approaches.explain_util import DeepLiftUtil
from graph.explainers.approaches.common import WalkBase

EPS = 1e-15


class DeepLIFT(WalkBase):
    def __init__(self, model: nn.Module, epochs=0, lr=0, explain_graph=True, molecule=False, device="cpu"):
        super().__init__(model=model, epochs=epochs, lr=lr, explain_graph=explain_graph, molecule=molecule, device=device)

    def forward(self, x: Tensor, edge_index: Tensor, **kwargs):
        # --- run the model once ---
        super().forward(x=x, edge_index=edge_index, **kwargs)
        self.model.eval()

        # --- add shap calculation hook ---
        shap = DeepLiftUtil(self.model)
        self.model.apply(shap._register_hooks)

        inp_with_ref = torch.cat([x, torch.zeros(x.shape, device=self.device, dtype=torch.float)], dim=0).requires_grad_(True)
        edge_index_with_ref = torch.cat([edge_index, edge_index + x.shape[0]], dim=1)  # 复制一份图
        batch = torch.arange(2, dtype=torch.long, device=self.device).view(2, 1).repeat(1, x.shape[0]).reshape(-1)

        out = self.model(x=inp_with_ref, edge_index=edge_index_with_ref, batch=batch)  # [2, 2]
        ex_label = 1
        f = torch.unbind(out[:, ex_label])

        (m, ) = torch.autograd.grad(outputs=f, inputs=inp_with_ref, retain_graph=True)  # m: [node_num * 2, node_features]
        inp, inp_ref = torch.chunk(inp_with_ref, 2)
        attr_wo_relu = (torch.chunk(m, 2)[0] * (inp - inp_ref)).sum(1)

        mask = attr_wo_relu.squeeze()  # [num_nodes]
        # mask = (mask[self_loop_edge_index[0]] + mask[self_loop_edge_index[1]]) / 2  # [edge_num + node_num]
        # mask = self.control_sparsity(mask, kwargs.get('sparsity'))
        # Store related predictions for further evaluation.
        shap._remove_hooks()
        sorted_results = mask.sort(descending=True)

        # with torch.no_grad():
        #     with self.connect_mask(self):
        #         related_preds = self.eval_related_pred(x, edge_index, masks, **kwargs)

        return mask.detach(), sorted_results.indices.cpu(), edge_index.cpu()