from tap import Tap
import torch
from torch import Tensor

from torch_geometric.data import Batch, Data
from torch.nn.functional import cross_entropy
from graph.explainers.approaches.common import ExplainerBase

class XParser(Tap):
    vis: bool = False
    lr: float = 0.001
    epoch: int = 300
    sparsity: float = 0.5
    walk: bool = False
    debug: bool = False
    nolabel: bool = False
    list_sample: bool = False

x_args = XParser().parse_args(known_only=True)


EPS = 1e-15


class GNNExplainer(ExplainerBase):
    r"""The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and small subsets node features that play a crucial role in a
    GNN’s node-predictions.

    .. note::

        For an example of using GNN-Explainer, see `examples/gnn_explainer.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        gnn_explainer.py>`_.

    Args:
        model (torch.nn.Module): The GNN module to explain.
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        log (bool, optional): If set to :obj:`False`, will not log any learning
            progress. (default: :obj:`True`)
    """

    coeffs = {
        'edge_size': 0.005,
        'node_feat_size': 1.0,
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
    }

    def __init__(self, model, epochs=x_args.epoch, lr=x_args.lr, explain_graph=True, molecule=False, device="cpu"):
        super(GNNExplainer, self).__init__(model, epochs, lr, explain_graph, molecule, device)

    def __loss__(self, raw_preds, x_label):
        loss = cross_entropy(raw_preds, x_label)
        m = self.edge_mask.sigmoid()
        loss = loss + self.coeffs['edge_size'] * m.sum()
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()

        if self.mask_features:
            m = self.node_feat_mask.sigmoid()
            loss = loss + self.coeffs['node_feat_size'] * m.sum()
            ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
            loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        loss = loss.requires_grad_(True)
        return loss

    def gnn_explainer_alg(self, x: Tensor, edge_index: Tensor, ex_label: Tensor, mask_features: bool = False, **kwargs) -> None:
        # initialize a mask
        patience = 10
        self.to(x.device)
        self.mask_features = mask_features

        # train to get the mask
        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask],
                                     lr=self.lr)

        best_loss = 4.0
        count = 0
        for epoch in range(1, self.epochs + 1):
            if mask_features:
                h = x * self.node_feat_mask.view(1, -1).sigmoid()
            else:
                h = x
            raw_preds = self.model(data=Batch.from_data_list([Data(x=h, edge_index=edge_index)]))
            loss = self.__loss__(raw_preds, ex_label)
            # if epoch % 10 == 0:
            #     print(f'#D#Loss:{loss.item()}')

            is_best = (loss < best_loss)

            if not is_best:
                count += 1
            else:
                count = 0
                best_loss = loss

            if count >= patience:
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return self.edge_mask.sigmoid().data

    def forward(self, x, edge_index, mask_features=False,
                positive=True, **kwargs):
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for node
        :attr:`node_idx`.

        Args:
            data (Batch): batch from dataloader
            edge_index (LongTensor): The edge indices.
            pos_neg (Literal['pos', 'neg']) : get positive or negative mask
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """
        self.model.eval()
        # self_loop_edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
        # Only operate on a k-hop subgraph around `node_idx`.
        # Calculate mask

        ex_label = torch.tensor([1]).to(self.device)
        self.__clear_masks__()
        self.__set_masks__(x, edge_index)
        edge_mask = self.gnn_explainer_alg(x, edge_index, ex_label)
        # edge_masks.append(self.gnn_explainer_alg(x, edge_index, ex_label))


        # with torch.no_grad():
        #     related_preds = self.eval_related_pred(x, edge_index, edge_masks, **kwargs)
        self.__clear_masks__()
        # sorted_results = edge_mask.sort(descending=True)
        return edge_mask.detach()

    def explain(self, x, edge_index):
        edge_mask = self.forward(x, edge_index)
        sorted_indices = edge_mask.sort(descending=True).indices.cpu()
        return sorted_indices, False


    def __repr__(self):
        return f'{self.__class__.__name__}()'