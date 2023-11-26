import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_sparse import SparseTensor
from torch.tensor import Tensor

from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.conv import GCNConv, gcn_conv

type_map = {
    'AndExpression': 1, 'Sizeof': 2, 'Identifier': 3, 'ForInit': 4, 'ReturnStatement': 5, 'SizeofOperand': 6,
    'InclusiveOrExpression': 7, 'PtrMemberAccess': 8, 'AssignmentExpr': 9, 'ParameterList': 10,
    'IdentifierDeclType': 11, 'SizeofExpr': 12, 'SwitchStatement': 13, 'IncDec': 14, 'Function': 15,
    'BitAndExpression': 16, 'UnaryOp': 17, 'DoStatement': 18, 'GotoStatement': 19, 'Callee': 20,
    'OrExpression': 21, 'ShiftExpression': 22, 'Decl': 23, 'CFGErrorNode': 24, 'WhileStatement': 25,
    'InfiniteForNode': 26, 'RelationalExpression': 27, 'CFGExitNode': 28, 'Condition': 29, 'BreakStatement': 30,
    'CompoundStatement': 31, 'UnaryOperator': 32, 'CallExpression': 33, 'CastExpression': 34,
    'ConditionalExpression': 35, 'ArrayIndexing': 36, 'PostfixExpression': 37, 'Label': 38,
    'ArgumentList': 39, 'EqualityExpression': 40, 'ReturnType': 41, 'Parameter': 42, 'Argument': 43, 'Symbol': 44,
    'ParameterType': 45, 'Statement': 46, 'AdditiveExpression': 47, 'PrimaryExpression': 48, 'DeclStmt': 49,
    'CastTarget': 50, 'IdentifierDeclStatement': 51, 'IdentifierDecl': 52, 'CFGEntryNode': 53, 'TryStatement': 54,
    'Expression': 55, 'ExclusiveOrExpression': 56, 'ClassDef': 57, 'ClassStaticIdentifier': 58, 'ForRangeInit': 59,
    'ClassDefStatement': 60, 'FunctionDef': 61, 'IfStatement': 62, 'MultiplicativeExpression': 63,
    'ContinueStatement': 64, 'MemberAccess': 65, 'ExpressionStatement': 66, 'ForStatement': 67, 'InitializerList': 68,
    'ElseStatement': 69, 'ThrowExpression': 70, 'IncDecOp': 71, 'NewExpression': 72, 'DeleteExpression': 73, 'BoolExpression': 74,
    'CharExpression': 75, 'DoubleExpression': 76, 'IntegerExpression': 77, 'PointerExpression': 78, 'StringExpression': 79,
    'ExpressionHolderStatement': 80, 'AssignmentExpression': 81, 'SizeofExpression': 82, 'UnaryOperationExpression': 83,
    'UnaryExpression': 84, 'PostIncDecOperationExpression': 85, 'File': 86
}

# suit the API in DIG/xgraph
class GNNPool(nn.Module):
    def __init__(self):
        super().__init__()

class GlobalAddPool(GNNPool):
    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return global_add_pool(x, batch)

class GlobalMaxPool(GNNPool):
    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return global_max_pool(x, batch)

class GlobalMaxMeanPool(GNNPool):
    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return torch.cat((global_max_pool(x, batch), global_mean_pool(x, batch)), dim=1)


class GCNConvGrad(GCNConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_weight = None

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        """"""
        if self.normalize and edge_weight is None:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_conv.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_conv.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # --- add require_grad ---
        edge_weight.requires_grad_(True)
        x = torch.matmul(x, self.weight)
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
        if self.bias is not None:
            out += self.bias

        # --- My: record edge_weight ---
        self.edge_weight = edge_weight

        return out