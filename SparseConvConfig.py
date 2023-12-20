from abc import ABC
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math

from torch_scatter import scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import softmax


def genMI(y, A, norm=False):
    if norm:
        z = (y.flatten() - y.mean()) / y.std()
    else:
        z = y.flatten() - y.mean()
    z = z.reshape(-1, 1)
    MI = z.matmul(z.t()).matmul(A.t()).abs().diag()
    return MI



class ReverseAttentionConv(MessagePassing):
    def __init__(self, in_channels, out_channels: int,
                 heads: int = 1, concat: bool = True, beta: bool = False,
                 dropout: float = 0., edge_dim=None,
                 bias: bool = True, root_weight: bool = True,
                 threshold: float = 0.10, negative_slope: float = 0.01, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(ReverseAttentionConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.threshold = threshold
        self.negative_slope = negative_slope
        self.leakyrelu = nn.LeakyReLU(negative_slope=negative_slope)

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def sparse(self, a):
        threshold = self.threshold / (1 - self.dropout)
        a_temp = a - threshold
        a_temp = self.leakyrelu(a_temp)
        a_temp += threshold * (a_temp >= 0)
        a_temp += threshold * self.negative_slope * (a_temp < 0)
        # slope * (x - threshold) + slope * threshold
        # a_temp += self.threshold * (a_temp < 0)
        return a_temp

    def forward(self, x, edge_index, edge_attr=None):
        x = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r

        return out

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        """
        :param x_i: 入节点特征
        :param x_j: 出节点特征
        :param edge_attr: 边特征
        :param index: 入节点索引
        :param ptr: None
        :param size_i: None
        """

        # 对输入的 x_i, x_j 做线性变换
        # 输出的 query, key 维度为 (num_edge, heads, out_dim)
        query = self.lin_query(x_i).view(-1, self.heads, self.out_channels)  # (num_edge, heads, out_channel)
        key = self.lin_key(x_j).view(-1, self.heads, self.out_channels)  # (num_edge, heads, out_channel)

        if self.lin_edge is not None:
            assert edge_attr is not None
            # 对输入的 edge_attr 做线性变换，输出维度 (num_edge, heads, out_channel)
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)  # (num_edge, heads, out_channel)
            key += edge_attr

        # 对 query, key 内积：对应元素相乘，对最后一个维度（out_channel）求和
        # 输出为相似系数，维度 (num_edge, heads)
        alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)
        sign = torch.sign(alpha)
        alpha = softmax(torch.abs(alpha), index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        alpha = self.sparse(alpha)  # (num_edge, heads)

        alpha_sum = scatter(alpha, index, dim=0, dim_size=maybe_num_nodes(index, num_nodes=None),
                            reduce='sum').index_select(dim=0, index=index)  # (num_edge, heads)
        alpha /= (alpha_sum + 1e-6) / (1 - self.dropout)  # (num_edge, heads)

        out = self.lin_value(x_j).view(-1, self.heads, self.out_channels)  # (num_edge, heads, out_channel)
        if edge_attr is not None:
            out += edge_attr

        out *= (sign * alpha).view(-1, self.heads, 1)
        return out

class AttentionLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout, cuda=True):
        super(AttentionLayer, self).__init__()
        self.dropout = dropout
        self.is_cuda = cuda
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.W = nn.Parameter(nn.init.xavier_normal_(
            torch.Tensor(in_dim, hidden_dim).type(torch.cuda.FloatTensor if cuda else torch.FloatTensor),
            gain=np.sqrt(1.0)), requires_grad=True)
        self.W2 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(hidden_dim, hidden_dim).type(
            torch.cuda.FloatTensor if cuda else torch.FloatTensor), gain=np.sqrt(1.0)),
            requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, bias, emb_dest, emb_src, feature_src):
        """
        emb_dest: Tensor, 待插值格点
        emb_src: Tensor, 观测站点 (训练集站点)
        feature_src: Tensor, dest_node 的观测值
        bias: Tensor, shape (dest_node_num + src_node_num, dest_node_num + src_node_num), 邻接矩阵
        """

        h_1 = torch.mm(emb_src, self.W)  # (src_node_num, hidden_dim)
        h_2 = torch.mm(emb_dest, self.W)  # (dest_node_num, hidden_dim)
        nan_ind = torch.isnan(feature_src.flatten())
        ft_src = torch.clone(feature_src).flatten()
        ft_src[nan_ind] = 0.

        e = self.leakyrelu(torch.mm(torch.mm(h_2, self.W2), h_1.t()))  # (dest_node_num, src_node_num)
        e = F.dropout(e, self.dropout, training=self.training)
        zero_vec = -9e15 * torch.ones_like(e)

        attention = torch.where(bias > 0, e, zero_vec)
        attention[:, nan_ind] = -9e15
        attention = F.softmax(attention / np.sqrt(h_1.shape[1]), dim=1)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, ft_src.reshape(-1, 1))
        return h_prime


class SparseConv(MessagePassing, ABC):
    def __init__(self, in_channels, out_channels: int,
                 scale: int = 30, heads: int = 1, concat: bool = True,
                 beta: bool = True, dropout: float = 0., edge_dim=None,
                 bias: bool = True, root_weight: bool = True,
                 threshold: float = 0.10, negative_slope: float = 0.01,
                 sparse_train: bool = True, idx=0, file=None, **kwargs):

        self._alpha = None
        self._sparse_alpha = None
        self._index = None
        kwargs.setdefault('aggr', 'add')
        super(SparseConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.idx = idx
        self.file = file

        self.scale = scale
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.threshold = threshold
        self.negative_slope = negative_slope
        self.leakyrelu = nn.LeakyReLU(negative_slope=negative_slope)
        self.sparse_train = sparse_train

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def sparse(self, a):
        threshold = self.threshold / (1 - self.dropout * self.sparse_train)
        # a[a < threshold] *= 0.01
        a_temp = a - threshold
        a_temp = self.leakyrelu(a_temp)
        a_temp += threshold * (a_temp >= 0)
        a_temp += threshold * self.negative_slope * (a_temp < 0)
        # a_temp += self.threshold * (a_temp < 0)
        return a_temp

    def forward(self, x, edge_index, edge_attr=None):
        x = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                # print(beta)
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r
        return out

    def message(self, x_i, x_j, edge_attr,
                index, ptr, size_i):
        """
        :param x_i: 入节点特征
        :param x_j: 出节点特征
        :param edge_attr: 边特征
        :param index: 入节点索引
        :param ptr: None
        :param size_i: None
        """

        # 对输入的 x_i, x_j 做线性变换
        query = self.lin_query(x_i).view(-1, self.heads, self.out_channels)  # (edge_num, heads, out_channel)
        key = self.lin_key(x_j).view(-1, self.heads, self.out_channels)  # (edge_num, heads, out_channel)
        if self.scale < 1:
            suffix = 'scaled'
        else:
            suffix = ''
        # print(self.scale)
        self.scale = self.scale.to(query.device)

        if self.lin_edge is not None:
            assert edge_attr is not None
            # 对输入的 edge_attr 做线性变换
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)  # (edge_num, heads, out_channel)
            key += edge_attr

        # 对 query, key 内积：对应元素相乘，对最后一个维度（out_channel）求和
        # 输出为相似系数，维度 (edge_num, heads)
        alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)
        num_neighbors = 1 + torch.log(1 + scatter(torch.ones(alpha.shape).to(alpha.device), index, dim=0,
                                                  dim_size=maybe_num_nodes(index, num_nodes=None),
                                                  reduce='sum').index_select(dim=0, index=index))
        alpha *= self.scale * num_neighbors
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        self._alpha = alpha

        # 注意力系数稀疏化
        alpha = self.sparse(alpha)  # (num_edge, heads)

        # 重新归一化, 维数 (edge_num, heads)
        alpha_sum = scatter(alpha, index, dim=0, dim_size=maybe_num_nodes(index, num_nodes=None),
                            reduce='sum').index_select(dim=0, index=index)  # (edge_num, heads)
        alpha_sum[alpha_sum == 0] += 1e-6
        alpha /= alpha_sum / (1 - self.dropout * self.sparse_train)
        self._sparse_alpha = alpha
        self._index = index

        out = self.lin_value(x_j).view(-1, self.heads, self.out_channels)  # (edge_num, heads, out_channel)
        if edge_attr is not None:
            out += edge_attr

        out *= alpha.view(-1, self.heads, 1)

        return out


