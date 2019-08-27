import numpy as np
import copy
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..nn import TreeCRF, VarMaskedGRU, VarMaskedRNN, VarMaskedLSTM, VarMaskedFastLSTM
from ..nn import SkipConnectFastLSTM, SkipConnectGRU, SkipConnectLSTM, SkipConnectRNN
from ..nn import BiAAttention, BiLinear
from neuronlp2.tasks import parser
from .parsing import StackPtrNet, PriorOrder

class HPtrNetPSTGate(StackPtrNet):
    """ receive hidden state from sibling, parent and previous step. Using gate. """
    def __init__(self, *args, **kwargs):
        super(HPtrNetPSTGate, self).__init__(*args, **kwargs)
        self.parent_hn_dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.sib_hn_dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.pre_hn_dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.parent_hn_dense_gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.sib_hn_dense_gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.pre_hn_dense_gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.bias_gate = nn.Parameter(torch.Tensor(self.hidden_size))
        self.dropout_hn = nn.Dropout2d(p=0.33)

    def _get_decoder_output(self, output_enc, heads, heads_stack, siblings, hx, mask_d=None, length_d=None):
        batch, _, _ = output_enc.size()
        # create batch index [batch]
        batch_index = torch.arange(0, batch).type_as(output_enc).long()
        # get vector for heads [batch, length_decoder, input_dim],
        src_encoding = output_enc[batch_index, heads_stack.t()].transpose(0, 1)

        if self.sibling:
            # [batch, length_decoder, hidden_size * 2]
            mask_sibs = siblings.ne(0).float().unsqueeze(2)
            output_enc_sibling = output_enc[batch_index, siblings.t()].transpose(0, 1) * mask_sibs
            src_encoding = src_encoding + output_enc_sibling

        if self.grandPar:
            # [length_decoder, batch]
            gpars = heads[batch_index, heads_stack.t()]
            # [batch, length_decoder, hidden_size * 2]
            output_enc_gpar = output_enc[batch_index, gpars].transpose(0, 1)
            src_encoding = src_encoding + output_enc_gpar

        # transform to decoder input
        # [batch, length_decoder, dec_dim]
        src_encoding = F.elu(self.src_dense(src_encoding))

        _, length_decoder, _ = src_encoding.shape
        # output from rnn [batch, length, hidden_size]      
        output = torch.zeros(batch, length_decoder, self.hidden_size).type_as(src_encoding)
        hn = hx
        parent_hn_dict = {}
        parent_hn_dict[0] = {}
        for b in range(batch):
            parent_hn_dict[0][b] = []
            for i in range(len(hn)):
                parent_hn_dict[0][b].append(hn[i][:,b].clone())

        sib_hn_dict = {}
        for step in range(length_decoder):
            head = heads_stack[:, step]
            hn_pre = [hn[0].clone(), hn[1].clone()]
            hn_parent = [torch.zeros(hn[0].shape).type_as(hn[0]), torch.zeros(hn[1].shape).type_as(hn[1])]
            hn_sib = [torch.zeros(hn[0].shape).type_as(hn[0]), torch.zeros(hn[1].shape).type_as(hn[1])]

            for b in range(batch):
                curr_head = int(head[b])
                if (curr_head in parent_hn_dict) and (b in parent_hn_dict[curr_head]):
                    for i in range(len(hn)):
                        hn_parent[i][:,b] = parent_hn_dict[curr_head][b][i].clone()
                if (curr_head in sib_hn_dict) and (b in sib_hn_dict[curr_head]):
                    for i in range(len(hn)):
                        hn_sib[i][:,b] = sib_hn_dict[curr_head][b][i].clone()

            hn = list(hn)
            for i in range(len(hn)):
                tmp_hn_pre = self.pre_hn_dense(hn_pre[i])
                tmp_hn_parent = self.parent_hn_dense(hn_parent[i].clone())
                tmp_hn_sib = self.sib_hn_dense(hn_sib[i].clone())
                tmp_hn_pre_gate = self.pre_hn_dense_gate(hn_pre[i])
                tmp_hn_parent_gate = self.parent_hn_dense_gate(hn_parent[i].clone())
                tmp_hn_sib_gate = self.sib_hn_dense_gate(hn_sib[i].clone())
                gate = F.sigmoid(tmp_hn_pre_gate + tmp_hn_parent_gate + tmp_hn_sib_gate + self.bias_gate)
                hn[i] = self.dropout_hn(F.tanh(tmp_hn_pre * gate + tmp_hn_parent * gate + tmp_hn_sib * gate))
            hn = tuple(hn)


            for b in range(batch):
                curr_head = int(head[b])
                if curr_head not in parent_hn_dict:
                    parent_hn_dict[curr_head] = {}
                if b not in parent_hn_dict[curr_head]:
                    parent_hn_dict[curr_head][b] = []
                    for i in range(len(hn_parent)):
                        parent_hn_dict[curr_head][b].append(hn_parent[i][:,b].clone())


            step_output, hn = self.decoder(src_encoding[:, step, :].unsqueeze(1),
                                                mask_d[:, step].unsqueeze(1),
                                                hx=hn)

            for b in range(batch):
                curr_head = int(head[b])
                if curr_head not in sib_hn_dict:
                    sib_hn_dict[curr_head] = {}
                sib_hn_dict[curr_head][b] = []
                for i in range(len(hn)):
                   sib_hn_dict[curr_head][b].append(hn[i][:,b].clone())

            for b in range(batch):
                output[b, step, :] = step_output[b]

        # apply dropout
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)

        return output, hn, mask_d, length_d

    def _decode_per_sentence(self, output_enc, arc_c, type_c, hx, length, beam, ordered, leading_symbolic):
        def valid_hyp(base_id, child_id, head):
            if constraints[base_id, child_id]:
                return False
            elif not ordered or self.prior_order == PriorOrder.DEPTH or child_orders[base_id, head] == 0:
                return True
            elif self.prior_order == PriorOrder.LEFT2RIGTH:
                return child_id > child_orders[base_id, head]
            else:
                if child_id < head:
                    return child_id < child_orders[base_id, head] < head
                else:
                    return child_id > child_orders[base_id, head]

        # output_enc [length, hidden_size * 2]
        # arc_c [length, arc_space]
        # type_c [length, type_space]
        # hx [decoder_layers, hidden_size]
        if length is not None:
            output_enc = output_enc[:length]
            arc_c = arc_c[:length]
            type_c = type_c[:length]
        else:
            length = output_enc.size(0)

        # [decoder_layers, 1, hidden_size]
        # hack to handle LSTM
        if isinstance(hx, tuple):
            hx, cx = hx
            hx = hx.unsqueeze(1)
            cx = cx.unsqueeze(1)
            h0 = hx
            hx = (hx, cx)
        else:
            hx = hx.unsqueeze(1)
            h0 = hx

        stacked_heads = [[0] for _ in range(beam)]
        grand_parents = [[0] for _ in range(beam)] if self.grandPar else None
        siblings = [[0] for _ in range(beam)] if self.sibling else None
        skip_connects = [[h0] for _ in range(beam)] if self.skipConnect else None
        children = torch.zeros(beam, int(2 * length - 1)).type_as(output_enc).long()
        stacked_types = children.new_zeros(children.size())
        hypothesis_scores = output_enc.new_zeros(beam)
        constraints = np.zeros([beam, length], dtype=np.bool)
        constraints[:, 0] = True
        child_orders = np.zeros([beam, length], dtype=np.int64)

        # temporal tensors for each step.
        new_stacked_heads = [[] for _ in range(beam)]
        new_grand_parents = [[] for _ in range(beam)] if self.grandPar else None
        new_siblings = [[] for _ in range(beam)] if self.sibling else None
        new_skip_connects = [[] for _ in range(beam)] if self.skipConnect else None
        new_children = children.new_zeros(children.size())
        new_stacked_types = stacked_types.new_zeros(stacked_types.size())
        num_hyp = 1
        num_step = 2 * length - 1

        parent_hn_dict = [{} for i in range(beam)]
        sib_hn_dict = [{} for i in range(beam)]
        # init parent_hn_dict for first step
        for n in range(num_hyp):
            parent_hn_dict[n][0] = []
            for i in range(len(hx)):
                parent_hn_dict[n][0].append(hx[i][:,n,:].clone())

        for t in range(num_step):
            # [num_hyp]
            heads = torch.LongTensor([stacked_heads[i][-1] for i in range(num_hyp)]).type_as(children)
            gpars = torch.LongTensor([grand_parents[i][-1] for i in range(num_hyp)]).type_as(children) if self.grandPar else None
            sibs = torch.LongTensor([siblings[i].pop() for i in range(num_hyp)]).type_as(children) if self.sibling else None

            # [decoder_layers, num_hyp, hidden_size]
            hs = torch.cat([skip_connects[i].pop() for i in range(num_hyp)], dim=1) if self.skipConnect else None

            # [num_hyp, hidden_size * 2]
            src_encoding = output_enc[heads]

            if self.sibling:
                mask_sibs = sibs.ne(0).float().unsqueeze(1)
                output_enc_sibling = output_enc[sibs] * mask_sibs
                src_encoding = src_encoding + output_enc_sibling

            if self.grandPar:
                output_enc_gpar = output_enc[gpars]
                src_encoding = src_encoding + output_enc_gpar

            # transform to decoder input
            # [num_hyp, dec_dim]
            src_encoding = F.elu(self.src_dense(src_encoding))

            hx_pre = (hx[0].clone(), hx[1].clone())
            hx_parent = [torch.zeros(hx[0].shape).type_as(hx[0]), torch.zeros(hx[1].shape).type_as(hx[1])]
            hx_sib = [torch.zeros(hx[0].shape).type_as(hx[0]), torch.zeros(hx[1].shape).type_as(hx[1])]

            # update parent hidden states
            for n in range(num_hyp):
                head = int(heads[n])
                if head in parent_hn_dict[n]:
                    for i in range(len(hx)):
                        hx_parent[i][:,n,:] = parent_hn_dict[n][head][i].clone()
                if head in sib_hn_dict[n]:
                    for i in range(len(hx)):
                        hx_sib[i][:,n,:] = sib_hn_dict[n][head][i].clone()

            hx = list(hx)
            # update hidden states
            for i in range(len(hx)):
                tmp_hn_pre = self.pre_hn_dense(hx_pre[i])
                tmp_hn_parent = self.parent_hn_dense(hx_parent[i].clone())
                tmp_hn_sib= self.sib_hn_dense(hx_sib[i].clone())
                tmp_hn_pre_gate = self.pre_hn_dense_gate(hx_pre[i])
                tmp_hn_parent_gate = self.parent_hn_dense_gate(hx_parent[i].clone())
                tmp_hn_sib_gate = self.sib_hn_dense_gate(hx_sib[i].clone())
                gate = F.sigmoid(tmp_hn_pre_gate + tmp_hn_parent_gate + tmp_hn_sib_gate + self.bias_gate)
                hx[i] = self.dropout_hn(F.tanh(tmp_hn_pre * gate + tmp_hn_parent * gate + tmp_hn_sib * gate))
            hx = tuple(hx)

            # output [num_hyp, hidden_size]
            # hx [decoder_layer, num_hyp, hidden_size]
            output_dec, hx = self.decoder.step(src_encoding, hx=hx, hs=hs) if self.skipConnect else self.decoder.step(src_encoding, hx=hx)

            # arc_h size [num_hyp, 1, arc_space]
            arc_h = F.elu(self.arc_h(output_dec.unsqueeze(1)))
            # type_h size [num_hyp, type_space]
            type_h = F.elu(self.type_h(output_dec))

            # [num_hyp, length_encoder]
            out_arc = self.attention(arc_h, arc_c.expand(num_hyp, *arc_c.size())).squeeze(dim=1).squeeze(dim=1)

            # [num_hyp, length_encoder]
            hyp_scores = F.log_softmax(out_arc, dim=1)

            new_hypothesis_scores = hypothesis_scores[:num_hyp].unsqueeze(1) + hyp_scores
            # [num_hyp * length_encoder]
            new_hypothesis_scores, hyp_index = torch.sort(new_hypothesis_scores.view(-1), dim=0, descending=True)
            base_index = hyp_index / length
            child_index = hyp_index % length

            cc = 0
            ids = []
            new_constraints = np.zeros([beam, length], dtype=np.bool)
            new_child_orders = np.zeros([beam, length], dtype=np.int64)
            new_parent_hn_dict = [{} for i in range(beam)]
            new_sib_hn_dict = [{} for i in range(beam)]
            for id in range(num_hyp * length):
                base_id = base_index[id]
                child_id = child_index[id]
                head = heads[base_id]
                new_hyp_score = new_hypothesis_scores[id]
                if child_id == head:
                    assert constraints[base_id, child_id], 'constrains error: %d, %d' % (base_id, child_id)
                    if head != 0 or t + 1 == num_step:
                        int_head = int(head)
                        new_parent_hn_dict[cc] = copy.copy(parent_hn_dict[base_id])
                        if int_head not in new_parent_hn_dict[cc]:
                            new_parent_hn_dict[cc][int_head] = []
                            for i in range(len(hx_parent)):
                                new_parent_hn_dict[cc][int_head].append(hx_parent[i][:,base_id,:].clone())
                        new_sib_hn_dict[cc] = copy.copy(sib_hn_dict[base_id])
                        new_sib_hn_dict[cc][int_head] = []
                        for i in range(len(hx)):
                            new_sib_hn_dict[cc][int_head].append(hx[i][:,base_id,:].clone())

                        new_constraints[cc] = constraints[base_id]
                        new_child_orders[cc] = child_orders[base_id]

                        new_stacked_heads[cc] = [stacked_heads[base_id][i] for i in range(len(stacked_heads[base_id]))]
                        new_stacked_heads[cc].pop()

                        if self.grandPar:
                            new_grand_parents[cc] = [grand_parents[base_id][i] for i in range(len(grand_parents[base_id]))]
                            new_grand_parents[cc].pop()

                        if self.sibling:
                            new_siblings[cc] = [siblings[base_id][i] for i in range(len(siblings[base_id]))]

                        if self.skipConnect:
                            new_skip_connects[cc] = [skip_connects[base_id][i] for i in range(len(skip_connects[base_id]))]

                        new_children[cc] = children[base_id]
                        new_children[cc, t] = child_id

                        hypothesis_scores[cc] = new_hyp_score
                        ids.append(id)
                        cc += 1
                elif valid_hyp(base_id, child_id, head):
                    int_head = int(head)
                    new_parent_hn_dict[cc] = copy.copy(parent_hn_dict[base_id])
                    if int_head not in new_parent_hn_dict[cc]:
                        new_parent_hn_dict[cc][int_head] = []
                        for i in range(len(hx_parent)):
                            new_parent_hn_dict[cc][int_head].append(hx_parent[i][:,base_id,:].clone())
                    new_sib_hn_dict[cc] = copy.copy(sib_hn_dict[base_id])
                    new_sib_hn_dict[cc][int_head] = []
                    for i in range(len(hx)):
                        new_sib_hn_dict[cc][int_head].append(hx[i][:,base_id,:].clone())

                    new_constraints[cc] = constraints[base_id]
                    new_constraints[cc, child_id] = True

                    new_child_orders[cc] = child_orders[base_id]
                    new_child_orders[cc, head] = child_id

                    new_stacked_heads[cc] = [stacked_heads[base_id][i] for i in range(len(stacked_heads[base_id]))]
                    new_stacked_heads[cc].append(child_id)

                    if self.grandPar:
                        new_grand_parents[cc] = [grand_parents[base_id][i] for i in range(len(grand_parents[base_id]))]
                        new_grand_parents[cc].append(head)

                    if self.sibling:
                        new_siblings[cc] = [siblings[base_id][i] for i in range(len(siblings[base_id]))]
                        new_siblings[cc].append(child_id)
                        new_siblings[cc].append(0)

                    if self.skipConnect:
                        new_skip_connects[cc] = [skip_connects[base_id][i] for i in range(len(skip_connects[base_id]))]
                        # hack to handle LSTM
                        if isinstance(hx, tuple):
                            new_skip_connects[cc].append(hx[0][:, base_id, :].unsqueeze(1))
                        else:
                            new_skip_connects[cc].append(hx[:, base_id, :].unsqueeze(1))
                        new_skip_connects[cc].append(h0)

                    new_children[cc] = children[base_id]
                    new_children[cc, t] = child_id

                    hypothesis_scores[cc] = new_hyp_score
                    ids.append(id)
                    cc += 1

                if cc == beam:
                    break

            # [num_hyp]
            num_hyp = len(ids)
            if num_hyp == 0:
                return None
            else:
                index = torch.from_numpy(np.array(ids)).type_as(base_index)
            base_index = base_index[index]
            child_index = child_index[index]

            # predict types for new hypotheses
            # compute output for type [num_hyp, num_labels]
            out_type = self.bilinear(type_h[base_index], type_c[child_index])
            hyp_type_scores = F.log_softmax(out_type, dim=1)
            # compute the prediction of types [num_hyp]
            hyp_type_scores, hyp_types = hyp_type_scores.max(dim=1)
            hypothesis_scores[:num_hyp] = hypothesis_scores[:num_hyp] + hyp_type_scores

            sib_hn_dict = new_sib_hn_dict
            parent_hn_dict = new_parent_hn_dict
            for i in range(num_hyp):
                base_id = base_index[i]
                new_stacked_types[i] = stacked_types[base_id]
                new_stacked_types[i, t] = hyp_types[i]

            stacked_heads = [[new_stacked_heads[i][j] for j in range(len(new_stacked_heads[i]))] for i in range(num_hyp)]
            if self.grandPar:
                grand_parents = [[new_grand_parents[i][j] for j in range(len(new_grand_parents[i]))] for i in range(num_hyp)]
            if self.sibling:
                siblings = [[new_siblings[i][j] for j in range(len(new_siblings[i]))] for i in range(num_hyp)]
            if self.skipConnect:
                skip_connects = [[new_skip_connects[i][j] for j in range(len(new_skip_connects[i]))] for i in range(num_hyp)]
            constraints = new_constraints
            child_orders = new_child_orders
            children.copy_(new_children)
            stacked_types.copy_(new_stacked_types)
            # hx [decoder_layers, num_hyp, hidden_size]
            # hack to handle LSTM
            if isinstance(hx, tuple):
                hx, cx = hx
                hx = hx[:, base_index, :]
                cx = cx[:, base_index, :]
                hx = (hx, cx)
            else:
                hx = hx[:, base_index, :]

        children = children.cpu().numpy()[0]
        stacked_types = stacked_types.cpu().numpy()[0]
        heads = np.zeros(length, dtype=np.int32)
        types = np.zeros(length, dtype=np.int32)
        stack = [0]
        for i in range(num_step):
            head = stack[-1]
            child = children[i]
            type = stacked_types[i]
            if child != head:
                heads[child] = head
                types[child] = type
                stack.append(child)
            else:
                stacked_types[i] = 0
                stack.pop()

        return heads, types, length, children, stacked_types


class HPtrNetPSTSGate(StackPtrNet):
    """ receive hidden state from sibling, parent and previous step. Using SGate. """
    def __init__(self, *args, **kwargs):
        super(HPtrNetPSTSGate, self).__init__(*args, **kwargs)
        self.parent_hn_dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.sib_hn_dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.pre_hn_dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.parent_hn_dense_gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.sib_hn_dense_gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.pre_hn_dense_gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.bias_gate = nn.Parameter(torch.Tensor(self.hidden_size))
        self.dropout_hn = nn.Dropout2d(p=0.33)

    def _get_decoder_output(self, output_enc, heads, heads_stack, siblings, hx, mask_d=None, length_d=None):
        batch, _, _ = output_enc.size()
        # create batch index [batch]
        batch_index = torch.arange(0, batch).type_as(output_enc).long()
        # get vector for heads [batch, length_decoder, input_dim],
        src_encoding = output_enc[batch_index, heads_stack.t()].transpose(0, 1)

        if self.sibling:
            # [batch, length_decoder, hidden_size * 2]
            mask_sibs = siblings.ne(0).float().unsqueeze(2)
            output_enc_sibling = output_enc[batch_index, siblings.t()].transpose(0, 1) * mask_sibs
            src_encoding = src_encoding + output_enc_sibling

        if self.grandPar:
            # [length_decoder, batch]
            gpars = heads[batch_index, heads_stack.t()]
            # [batch, length_decoder, hidden_size * 2]
            output_enc_gpar = output_enc[batch_index, gpars].transpose(0, 1)
            src_encoding = src_encoding + output_enc_gpar

        # transform to decoder input
        # [batch, length_decoder, dec_dim]
        src_encoding = F.elu(self.src_dense(src_encoding))

        _, length_decoder, _ = src_encoding.shape
        # output from rnn [batch, length, hidden_size]      
        output = torch.zeros(batch, length_decoder, self.hidden_size).type_as(src_encoding)
        hn = hx
        parent_hn_dict = {}
        parent_hn_dict[0] = {}
        for b in range(batch):
            parent_hn_dict[0][b] = []
            for i in range(len(hn)):
                parent_hn_dict[0][b].append(hn[i][:,b].clone())

        sib_hn_dict = {}
        for step in range(length_decoder):
            head = heads_stack[:, step]
            hn_pre = [hn[0].clone(), hn[1].clone()]
            hn_parent = [torch.zeros(hn[0].shape).type_as(hn[0]), torch.zeros(hn[1].shape).type_as(hn[1])]
            hn_sib = [torch.zeros(hn[0].shape).type_as(hn[0]), torch.zeros(hn[1].shape).type_as(hn[1])]

            for b in range(batch):
                curr_head = int(head[b])
                if (curr_head in parent_hn_dict) and (b in parent_hn_dict[curr_head]):
                    for i in range(len(hn)):
                        hn_parent[i][:,b] = parent_hn_dict[curr_head][b][i].clone()
                if (curr_head in sib_hn_dict) and (b in sib_hn_dict[curr_head]):
                    for i in range(len(hn)):
                        hn_sib[i][:,b] = sib_hn_dict[curr_head][b][i].clone()

            hn = list(hn)
            for i in range(len(hn)):
                tmp_hn_pre = self.pre_hn_dense(hn_pre[i])
                tmp_hn_parent = self.parent_hn_dense(hn_parent[i].clone())
                tmp_hn_sib = self.sib_hn_dense(hn_sib[i].clone())
                tmp_hn_parent_gate = self.parent_hn_dense_gate(hn_parent[i].clone() * hn_pre[i])
                tmp_hn_sib_gate = self.sib_hn_dense_gate(hn_sib[i].clone() * hn_pre[i])
                gate = F.sigmoid(tmp_hn_parent_gate + tmp_hn_sib_gate + self.bias_gate)
                hn[i] = self.dropout_hn(F.tanh(tmp_hn_pre * gate + tmp_hn_parent * gate + tmp_hn_sib * gate))
            hn = tuple(hn)


            for b in range(batch):
                curr_head = int(head[b])
                if curr_head not in parent_hn_dict:
                    parent_hn_dict[curr_head] = {}
                if b not in parent_hn_dict[curr_head]:
                    parent_hn_dict[curr_head][b] = []
                    for i in range(len(hn_parent)):
                        parent_hn_dict[curr_head][b].append(hn_parent[i][:,b].clone())


            step_output, hn = self.decoder(src_encoding[:, step, :].unsqueeze(1),
                                                mask_d[:, step].unsqueeze(1),
                                                hx=hn)

            for b in range(batch):
                curr_head = int(head[b])
                if curr_head not in sib_hn_dict:
                    sib_hn_dict[curr_head] = {}
                sib_hn_dict[curr_head][b] = []
                for i in range(len(hn)):
                   sib_hn_dict[curr_head][b].append(hn[i][:,b].clone())

            for b in range(batch):
                output[b, step, :] = step_output[b]

        # apply dropout
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)

        return output, hn, mask_d, length_d

    def _decode_per_sentence(self, output_enc, arc_c, type_c, hx, length, beam, ordered, leading_symbolic):
        def valid_hyp(base_id, child_id, head):
            if constraints[base_id, child_id]:
                return False
            elif not ordered or self.prior_order == PriorOrder.DEPTH or child_orders[base_id, head] == 0:
                return True
            elif self.prior_order == PriorOrder.LEFT2RIGTH:
                return child_id > child_orders[base_id, head]
            else:
                if child_id < head:
                    return child_id < child_orders[base_id, head] < head
                else:
                    return child_id > child_orders[base_id, head]

        # output_enc [length, hidden_size * 2]
        # arc_c [length, arc_space]
        # type_c [length, type_space]
        # hx [decoder_layers, hidden_size]
        if length is not None:
            output_enc = output_enc[:length]
            arc_c = arc_c[:length]
            type_c = type_c[:length]
        else:
            length = output_enc.size(0)

        # [decoder_layers, 1, hidden_size]
        # hack to handle LSTM
        if isinstance(hx, tuple):
            hx, cx = hx
            hx = hx.unsqueeze(1)
            cx = cx.unsqueeze(1)
            h0 = hx
            hx = (hx, cx)
        else:
            hx = hx.unsqueeze(1)
            h0 = hx

        stacked_heads = [[0] for _ in range(beam)]
        grand_parents = [[0] for _ in range(beam)] if self.grandPar else None
        siblings = [[0] for _ in range(beam)] if self.sibling else None
        skip_connects = [[h0] for _ in range(beam)] if self.skipConnect else None
        children = torch.zeros(beam, int(2 * length - 1)).type_as(output_enc).long()
        stacked_types = children.new_zeros(children.size())
        hypothesis_scores = output_enc.new_zeros(beam)
        constraints = np.zeros([beam, length], dtype=np.bool)
        constraints[:, 0] = True
        child_orders = np.zeros([beam, length], dtype=np.int64)

        # temporal tensors for each step.
        new_stacked_heads = [[] for _ in range(beam)]
        new_grand_parents = [[] for _ in range(beam)] if self.grandPar else None
        new_siblings = [[] for _ in range(beam)] if self.sibling else None
        new_skip_connects = [[] for _ in range(beam)] if self.skipConnect else None
        new_children = children.new_zeros(children.size())
        new_stacked_types = stacked_types.new_zeros(stacked_types.size())
        num_hyp = 1
        num_step = 2 * length - 1

        parent_hn_dict = [{} for i in range(beam)]
        sib_hn_dict = [{} for i in range(beam)]
        # init parent_hn_dict for first step
        for n in range(num_hyp):
            parent_hn_dict[n][0] = []
            for i in range(len(hx)):
                parent_hn_dict[n][0].append(hx[i][:,n,:].clone())

        for t in range(num_step):
            # [num_hyp]
            heads = torch.LongTensor([stacked_heads[i][-1] for i in range(num_hyp)]).type_as(children)
            gpars = torch.LongTensor([grand_parents[i][-1] for i in range(num_hyp)]).type_as(children) if self.grandPar else None
            sibs = torch.LongTensor([siblings[i].pop() for i in range(num_hyp)]).type_as(children) if self.sibling else None

            # [decoder_layers, num_hyp, hidden_size]
            hs = torch.cat([skip_connects[i].pop() for i in range(num_hyp)], dim=1) if self.skipConnect else None

            # [num_hyp, hidden_size * 2]
            src_encoding = output_enc[heads]

            if self.sibling:
                mask_sibs = sibs.ne(0).float().unsqueeze(1)
                output_enc_sibling = output_enc[sibs] * mask_sibs
                src_encoding = src_encoding + output_enc_sibling

            if self.grandPar:
                output_enc_gpar = output_enc[gpars]
                src_encoding = src_encoding + output_enc_gpar

            # transform to decoder input
            # [num_hyp, dec_dim]
            src_encoding = F.elu(self.src_dense(src_encoding))

            hx_pre = (hx[0].clone(), hx[1].clone())
            hx_parent = [torch.zeros(hx[0].shape).type_as(hx[0]), torch.zeros(hx[1].shape).type_as(hx[1])]
            hx_sib = [torch.zeros(hx[0].shape).type_as(hx[0]), torch.zeros(hx[1].shape).type_as(hx[1])]

            # update parent hidden states
            for n in range(num_hyp):
                head = int(heads[n])
                if head in parent_hn_dict[n]:
                    for i in range(len(hx)):
                        hx_parent[i][:,n,:] = parent_hn_dict[n][head][i].clone()
                if head in sib_hn_dict[n]:
                    for i in range(len(hx)):
                        hx_sib[i][:,n,:] = sib_hn_dict[n][head][i].clone()

            hx = list(hx)
            # update hidden states
            for i in range(len(hx)):
                tmp_hn_pre = self.pre_hn_dense(hx_pre[i])
                tmp_hn_parent = self.parent_hn_dense(hx_parent[i].clone())
                tmp_hn_sib= self.sib_hn_dense(hx_sib[i].clone())
                tmp_hn_parent_gate = self.parent_hn_dense_gate(hx_parent[i].clone() * hx_pre[i])
                tmp_hn_sib_gate = self.sib_hn_dense_gate(hx_sib[i].clone() * hx_pre[i])
                gate = F.sigmoid(tmp_hn_parent_gate + tmp_hn_sib_gate + self.bias_gate)
                hx[i] = self.dropout_hn(F.tanh(tmp_hn_pre * gate + tmp_hn_parent * gate + tmp_hn_sib * gate))
            hx = tuple(hx)

            # output [num_hyp, hidden_size]
            # hx [decoder_layer, num_hyp, hidden_size]
            output_dec, hx = self.decoder.step(src_encoding, hx=hx, hs=hs) if self.skipConnect else self.decoder.step(src_encoding, hx=hx)

            # arc_h size [num_hyp, 1, arc_space]
            arc_h = F.elu(self.arc_h(output_dec.unsqueeze(1)))
            # type_h size [num_hyp, type_space]
            type_h = F.elu(self.type_h(output_dec))

            # [num_hyp, length_encoder]
            out_arc = self.attention(arc_h, arc_c.expand(num_hyp, *arc_c.size())).squeeze(dim=1).squeeze(dim=1)

            # [num_hyp, length_encoder]
            hyp_scores = F.log_softmax(out_arc, dim=1)

            new_hypothesis_scores = hypothesis_scores[:num_hyp].unsqueeze(1) + hyp_scores
            # [num_hyp * length_encoder]
            new_hypothesis_scores, hyp_index = torch.sort(new_hypothesis_scores.view(-1), dim=0, descending=True)
            base_index = hyp_index / length
            child_index = hyp_index % length

            cc = 0
            ids = []
            new_constraints = np.zeros([beam, length], dtype=np.bool)
            new_child_orders = np.zeros([beam, length], dtype=np.int64)
            new_parent_hn_dict = [{} for i in range(beam)]
            new_sib_hn_dict = [{} for i in range(beam)]
            for id in range(num_hyp * length):
                base_id = base_index[id]
                child_id = child_index[id]
                head = heads[base_id]
                new_hyp_score = new_hypothesis_scores[id]
                if child_id == head:
                    assert constraints[base_id, child_id], 'constrains error: %d, %d' % (base_id, child_id)
                    if head != 0 or t + 1 == num_step:
                        int_head = int(head)
                        new_parent_hn_dict[cc] = copy.copy(parent_hn_dict[base_id])
                        if int_head not in new_parent_hn_dict[cc]:
                            new_parent_hn_dict[cc][int_head] = []
                            for i in range(len(hx_parent)):
                                new_parent_hn_dict[cc][int_head].append(hx_parent[i][:,base_id,:].clone())
                        new_sib_hn_dict[cc] = copy.copy(sib_hn_dict[base_id])
                        new_sib_hn_dict[cc][int_head] = []
                        for i in range(len(hx)):
                            new_sib_hn_dict[cc][int_head].append(hx[i][:,base_id,:].clone())

                        new_constraints[cc] = constraints[base_id]
                        new_child_orders[cc] = child_orders[base_id]

                        new_stacked_heads[cc] = [stacked_heads[base_id][i] for i in range(len(stacked_heads[base_id]))]
                        new_stacked_heads[cc].pop()

                        if self.grandPar:
                            new_grand_parents[cc] = [grand_parents[base_id][i] for i in range(len(grand_parents[base_id]))]
                            new_grand_parents[cc].pop()

                        if self.sibling:
                            new_siblings[cc] = [siblings[base_id][i] for i in range(len(siblings[base_id]))]

                        if self.skipConnect:
                            new_skip_connects[cc] = [skip_connects[base_id][i] for i in range(len(skip_connects[base_id]))]

                        new_children[cc] = children[base_id]
                        new_children[cc, t] = child_id

                        hypothesis_scores[cc] = new_hyp_score
                        ids.append(id)
                        cc += 1
                elif valid_hyp(base_id, child_id, head):
                    int_head = int(head)
                    new_parent_hn_dict[cc] = copy.copy(parent_hn_dict[base_id])
                    if int_head not in new_parent_hn_dict[cc]:
                        new_parent_hn_dict[cc][int_head] = []
                        for i in range(len(hx_parent)):
                            new_parent_hn_dict[cc][int_head].append(hx_parent[i][:,base_id,:].clone())
                    new_sib_hn_dict[cc] = copy.copy(sib_hn_dict[base_id])
                    new_sib_hn_dict[cc][int_head] = []
                    for i in range(len(hx)):
                        new_sib_hn_dict[cc][int_head].append(hx[i][:,base_id,:].clone())

                    new_constraints[cc] = constraints[base_id]
                    new_constraints[cc, child_id] = True

                    new_child_orders[cc] = child_orders[base_id]
                    new_child_orders[cc, head] = child_id

                    new_stacked_heads[cc] = [stacked_heads[base_id][i] for i in range(len(stacked_heads[base_id]))]
                    new_stacked_heads[cc].append(child_id)

                    if self.grandPar:
                        new_grand_parents[cc] = [grand_parents[base_id][i] for i in range(len(grand_parents[base_id]))]
                        new_grand_parents[cc].append(head)

                    if self.sibling:
                        new_siblings[cc] = [siblings[base_id][i] for i in range(len(siblings[base_id]))]
                        new_siblings[cc].append(child_id)
                        new_siblings[cc].append(0)

                    if self.skipConnect:
                        new_skip_connects[cc] = [skip_connects[base_id][i] for i in range(len(skip_connects[base_id]))]
                        # hack to handle LSTM
                        if isinstance(hx, tuple):
                            new_skip_connects[cc].append(hx[0][:, base_id, :].unsqueeze(1))
                        else:
                            new_skip_connects[cc].append(hx[:, base_id, :].unsqueeze(1))
                        new_skip_connects[cc].append(h0)

                    new_children[cc] = children[base_id]
                    new_children[cc, t] = child_id

                    hypothesis_scores[cc] = new_hyp_score
                    ids.append(id)
                    cc += 1

                if cc == beam:
                    break

            # [num_hyp]
            num_hyp = len(ids)
            if num_hyp == 0:
                return None
            else:
                index = torch.from_numpy(np.array(ids)).type_as(base_index)
            base_index = base_index[index]
            child_index = child_index[index]

            # predict types for new hypotheses
            # compute output for type [num_hyp, num_labels]
            out_type = self.bilinear(type_h[base_index], type_c[child_index])
            hyp_type_scores = F.log_softmax(out_type, dim=1)
            # compute the prediction of types [num_hyp]
            hyp_type_scores, hyp_types = hyp_type_scores.max(dim=1)
            hypothesis_scores[:num_hyp] = hypothesis_scores[:num_hyp] + hyp_type_scores

            sib_hn_dict = new_sib_hn_dict
            parent_hn_dict = new_parent_hn_dict
            for i in range(num_hyp):
                base_id = base_index[i]
                new_stacked_types[i] = stacked_types[base_id]
                new_stacked_types[i, t] = hyp_types[i]

            stacked_heads = [[new_stacked_heads[i][j] for j in range(len(new_stacked_heads[i]))] for i in range(num_hyp)]
            if self.grandPar:
                grand_parents = [[new_grand_parents[i][j] for j in range(len(new_grand_parents[i]))] for i in range(num_hyp)]
            if self.sibling:
                siblings = [[new_siblings[i][j] for j in range(len(new_siblings[i]))] for i in range(num_hyp)]
            if self.skipConnect:
                skip_connects = [[new_skip_connects[i][j] for j in range(len(new_skip_connects[i]))] for i in range(num_hyp)]
            constraints = new_constraints
            child_orders = new_child_orders
            children.copy_(new_children)
            stacked_types.copy_(new_stacked_types)
            # hx [decoder_layers, num_hyp, hidden_size]
            # hack to handle LSTM
            if isinstance(hx, tuple):
                hx, cx = hx
                hx = hx[:, base_index, :]
                cx = cx[:, base_index, :]
                hx = (hx, cx)
            else:
                hx = hx[:, base_index, :]

        children = children.cpu().numpy()[0]
        stacked_types = stacked_types.cpu().numpy()[0]
        heads = np.zeros(length, dtype=np.int32)
        types = np.zeros(length, dtype=np.int32)
        stack = [0]
        for i in range(num_step):
            head = stack[-1]
            child = children[i]
            type = stacked_types[i]
            if child != head:
                heads[child] = head
                types[child] = type
                stack.append(child)
            else:
                stacked_types[i] = 0
                stack.pop()

        return heads, types, length, children, stacked_types


class HPtrNetPSGate(StackPtrNet):
    """ receive hidden state from sibling and parent. Using gate. """
    def __init__(self, *args, **kwargs):
        super(HPtrNetPSGate, self).__init__(*args, **kwargs)
        self.parent_hn_dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.sib_hn_dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.pre_hn_dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.parent_hn_dense_gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.sib_hn_dense_gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.pre_hn_dense_gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.bias_gate = nn.Parameter(torch.Tensor(self.hidden_size))
        self.dropout_hn = nn.Dropout2d(p=0.33)

    def _get_decoder_output(self, output_enc, heads, heads_stack, siblings, hx, mask_d=None, length_d=None):
        batch, _, _ = output_enc.size()
        # create batch index [batch]
        batch_index = torch.arange(0, batch).type_as(output_enc).long()
        # get vector for heads [batch, length_decoder, input_dim],
        src_encoding = output_enc[batch_index, heads_stack.t()].transpose(0, 1)

        if self.sibling:
            # [batch, length_decoder, hidden_size * 2]
            mask_sibs = siblings.ne(0).float().unsqueeze(2)
            output_enc_sibling = output_enc[batch_index, siblings.t()].transpose(0, 1) * mask_sibs
            src_encoding = src_encoding + output_enc_sibling

        if self.grandPar:
            # [length_decoder, batch]
            gpars = heads[batch_index, heads_stack.t()]
            # [batch, length_decoder, hidden_size * 2]
            output_enc_gpar = output_enc[batch_index, gpars].transpose(0, 1)
            src_encoding = src_encoding + output_enc_gpar

        # transform to decoder input
        # [batch, length_decoder, dec_dim]
        src_encoding = F.elu(self.src_dense(src_encoding))

        _, length_decoder, _ = src_encoding.shape
        # output from rnn [batch, length, hidden_size]      
        output = torch.zeros(batch, length_decoder, self.hidden_size).type_as(src_encoding)
        hn = hx
        parent_hn_dict = {}
        parent_hn_dict[0] = {}
        for b in range(batch):
            parent_hn_dict[0][b] = []
            for i in range(len(hn)):
                parent_hn_dict[0][b].append(hn[i][:,b].clone())

        sib_hn_dict = {}
        for step in range(length_decoder):
            head = heads_stack[:, step]
            hn_pre = [hn[0].clone(), hn[1].clone()]
            hn_parent = [torch.zeros(hn[0].shape).type_as(hn[0]), torch.zeros(hn[1].shape).type_as(hn[1])]
            hn_sib = [torch.zeros(hn[0].shape).type_as(hn[0]), torch.zeros(hn[1].shape).type_as(hn[1])]

            for b in range(batch):
                curr_head = int(head[b])
                if (curr_head in parent_hn_dict) and (b in parent_hn_dict[curr_head]):
                    for i in range(len(hn)):
                        hn_parent[i][:,b] = parent_hn_dict[curr_head][b][i].clone()
                if (curr_head in sib_hn_dict) and (b in sib_hn_dict[curr_head]):
                    for i in range(len(hn)):
                        hn_sib[i][:,b] = sib_hn_dict[curr_head][b][i].clone()

            hn = list(hn)
            for i in range(len(hn)):
                tmp_hn_pre = self.pre_hn_dense(hn_pre[i])
                tmp_hn_parent = self.parent_hn_dense(hn_parent[i].clone())
                tmp_hn_sib = self.sib_hn_dense(hn_sib[i].clone())
                tmp_hn_pre_gate = self.pre_hn_dense_gate(hn_pre[i])
                tmp_hn_parent_gate = self.parent_hn_dense_gate(hn_parent[i].clone())
                tmp_hn_sib_gate = self.sib_hn_dense_gate(hn_sib[i].clone())
                gate = F.sigmoid(tmp_hn_parent_gate + tmp_hn_sib_gate + self.bias_gate)
                hn[i] = self.dropout_hn(F.tanh(tmp_hn_parent * gate + tmp_hn_sib * gate))
            hn = tuple(hn)


            for b in range(batch):
                curr_head = int(head[b])
                if curr_head not in parent_hn_dict:
                    parent_hn_dict[curr_head] = {}
                if b not in parent_hn_dict[curr_head]:
                    parent_hn_dict[curr_head][b] = []
                    for i in range(len(hn_parent)):
                        parent_hn_dict[curr_head][b].append(hn_parent[i][:,b].clone())


            step_output, hn = self.decoder(src_encoding[:, step, :].unsqueeze(1),
                                                mask_d[:, step].unsqueeze(1),
                                                hx=hn)

            for b in range(batch):
                curr_head = int(head[b])
                if curr_head not in sib_hn_dict:
                    sib_hn_dict[curr_head] = {}
                sib_hn_dict[curr_head][b] = []
                for i in range(len(hn)):
                   sib_hn_dict[curr_head][b].append(hn[i][:,b].clone())

            for b in range(batch):
                output[b, step, :] = step_output[b]

        # apply dropout
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)

        return output, hn, mask_d, length_d

    def _decode_per_sentence(self, output_enc, arc_c, type_c, hx, length, beam, ordered, leading_symbolic):
        def valid_hyp(base_id, child_id, head):
            if constraints[base_id, child_id]:
                return False
            elif not ordered or self.prior_order == PriorOrder.DEPTH or child_orders[base_id, head] == 0:
                return True
            elif self.prior_order == PriorOrder.LEFT2RIGTH:
                return child_id > child_orders[base_id, head]
            else:
                if child_id < head:
                    return child_id < child_orders[base_id, head] < head
                else:
                    return child_id > child_orders[base_id, head]

        # output_enc [length, hidden_size * 2]
        # arc_c [length, arc_space]
        # type_c [length, type_space]
        # hx [decoder_layers, hidden_size]
        if length is not None:
            output_enc = output_enc[:length]
            arc_c = arc_c[:length]
            type_c = type_c[:length]
        else:
            length = output_enc.size(0)

        # [decoder_layers, 1, hidden_size]
        # hack to handle LSTM
        if isinstance(hx, tuple):
            hx, cx = hx
            hx = hx.unsqueeze(1)
            cx = cx.unsqueeze(1)
            h0 = hx
            hx = (hx, cx)
        else:
            hx = hx.unsqueeze(1)
            h0 = hx

        stacked_heads = [[0] for _ in range(beam)]
        grand_parents = [[0] for _ in range(beam)] if self.grandPar else None
        siblings = [[0] for _ in range(beam)] if self.sibling else None
        skip_connects = [[h0] for _ in range(beam)] if self.skipConnect else None
        children = torch.zeros(beam, int(2 * length - 1)).type_as(output_enc).long()
        stacked_types = children.new_zeros(children.size())
        hypothesis_scores = output_enc.new_zeros(beam)
        constraints = np.zeros([beam, length], dtype=np.bool)
        constraints[:, 0] = True
        child_orders = np.zeros([beam, length], dtype=np.int64)

        # temporal tensors for each step.
        new_stacked_heads = [[] for _ in range(beam)]
        new_grand_parents = [[] for _ in range(beam)] if self.grandPar else None
        new_siblings = [[] for _ in range(beam)] if self.sibling else None
        new_skip_connects = [[] for _ in range(beam)] if self.skipConnect else None
        new_children = children.new_zeros(children.size())
        new_stacked_types = stacked_types.new_zeros(stacked_types.size())
        num_hyp = 1
        num_step = 2 * length - 1

        parent_hn_dict = [{} for i in range(beam)]
        sib_hn_dict = [{} for i in range(beam)]
        # init parent_hn_dict for first step
        for n in range(num_hyp):
            parent_hn_dict[n][0] = []
            for i in range(len(hx)):
                parent_hn_dict[n][0].append(hx[i][:,n,:].clone())

        for t in range(num_step):
            # [num_hyp]
            heads = torch.LongTensor([stacked_heads[i][-1] for i in range(num_hyp)]).type_as(children)
            gpars = torch.LongTensor([grand_parents[i][-1] for i in range(num_hyp)]).type_as(children) if self.grandPar else None
            sibs = torch.LongTensor([siblings[i].pop() for i in range(num_hyp)]).type_as(children) if self.sibling else None

            # [decoder_layers, num_hyp, hidden_size]
            hs = torch.cat([skip_connects[i].pop() for i in range(num_hyp)], dim=1) if self.skipConnect else None

            # [num_hyp, hidden_size * 2]
            src_encoding = output_enc[heads]

            if self.sibling:
                mask_sibs = sibs.ne(0).float().unsqueeze(1)
                output_enc_sibling = output_enc[sibs] * mask_sibs
                src_encoding = src_encoding + output_enc_sibling

            if self.grandPar:
                output_enc_gpar = output_enc[gpars]
                src_encoding = src_encoding + output_enc_gpar

            # transform to decoder input
            # [num_hyp, dec_dim]
            src_encoding = F.elu(self.src_dense(src_encoding))

            hx_pre = (hx[0].clone(), hx[1].clone())
            hx_parent = [torch.zeros(hx[0].shape).type_as(hx[0]), torch.zeros(hx[1].shape).type_as(hx[1])]
            hx_sib = [torch.zeros(hx[0].shape).type_as(hx[0]), torch.zeros(hx[1].shape).type_as(hx[1])]

            # update parent hidden states
            for n in range(num_hyp):
                head = int(heads[n])
                if head in parent_hn_dict[n]:
                    for i in range(len(hx)):
                        hx_parent[i][:,n,:] = parent_hn_dict[n][head][i].clone()
                if head in sib_hn_dict[n]:
                    for i in range(len(hx)):
                        hx_sib[i][:,n,:] = sib_hn_dict[n][head][i].clone()

            hx = list(hx)
            # update hidden states
            for i in range(len(hx)):
                tmp_hn_pre = self.pre_hn_dense(hx_pre[i])
                tmp_hn_parent = self.parent_hn_dense(hx_parent[i].clone())
                tmp_hn_sib= self.sib_hn_dense(hx_sib[i].clone())
                tmp_hn_pre_gate = self.pre_hn_dense_gate(hx_pre[i])
                tmp_hn_parent_gate = self.parent_hn_dense_gate(hx_parent[i].clone())
                tmp_hn_sib_gate = self.sib_hn_dense_gate(hx_sib[i].clone())
                gate = F.sigmoid(tmp_hn_parent_gate + tmp_hn_sib_gate + self.bias_gate)
                hx[i] = self.dropout_hn(F.tanh(tmp_hn_parent * gate + tmp_hn_sib * gate))
            hx = tuple(hx)

            # output [num_hyp, hidden_size]
            # hx [decoder_layer, num_hyp, hidden_size]
            output_dec, hx = self.decoder.step(src_encoding, hx=hx, hs=hs) if self.skipConnect else self.decoder.step(src_encoding, hx=hx)

            # arc_h size [num_hyp, 1, arc_space]
            arc_h = F.elu(self.arc_h(output_dec.unsqueeze(1)))
            # type_h size [num_hyp, type_space]
            type_h = F.elu(self.type_h(output_dec))

            # [num_hyp, length_encoder]
            out_arc = self.attention(arc_h, arc_c.expand(num_hyp, *arc_c.size())).squeeze(dim=1).squeeze(dim=1)

            # [num_hyp, length_encoder]
            hyp_scores = F.log_softmax(out_arc, dim=1)

            new_hypothesis_scores = hypothesis_scores[:num_hyp].unsqueeze(1) + hyp_scores
            # [num_hyp * length_encoder]
            new_hypothesis_scores, hyp_index = torch.sort(new_hypothesis_scores.view(-1), dim=0, descending=True)
            base_index = hyp_index / length
            child_index = hyp_index % length

            cc = 0
            ids = []
            new_constraints = np.zeros([beam, length], dtype=np.bool)
            new_child_orders = np.zeros([beam, length], dtype=np.int64)
            new_parent_hn_dict = [{} for i in range(beam)]
            new_sib_hn_dict = [{} for i in range(beam)]
            for id in range(num_hyp * length):
                base_id = base_index[id]
                child_id = child_index[id]
                head = heads[base_id]
                new_hyp_score = new_hypothesis_scores[id]
                if child_id == head:
                    assert constraints[base_id, child_id], 'constrains error: %d, %d' % (base_id, child_id)
                    if head != 0 or t + 1 == num_step:
                        int_head = int(head)
                        new_parent_hn_dict[cc] = copy.copy(parent_hn_dict[base_id])
                        if int_head not in new_parent_hn_dict[cc]:
                            new_parent_hn_dict[cc][int_head] = []
                            for i in range(len(hx_parent)):
                                new_parent_hn_dict[cc][int_head].append(hx_parent[i][:,base_id,:].clone())
                        new_sib_hn_dict[cc] = copy.copy(sib_hn_dict[base_id])
                        new_sib_hn_dict[cc][int_head] = []
                        for i in range(len(hx)):
                            new_sib_hn_dict[cc][int_head].append(hx[i][:,base_id,:].clone())

                        new_constraints[cc] = constraints[base_id]
                        new_child_orders[cc] = child_orders[base_id]

                        new_stacked_heads[cc] = [stacked_heads[base_id][i] for i in range(len(stacked_heads[base_id]))]
                        new_stacked_heads[cc].pop()

                        if self.grandPar:
                            new_grand_parents[cc] = [grand_parents[base_id][i] for i in range(len(grand_parents[base_id]))]
                            new_grand_parents[cc].pop()

                        if self.sibling:
                            new_siblings[cc] = [siblings[base_id][i] for i in range(len(siblings[base_id]))]

                        if self.skipConnect:
                            new_skip_connects[cc] = [skip_connects[base_id][i] for i in range(len(skip_connects[base_id]))]

                        new_children[cc] = children[base_id]
                        new_children[cc, t] = child_id

                        hypothesis_scores[cc] = new_hyp_score
                        ids.append(id)
                        cc += 1
                elif valid_hyp(base_id, child_id, head):
                    int_head = int(head)
                    new_parent_hn_dict[cc] = copy.copy(parent_hn_dict[base_id])
                    if int_head not in new_parent_hn_dict[cc]:
                        new_parent_hn_dict[cc][int_head] = []
                        for i in range(len(hx_parent)):
                            new_parent_hn_dict[cc][int_head].append(hx_parent[i][:,base_id,:].clone())
                    new_sib_hn_dict[cc] = copy.copy(sib_hn_dict[base_id])
                    new_sib_hn_dict[cc][int_head] = []
                    for i in range(len(hx)):
                        new_sib_hn_dict[cc][int_head].append(hx[i][:,base_id,:].clone())

                    new_constraints[cc] = constraints[base_id]
                    new_constraints[cc, child_id] = True

                    new_child_orders[cc] = child_orders[base_id]
                    new_child_orders[cc, head] = child_id

                    new_stacked_heads[cc] = [stacked_heads[base_id][i] for i in range(len(stacked_heads[base_id]))]
                    new_stacked_heads[cc].append(child_id)

                    if self.grandPar:
                        new_grand_parents[cc] = [grand_parents[base_id][i] for i in range(len(grand_parents[base_id]))]
                        new_grand_parents[cc].append(head)

                    if self.sibling:
                        new_siblings[cc] = [siblings[base_id][i] for i in range(len(siblings[base_id]))]
                        new_siblings[cc].append(child_id)
                        new_siblings[cc].append(0)

                    if self.skipConnect:
                        new_skip_connects[cc] = [skip_connects[base_id][i] for i in range(len(skip_connects[base_id]))]
                        # hack to handle LSTM
                        if isinstance(hx, tuple):
                            new_skip_connects[cc].append(hx[0][:, base_id, :].unsqueeze(1))
                        else:
                            new_skip_connects[cc].append(hx[:, base_id, :].unsqueeze(1))
                        new_skip_connects[cc].append(h0)

                    new_children[cc] = children[base_id]
                    new_children[cc, t] = child_id

                    hypothesis_scores[cc] = new_hyp_score
                    ids.append(id)
                    cc += 1

                if cc == beam:
                    break

            # [num_hyp]
            num_hyp = len(ids)
            if num_hyp == 0:
                return None
            else:
                index = torch.from_numpy(np.array(ids)).type_as(base_index)
            base_index = base_index[index]
            child_index = child_index[index]

            # predict types for new hypotheses
            # compute output for type [num_hyp, num_labels]
            out_type = self.bilinear(type_h[base_index], type_c[child_index])
            hyp_type_scores = F.log_softmax(out_type, dim=1)
            # compute the prediction of types [num_hyp]
            hyp_type_scores, hyp_types = hyp_type_scores.max(dim=1)
            hypothesis_scores[:num_hyp] = hypothesis_scores[:num_hyp] + hyp_type_scores

            sib_hn_dict = new_sib_hn_dict
            parent_hn_dict = new_parent_hn_dict
            for i in range(num_hyp):
                base_id = base_index[i]
                new_stacked_types[i] = stacked_types[base_id]
                new_stacked_types[i, t] = hyp_types[i]

            stacked_heads = [[new_stacked_heads[i][j] for j in range(len(new_stacked_heads[i]))] for i in range(num_hyp)]
            if self.grandPar:
                grand_parents = [[new_grand_parents[i][j] for j in range(len(new_grand_parents[i]))] for i in range(num_hyp)]
            if self.sibling:
                siblings = [[new_siblings[i][j] for j in range(len(new_siblings[i]))] for i in range(num_hyp)]
            if self.skipConnect:
                skip_connects = [[new_skip_connects[i][j] for j in range(len(new_skip_connects[i]))] for i in range(num_hyp)]
            constraints = new_constraints
            child_orders = new_child_orders
            children.copy_(new_children)
            stacked_types.copy_(new_stacked_types)
            # hx [decoder_layers, num_hyp, hidden_size]
            # hack to handle LSTM
            if isinstance(hx, tuple):
                hx, cx = hx
                hx = hx[:, base_index, :]
                cx = cx[:, base_index, :]
                hx = (hx, cx)
            else:
                hx = hx[:, base_index, :]

        children = children.cpu().numpy()[0]
        stacked_types = stacked_types.cpu().numpy()[0]
        heads = np.zeros(length, dtype=np.int32)
        types = np.zeros(length, dtype=np.int32)
        stack = [0]
        for i in range(num_step):
            head = stack[-1]
            child = children[i]
            type = stacked_types[i]
            if child != head:
                heads[child] = head
                types[child] = type
                stack.append(child)
            else:
                stacked_types[i] = 0
                stack.pop()

        return heads, types, length, children, stacked_types


