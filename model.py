"""
Code adapted from MAC implementation: https://github.com/rosinality/mac-network-pytorch/blob/master/model.py
License: MIT
"""

import torch
from torch import nn
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F


def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()

    return lin


class ControlUnit(nn.Module):
    def __init__(self, dim, max_step):
        super().__init__()

        self.position_aware = nn.ModuleList()
        for i in range(max_step):
            self.position_aware.append(linear(dim * 2, dim))

        self.control_question = linear(dim * 2, dim)
        self.attn = linear(dim, 1)

        self.dim = dim

    def forward(self, step, context, question, control):
        position_aware = self.position_aware[step](question)

        control_question = torch.cat([control, position_aware], 1)
        control_question = self.control_question(control_question)
        control_question = control_question.unsqueeze(1)

        context_prod = control_question * context
        attn_weight = self.attn(context_prod)

        attn = F.softmax(attn_weight, 1)

        next_control = (attn * context).sum(1)

        return next_control


class ReadUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.mem = linear(dim, dim)
        self.concat = linear(dim * 2, dim)
        self.attn = linear(dim, 1)

    def forward(self, memory, know, control):
        mem = self.mem(memory[-1]).unsqueeze(2)
        concat = self.concat(torch.cat([mem * know, know], 1)
                                .permute(0, 2, 1))
        attn = concat * control[-1].unsqueeze(1)
        attn = self.attn(attn).squeeze(2)
        attn = F.softmax(attn, 1).unsqueeze(1)

        read = (attn * know).sum(2)

        return read


class WriteUnit(nn.Module):
    def __init__(self, dim, self_attention=False, memory_gate=False):
        super().__init__()

        self.concat = linear(dim * 2, dim)

        if self_attention:
            self.attn = linear(dim, 1)
            self.mem = linear(dim, dim)

        if memory_gate:
            self.control = linear(dim, 1)

        self.self_attention = self_attention
        self.memory_gate = memory_gate

    def forward(self, memories, retrieved, controls):
        prev_mem = memories[-1]
        concat = self.concat(torch.cat([retrieved, prev_mem], 1))
        next_mem = concat

        if self.self_attention:
            controls_cat = torch.stack(controls[:-1], 2)
            attn = controls[-1].unsqueeze(2) * controls_cat
            attn = self.attn(attn.permute(0, 2, 1))
            attn = F.softmax(attn, 1).permute(0, 2, 1)

            memories_cat = torch.stack(memories, 2)
            attn_mem = (attn * memories_cat).sum(2)
            next_mem = self.mem(attn_mem) + concat

        if self.memory_gate:
            control = self.control(controls[-1])
            gate = F.sigmoid(control)
            next_mem = gate * prev_mem + (1 - gate) * next_mem

        return next_mem


class MACUnit(nn.Module):
    def __init__(self, dim, max_step=12,
                self_attention=False, memory_gate=False,
                dropout=0.15):
        super().__init__()

        self.control = ControlUnit(dim, max_step)
        self.read = ReadUnit(dim)
        self.write = WriteUnit(dim, self_attention, memory_gate)

        self.mem_0 = nn.Parameter(torch.zeros(1, dim))
        self.control_0 = nn.Parameter(torch.zeros(1, dim))

        self.dim = dim
        self.max_step = max_step
        self.dropout = dropout

    def get_mask(self, x, dropout):
        mask = torch.empty_like(x).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)

        return mask

    def forward(self, context, question, knowledge):
        b_size = question.size(0)

        control = self.control_0.expand(b_size, self.dim)
        memory = self.mem_0.expand(b_size, self.dim)

        if self.training:
            control_mask = self.get_mask(control, self.dropout)
            memory_mask = self.get_mask(memory, self.dropout)
            control = control * control_mask
            memory = memory * memory_mask

        controls = [control]
        memories = [memory]

        for i in range(self.max_step):
            control = self.control(i, context, question, control)
            if self.training:
                control = control * control_mask
            controls.append(control)

            read = self.read(memories, knowledge, controls)
            memory = self.write(memories, read, controls)
            if self.training:
                memory = memory * memory_mask
            memories.append(memory)

        # normalization to [-1, 1] - values increase too much without it
        memory = memory - memory.min()
        memory = memory / memory.max()
        memory = 2*memory - 1
        
        return memory


class MACNetwork(nn.Module):
    def __init__(self, kb_dim, dim, embed_hidden=768,
                max_step=12, self_attention=False, memory_gate=False, dropout=0.15, 
                classes=2):
        super().__init__()
        
        self.lstm = nn.LSTM(embed_hidden, dim, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(kb_dim, dim, batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(dim*2, dim)
        
        self.mac = MACUnit(dim, max_step, self_attention, memory_gate, dropout)

        self.classifier = nn.Sequential(linear(dim + 2*dim + classes*2*dim, dim), nn.ELU(), linear(dim, classes)) 

        self.max_step = max_step
        self.dim = dim

    def forward(self, kb, question, answers):
        b_size = question.shape[0]
        
        lstm_out_q, (h_q, _) = self.lstm(question)
        h_q = h_q.permute(1, 0, 2).contiguous().view(b_size, -1)
        
        h_as = []
        for answer in answers:
            lstm_out_a, (h_a, _) = self.lstm(answer)
            h_as.append(h_a.permute(1, 0, 2).contiguous().view(b_size, -1))
        
        lstm_out_kb, (h_kb, _) = self.lstm2(kb)
        h_kb = h_kb.permute(1, 0, 2).contiguous().view(b_size, -1)
                
        lstm_out_q = self.lstm_proj(lstm_out_q)
        lstm_out_kb = self.lstm_proj(lstm_out_kb).view(b_size, self.dim, -1)
        
        memory = self.mac(lstm_out_q, h_q, lstm_out_kb)
        
        return self.classifier(torch.cat([h_q, memory, *h_as], 1))


class MACUnit_2RUs(nn.Module):
    def __init__(self, dim, max_step=12,
                self_attention=False, memory_gate=False,
                dropout=0.15):
        super().__init__()

        self.control = ControlUnit(dim, max_step)
        self.ru1 = ReadUnit(dim)
        self.ru2 = ReadUnit(dim)
        self.proj = linear(2*dim, dim)
        
        self.write = WriteUnit(dim, self_attention, memory_gate)

        self.mem_0 = nn.Parameter(torch.zeros(1, dim))
        self.control_0 = nn.Parameter(torch.zeros(1, dim))

        self.dim = dim
        self.max_step = max_step
        self.dropout = dropout

    def get_mask(self, x, dropout):
        mask = torch.empty_like(x).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)

        return mask

    def forward(self, context, question, kb1, kb2):
        b_size = question.size(0)

        control = self.control_0.expand(b_size, self.dim)
        memory = self.mem_0.expand(b_size, self.dim)

        if self.training:
            control_mask = self.get_mask(control, self.dropout)
            memory_mask = self.get_mask(memory, self.dropout)
            control = control * control_mask
            memory = memory * memory_mask

        controls = [control]
        memories = [memory]

        for i in range(self.max_step):
            control = self.control(i, context, question, control)
            if self.training:
                control = control * control_mask
            controls.append(control)

            read1 = self.ru1(memories, kb1, controls)
            read2 = self.ru2(memories, kb2, controls)
            
            # for concat fusion
            read = self.proj(torch.cat([read1, read2], 1))
            memory = self.write(memories, read, controls)
            if self.training:
                memory = memory * memory_mask
            memories.append(memory)

            # for iterative
#             memory = self.write(memories, read1, controls)
#             if self.training:
#                 memory = memory * memory_mask
#             memories.append(memory)
#             memory = self.write(memories, read2, controls) # new memories, same controls
#             if self.training:
#                 memory = memory * memory_mask
#             memories.append(memory)

        # normalization to [-1, 1] - values increase too much without it
        memory = memory - memory.min()
        memory = memory / memory.max()
        memory = 2*memory - 1
        
        return memory

    
class MACUnit_3RUs(nn.Module):
    def __init__(self, dim, max_step=12,
                self_attention=False, memory_gate=False,
                dropout=0.15):
        super().__init__()

        self.control = ControlUnit(dim, max_step)
        self.ru1 = ReadUnit(dim)
        self.ru2 = ReadUnit(dim)
        self.ru3 = ReadUnit(dim)
        self.proj = linear(3*dim, dim)
        
        self.write = WriteUnit(dim, self_attention, memory_gate)

        self.mem_0 = nn.Parameter(torch.zeros(1, dim))
        self.control_0 = nn.Parameter(torch.zeros(1, dim))

        self.dim = dim
        self.max_step = max_step
        self.dropout = dropout

    def get_mask(self, x, dropout):
        mask = torch.empty_like(x).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)

        return mask

    def forward(self, context, question, kb1, kb2, kb3):
        b_size = question.size(0)

        control = self.control_0.expand(b_size, self.dim)
        memory = self.mem_0.expand(b_size, self.dim)

        if self.training:
            control_mask = self.get_mask(control, self.dropout)
            memory_mask = self.get_mask(memory, self.dropout)
            control = control * control_mask
            memory = memory * memory_mask

        controls = [control]
        memories = [memory]

        for i in range(self.max_step):
            control = self.control(i, context, question, control)
            if self.training:
                control = control * control_mask
            controls.append(control)

            read1 = self.ru1(memories, kb1, controls)
            read2 = self.ru2(memories, kb2, controls)
            read3 = self.ru3(memories, kb3, controls)
            
            # for concat fusion
            read = self.proj(torch.cat([read1, read2, read3], 1))
            memory = self.write(memories, read, controls)
            if self.training:
                memory = memory * memory_mask
            memories.append(memory)

            # for iterative
#             memory = self.write(memories, read1, controls)
#             if self.training:
#                 memory = memory * memory_mask
#             memories.append(memory)
#             memory = self.write(memories, read2, controls) # new memories, same controls
#             if self.training:
#                 memory = memory * memory_mask
#             memories.append(memory)
        
        # normalization to [-1, 1] - values increase too much without it
        memory = memory - memory.min()
        memory = memory / memory.max()
        memory = 2*memory - 1
        
        return memory
    
    
class MACNetwork_2RUs(nn.Module):
    def __init__(self, visual_dim, dim, embed_hidden=768,
                max_step=12, self_attention=False, memory_gate=False, dropout=0.15, 
                classes=2):
        super().__init__()

        self.lstm = nn.LSTM(embed_hidden, dim, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(visual_dim, dim, batch_first=True, bidirectional=True)
        
        self.lstm_proj = nn.Linear(dim * 2, dim)
        
        self.mac = MACUnit_2RUs(dim, max_step, self_attention, memory_gate, dropout)

        self.classifier = nn.Sequential(linear(dim + 2*dim + classes*2*dim, dim), nn.ELU(), linear(dim, classes)) 

        self.max_step = max_step
        self.dim = dim

    def forward(self, kb1, kb2, question, answers):
        b_size = question.shape[0]
        
        lstm_out_q, (h_q, _) = self.lstm(question)
        h_q = h_q.permute(1, 0, 2).contiguous().view(b_size, -1)
        
        h_as = []
        for answer in answers:
            lstm_out_a, (h_a, _) = self.lstm(answer)
            h_as.append(h_a.permute(1, 0, 2).contiguous().view(b_size, -1))
        
        lstm_out_kb1, (h_kb1, _) = self.lstm(kb1)
        h_kb1 = h_kb1.permute(1, 0, 2).contiguous().view(b_size, -1)
        
        lstm_out_kb2, (h_kb2, _) = self.lstm2(kb2)
        h_kb2 = h_kb2.permute(1, 0, 2).contiguous().view(b_size, -1)
                
        lstm_out_q = self.lstm_proj(lstm_out_q)
        lstm_out_kb1 = self.lstm_proj(lstm_out_kb1).view(b_size, self.dim, -1)
        lstm_out_kb2 = self.lstm_proj(lstm_out_kb2).view(b_size, self.dim, -1)
        
        memory = self.mac(lstm_out_q, h_q, lstm_out_kb1, lstm_out_kb2)
        
        return self.classifier(torch.cat([h_q, memory, *h_as], 1))


class MACNetwork_3RUs(nn.Module):
    def __init__(self, visual_dim, ac_dim, dim, embed_hidden=768,
                max_step=12, self_attention=False, memory_gate=False, dropout=0.15, 
                classes=2):
        super().__init__()

        self.lstm = nn.LSTM(embed_hidden, dim, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(visual_dim, dim, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(ac_dim, dim, batch_first=True, bidirectional=True)
        
        self.lstm_proj = nn.Linear(dim * 2, dim)
        
        self.mac = MACUnit_3RUs(dim, max_step, self_attention, memory_gate, dropout)

        self.classifier = nn.Sequential(linear(dim + 2*dim + classes*2*dim, dim), nn.ELU(), linear(dim, classes)) 

        self.max_step = max_step
        self.dim = dim

    def forward(self, kb1, kb2, kb3, question, answers):
        b_size = question.shape[0]
        
        lstm_out_q, (h_q, _) = self.lstm(question)
        h_q = h_q.permute(1, 0, 2).contiguous().view(b_size, -1)
        
        h_as = []
        for answer in answers:
            lstm_out_a, (h_a, _) = self.lstm(answer)
            h_as.append(h_a.permute(1, 0, 2).contiguous().view(b_size, -1))
        
        lstm_out_kb1, (h_kb1, _) = self.lstm(kb1)
        h_kb1 = h_kb1.permute(1, 0, 2).contiguous().view(b_size, -1)
        
        lstm_out_kb2, (h_kb2, _) = self.lstm2(kb2)
        h_kb2 = h_kb2.permute(1, 0, 2).contiguous().view(b_size, -1)
        
        lstm_out_kb3, (h_kb3, _) = self.lstm3(kb3)
        h_kb3 = h_kb3.permute(1, 0, 2).contiguous().view(b_size, -1)
                
        lstm_out_q = self.lstm_proj(lstm_out_q)
        lstm_out_kb1 = self.lstm_proj(lstm_out_kb1).view(b_size, self.dim, -1)
        lstm_out_kb2 = self.lstm_proj(lstm_out_kb2).view(b_size, self.dim, -1)
        lstm_out_kb3 = self.lstm_proj(lstm_out_kb3).view(b_size, self.dim, -1)
        
        memory = self.mac(lstm_out_q, h_q, lstm_out_kb1, lstm_out_kb2, lstm_out_kb3)
        
        return self.classifier(torch.cat([h_q, memory, *h_as], 1))
    
    
class MACNetwork_LateFuse(nn.Module):
    def __init__(self, visual_dim, ac_dim, dim, embed_hidden=768,
                max_step=12, self_attention=False, memory_gate=False, dropout=0.15, 
                classes=2):
        super().__init__()

        self.lstm = nn.LSTM(embed_hidden, dim, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(visual_dim, dim, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(ac_dim, dim, batch_first=True, bidirectional=True)
        
        self.lstm_proj = nn.Linear(dim * 2, dim)
        
        self.mem_late_proj = nn.Linear(dim * 3, dim)
        
        self.mac = MACUnit(dim, max_step, self_attention, memory_gate, dropout)

        self.classifier = nn.Sequential(linear(dim + 2*dim + classes*2*dim, dim), nn.ELU(), linear(dim, classes)) 

        self.max_step = max_step
        self.dim = dim

    def forward(self, kb1, kb2, kb3, question, answers):
        b_size = question.shape[0]
        
        lstm_out_q, (h_q, _) = self.lstm(question)
        h_q = h_q.permute(1, 0, 2).contiguous().view(b_size, -1)
        
        h_as = []
        for answer in answers:
            lstm_out_a, (h_a, _) = self.lstm(answer)
            h_as.append(h_a.permute(1, 0, 2).contiguous().view(b_size, -1))
        
        lstm_out_kb1, (h_kb1, _) = self.lstm(kb1)
        h_kb1 = h_kb1.permute(1, 0, 2).contiguous().view(b_size, -1)
        
        lstm_out_kb2, (h_kb2, _) = self.lstm2(kb2)
        h_kb2 = h_kb2.permute(1, 0, 2).contiguous().view(b_size, -1)
        
        lstm_out_kb3, (h_kb3, _) = self.lstm3(kb3)
        h_kb3 = h_kb3.permute(1, 0, 2).contiguous().view(b_size, -1)
                
        lstm_out_q = self.lstm_proj(lstm_out_q)
        lstm_out_kb1 = self.lstm_proj(lstm_out_kb1).view(b_size, self.dim, -1)
        lstm_out_kb2 = self.lstm_proj(lstm_out_kb2).view(b_size, self.dim, -1)
        lstm_out_kb3 = self.lstm_proj(lstm_out_kb3).view(b_size, self.dim, -1)
        
        memory1 = self.mac(lstm_out_q, h_q, lstm_out_kb1)
        memory2 = self.mac(lstm_out_q, h_q, lstm_out_kb2)
        memory3 = self.mac(lstm_out_q, h_q, lstm_out_kb3)
        
        memory = self.mem_late_proj(torch.cat([memory1, memory2, memory3], 1))
        
        return self.classifier(torch.cat([h_q, memory, *h_as], 1))
