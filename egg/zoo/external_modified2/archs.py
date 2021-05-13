# adapted from https://github.com/facebookresearch/EGG

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import RelaxedOneHotCategorical
from torch.distributions import Categorical

from egg.core.gs_wrappers import gumbel_softmax_sample, RnnReceiverGS
from egg.core.interaction import LoggingStrategy


class Receiver(nn.Module):
    def __init__(self, output_size, n_hidden):
        super(Receiver, self).__init__()
        self.output = nn.Linear(n_hidden, output_size)

    def forward(self, x, _input):
        return torch.sigmoid(self.output(x))


class Sender(nn.Module):
    def __init__(self, n_hidden, n_features):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)

    def forward(self, x):
        x = self.fc1(x)
        return x

class RnnSenderGS(nn.Module):
    def __init__(
        self,
        agent,
        vocab_size,
        embed_dim,
        hidden_size,
        max_len,
        temperature,
        cell="rnn",
        trainable_temperature=False,
        straight_through=False,
    ):
        super(RnnSenderGS, self).__init__()
        self.agent = agent

        assert max_len >= 1, "Cannot have a max_len below 1"
        self.max_len = max_len

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.embedding = nn.Linear(vocab_size, embed_dim)
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        if not trainable_temperature:
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature]), requires_grad=True
            )

        self.straight_through = straight_through
        self.cell = None

        cell = cell.lower()

        if cell == "rnn":
            self.cell = nn.RNNCell(input_size=embed_dim, hidden_size=hidden_size)
        elif cell == "gru":
            self.cell = nn.GRUCell(input_size=embed_dim, hidden_size=hidden_size)
        elif cell == "lstm":
            self.cell = nn.LSTMCell(input_size=embed_dim, hidden_size=hidden_size)
        else:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def forward(self, x):
        prev_hidden = self.agent(x)
        prev_c = torch.zeros_like(prev_hidden)  # only for LSTM

        e_t = torch.stack([self.sos_embedding] * prev_hidden.size(0))
        sequence = []
        logits = []
        for step in range(self.max_len):
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = self.cell(e_t, (prev_hidden, prev_c))
            else:
                h_t = self.cell(e_t, prev_hidden)

            step_logits = self.hidden_to_output(h_t)

            x = gumbel_softmax_sample(
                step_logits, self.temperature, self.training, self.straight_through
            )
            prev_hidden = h_t
            e_t = self.embedding(x)
            sequence.append(x)
            logits.append(step_logits)

        sequence = torch.stack(sequence).permute(1, 0, 2)
        logits = torch.stack(logits).permute(1, 0, 2)

        eos = torch.zeros_like(sequence[:, 0, :]).unsqueeze(1)
        eos[:, 0, 0] = 1
        sequence = torch.cat([sequence, eos], dim=1)

        return sequence, logits


class SenderReceiverRnnGS(nn.Module):
    def __init__(
        self,
        sender,
        receiver,
        loss,
        train_logging_strategy: Optional[LoggingStrategy] = None,
        test_logging_strategy: Optional[LoggingStrategy] = None,
    ):
        super(SenderReceiverRnnGS, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss
        self.train_logging_strategy = (
            LoggingStrategy()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy()
            if test_logging_strategy is None
            else test_logging_strategy
        )

    def forward(self, sender_input, labels, receiver_input=None):
        message, logits = self.sender(sender_input)
        receiver_output = self.receiver(message, receiver_input)

        loss = 0.0
        not_eosed_before = torch.ones(receiver_output.size(0)).to(
            receiver_output.device
        )
        expected_length = 0.0

        aux_info = {}
        z = 0.0
        for step in range(receiver_output.size(1)):
            eos_mask = message[:, step, 0]  # always eos == 0

            add_mask = eos_mask * not_eosed_before
            z += add_mask

            expected_length += add_mask.detach() * (1.0 + step)

            step_loss, step_aux = self.loss(
                sender_input,
                step+1,
                message[:, 0:step+1, ...],
                logits[:, 0:step+1, ...],
                receiver_input,
                receiver_output[:, step, ...],
                labels,
            )
            loss += step_loss * add_mask
            for name, value in step_aux.items():
                aux_info[name] = value * add_mask + aux_info.get(name, 0.0)

            not_eosed_before = not_eosed_before * (1.0 - eos_mask)

        # the remainder of the probability mass
        loss += (step_loss * not_eosed_before)
        expected_length += (step + 1) * not_eosed_before

        z += not_eosed_before
        assert z.allclose(
            torch.ones_like(z)
        ), f"lost probability mass, {z.min()}, {z.max()}"

        for name, value in step_aux.items():
            aux_info[name] = value * not_eosed_before + aux_info.get(name, 0.0)

        aux_info["length"] = expected_length

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            receiver_output=receiver_output.detach(),
            message=message.detach(),
            message_length=expected_length.detach(),
            aux=aux_info,
        )

        return loss.mean(), interaction

class BiReceiver(nn.Module):
    def __init__(self, output_size, n_hidden):
        super(BiReceiver, self).__init__()
        self.output = nn.Linear(2 * n_hidden, output_size)

    def forward(self, x, _input):
        return torch.sigmoid(self.output(x))

class BiRnnReceiverSTGS(nn.Module):
    def __init__(self, agent, vocab_size, embed_dim, hidden_size, cell="rnn", num_layers=1):
        super(BiRnnReceiverSTGS, self).__init__()
        self.agent = agent

        if cell == 'rnn':
            self.rnn = nn.RNN(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        elif cell == 'gru':
            self.rnn = nn.GRU(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        elif cell == 'lstm':
            self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        else:
            raise ValueError(f"Unknown RNN: {cell}") 

        self.embedding = nn.Linear(vocab_size, embed_dim)

    def forward(self, message, input=None):
        eosed = torch.zeros(message.size(0)).to(
            message.device
        )
        F_lens = torch.ones(message.size(0)).to(
            message.device
        )
        for step in range(message.size(1)):
            eos_mask = message[:, step, 0]  # always eos == 0
            eosed += eos_mask * (1 - eosed)
            F_lens += (1 - eosed)

        x = self.embedding(message)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, F_lens, enforce_sorted=False, batch_first=True)
        x, h = self.rnn(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, padding_value=0, batch_first=True)

        N, S, H = x.size()
        forward = torch.stack([x[i, int(F_lens[i])-1, 0:H//2] for i in range(N)])
        backward = x[:, 0, H//2:H]

        return self.agent(torch.cat((forward, backward), dim=1), input)

class SenderReceiverBiRnnSTGS(nn.Module):
    def __init__(
        self,
        sender,
        receiver,
        loss,
        train_logging_strategy: Optional[LoggingStrategy] = None,
        test_logging_strategy: Optional[LoggingStrategy] = None,
    ):
        super(SenderReceiverBiRnnSTGS, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss
        self.train_logging_strategy = (
            LoggingStrategy()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy()
            if test_logging_strategy is None
            else test_logging_strategy
        )

    def forward(self, sender_input, labels, receiver_input=None):
        message, logits = self.sender(sender_input)
        receiver_output = self.receiver(message, receiver_input)

        aux_info = {}
        loss, aux = self.loss(
            sender_input,
            0,
            message,
            logits,
            receiver_input,
            receiver_output,
            labels,
        )
        for name, value in aux.items():
            aux_info[name] = value

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            receiver_output=receiver_output.detach(),
            message=message.detach(),
            message_length=None,
            aux=aux_info,
        )

        return loss.mean(), interaction


