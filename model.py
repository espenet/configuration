from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
"""Sequential implementation of Recurrent Neural Network Language Model."""
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
from typeguard import check_argument_types

from espnet2.lm.abs_model import AbsLM


class SequentialRNNLM(AbsLM):

    def __init__(
        self,
        vocab_size: int,
        unit: int = 650,
        nhid: int = None,
        nlayers: int = 2,
        dropout_rate: float = 0.0,
        tie_weights: bool = False,
        rnn_type: str = "lstm",
        ignore_id: int = 0,
    ):
        assert check_argument_types()
        super().__init__()

        ninp = unit
        if nhid is None:
            nhid = unit
        rnn_type = rnn_type.upper()

        self.drop = nn.Dropout(dropout_rate)
        self.encoder = nn.Embedding(vocab_size, ninp, padding_idx=ignore_id)
        if rnn_type in ["LSTM", "GRU"]:
            rnn_class = getattr(nn, rnn_type)
            self.rnn = rnn_class(
                ninp, nhid, nlayers, dropout=dropout_rate, batch_first=True
            )
        else:
            try:
                nonlinearity = {"RNN_TANH": "tanh", "RNN_RELU": "relu"}[rnn_type]
            except KeyError:
                raise ValueError(
                    """An invalid option for `--model` was supplied,
                    options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']"""
                )
            self.rnn = nn.RNN(
                ninp,
                nhid,
                nlayers,
                nonlinearity=nonlinearity,
                dropout=dropout_rate,
                batch_first=True,
            )
        self.decoder = nn.Linear(nhid, vocab_size)

        if tie_weights:
            if nhid != ninp:
                raise ValueError(
                    "When using the tied flag, nhid must be equal to emsize"
                )
            self.decoder.weight = self.encoder.weight

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

   def forward(
        self, input: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(
            output.contiguous().view(output.size(0) * output.size(1), output.size(2))
        )
        return (
            decoded.view(output.size(0), output.size(1), decoded.size(1)),
            hidden,
        )

    def score(
        self,
        y: torch.Tensor,
        state: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        
        y, new_state = self(y[-1].view(1, 1), state)
        logp = y.log_softmax(dim=-1).view(-1)
        return logp, new_state


    def batch_score(
        self, ys: torch.Tensor, states: torch.Tensor, xs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states[0] is None:
            states = None
        elif isinstance(self.rnn, torch.nn.LSTM):
            # states: Batch x 2 x (Nlayers, Dim) -> 2 x (Nlayers, Batch, Dim)
            h = torch.stack([h for h, c in states], dim=1)
            c = torch.stack([c for h, c in states], dim=1)
            states = h, c
        else:
            # states: Batch x (Nlayers, Dim) -> (Nlayers, Batch, Dim)
            states = torch.stack(states, dim=1)

        ys, states = self(ys[:, -1:], states)
        # ys: (Batch, 1, Nvocab) -> (Batch, NVocab)
        assert ys.size(1) == 1, ys.shape
        ys = ys.squeeze(1)
        logp = ys.log_softmax(dim=-1)

        # state: Change to batch first
        if isinstance(self.rnn, torch.nn.LSTM):
            # h, c: (Nlayers, Batch, Dim)
            h, c = states
            # states: Batch x 2 x (Nlayers, Dim)
            states = [(h[:, i], c[:, i]) for i in range(h.size(1))]
        else:
            # states: (Nlayers, Batch, Dim) -> Batch x (Nlayers, Dim)
            states = [states[:, i] for i in range(states.size(1))]

        return logp, states
def training():
    training_args = TrainingArguments(
  	group_by_length=True,
  	per_device_train_batch_size=16,
  	gradient_accumulation_steps=2,
  	evaluation_strategy="steps",
  	num_train_epochs=30,
  	fp16=True,
  	save_steps=100,
  	eval_steps=100,
  	logging_steps=10,
  	learning_rate=3e-4,
  	warmup_steps=500,
  	save_total_limit=2,
     )

trainer = Trainer(
    model=SequentialRNNLM,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=common_voice_train,
    eval_dataset=common_voice_test,
    tokenizer=processor.feature_extractor,
)
trainer.train()
