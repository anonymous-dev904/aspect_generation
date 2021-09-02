# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from typing import Optional, Tuple
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import BartClassificationHead

class BartAttentionMulti(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_2: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        key_states = self._shape(self.k_proj(hidden_states_2), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states_2), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask_2 is not None:
            if attention_mask_2.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask_2.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask_2
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, None


def _expand_mask_multi(mask: torch.Tensor, mask_2: torch.Tensor, dtype: torch.dtype):
    bsz, src_len = mask.size()
    bsz_2, src_len_2 = mask_2.size()
    expanded_mask_1 = mask[:, None, :].expand(bsz, 1, src_len).to(dtype)
    expanded_mask_2 = mask_2[:, None, :].expand(bsz, 1, src_len_2).to(dtype)
    expanded_mask = torch.bmm(expanded_mask_1.transpose(1, 2), expanded_mask_2)
    expanded_mask = expanded_mask[:, None, :, :].expand(bsz, 1, expanded_mask.shape[1], expanded_mask.shape[2]).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)

class BartEncoderLayerMulti(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BartAttentionMulti(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            # dropout=config.attention_dropout,
            dropout=0.2
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # self.dropout = config.dropout
        self.dropout = 0.2
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_2: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_2: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states_2 = self.self_attn_layer_norm(hidden_states_2)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            hidden_states_2=hidden_states_2,
            attention_mask=attention_mask,
            attention_mask_2=attention_mask_2,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        # residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        # hidden_states = self.final_layer_norm(hidden_states)

        # if hidden_states.dtype == torch.float16 and (
        #     torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        # ):
        #     clamp_value = torch.finfo(hidden_states.dtype).max - 1000
        #     hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        return outputs

class Seq2Seq(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model. 
        * `beam_size`- beam size for beam search. 
        * `max_length`- max length of target for beam search. 
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """
    def __init__(self, encoder,bart_model,config,config_2,beam_size=None,max_length=None,sos_id=None,eos_id=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.bart_model=bart_model.model
        self.config=config
        self.config_2 = config_2
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config_2.hidden_size, config_2.hidden_size)
        self.lm_head = nn.Linear(config_2.hidden_size, config_2.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()
        
        self.beam_size=beam_size
        self.max_length=max_length
        self.sos_id=sos_id
        self.eos_id=eos_id

        # self.encode_mix = BartAttentionMulti(
        #     embed_dim=config.d_model,
        #     num_heads=config.encoder_attention_heads,
        #     dropout=config.attention_dropout,
        # )

        self.encode_mix = BartEncoderLayerMulti(
            config
        )
        self.type_classes = 2
        self.patch_classifier = BartClassificationHead(input_dim=config.d_model, inner_dim=config.d_model, num_classes=self.type_classes, pooler_dropout=0.1)
        self.bart_model._init_weights(self.patch_classifier.dense)
        self.bart_model._init_weights(self.patch_classifier.out_proj)


    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config_2.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight
                  
    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        # self._tie_or_clone_weights(self.lm_head,
        #                            self.bart_model.encoder.embeddings.word_embeddings)
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)

    def patch_classfifcation(self, encoder_output):
        classifier_input = torch.mean(encoder_output[:, :, :], dim=1)
        output = self.patch_classifier(classifier_input)
        return output
        
    def forward(self, source_ids=None,source_mask=None, source_ids_2=None,source_mask_2=None,target_ids=None,target_mask=None,extra_labels=None):
        outputs_2 = self.encoder(source_ids_2, attention_mask=source_mask_2)
        encoder_outputs_1 = self.bart_model.encoder(source_ids, attention_mask=source_mask)

        if source_mask_2 is not None and source_mask is not None:
            expand_attention_mask_2 = _expand_mask_multi(source_mask, source_mask_2, torch.float32)

        outputs =self.encode_mix(hidden_states=encoder_outputs_1[0], hidden_states_2=outputs_2[0],
                                              attention_mask_2=expand_attention_mask_2)
        # outputs = encoder_outputs_1[0] + outputs_2[0]
        # encoder_output = outputs[0].permute([1,0,2]).contiguous()
        encoder_output = outputs[0]
        patch_outputs = self.patch_classfifcation(encoder_output)
        # if source_mask[0]
        if target_ids is not None:
            # attn_mask=-1e4 *(1-self.bias[:target_ids.shape[1],:target_ids.shape[1]])
            # tgt_embeddings = self.encoder.embeddings(target_ids).permute([1,0,2]).contiguous()
            # tgt_embeddings = self.bart_model.encoder.embed_tokens(target_ids)
            # out = self.bart_model.decoder(input_ids=target_ids,encoder_hidden_states=encoder_output,attention_mask=target_mask,encoder_attention_mask=source_mask)
            # # hidden_states = torch.tanh(self.dense(out)).permute([1,0,2]).contiguous()
            # hidden_states = torch.tanh(self.dense(out['last_hidden_state']))
            # lm_logits = self.lm_head(hidden_states)
            # # Shift so that tokens < n predict n
            # active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            # shift_logits = lm_logits[..., :-1, :].contiguous()
            # shift_labels = target_ids[..., 1:].contiguous()
            # # Flatten the tokens
            # loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
            #                 shift_labels.view(-1)[active_loss])
            #
            # outputs = loss,loss*active_loss.sum(),active_loss.sum()
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(patch_outputs, extra_labels.view(-1))
            return masked_lm_loss, None, None
        else:
            #Predict 
            # preds=[]
            # zero=torch.cuda.LongTensor(1).fill_(0)
            # for i in range(source_ids.shape[0]):
            #     context=encoder_output[i:i+1,:,:]
            #     context_mask=source_mask[i:i+1,:]
            #     beam = Beam(self.beam_size,self.sos_id,self.eos_id)
            #     input_ids=beam.getCurrentState()
            #     context=context.repeat(self.beam_size,1, 1)
            #     context_mask=context_mask.repeat(self.beam_size,1)
            #     for _ in range(self.max_length):
            #         if beam.done():
            #             break
            #         # attn_mask=-1e4 *(1-self.bias[:input_ids.shape[1],:input_ids.shape[1]])
            #         # tgt_embeddings = self.bart_model.encoder.embeddings(input_ids)
            #         out = self.bart_model.decoder(input_ids=input_ids,encoder_hidden_states=context,attention_mask=None,encoder_attention_mask=context_mask)
            #         out = torch.tanh(self.dense(out['last_hidden_state']))
            #         hidden_states=out[:,-1,:]
            #         out = self.lsm(self.lm_head(hidden_states)).data
            #         beam.advance(out)
            #         input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
            #         input_ids=torch.cat((input_ids,beam.getCurrentState()),-1)
            #     hyp= beam.getHyp(beam.getFinal())
            #     pred=beam.buildTargetTokens(hyp)[:self.beam_size]
            #     pred=[torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred]
            #     preds.append(torch.cat(pred,0).unsqueeze(0))
            #
            # preds=torch.cat(preds,0)
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(patch_outputs, extra_labels.view(-1))
            return masked_lm_loss, patch_outputs
        
        

class Beam(object):
    def __init__(self, size,sos,eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))


        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >=self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished=[]
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i)) 
            unfinished.sort(key=lambda a: -a[0])
            self.finished+=unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps=[]
        for _,timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps
    
    def buildTargetTokens(self, preds):
        sentence=[]
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok==self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
        
