import torch
import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel


class Model(BertPreTrainedModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config=config)

        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size * 3 , config.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask, e1_mask, e2_mask):
        bert_outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output, pooled_output = bert_outputs[:2]
        # Average
        e1_h = self.entity_pooling(sequence_output, e1_mask, pooling_function='average')
        e2_h = self.entity_pooling(sequence_output, e2_mask, pooling_function='average')

        pooled_output = self.dropout(pooled_output)

        concat_h = torch.cat([pooled_output,e1_h,e2_h], dim=-1)

        output = self.classifier(concat_h)

        return output

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]

        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    @staticmethod
    def entity_pooling(hidden_output, e_mask, pooling_function='max'):
        """
        Pool the entity hidden state vectors (H_i ~ H_j) with max-pooling in default.
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :param pooling_function: str in ['max','average','start']
        :return: [batch_size, dim]
        """
        hidden_size = hidden_output.size()[-1]
        e_mask = e_mask.unsqueeze(-1).bool()
        e_mask = e_mask.expand_as(hidden_output)

        pooled_hidden_output = []

        for h, m in zip(hidden_output, e_mask):
            maked_hidden = torch.masked_select(h, m)
            maked_hidden = maked_hidden.view(-1, hidden_size)
            # pooling function
            if pooling_function == 'start':
                pooled_hidden = maked_hidden[0, :]
            elif pooling_function == 'average':
                pooled_hidden = torch.mean(maked_hidden, 0)
            else:
                pooled_hidden, _ = torch.max(maked_hidden, 0)
            pooled_hidden_output.append(pooled_hidden.unsqueeze(0))

        hidden_output = torch.cat(pooled_hidden_output)

        return hidden_output
