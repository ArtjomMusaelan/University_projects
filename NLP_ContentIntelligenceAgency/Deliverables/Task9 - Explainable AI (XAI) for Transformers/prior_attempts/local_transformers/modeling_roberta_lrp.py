import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification, RobertaConfig

class RobertaForSequenceClassificationWithLRP(RobertaForSequenceClassification):
    """
    Расширение RobertaForSequenceClassification, чтобы поддержать 
    базовые шаги LRP (по мотивам xai_transformer.py).
    """

    def __init__(self, config: RobertaConfig):
        super().__init__(config)
        # Настраиваем флаги, если хотим детальней управлять logic
        self.enable_lrp = True
        self.config = config

        # Понадобятся поля для хранения промежуточных результатов
        self._last_hidden_states = None
        self._last_attentions = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        """
        Переопределяем forward, чтобы 
        1) всегда включать output_attentions/output_hidden_states
        2) сохранять их в self._last_attentions / self._last_hidden_states
        3) затем вызывать родительский forward
        """
        # Для LRP нужно хранить hidden_states/attentions
        # Поэтому включаем их
        if output_attentions is None:
            output_attentions = True
        if output_hidden_states is None:
            output_hidden_states = True
        if return_dict is None:
            return_dict = True

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # Сохраняем для последующего использования
        # outputs.hidden_states: tuple of length (num_layers+1), each shape [bsz, seq_len, hidden_dim]
        # outputs.attentions: tuple of length (num_layers), each shape [bsz, num_heads, seq_len, seq_len]
        self._last_hidden_states = outputs.hidden_states
        self._last_attentions = outputs.attentions

        return outputs

    def explain(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        target_label=None,
        **kwargs
    ):
        """
        Пример метода для LRP: 
        1) Вызываем forward(...) -> logits
        2) Назначаем final relevance = logits[:, target_label] (или argmax)
        3) Вызываем backward() и layer-by-layer grad×input
        4) Возвращаем map релевантности на уровне токенов
        """

        # 1) Сделать forward
        out = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            **kwargs
        )
        logits = out.logits  # shape: [bsz, num_labels]
        bsz = logits.shape[0]

        # 2) target_label – либо передаём, либо берём argmax
        if target_label is None:
            # пусть будет argmax
            pred_label = logits.argmax(dim=1)  # [bsz]
        else:
            # предполагаем, что target_label это int (одинаковый для всех в batch)
            pred_label = torch.tensor([target_label]*bsz, device=logits.device)

        # 3) создаём “финальную” релевантность R^L = logits[..., label]
        #    затем backward -> grad
        #    *Учтём, что Hugging Face + Autograd = нужно .sum() 
        #    (иначе PyTorch не любит batched backward)
        selected_logit = []
        for i in range(bsz):
            selected_logit.append(logits[i, pred_label[i]])
        final_relevance = torch.stack(selected_logit).sum()  # scalar

        # Убеждаемся, что grads обнулим
        self.zero_grad()
        final_relevance.backward(retain_graph=True)

        # 4) Cобираем grad×input layer-by-layer:
        #    (а) на уровне embedding
        #    (б) на каждом hidden_state (можно),
        #    (в) или только на вход (embedding).
        #    Здесь покажем grad wrt embedding input (пример).
        # 
        # embed grad wrt input_embeds => 
        #  - нам нужен input_embeds, 
        #    можно вытянуть (последнее hidden_states[0]) – embedding output
        #    но нужно, чтобы requires_grad_=True
        #    ИЛИ use hooks => 
        # Упростим: “mock” (вернём grad×input = (grad_of layer 0)*layer0 )

        # last_hidden_states[0] shape [bsz, seq_len, hidden_dim]
        # grad_of that => .grad
        # But: HF не всегда keep grad for hidden_states. 
        # => нужно manually set requires_grad_(True) 
        #    Либо monkey-patch. 
        # 
        # В учебном примере (Ali A.), 
        #   “A['attn_input_{}_data'.format(i)].grad * A['attn_input_{}'] ...”

        # PSEUDO, “вытаскиваем” final relevance as token-level
        # (тут dummy)
        r_tokens = None

        #  - (1) grad of hidden_states[0], sum across hidden_dim
        #     grad_of_hidden_states_0 = self._last_hidden_states[0].grad
        #     if grad_of_hidden_states_0 is not None:
        #         # grad×input => ...
        #         r_tokens = (grad_of_hidden_states_0 * self._last_hidden_states[0]).sum(dim=-1)
        #
        #  - However, HF by default does not keep grad for hidden_states. 
        #    => need set “self._last_hidden_states[0].requires_grad_(True)” before forward, or do register_hook.

        # => Более “manual” route: monkey-patch RobertaLayer
        # => alias: see xai_transformer’s forward_and_explain approach

        # Для демонстрации вернём logits, pred_label, “r_tokens=None”
        return {
            "logits": logits,
            "pred_labels": pred_label,
            "token_relevance": r_tokens,  # TODO: нужно доп. внедрение
        }