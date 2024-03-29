from math import isnan

import torch
from torch.distributions import Categorical
from transformers import Seq2SeqTrainer

from tw_rouge import get_rouge
from collections import deque


class RLSeq2SeqTrainer(Seq2SeqTrainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        if isnan(outputs.loss.item()):
            print("warning: ### loss is nan")
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        loss = self._policy_gradient(outputs.logits, inputs['labels'])

        return (loss, outputs) if return_outputs else loss

    def _policy_gradient(self, logits, labels, gamma=0.9):
        '''
        Vanilla policy gradient
        '''
        def get_reward(prediction, label):
            try:
                baseline = {
                    'rouge-1': 0.22,
                    'rouge-2': 0.085,
                    'rouge-l': 0.020,
                }

                rouge_scores = get_rouge(prediction, label)
                rouge_scores = {k: v['f'] for k, v in rouge_scores.items()}
                combined_rouge = 0
                for k, v in rouge_scores.items():
                    if k in baseline:
                        combined_rouge += v / baseline[k]
                    else:
                        raise Exception(f"{k} not in get_rouge")

                return combined_rouge
            except ValueError:
                return 0

        device = logits.device
        eps = 1e-6

        all_policys = []
        all_rewards = []
        for logit, label_ids in zip(logits, labels):
            generated_sequence = []
            for idx, single_logit in enumerate(logit):
                word_distribution = Categorical(logits=single_logit)
                next_word = word_distribution.sample()
                generated_sequence.append(next_word)
                all_policys.append(
                    word_distribution.log_prob(next_word).unsqueeze(0))
                if generated_sequence[-1] == self.tokenizer.eos_token_id or idx >= self.args.generation_max_length:
                    break
            generated_sequence = torch.tensor(generated_sequence).to()
            generated_sequence = self.tokenizer.decode(
                generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            label_ids = label_ids.detach().cpu().tolist()
            label_ids = [l for l in label_ids if l > 0]
            label_text = self.tokenizer.decode(
                label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            R = 0 if generated_sequence == "" else get_reward(
                generated_sequence, label_text)
            rewards = deque()
            for _ in range(idx + 1):
                rewards.appendleft(R)
                R *= gamma
            all_rewards.extend(rewards)

        all_policys = torch.cat(all_policys)

        all_rewards = torch.tensor(all_rewards).to(device)
        all_rewards = (all_rewards - all_rewards.mean()) / \
            (all_rewards.std() + eps)  # standardize
        # \times -1 = maximization
        loss = torch.sum(-1 * all_policys * all_rewards, -1)
        return loss
