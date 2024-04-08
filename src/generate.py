import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from .model import NanoLlama
from .utils import load_checkpoint, get_model_size


class ArticleGenerator:

    def __init__(self, model, tokenizer, end_token):
        self.model = model
        self.tokenizer = tokenizer
        self.end_token = end_token

    @classmethod
    def from_config(
        cls, 
        checkpoint_path, 
        tokenizer_path,
        model_size,
        context_size = 256, 
        end_token = "[END-OF-TEXT]", 
    ):
        tokenizer = Tokenizer.from_file(tokenizer_path)
        vocab_size = tokenizer.get_vocab_size()
        model_params = get_model_size(model_size)
        model = NanoLlama(vocab_size, context_size, **model_params)
        _ = load_checkpoint(checkpoint_path, model)
        model.eval()
        return cls(model, tokenizer, end_token)

    @torch.no_grad()
    def generate(
        self,
        text: str,
        max_tokens: int = 256,
        beam_size: int = 1,
        top_k: int = 0,
        top_p: float = 0.0,
        temperature: float = 1.0,
        length_penalty: float = 0.0,
        alpha: float = 0.0
    ):
        tokens = torch.tensor([self.tokenizer.encode(text).ids], dtype=torch.long)

        # Based on given parameters choose the appropriate generation method.
        if beam_size > 0 and top_k == 0 and top_p == 0:
            tokens = self.beam_search(tokens, max_tokens=max_tokens, beam_size=beam_size, length_penalty=length_penalty)

        elif top_k > 0 and top_p == 0 and alpha == 0:
            tokens = self.top_k(tokens, max_tokens=max_tokens, top_k=top_k, temperature=temperature)

        elif top_k == 0 and top_p > 0 and alpha == 0:
            tokens = self.top_p(tokens, max_tokens=max_tokens, top_p=top_p, temperature=temperature)

        elif top_k > 0 and top_p == 0 and alpha > 0:
            tokens = self.contrastive_search(tokens, max_tokens=max_tokens, top_k=top_k, alpha=alpha)

        article = self.tokenizer.decode(tokens)
        return article
    
    def beam_search(self, tokens, max_tokens, beam_size, length_penalty):
        beam_candidates = [{'text': tokens, 'score': 0.0}]

        for i in range(tokens.shape[1], max_tokens):
            next_beam_candidates = []
            for candidate in beam_candidates:
                partial_text = candidate['text']
                logits = self.model(partial_text)
                probabilities = F.softmax(logits[0, i - 1], dim=-1)

                if beam_size > 1:
                    top_tokens = probabilities.topk(beam_size)[1].unsqueeze(1).unsqueeze(1)
                else:
                    # If beam size is 1, directly choose the token with the highest probability.
                    # This is equivalent to greedy search.
                    top_token = probabilities.argmax()
                    top_tokens = torch.tensor([[[top_token.item()]]])

                for token in top_tokens:
                    ne_text = partial_text.clone()
                    ne_text = torch.cat((ne_text, token), dim=-1)
                    new_score = candidate['score'] + torch.log(probabilities[token])

                    # Apply length penalty to the score.
                    penalty = ((5 + i) / 6) ** length_penalty
                    new_score /= penalty
                    next_beam_candidates.append({'text': ne_text, 'score': new_score})

            next_beam_candidates.sort(key=lambda x: x['score'], reverse=True)
            beam_candidates = next_beam_candidates[:beam_size]
            if all(self.__check_end_token(candidate['text'][:, i].item()) for candidate in beam_candidates):
                break

        best_text = beam_candidates[0]['text']
        return best_text.squeeze(0).numpy()
    
    def top_k(self, tokens, max_tokens, top_k, temperature):

        for i in range(tokens.shape[1], max_tokens):
            logits = self.model(tokens)
            logits = logits[0, i - 1] / temperature
            probabilities = F.softmax(logits, dim=-1)

            top_k_values, top_k_indices = probabilities.topk(top_k)
            top_k_values = top_k_values / torch.sum(top_k_values)
            next_token_index = torch.multinomial(top_k_values, num_samples=1)
            next_token = top_k_indices[next_token_index].unsqueeze(0)
            tokens = torch.cat((tokens, next_token), dim=-1)
            if self.__check_end_token(next_token):
                break

        return tokens.squeeze(0).numpy()
    
    def top_p(self, tokens, max_tokens, top_p, temperature):

        for i in range(tokens.shape[1], max_tokens):
            logits = self.model(tokens)
            logits = logits[0, i - 1] / temperature
            probs = F.softmax(logits, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            idx_to_remove = cum_probs > top_p

            # If small value of top_p is passed, make sure to always keep at least one token.
            idx_to_remove[1:] = idx_to_remove[:-1].clone()
            idx_to_remove[0] = False

            sorted_probs[idx_to_remove] = 0
            norm_probs = sorted_probs / torch.sum(sorted_probs)
            next_token_index = torch.multinomial(norm_probs, num_samples=1)
            next_token = sorted_indices[next_token_index].unsqueeze(0)
            tokens = torch.cat((tokens, next_token), dim=-1)
            if self.__check_end_token(next_token):
                break
        
        return tokens.squeeze(0).numpy()
    
    def contrastive_search(self, tokens, max_tokens, top_k, alpha):
        # Initialize the list of word embeddings with the embedding of the first tokens.
        word_embeddings_matrix = self.model.embeddings.weight
        word_embeddings = [word_embeddings_matrix[i] for i in tokens.squeeze(0)]

        for i in range(tokens.shape[1], max_tokens):
            logits = self.model(tokens)
            probabilities = F.softmax(logits[0, i - 1], dim=-1)
            confidences, indices = probabilities.topk(top_k)
            scores = []
            for conf, ind in zip(confidences, indices):
                degeneration_penalty = max([torch.cosine_similarity(word_embeddings_matrix[ind], word, dim=-1) for word in word_embeddings])
                scores.append((1 - alpha) * conf - alpha * degeneration_penalty)
            index = scores.index(max(scores))
            next_token = indices[index].unsqueeze(0).unsqueeze(0)
            tokens = torch.cat((tokens, next_token), dim=-1)
            word_embeddings.append(word_embeddings_matrix[next_token].squeeze(0))
            if self.__check_end_token(next_token):
                break
        
        return tokens.squeeze(0).numpy()


    def __check_end_token(self, token):
        return token == self.tokenizer.token_to_id(self.end_token)
