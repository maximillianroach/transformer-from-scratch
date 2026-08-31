import torch
from .softmax import softmax


def decode(model, prompt, max_tokens, temperature, top_p, eos_token_id):
    tokens = prompt
    for i in range(max_tokens):
        # get the logits for the next token
        next_token_logits = model(tokens)[-1, :]

        # apply temperature scaling
        next_token_logits_temp = next_token_logits / temperature

        # apply softmax to logits
        probs = softmax(next_token_logits_temp, -1)

        # top-p sampling
        sorted_probs, indices = torch.sort(probs, descending=True, dim=-1)
        cur_p = 0
        j = 0
        nucleus = []
        while j < indices.shape[-1] and cur_p < top_p:
            nucleus.append(indices[0, 0, j].item())
            cur_p += sorted_probs[0, 0, j].item()
            j += 1

        nucleus = torch.tensor(nucleus, dtype=torch.long)

        # set all non-nucleus values to 0
        top_p_probs = torch.zeros((10000))
        top_p_probs[nucleus] = probs[0, 0][nucleus]

        # normalize so probabilities sum to 1
        top_p_probs /= torch.sum(top_p_probs, dim=-1)

        # sample from probs
        next_token = torch.multinomial(top_p_probs, num_samples=1)

        # append token to running tokens
        tokens = torch.cat([tokens, next_token.unsqueeze(1).unsqueeze(2)], dim=0)

        # if we get the eos token, then stop
        if next_token.item() == eos_token_id:
            break

    tokens = tokens.squeeze(2).squeeze(1).tolist()
    print(tokens)


    return tokens





    
