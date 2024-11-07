import random

from torch import nn


class MaskGenerator(nn.Module):
    """Mask generator."""

    def __init__(self,):
        super().__init__()
        self.sort = True

    def uniform_rand(self, num_tokens, mask_ratio):
        mask = list(range(int(num_tokens)))
        random.shuffle(mask)
        mask_len = int(num_tokens * mask_ratio)
        self.masked_tokens = mask[:mask_len]
        self.unmasked_tokens = mask[mask_len:]
        if self.sort:
            self.masked_tokens = sorted(self.masked_tokens)
            self.unmasked_tokens = sorted(self.unmasked_tokens)
        return self.unmasked_tokens, self.masked_tokens

    def forward(self, num_tokens, mask_ratio):
        self.unmasked_tokens, self.masked_tokens = self.uniform_rand(num_tokens, mask_ratio)
        return self.unmasked_tokens, self.masked_tokens
