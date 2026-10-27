"""Train a byte-level BPE tokenizer from the corpus, mirroring nanochat's recipe.

Same GPT-4-style pre-tokenization and byte-fallback BPE nanochat uses, so the
tokenizer is generated from whatever corpus you plug in rather than borrowed from
another model. Adds [MASK]/[PAD] for the MLM encoder and returns a
PreTrainedTokenizerFast (standard HF interface).
"""
from tokenizers import Tokenizer, pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

# GPT-4 split pattern (matches nanochat; \p{N}{1,2} for smaller vocabularies).
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = ["<|bos|>", "[PAD]", "[MASK]"]


def train_tokenizer(text_iterator, vocab_size):
    """Train a byte-level BPE on an iterator of strings; return PreTrainedTokenizerFast."""
    tok = Tokenizer(BPE(byte_fallback=True, unk_token=None, fuse_unk=False))
    tok.normalizer = None
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=Regex(SPLIT_PATTERN), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(
        vocab_size=vocab_size, show_progress=True, min_frequency=0,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=SPECIAL_TOKENS,
    )
    tok.train_from_iterator(text_iterator, trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token="<|bos|>", pad_token="[PAD]", mask_token="[MASK]",
    )
