from dataclasses import dataclass
from typing import Optional
import struct
import sys
import random
import time
import argparse

import numpy as np


@dataclass
class Config:
    dim: int
    hidden_dim: Optional[int]
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    seq_len: int
    # multiple_of: int
    # norm_eps: float
    # drop_out: float


# @dataclass
class Weights:
    token_embedding_table: np.array
    rms_att_weight: np.array
    wq: np.array
    wk: np.array
    wv: np.array
    wo: np.array
    rms_ffn_weight: np.array
    w1: np.array
    w2: np.array
    w3: np.array
    rms_final_weight: np.array
    freq_cis_real: np.array
    freq_cis_imag: np.array
    wcls: np.array


@dataclass
class RunState:
    x: np.array  # (dim, )
    xb: np.array  # (dim, )
    xb2: np.array  # (dim, )
    hb: np.array  # (hidden_dim, )
    hb2: np.array  # (hidden_dim, )
    q: np.array  # (dim, )
    k: np.array  # (dim, )
    v: np.array  # (dim, )
    att: np.array  # (n_heads, seq_len)
    logits: np.array  # (dim, )
    key_cache: np.array  # (layer, seq_len, kv_dim)
    value_cache: np.array  # (layer, seq_len, kv_dim)


@dataclass
class Transformer:
    config: Config
    weights: Weights
    state: RunState


def init_weight(weights: Weights, arg: Config, i: int, data: np.array, shared_weights):
    # def read(count):
    #     values = struct.unpack(str(count)+'f', file.read(count * 4))
    #     return np.array(values)
    head_size = arg.dim // arg.n_heads
    n_layers = arg.n_layers
    weights.token_embedding_table = data[i : i + arg.vocab_size * arg.dim]
    i += arg.vocab_size * arg.dim
    weights.rms_att_weight = data[i : i + n_layers * arg.dim]
    i += n_layers * arg.dim
    weights.wq = data[i : i + n_layers * arg.dim * (arg.n_heads * head_size)]
    i += n_layers * arg.dim * (arg.n_heads * head_size)

    weights.wk = data[i : i + n_layers * arg.dim * (arg.n_kv_heads * head_size)]
    i += n_layers * arg.dim * (arg.n_kv_heads * head_size)

    weights.wv = data[i : i + n_layers * arg.dim * (arg.n_kv_heads * head_size)]
    i += n_layers * arg.dim * (arg.n_kv_heads * head_size)

    weights.wo = data[i : i + n_layers * (arg.n_heads * head_size) * arg.dim]
    i += n_layers * (arg.n_heads * head_size) * arg.dim

    weights.rms_ffn_weight = data[i : i + n_layers * arg.dim]
    i += n_layers * arg.dim

    weights.w1 = data[i : i + n_layers * arg.dim * arg.hidden_dim]
    i += n_layers * arg.dim * arg.hidden_dim
    weights.w2 = data[i : i + n_layers * arg.hidden_dim * arg.dim]
    i += n_layers * arg.hidden_dim * arg.dim
    weights.w3 = data[i : i + n_layers * arg.dim * arg.hidden_dim]
    i += n_layers * arg.dim * arg.hidden_dim
    weights.rms_final_weight = data[i : i + arg.dim]
    i += arg.dim
    weights.freq_cis_real = data[i : i + arg.seq_len * head_size // 2]
    i += arg.seq_len * head_size // 2

    weights.freq_cis_imag = data[i : i + arg.seq_len * head_size // 2]
    i += arg.seq_len * head_size // 2

    weights.wcls = weights.token_embedding_table if shared_weights else data[i:]


def rmsnorm(x, weight):
    # size =  len(x)
    ss = np.sum(x**2) / len(x) + 1e-5
    ss = 1.0 / np.sqrt(ss)
    return weight * (ss * x)


def softmax(x):
    x = np.exp(x - np.max(x, -1, keepdims=True))
    x /= np.sum(x, -1, keepdims=True)
    return x


def apply_rotary(q, k, freq_cis_real, freq_cis_imag, n_heads, n_kv_heads):
    """q: (dim, ) k: (dim, ) head_size = dim / n_heads
    freq_cis_real: (head_size/2, )
    freq_cis_imag: (head_size/2, )
    """
    q0, q1 = q[::2], q[1::2]
    k0, k1 = k[::2], k[1::2]
    qout_r = q0 * np.tile(freq_cis_real, n_heads) - q1 * np.tile(freq_cis_imag, n_heads)
    qout_i = q0 * np.tile(freq_cis_imag, n_heads) + q1 * np.tile(freq_cis_real, n_heads)
    kout_r = k0 * np.tile(freq_cis_real, n_kv_heads) - k1 * np.tile(
        freq_cis_imag, n_kv_heads
    )
    kout_i = k0 * np.tile(freq_cis_imag, n_kv_heads) + k1 * np.tile(
        freq_cis_real, n_kv_heads
    )

    qout = np.stack((qout_r, qout_i), axis=1).flatten()
    kout = np.stack((kout_r, kout_i), axis=1).flatten()

    return qout, kout


def repeat_kv(x: np.array, n_rep):
    # seq_len, n_kv_heads, head_size = x.shape

    if n_rep == 1:
        return x
    return np.repeat(x, n_rep, axis=1)


def forward(transformer: Transformer, token: int, pos: int):
    arg = transformer.config
    weights = transformer.weights
    state = transformer.state

    x = state.x
    dim = arg.dim
    assert dim % arg.n_heads == 0, "dim / n_heads != 0"
    assert arg.n_heads % arg.n_kv_heads == 0, "n_heads / n_kv_heads != 0"

    head_size = dim // arg.n_heads
    kv_dim = head_size * arg.n_kv_heads
    kv_mul = arg.n_heads // arg.n_kv_heads
    hidden_dim = arg.hidden_dim

    # state.x = weights.token_embedding_table[token * dim : (token + 1) * dim]
    content_row = weights.token_embedding_table[token * dim : (token + 1) * dim]
    x[:] = content_row
    # print(state.x.shape)

    freq_cis_real = weights.freq_cis_real[
        pos * head_size // 2 : (pos + 1) * head_size // 2
    ]
    freq_cis_imag = weights.freq_cis_imag[
        pos * head_size // 2 : (pos + 1) * head_size // 2
    ]

    for l in range(arg.n_layers):
        state.xb = rmsnorm(x, weights.rms_att_weight[l * dim : (l + 1) * dim])
        # print('xb', state.xb.shape)

        state.q = np.matmul(weights.wq[l * dim * dim : (l + 1) * dim * dim].reshape(dim, dim), state.xb)
        state.k = np.matmul(weights.wk[l * dim * kv_dim : (l + 1) * dim * kv_dim].reshape(kv_dim, dim), state.xb,)
        # print(weights.wk.shape)
        # print('kv_dim:', kv_dim)
        # print('dim :', dim)
        # print(weights.wq.shape)
        state.v = np.matmul(weights.wv[l * dim * kv_dim : (l + 1) * dim * kv_dim].reshape(kv_dim, dim), state.xb,)
        # print(state.v)

        # print(state.q)

        # Add positonal embedding
        state.q, state.k = apply_rotary(
            state.q, state.k, freq_cis_real, freq_cis_imag, arg.n_heads, arg.n_kv_heads
        )
        # print(state.q.shape)
        # print(state.k.shape)

        # loff = l * arg.seq_len * dim
        # store key, value at this pos to kv cache
        # state.key_cache (layer, seq_len, dim)
        state.key_cache[l, pos, :] = state.k
        state.value_cache[l, pos, :] = state.v
        # state.value_cache[loff + pos * dim: loff + (pos + 1) * dim] = state.v

        # np.repeat(state.key_cache[l].reshape(arg.seq_len, arg.n_heads, -1), kv_mul, axis=1)
        #  (seq_len, n_kv_heads, head_size)  ---> (seq_len, n_heads, head_size)
        xk = repeat_kv(
            state.key_cache[l].reshape(arg.seq_len, arg.n_kv_heads, -1), kv_mul
        )  # seq_len, n_heads, head_size
        xv = repeat_kv(
            state.value_cache[l].reshape(arg.seq_len, arg.n_kv_heads, -1), kv_mul
        )  # seq_len, n_heads, head_size

        xk = xk.transpose(1, 0, 2)  # n_heads, seq_len, head_size
        xv = xv.transpose(1, 0, 2)  # n_heads, seq_len, head_size

        # (n_heads, seq_len, head_size) @ (n_heads, head_size, 1) ---> (n_heads, seq_len, 1) ---> (n_heads, seq_len)
        # (n_heads, 1, head_size) @ (n_heads, head_size, seq_len) ---> (n_heads, 1, seq_len)
        scores = np.matmul(
            state.q.reshape(arg.n_heads, 1, head_size), xk.transpose(0, 2, 1)
        ) / np.sqrt(head_size)

        # print(scores)
        # scores += np.triu(np.full(arg.seq_len, float('-inf')))
        scores[:, :, pos+1:] = float("-inf")


        scores = softmax(scores)  # (n_heads, 1, seq_len)
        state.att = scores.squeeze(
            1
        )  # (n_heads, seq_len) store score to buffer, why store?

        state.xb = (
            scores @ xv
        )  # (n_heads, 1, seq_len) @ (n_heads, seq_len, head_size) ---> (n_heads, 1, head_size)

        # print(state.xb.shape)
        # print(state.xb.flatten().shape)

        state.xb2 = np.matmul(
            weights.wo[l * dim * dim : (l + 1) * dim * dim].reshape(dim, dim),
            state.xb.reshape(
                dim,
            ),
        )
        # print(state.xb2.shape)

        x += state.xb2  # residual connection

        state.xb = rmsnorm(
            state.x, weights.rms_ffn_weight[l * dim : (l + 1) * dim]
        )  # ffn rmsnorm

        #  w2(silu(w1(x)) * w3(x))
        ## hb = w1(x), hb2 = w3(x)
        state.hb = np.matmul(
            weights.w1[l * hidden_dim * dim : (l + 1) * hidden_dim * dim].reshape(
                hidden_dim, dim
            ),
            state.xb.reshape(
                dim,
            ),
        )
        # print(state.hb.shape)
        state.hb2 = np.matmul(weights.w3[l * hidden_dim * dim : (l + 1) * hidden_dim * dim].reshape(hidden_dim, dim),state.xb.reshape(dim,),)
        # print(state.hb2)
        # print(state.hb2.shape)
        state.hb *= 1.0 / (1.0 + np.exp(-state.hb))  # silu(w1(x)) * w3(x)
        state.hb *= state.hb2

        state.xb = np.matmul(
            weights.w2[l * dim * hidden_dim : (l + 1) * dim * hidden_dim].reshape(
                dim, hidden_dim
            ),
            state.hb.reshape(
                hidden_dim,
            ),
        )

        x += state.xb  # residual connection
        # print(state.x)
        # break

        # final rmsnorm
    x = rmsnorm(x, weights.rms_final_weight)

        # Calssifier into logits
    state.logits = np.matmul(weights.wcls.reshape(arg.vocab_size, dim), x)
    # print(state.logits)

    return state.logits


def tokenizer_init(conf: Config, file):
    vocab, vocab_scores, max_token_length = [], [], 0

    max_token_length = struct.unpack("i", file.read(4))[0]
    for i in range(0, conf.vocab_size):
        vocab_scores.append(struct.unpack("f", file.read(4))[0])
        len = struct.unpack("i", file.read(4))[0]
        bstr = file.read(len)
        if type(bstr) is not str:
            bstr = bstr.decode("utf8")
        vocab.append(bstr)
    return vocab, vocab_scores, max_token_length

def time_in_ms():
    # Returns time in milliseconds for benchmarking the model speed
    return int(time.time() * 1000)

def str_lookup(string, vocab):
    try:
        index = vocab.index(string)
        return index
    except ValueError:
        return -1


def bpe_encode(text, vocab, vocab_scores):
    # add optional (bos) tokens
    # add dummy_prefix.  llama add_dummy_prefix is true by default
    tokens = []
    for pos, char in enumerate(text):
        string = char
        if char == '\n':
            id = vocab.index('<0x0A>')
        else:
            id = str_lookup(string, vocab)
        if id == -1:
            print(f"cannot find text at pos {pos}")
            sys.exit(1)
        tokens.append(id)

    while True:
        best_score = -1e10
        best_id = -1
        best_idx = -1

        for i in range(len(tokens) - 1):
            # Check if we can merge the pair (tokens[i], tokens[i+1])
            # string = vocab[tokens[i]].rstrip(b'\x00') + vocab[tokens[i + 1]].rstrip(b'\x00')
            string = vocab[tokens[i]] + vocab[tokens[i + 1]]
            id = str_lookup(string, vocab)
            if id != -1 and vocab_scores[id] > best_score:
                # This merge pair exists in vocab! Record its score and position
                best_score = vocab_scores[id]
                best_id = id
                best_idx = i

        if best_idx == -1:
            break  # We couldn't find any more pairs to merge, so we're done

        # Merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id
        # Delete token at position best_idx+1, shift the entire sequence back 1
        tokens = tokens[0 : best_idx + 1] + tokens[best_idx + 2 :]

    return tokens


def decode(vocab, token: int, pre_token: int):
    if pre_token == 1 and vocab[token][0] == " ":
        token_str = vocab[token].lstrip()
    token_str = vocab[token]
    ## investage some cases like e.g. '<0x01>'
    ## needs to convert them to actually bytes
    return token_str

def print_bytes(s):
    def string_bytes(s:str):
        if ord(s) >= ord("A"):
            return ord(s) - ord("A") + 10
        return ord(s) - ord("0")

    # if len(s) >=6:
    if s[:3] == '<0x':
            to_print = chr(string_bytes(s[3]) * 16 + string_bytes(s[4]))
            print(to_print)
    else:
        print(s, end='')
    


def generate(checkpoints, temperature, steps, prompt, tokenization):
    random.seed(0)

    weights = Weights()

    with open(checkpoints, "rb") as file:
        # config header
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = (
            struct.unpack("7i", file.read(struct.calcsize("7i")))
        )
        print(dim, n_heads, n_kv_heads)
        arg = Config(
            dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
        )
        shared_weights = 1 if arg.vocab_size > 0 else 0
        arg.vocab_size = abs(arg.vocab_size)

    data = np.fromfile(
        checkpoints, dtype="float32", offset=28
    ).astype(np.float64)  # offset the first 7 intergers above, 28 = 7 * 4
    init_weight(weights=weights, arg=arg, i=0, data=data, shared_weights=shared_weights)

    # Read in the tokenizer.bin file
    with open(tokenization, "rb") as file:
        vocab, vocab_scores, max_token_length = tokenizer_init(arg, file)

    runState_init = dict(
        x=np.zeros(dim),
        xb=np.zeros(dim),
        xb2=np.zeros(dim),
        hb=np.zeros(hidden_dim),
        hb2=np.zeros(hidden_dim),
        q=np.zeros(dim),
        k=np.zeros(dim),
        v=np.zeros(dim),
        att=np.zeros((n_heads, seq_len)),
        logits=np.zeros(arg.vocab_size),
        key_cache=np.zeros((n_layers, seq_len, dim//n_heads * n_kv_heads)),
        value_cache=np.zeros((n_layers, seq_len, dim//n_heads * n_kv_heads)),
    )

    state = RunState(**runState_init)
    # print(state.logits)

    # Process the prompt, if any
    prompt_tokens = []
    if prompt:
        prompt_tokens = bpe_encode(prompt, vocab, vocab_scores)

    # Start the main loop
    start = 0  # Used to time our code, only initialized after the first iteration
    next_token = 0  # Will store the next token in the sequence
    # Initialize with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
    token = 1
    pos = 0  # Position in the sequence
    # Explicitly print the initial BOS token for stylistic symmetry reasons

    # print("<s>")

    transformer = Transformer(config=arg, weights=weights, state=state)
    while pos < steps:
        # Forward the transformer to get logits for the next token
        # transformer(token, pos, arg, state, weights)
        state.logits = forward(transformer=transformer, token=token, pos=pos)
        # print(state.logits[-3:])
        # print(state.logits.shape)

        if pos < len(prompt_tokens):
            # If we are still processing the input prompt, force the next prompt token
            next_token = prompt_tokens[pos]
        else:
            # Sample the next token
            if temperature == 0.0:
                # Greedy argmax sampling: take the token with the highest probability
                next_token = np.argmax(state.logits)
            else:
                # Apply the temperature to the logits
                state.logits /= temperature
                # Apply softmax to the logits to get the probabilities for the next token
                state.logits = softmax(state.logits)
                # Sample from this distribution to get the next token
                # next_token = sample(state.logits)
                next_token = np.where(np.random.multinomial(1, state.logits))[0][0]
                # print(next_token)
            if next_token == 1 or next_token ==2:
                break

        # Following BOS token (1), sentencepiece decoder strips any leading whitespace
        token_str = (
            vocab[next_token].lstrip()
            if token == 1 and vocab[next_token][0] == " "
            else vocab[next_token]
        )

        # print(token_str, end="")
        print_bytes(token_str)
        sys.stdout.flush()

        if next_token == 1:
            break

        # Advance forward
        token = next_token
        pos += 1
        
        # Initialize our timer here because the first iteration could be time consuming due to IO operations
        if start == 0:
           start = time_in_ms()

    # Report achieved tok/s
    end = time_in_ms()
    print(f"\nachieved tok/s: {(steps - 1) / (end - start) * 1000}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Llama2 Numpy inference')
    parser.add_argument('--checkpoints', '-c', default='stories15M.bin', help='model checkpoint file')
    parser.add_argument('--tokenization', '-t', default='tokenizer.bin', help='tokenizer')
    parser.add_argument('--temperature', type=float, default=0.8, help='temperature')
    parser.add_argument('--prompt', default='Dream comes true this day', help='prompt')
    parser.add_argument('--steps', type=int, default=256, help='steps')
    args = parser.parse_args()
    # args = {
    #     "checkpoint": "./out/stories15M.bin",
    #     "temperature": "0.0",
    #     "steps": "256",
    #     "prompt": None,
    # }
    # checkpoints = "stories15M.bin"
    # temperature = 0.0
    # steps = 256
    # prompt = "hello"
    # # if len(sys.argv) < 2:
    # #     print(
    # #         "Usage: python script.py <checkpoint_file> [temperature] [steps] [prompt]")
    # #     sys.exit(1)

    # if len(sys.argv) >= 2:
    #     checkpoints = sys.argv[1]

    # if len(sys.argv) >= 3:
    #     temperature = float(sys.argv[2])

    # if len(sys.argv) >= 4:
    #     steps = int(sys.argv[3])

    # if len(sys.argv) >= 5:
    #     prompt = sys.argv[4]

    generate(args.checkpoints, args.temperature, args.steps, args.prompt, args.tokenization)
    # generate(checkpoints, temperature, steps, prompt)
