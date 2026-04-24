"""Generate the missing phrase embedding file for CSFCube (or DORISMAE)."""

import os
import sys
import pickle
import torch
from tqdm import trange
from transformers import AutoTokenizer
from adapters import AutoAdapterModel


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def generate(data_dir, batch_size=32):
    out_path = os.path.join(data_dir, 'specter2_corpus_with-topic-terms.json.phrase-enc-specter.pt')
    if os.path.isfile(out_path):
        print(f'Already exists, nothing to do: {out_path}')
        return

    phrase_idx_path = os.path.join(data_dir, 'specter2_corpus_with-topic-terms.json.phrase_idx.pkl')
    if not os.path.isfile(phrase_idx_path):
        print(f'ERROR: missing phrase index pkl: {phrase_idx_path}', file=sys.stderr)
        sys.exit(1)

    id2phrase, _ = pickle.load(open(phrase_idx_path, 'rb'))
    print(f'Phrases to encode: {len(id2phrase)}')

    device = pick_device()
    print(f'Device: {device}')

    backbone = 'allenai/specter2_base'
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = AutoAdapterModel.from_pretrained(backbone)
    model.load_adapter('allenai/specter2', source='hf', load_as='proximity')
    model.set_active_adapters('proximity')
    model.to(device)
    model.eval()

    phrase_embeddings = []
    with torch.no_grad():
        for start in trange(0, len(id2phrase), batch_size, desc='encoding phrases'):
            batch = id2phrase[start:start + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt',
                return_token_type_ids=False,
            ).to(device)
            out = model(**inputs)
            emb = out.last_hidden_state[:, 0, :].cpu()
            phrase_embeddings.append(emb)

    phrase_embeddings = torch.cat(phrase_embeddings, dim=0)
    print(f'Embedding shape: {phrase_embeddings.shape}')
    torch.save(phrase_embeddings, out_path)
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./CSFCube')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    generate(args.data_dir, args.batch_size)
