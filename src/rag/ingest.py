import json
import re
from pathlib import Path
from uuid import uuid4

import cloudflare
from tqdm import tqdm


def stream(path: Path, size: int, overlap: int) -> str:
  buffer = ''
  step = size - overlap

  if step <= 0:
    raise ValueError('Overlap harus lebih kecil dari ukuran chunk.')

  with (
    path.open(mode='r', encoding='utf-8') as file,
    tqdm(
      total=path.stat().st_size,
      unit='B',
      unit_scale=True,
      desc=f'Stream {path.name}',
      ncols=80,
      leave=False,
    ) as bar,
  ):
    while True:
      needed = size - len(buffer)

      if needed > 0:
        chunk = file.read(needed)

        if not chunk:  # EOF
          if buffer:
            yield buffer

          break

        buffer += chunk
        bar.update(len(chunk))

      if len(buffer) >= size:
        yield buffer[:size]

        buffer = buffer[step:]


def clean(text: str) -> str:
  lines = []

  for line in text.splitlines():
    line = line.strip()

    if not line:
      continue

    line = re.sub(r'\s+', ' ', line)
    lines.append(line)

  return '\n'.join(lines)


def ingestion(
  documents: Path,
  account_id: str,
  api_token: str,
  index_name: str,
  model_name: str,
  top_k: int,
  size: int,
  overlap: int,
) -> int:
  if not documents.exists():
    raise FileNotFoundError(f'Folder tidak ditemukan: {documents}')

  cf = cloudflare.Client(api_token=api_token)

  try:
    cf.vectorize.indexes.get(index_name=index_name, account_id=account_id)
  except cloudflare.NotFoundError:
    cf.vectorize.indexes.create(
      account_id=account_id,
      config={
        'dimensions': 768,
        'metric': 'cosine',
      },
      name=index_name,
    )

  t = []

  for path in documents.rglob('*'):
    if path.suffix.lower() in {'.md', '.txt'} and path.is_file():
      for index, text in enumerate(stream(path, size, overlap)):
        text = clean(text)

        if not text:
          continue

        t.append(text)

        if len(t) >= 32:
          embeddings = map(
            lambda embedding: {
              'id': f'{path.stem}-{index}-{uuid4()}'.replace(' ', '-'),
              'values': embedding,
              'metadata': {
                'source': str(path),
                'text': text,
                'index': index,
              },
            },
            cf.ai.run(model_name, account_id=account_id, text=t),
          )

          cf.vectorize.indexes.insert(
            index_name=index_name,
            account_id=account_id,
            body='\n'.join(json.dumps(embedding, ensure_ascii=False) for embedding in embeddings) + '\n',
          )

          t.clear()

    return 0


__all__ = ['ingestion']
