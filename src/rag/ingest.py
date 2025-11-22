import re
import unicodedata
from collections.abc import Generator
from pathlib import Path

import cloudflare
from tqdm import tqdm

ZERO_WIDTH_RE = re.compile(r'[\u200B\u200C\u200D\uFEFF]')
CONTROL_RE = re.compile(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]')


def stream(path: Path, size: int, overlap: int) -> Generator[str]:
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
  if not text:
    return ''

  text = unicodedata.normalize('NFKC', text)
  text = ZERO_WIDTH_RE.sub('', text)
  text = CONTROL_RE.sub(' ', text)
  text = text.replace('\u2013', '-').replace('\u2014', '-')
  text = text.replace('\u2018', "'").replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
  text = re.sub(r'\s+', ' ', text)
  text = text.strip()

  return text


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

  for path in documents.rglob('*'):
    if path.suffix.lower() in {'.md', '.txt'} and path.is_file():
      for index, text in enumerate(stream(path, size, overlap)):
        text = clean(text)

        if not text:
          continue

        # noinspection PyTypeChecker
        cf.vectorize.indexes.insert(
          index_name=index_name,
          account_id=account_id,
          extra_headers={
            'Content-Type': 'application/x-ndjson',
          },
          body={
            'id': str(index),
            'values': cf.ai.run(model_name, account_id=account_id, text=text)['data'][0],
            'namespace': 'text',
            'metadata': {
              'source': str(path),
              'text': text,
              'index': index,
            },
          },
        )

  return 0


__all__ = ['ingestion']
