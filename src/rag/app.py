import cloudflare
from flask import Flask, request

from .schema import ChatRequest

flask = Flask(__name__, static_folder=None)

PROMPT = """
Anda adalah asisten AI yang ahli dalam bidang hukum dan undang-undang
Sebagai asisten AI anda diminta untuk menjawab pertanyaan berdasarkan dokumen hukum Indonesia
misalnya undang-undang, peraturan pemerintah, peraturan menteri, atau putusan pengadilan.

ATURAN:
1. Jawab HANYA berdasarkan "Konteks Dokumen" di bawah.
2. Jawab dengan Bahasa Indonesia yang baku tetapi tidak kaku.
3. Jika pertanyaan atau Konteks Dokumen memiliki struktur POIN, jawab dengan deskriptif singkat.
4. Jika informasi TIDAK ADA dalam konteks, jawab: "Saya tidak menemukan pengaturan yang relevan dalam dokumen hukum."
5. Jika pasal/ayat jelas, sebutkan undang-undang nya beserta nomor pasal/ayat dan tahun.
6. JANGAN menambahkan informasi di luar konteks dokumen atau asumsi pribadi.

Konteks Dokumen:
{context}

Pertanyaan: {question}

Jawaban singkat:
"""


@flask.post('/chat')
async def chat():
  cf = cloudflare.AsyncClient(api_token=flask.config['cloudflare']['api_token'])
  serializer = ChatRequest(**request.get_json())

  embedding = await cf.ai.run(
    flask.config['cloudflare']['embedding_model_name'],
    account_id=flask.config['cloudflare']['account_id'],
    text=serializer.text,
  )

  # noinspection PyTypeChecker
  query = await cf.vectorize.indexes.query(
    index_name=flask.config['cloudflare']['index_name'],
    account_id=flask.config['cloudflare']['account_id'],
    vector=embedding['data'][0],
    return_metadata='all',
    return_values=False,
    top_k=flask.config['top-k'],
  )

  matches = query.matches or []

  contexts: list[str] = []

  for m in matches:
    metadata = m.metadata or {}

    if metadata:
      contexts.append(metadata['text'])

  context = '\n\n---\n\n'.join(contexts) if contexts else ''

  content = PROMPT.format(
    context=context,
    question=serializer.text,
  )

  # noinspection PyTypeChecker
  llm = await cf.ai.run(
    flask.config['cloudflare']['generation_model_name'],
    account_id=flask.config['cloudflare']['account_id'],
    messages=[
      {'role': 'user', 'content': content},
    ],
    top_k=flask.config['top-k'],
  )

  # noinspection PyTypeChecker
  answer = llm['response']

  return {
    'answer': answer,
  }


def create(
  cloudflare_account_id,
  cloudflare_api_token,
  cloudflare_index_name,
  cloudflare_generation_model_name,
  cloudflare_embedding_model_name,
  top_k,
) -> None:
  flask.config['top-k'] = top_k
  flask.config['cloudflare'] = {
    'account_id': cloudflare_account_id,
    'api_token': cloudflare_api_token,
    'index_name': cloudflare_index_name,
    'generation_model_name': cloudflare_generation_model_name,
    'embedding_model_name': cloudflare_embedding_model_name,
  }

  flask.run(host='127.0.0.1', port=8080)


__all__ = ['create']
