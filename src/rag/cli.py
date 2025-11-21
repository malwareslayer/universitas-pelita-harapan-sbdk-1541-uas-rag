import argparse
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from .app import fastapi
from .ingest import ingestion

parser = argparse.ArgumentParser(prog='rag', description='Policy RAG')
command = parser.add_subparsers(dest='command')


def main(argv: Sequence[str] | None = None) -> int:
  serve = command.add_parser('serve', help='Run FastAPI server')
  serve.add_argument('--host', type=str, default='127.0.0.1', help='FastAPI host')
  serve.add_argument('--port', type=int, default=8080, help='FastAPI port')
  serve.add_argument('--openai-api-key', required=True, help='OpenAI API key')
  serve.add_argument('--openai-chat-model', default='gpt-4o-mini', help='OpenAI chat model')
  serve.add_argument('--openai-embedding-model', default='text-embedding-3-large', help='OpenAI embedding model')
  serve.add_argument('--cloudflare-account-id', required=True, help='Cloudflare account ID')
  serve.add_argument('--cloudflare-index', required=True, help='Cloudflare Vectorize index name')
  serve.add_argument('--cloudflare-api-token', required=True, help='Cloudflare API token')
  serve.add_argument('--top-k', type=int, default=5, help='Top-K vector search results')

  ingest_cmd = command.add_parser('ingest', help='Ingest markdown/txt documents into Cloudflare Vectorize')
  ingest_cmd.add_argument('--base', required=True, type=Path, help='Directory containing documents')
  ingest_cmd.add_argument('--openai-api-key', required=True, help='OpenAI API key')
  ingest_cmd.add_argument('--openai-embedding-model', default='text-embedding-3-large', help='OpenAI embedding model')
  ingest_cmd.add_argument('--cloudflare-account-id', required=True, help='Cloudflare account ID')
  ingest_cmd.add_argument('--cloudflare-index', required=True, help='Cloudflare Vectorize index name')
  ingest_cmd.add_argument('--cloudflare-api-token', required=True, help='Cloudflare API token')
  ingest_cmd.add_argument('--chunk-size', type=int, default=500, help='Characters per chunk')
  ingest_cmd.add_argument('--chunk-overlap', type=int, default=100, help='Chunk overlap in characters')

  args = parser.parse_args(argv)

  match args.command:
    case 'serve':
      fastapi.state.config = {
        'openai_api_key': args.openai_api_key,
        'openai_chat_model': args.openai_chat_model,
        'openai_embedding_model': args.openai_embedding_model,
        'cloudflare_account_id': args.cloudflare_account_id,
        'cloudflare_index': args.cloudflare_index,
        'cloudflare_api_token': args.cloudflare_api_token,
        'top_k': args.top_k,
      }

      uvicorn.run(fastapi, host=args.host, port=args.port)
    case 'ingest':
      ingestion(
        base=args.base,
        openai_api_key=args.openai_api_key,
        openai_embedding_model=args.openai_embedding_model,
        cloudflare_account_id=args.cloudflare_account_id,
        cloudflare_index=args.cloudflare_index,
        cloudflare_api_token=args.cloudflare_api_token,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
      )
    case _:
      raise AttributeError(f'Unknown command: {argv}')

  return 0


__all__ = ['main']
