#!/usr/bin/env python3
import os
import sys
import pickle
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from config import TARGET


def load_vectorstore(path):
    """ベクトルストアを読み込む"""
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    # ベクトルストアのパス
    vectorstore_path = Path("data") / TARGET / "vectorstore.pkl"
    
    # ベクトルストアが存在するか確認
    if not vectorstore_path.exists():
        print(f"Vector store not found at {vectorstore_path}")
        print("Please run txt2vec.py first to create the vector store.")
        return
    
    # ベクトルストアを読み込む
    print("Loading vector store...")
    vectorstore = load_vectorstore(vectorstore_path)
    
    # エンベディングモデルを初期化（txt2vec.pyと同じモデルを使用）
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # ベクトルストアのエンベディングを更新
    vectorstore.embedding_function = embeddings.embed_query
    
    print("Ready to search. Type 'exit' to quit.")
    print()
    
    try:
        while True:
            # プロンプトを表示して入力を受け取る
            try:
                query = input(" > ")
            except EOFError:  # Ctrl-D
                print()
                break
            except KeyboardInterrupt:  # Ctrl-C
                print()
                break
            
            # 空の入力は無視
            if not query.strip():
                continue
            
            # ESCキーチェック（最初の文字が\x1bの場合）
            if query and ord(query[0]) == 27:
                break
            
            # exitコマンド
            if query.lower() == "exit":
                break
            
            # 類似度検索を実行
            try:
                results = vectorstore.similarity_search(query, k=5)
                
                if results:
                    print("\n" + "="*80)
                    for i, doc in enumerate(results, 1):
                        print(f"\n[Result {i}]")
                        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
                        print(f"Chunk Index: {doc.metadata.get('chunk_index', 'Unknown')}")
                        print("-" * 40)
                        print(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                    print("\n" + "="*80 + "\n")
                else:
                    print("No results found.\n")
                    
            except Exception as e:
                print(f"Error during search: {e}\n")
    
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()