#!/usr/bin/env python3
import os
import pickle
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import TARGET, CHUNK_SIZE, CHUNK_OVERLAP


def main():
    # データディレクトリのパス
    data_dir = Path("data") / TARGET
    
    # テキストファイルを読み込む
    documents = []
    for txt_file in data_dir.glob("*.txt"):
        print(f"Reading {txt_file}...")
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read()
            documents.append({
                "content": content,
                "source": str(txt_file)
            })
    
    if not documents:
        print(f"No text files found in {data_dir}")
        return
    
    # テキストを分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", "。", ".", " ", ""]
    )
    
    # すべてのドキュメントを分割
    all_splits = []
    for doc in documents:
        splits = text_splitter.split_text(doc["content"])
        # メタデータを追加
        for i, split in enumerate(splits):
            all_splits.append({
                "page_content": split,
                "metadata": {
                    "source": doc["source"],
                    "chunk_index": i
                }
            })
    
    print(f"Created {len(all_splits)} chunks")
    
    # エンベディングモデルを初期化
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # FAISSベクトルストアを作成
    print("Creating vector store...")
    texts = [doc["page_content"] for doc in all_splits]
    metadatas = [doc["metadata"] for doc in all_splits]
    
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )
    
    # ベクトルストアを保存
    output_path = data_dir / "vectorstore.pkl"
    print(f"Saving vector store to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(vectorstore, f)
    
    print("Done!")


if __name__ == "__main__":
    main()