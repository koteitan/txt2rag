#!/usr/bin/env python3
import os
import re
import pickle
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import TARGET, CHUNK_SIZE, CHUNK_OVERLAP


def preprocess_japanese_text(text):
    """日本語テキストの前処理：単語途中の改行を除去"""
    # 日本語文字（ひらがな、カタカナ、漢字）の後に改行があり、
    # その次も日本語文字の場合は改行を削除
    pattern = r'([ぁ-んァ-ヶー一-龠々])\n([ぁ-んァ-ヶー一-龠々])'
    text = re.sub(pattern, r'\1\2', text)
    
    # 連続する改行は保持（段落の区切り）
    text = re.sub(r'\n\n+', '\n\n', text)
    
    return text


def main():
    # データディレクトリのパス
    data_dir = Path("data") / TARGET
    
    # テキストファイルを読み込む
    documents = []
    for txt_file in data_dir.glob("*.txt"):
        print(f"Reading {txt_file}...")
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read()
            # 日本語テキストの前処理
            content = preprocess_japanese_text(content)
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
        # 日本語の文章区切りを優先
        separators=["\n\n", "。\n", "。", "、", "\n", " ", ""]
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