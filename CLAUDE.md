# CLAUDE.md

## Specification
- there is a config file `config.py` that defines the target config TARGET.
- the source text of a book is at data/[config]/*.txt
  - the text files are in UTF-8 encoding
  - the text files are Japanese books
  - the text files can contain newlines in the middle of words

- txt2vec.py converts the text files into a vector store by LangChain
  - it uses RecursiveCharacterTextSplitter to split the text into chunks
  - it uses FAISS as the vector store
  - it saves the vector store to data/[config]/vectorstore.pkl
- search.py performs a search on the vector store and returns results
  - 1. load the vector store from data/[config]/vectorstore.pkl
  - 2. receive a query from the user by prompt ' > '
  - 3. perform a similarity search on the vector store
  - 4. print the results
  - 5. repeat from step 2, until the user enters 'exit' or ctrl-d or ctrl-c or ESC at the first char

## Reference

* **RecursiveCharacterTextSplitter**
  * How To: [Text Splitters](https://python.langchain.com/docs/concepts/text_splitters/)
  * Reference: [RecursiveCharacterTextSplitter API](https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html)

* **FAISS**
  * How To: [FAISS Vector Store](https://python.langchain.com/docs/integrations/vectorstores/faiss/)
  * Reference: [FAISS API](https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html)


## Web Search

When web search is needed, use:
```bash
gemini --prompt "WebSearch: <query>"
```

This command should be used via the Task Tool instead of the built-in Web_Search tool.
