# Vector Database Application
This project is a Vector Search Engine that enables similarity search on text/ documents based on sematic meaning rather than the key words using various similarity search algorithms- cosine similarity, kd tree, locality sensitive hashing.

# How it works:
1. Document Encoding: Converts documents into embeddings using Sentence Transformers and stored in numpy array
2. Indexing: Embeddings are indexed using kd-tree or LSH
3. search and retrieval: finds top-k similar embeddings from the stored embeddings based on the query embedding using the search algorithm - cosine similarity, kd tress, lsh
4. AI response: Uses Ollama api to generate a response based on prompt created using retrieved search documents  and user query 

# Technologies Used
- Python
- Sentence transfromers (all-MiniLM-L6-v2)
- Scipy (KD-Tress indexing)
- Datasketch (MinHash LSH)
- Ollama AI (llama2:7b)

# Installation:
git clone https://github.com/praria/EnggLab7-VectorDatabase.git 

# create a virtual environment
python -m venv env

# Activate the virtual environment
source env/bin/activate

# install dependencies
pip install -r requirements.txt 

# Usage
************
# start Ollama server
ollama serve

# run main application
run python3 app.py 

# choose search method
when promted, choose one of the following methods: 
- cosine -> uses cosine similarity
- kd_tree -> uses kd-tree for effective nearest neighbour search
- lsh -> uses locality-sensitive hashing for approximate search

# Ask a Question:
type a query related to the loaded document, and the system will return relevant responses based on the search method chosen. 
Example Questions:
1. opportunityCost.txt:
who receives the most money in interest?
what is opportunity cost?
what should people compare before they make a trade off?
what is simple interest?
what is compound interest?

2. lsh.txt
where is the bird?
what is on the floor?

3. kdtree.txt:
[0.7, 0.3, 0.2]
To exit the application, type /bye

# Trouble shooting

# restart the Ollama server
pkill ollama  # Stop the current instance
ollama serve  # Restart the Ollama server 

# for testing only
running the llama2 chat model in Terminal : ollama run llama2
exit: ctrl + d or /bye

# to look up PID
lsof -i tcp:11434
kill -9 PID


