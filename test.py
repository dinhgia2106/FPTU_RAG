from sentence_transformers import SentenceTransformer, util

# Tải mô hình
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

# Tính embedding cho câu
sentences = ["Xin chào, bạn khỏe không?", "Hello, how are you?"]
embeddings = model.encode(sentences)

# Tính độ tương đồng cosine giữa 2 embedding
similarity = util.cos_sim(embeddings[0], embeddings[1])
print(similarity)
