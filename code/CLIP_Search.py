import torch
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
    rbf_kernel
)
import clip

class CLIP_Search:
    def __init__(self, embedding_dir, root_dir, metric = 'cosine',device=None, model=None):
        self.embedding_dir = embedding_dir
        self.root_dir = root_dir
        self.metric = metric
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # CLIP 모델 초기화
        if model is None:
            self.model, _ = clip.load("ViT-B/32", device=self.device)
        else:
            self.model = model

    def compute_similarity(self, text_embedding, embedding):
        if self.metric == 'cosine':
            return cosine_similarity(text_embedding, embedding)[0][0]
        elif self.metric == 'euclidean':
            # 거리를 유사도로 변환 (1 / (1 + distance))
            return 1 / (1 + euclidean_distances(text_embedding, embedding)[0][0])
        elif self.metric == 'manhattan':
            return 1 / (1 + manhattan_distances(text_embedding, embedding)[0][0])
        elif self.metric == 'rbf':
            return rbf_kernel(text_embedding, embedding)[0][0]
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")


    def find_top_k_similar_images_from_text(self, input_text, top_k=5):
        # Compute text embedding
        with torch.no_grad():
            text_token = clip.tokenize([input_text]).to(self.device)
            text_embedding = self.model.encode_text(text_token).cpu().numpy()
        
        similarities = []

        # Iterate through each embedding file
        for embeddings_file in ["2017_embeddings.pt", "2018_embeddings.pt", "2019_embeddings.pt"]:
            embeddings_path = f"{self.embedding_dir}/{embeddings_file}"
            embeddings = torch.load(embeddings_path)
            
            # Compute cosine similarity for each embedding
            for file_name, embedding in embeddings.items():
                embedding = embedding.numpy().reshape(1, -1)
                similarity = self.compute_similarity(text_embedding,embedding)
                
                # Store the similarity with corresponding file info
                similarities.append((file_name, embeddings_file, similarity, 
                                  f"{self.root_dir}/{embeddings_file[:4]}/{file_name}"))
        
        # Sort by similarity in descending order and get the top K
        top_k_similarities = sorted(similarities, key=lambda x: x[2], reverse=True)[:top_k]
        
        return top_k_similarities
    
    def display_results_streamlit(self, top_k_results):
        
        import streamlit as st
        from PIL import Image

        if top_k_results:
            # 결과를 2열로 표시하기 위한 열 생성
            cols = st.columns(2)
            
            for idx, (file_name, embeddings_file, similarity, image_path) in enumerate(top_k_results):
                # 열 번갈아가며 사용
                col = cols[idx % 2]
                
                with col:
                    # 결과 정보 표시
                    st.subheader(f"Rank {idx + 1}")
                    st.text(f"File: {file_name}")
                    st.text(f"Found in: {embeddings_file}")
                    st.text(f"Similarity: {similarity:.4f}")
                    
                    try:
                        # 이미지 열기 및 표시
                        image = Image.open(image_path)
                        st.image(image, 
                                caption=f"Rank {idx + 1}: {file_name}",
                                use_container_width=True)
                    except Exception as e:
                        st.error(f"Error loading image: {image_path}\nError: {str(e)}")
                    
                    # 구분선 추가
                    st.divider()
        else:
            st.warning("No similar images found.")