from openai import OpenAI
import torch
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
    rbf_kernel
)
import numpy as np

bkms2_api_key = 

class Gemma_OpenAI_Search:
    def __init__(self, embedding_dir, root_dir, metric = 'cosine', api_key = bkms2_api_key):
        self.embedding_dir = embedding_dir
        self.root_dir = root_dir
        self.metric = metric
        self.client = OpenAI(api_key=api_key)
        self.embeddings_cache = {}
        self._load_embeddings()

    def _load_embeddings(self):
        """모든 PT 파일의 임베딩을 메모리에 로드"""
        print("Loading embeddings into memory...")
        for embeddings_file in ["2017_embeddings.pt", "2018_embeddings.pt", "2019_embeddings.pt"]:
            embeddings_path = f"{self.embedding_dir}/{embeddings_file}"
            try:
                self.embeddings_cache[embeddings_file] = torch.load(embeddings_path)
                print(f"Loaded {embeddings_file}")
            except Exception as e:
                print(f"Error loading {embeddings_file}: {e}")
    
    def get_text_embedding(self, text):
        """입력 텍스트의 임베딩을 생성"""
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        # numpy array로 변환하여 반환
        return np.array(response.data[0].embedding).reshape(1, -1)
    
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
        # 검색 쿼리의 임베딩 생성
        text_embedding = self.get_text_embedding(input_text)
        
        similarities = []
        
        # 각 연도별 임베딩 파일 순회
        for embeddings_file in ["2017_embeddings.pt", "2018_embeddings.pt", "2019_embeddings.pt"]:
            embeddings_path = f"{self.embedding_dir}/{embeddings_file}"
            embeddings = torch.load(embeddings_path)
            
            # 각 이미지 임베딩과의 유사도 계산
            for file_name, embedding in embeddings.items():
                embedding = embedding.numpy().reshape(1, -1)
                similarity = self.compute_similarity(text_embedding, embedding)
                
                similarities.append((file_name, embeddings_file, similarity, 
                                  f"{self.root_dir}/{embeddings_file[:4]}/{file_name}"))
        
        # 유사도 기준 상위 k개 반환
        top_k_similarities = sorted(similarities, key=lambda x: x[2], reverse=True)[:top_k]
        
        return top_k_similarities
    
    def find_top_k_similar_images_from_text_in_memory(self, input_text, top_k=5):
        text_embedding = self.get_text_embedding(input_text)
        similarities = []
        
        # 캐시된 임베딩 사용
        for embeddings_file, embeddings in self.embeddings_cache.items():
            for file_name, embedding in embeddings.items():
                embedding = embedding.numpy().reshape(1, -1)
                similarity = self.compute_similarity(text_embedding, embedding)
                similarities.append((file_name, embeddings_file, similarity, 
                                  f"{self.root_dir}/{embeddings_file[:4]}/{file_name}"))
        
        return sorted(similarities, key=lambda x: x[2], reverse=True)[:top_k]

    # Streamlit 디스플레이 함수는 CLIP_Search와 동일하게 사용 가능
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