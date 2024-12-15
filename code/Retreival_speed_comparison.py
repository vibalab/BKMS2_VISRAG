# search_benchmark.py
import time
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from openai import OpenAI
bkms2_api_key = 

class SearchBenchmark:
    def __init__(self, pt_searcher=None, chroma_collection=None, image_root_dir=None):
        self.pt_searcher = pt_searcher
        self.chroma_collection = chroma_collection
        self.image_root_dir = image_root_dir
        self.client = OpenAI(api_key = bkms2_api_key )

        if pt_searcher:
            self.pt_load_time = self._measure_pt_load_time()
        if chroma_collection:
            self.chroma_load_time = self._measure_chroma_load_time()

    def _measure_pt_load_time(self):
        """PT 파일 로딩 시간 측정"""
        start_time = time.time()
        # 이미 로드되어 있으므로 캐시 접근 시간만 측정
        _ = self.pt_searcher.embeddings_cache
        end_time = time.time()
        return end_time - start_time
    
    def _measure_chroma_load_time(self):
        """ChromaDB 초기 로딩 시간 측정"""
        start_time = time.time()
        _ = self.chroma_collection.count()  # 간단한 쿼리로 초기화 시간 측정
        end_time = time.time()
        return end_time - start_time


    def measure_pt_search_speed(self, query, n_runs=5):
        """PT 파일 기반 검색 속도 측정"""
        times = []
        results = None
        for _ in range(n_runs):
            start_time = time.time()
            results = self.pt_searcher.find_top_k_similar_images_from_text_in_memory(query)
            end_time = time.time()
            times.append(end_time - start_time)
        return times, results
    
    def get_text_embedding(self, text):
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding  # 직접 임베딩 리스트 반환


    def measure_chromadb_search_speed(self, query, n_runs=5):
        """ChromaDB 검색 속도 측정"""
        query_embedding = self.get_text_embedding(query)

        times = []
        results = None
        for _ in range(n_runs):
            start_time = time.time()
            results = self.chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=5
            )
            end_time = time.time()
            times.append(end_time - start_time)
        return times, results

    def run_comparison(self, query, methods, n_runs=5):
        """선택된 방법들의 속도 비교 실행"""
        speed_data = []
        results = {}

        if "PT File Search" in methods and self.pt_searcher:
            pt_times, pt_results = self.measure_pt_search_speed(query, n_runs)
            speed_data.extend([{"Method": "PT File Search", "Time": t} for t in pt_times])
            results['pt'] = pt_results

        if "ChromaDB Search" in methods and self.chroma_collection:
            chroma_times, chroma_results = self.measure_chromadb_search_speed(query, n_runs)
            speed_data.extend([{"Method": "ChromaDB Search", "Time": t} for t in chroma_times])
            results['chroma'] = chroma_results

        return speed_data, results

    def display_speed_metrics(self, speed_data):
        """속도 측정 결과 시각화"""
        st.subheader("Speed Comparison Results")
        
        # 검색 시간 통계
        df = pd.DataFrame(speed_data)
        stats = df.groupby("Method").agg({
            "Time": ["mean", "std", "min", "max"]
        }).round(4)
        
        # 초기화 시간 표시
        st.write("Initialization Times:")
        init_times = {
            "PT File Search": f"{self.pt_load_time:.4f}s",
            "ChromaDB Search": f"{self.chroma_load_time:.4f}s"
        }
        st.json(init_times)
        
        # 검색 시간 시각화
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(stats, width=800, height=200)
        
        with col2:
            fig = px.box(df, x="Method", y="Time", 
                        title="Search Time Distribution")
            fig.update_layout(width=800, height=400)
            st.plotly_chart(fig, use_container_width=True)


    def display_chroma_results(self, results):
        """ChromaDB 검색 결과 표시"""
        for idx, (id, metadata) in enumerate(zip(results['ids'][0], results['metadatas'][0])):
            year = metadata['year']
            image_path = f"{self.image_root_dir}/{year}/{metadata['image']}"
            
            col = st.columns(2)[idx % 2]
            with col:
                st.subheader(f"Rank {idx + 1}")
                st.text(f"File: {metadata['image']}")
                st.text(f"Year: {year}")
                
                try:
                    image = Image.open(image_path)
                    st.image(image, use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
                
                st.divider()

    def display_benchmark_results(self, pt_results, chroma_results):
        """PT와 ChromaDB 검색 결과를 나란히 표시"""
        st.subheader("Search Results Comparison")
        
        # 두 개의 컬럼 생성
        pt_col, chroma_col = st.columns(2)
        
        # PT 검색 결과 표시
        with pt_col:
            st.write("PT File Search Results")
            if pt_results:
                for file_name, embeddings_file, similarity, image_path in pt_results:
                    try:
                        image = Image.open(image_path)
                        st.text(f"File: {file_name}")
                        st.text(f"Similarity: {similarity:.4f}")
                        st.image(image, use_container_width=True)
                        st.divider()
                    except Exception as e:
                        st.error(f"Error loading image: {str(e)}")
        
        # ChromaDB 검색 결과 표시
        with chroma_col:
            st.write("ChromaDB Search Results")
            if chroma_results:
                for idx, (id, metadata) in enumerate(zip(chroma_results['ids'][0], chroma_results['metadatas'][0])):
                    try:
                        year = metadata['year']
                        image_path = f"{self.image_root_dir}/{year}/{metadata['image']}"
                        image = Image.open(image_path)
                        st.text(f"File: {metadata['image']}")
                        st.text(f"Year: {year}")
                        st.image(image, use_container_width=True)
                        st.divider()
                    except Exception as e:
                        st.error(f"Error loading image: {str(e)}")
