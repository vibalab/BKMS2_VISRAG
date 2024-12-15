import streamlit as st
import time
from CLIP_Search import CLIP_Search
from Gemma_OpenAI_search import Gemma_OpenAI_Search
from Retreival_speed_comparison import SearchBenchmark
import chromadb

CLIP_embeddings_dir = "/nas/dataset/BKMS2/clip_embeddings"
image_root_dir = "/nas/dataset/BKMS2/part2017-2019"  
Gemma_embeddings_dir = "/home/snuppy/data_link/BKMS2/embeddings"

st.sidebar.title("Demo for VISRAG projcets")
page = st.sidebar.radio("select a feature", ["Embedding Comparison", "Search Speed & Accuracy Comparison", "Chart Generation with LLM"])

if page == "Embedding Comparison":
    st.title("Compare chart search result based on embedding method")
    st.subheader("Choose Embedding Type:")
    search_type = st.radio(
        "Which embedding method would you like to use?",
        ("CLIP-based", "ChartGemma-based", "Hybrid Approach")
    )
    metric_type = st.radio(
        "Which distance metric would you like to use?",
        ("cosine", "euclidean", "manhattan", "rbf")
    )
    query = st.text_input("Enter your search query:", placeholder="e.g., a line chart about biology")

    if search_type == "CLIP-based" and query:
        
        searcher = CLIP_Search(embedding_dir = CLIP_embeddings_dir, root_dir = image_root_dir, metric=metric_type)
        results = searcher.find_top_k_similar_images_from_text(query)

        searcher.display_results_streamlit(results)

    if search_type == "ChartGemma-based" and query:
        
        searcher = Gemma_OpenAI_Search(embedding_dir= Gemma_embeddings_dir, root_dir=image_root_dir, metric = metric_type)
        results = searcher.find_top_k_similar_images_from_text(query)
        
        searcher.display_results_streamlit(results)
        
if page == "Search Speed & Accuracy Comparison":
    st.title("Search Speed & Accuracy Comparison")
    
    col1, col2 = st.columns(2)

    with col1:     
        query = st.text_input("Enter your search query:", placeholder="e.g., a line chart about biology")
        n_runs = st.slider("Number of test runs:", min_value=1, max_value=10, value=5)

    with col2:
        methods = st.multiselect(
            "Select search methods to compare:",
            ["PT File Search", "ChromaDB Search"],
            ["PT File Search", "ChromaDB Search"]
        )

    if st.button("Run Comparison"):
        if query:
            pt_searcher = Gemma_OpenAI_Search(embedding_dir= Gemma_embeddings_dir, root_dir=image_root_dir)
            chroma_client = chromadb.PersistentClient(path="/home/snuppy/data_link/BKMS2/chartgemma/chroma")
            chroma_collection = chroma_client.get_collection(name="jjs_embeddings")

            benchmark = SearchBenchmark(pt_searcher=pt_searcher, 
                                        chroma_collection=chroma_collection, 
                                        image_root_dir=image_root_dir)

            with st.spinner("Running speed tests..."):
                speed_data, results = benchmark.run_comparison(query, methods, n_runs)
                
                # 결과 표시
                benchmark.display_speed_metrics(speed_data)
                
                st.subheader("Search Results")
                if "PT File Search" and "ChromaDB Search" in methods:
                    benchmark.display_benchmark_results(results['pt'],results['chroma'])


                elif "PT File Search" in methods:
                    st.write("PT File Search Results:")
                    pt_searcher.display_results_streamlit(results['pt'])
                
                elif "ChromaDB Search" in methods:
                    st.write("ChromaDB Search Results:")
                    benchmark.display_chroma_results(results['chroma'])
    else:
        st.warning("Please enter a search query.")



    
