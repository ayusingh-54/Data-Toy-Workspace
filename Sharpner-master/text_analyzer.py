import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import re
import nltk
try:
    from nltk.corpus import stopwords
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False

import base64
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class TextAnalyzer:
    """Advanced text analysis toolkit for NLP and text mining"""
    
    @staticmethod
    def analyze_text_column(df, text_column):
        """Analyze a text column with advanced NLP techniques"""
        if df is None or text_column not in df.columns:
            st.error(f"Column '{text_column}' not found or no data loaded")
            return
        
        st.markdown("### 📝 Text Analysis Toolkit")
        
        # Extract text data
        text_data = df[text_column].astype(str).values
        
        # Remove empty or very short texts
        text_data = [text for text in text_data if len(text.strip()) > 5]
        
        if not text_data:
            st.warning("No valid text content found to analyze")
            return
        
        # Create tabs for different analysis types
        tabs = st.tabs([
            "📊 Basic Stats", "🔤 Word Frequency", 
            "☁️ Word Cloud", "📈 N-grams Analysis",
            "🔍 Topic Extraction", "🔎 Text Similarity"
        ])
        
        with tabs[0]:
            TextAnalyzer.text_basic_stats(text_data, text_column)
            
        with tabs[1]:
            TextAnalyzer.word_frequency_analysis(text_data)
            
        with tabs[2]:
            TextAnalyzer.generate_word_cloud(text_data)
            
        with tabs[3]:
            TextAnalyzer.ngram_analysis(text_data)
            
        with tabs[4]:
            TextAnalyzer.topic_extraction(text_data)
            
        with tabs[5]:
            TextAnalyzer.text_similarity_analysis(text_data)
    
    @staticmethod
    def text_basic_stats(text_data, column_name):
        """Generate basic text statistics"""
        st.markdown("#### 📊 Basic Text Statistics")
        
        # Process text data
        text_lengths = [len(text) for text in text_data]
        word_counts = [len(text.split()) for text in text_data]
        
        # Calculate statistics
        total_texts = len(text_data)
        avg_text_length = np.mean(text_lengths)
        avg_word_count = np.mean(word_counts)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", f"{total_texts:,}")
        with col2:
            st.metric("Avg Document Length", f"{avg_text_length:.1f} chars")
        with col3:
            st.metric("Avg Words per Document", f"{avg_word_count:.1f}")
        
        # Length distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=text_lengths,
            name="Text Length",
            nbinsx=30
        ))
        fig.update_layout(
            title=f"Document Length Distribution for '{column_name}'",
            xaxis_title="Character Count",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Word count distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=word_counts,
            name="Word Count",
            nbinsx=30
        ))
        fig.update_layout(
            title=f"Word Count Distribution for '{column_name}'",
            xaxis_title="Word Count",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional statistics
        with st.expander("📝 Detailed Text Statistics"):
            # Calculate more stats
            longest_text_idx = np.argmax(text_lengths)
            shortest_text_idx = np.argmin(text_lengths)
            
            # Word count per character
            words_per_char = [wc/max(tl, 1) for wc, tl in zip(word_counts, text_lengths)]
            avg_words_per_char = np.mean(words_per_char)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Longest Document", f"{text_lengths[longest_text_idx]:,} chars")
                st.metric("Most Words", f"{word_counts[np.argmax(word_counts)]:,}")
            with col2:
                st.metric("Shortest Document", f"{text_lengths[shortest_text_idx]} chars")
                st.metric("Fewest Words", f"{word_counts[np.argmin(word_counts)]}")
            with col3:
                st.metric("Words/Char Ratio", f"{avg_words_per_char:.3f}")
                
                # Calculate percentage of documents above average length
                pct_above_avg = (np.array(text_lengths) > avg_text_length).mean() * 100
                st.metric("% Above Avg Length", f"{pct_above_avg:.1f}%")
            
            # Show samples of longest and shortest documents
            st.write("**Sample of Longest Document:**")
            longest_text = text_data[longest_text_idx]
            st.text_area("", longest_text[:500] + ("..." if len(longest_text) > 500 else ""), height=100)
            
            st.write("**Sample of Shortest Document:**")
            shortest_text = text_data[shortest_text_idx]
            st.text_area("", shortest_text, height=50)
    
    @staticmethod
    def word_frequency_analysis(text_data):
        """Analyze word frequencies in text data"""
        st.markdown("#### 🔤 Word Frequency Analysis")
        
        # Preprocessing options
        st.write("**Preprocessing Options:**")
        col1, col2 = st.columns(2)
        with col1:
            remove_stopwords = st.checkbox("Remove stopwords", True)
            case_sensitive = st.checkbox("Case sensitive", False)
        with col2:
            remove_punctuation = st.checkbox("Remove punctuation", True)
            min_word_length = st.slider("Minimum word length", 1, 10, 3)
        
        # Process the text
        combined_text = " ".join(text_data)
        
        # Apply preprocessing
        if not case_sensitive:
            combined_text = combined_text.lower()
            
        if remove_punctuation:
            combined_text = re.sub(r'[^\w\s]', ' ', combined_text)
        
        # Tokenize
        words = combined_text.split()
        
        # Apply length filter
        words = [word for word in words if len(word) >= min_word_length]
        
        # Remove stopwords
        if remove_stopwords and NLTK_AVAILABLE:
            stops = set(stopwords.words('english'))
            words = [word for word in words if word.lower() not in stops]
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
        
        # Convert to dataframe for visualization
        word_freq_df = pd.DataFrame(list(word_freq.items()), columns=['Word', 'Frequency'])
        word_freq_df = word_freq_df.sort_values('Frequency', ascending=False)
        
        # Show top words
        top_n = st.slider("Number of top words to show", 10, 100, 30)
        
        # Bar chart of top words
        fig = px.bar(
            word_freq_df.head(top_n),
            x='Word',
            y='Frequency',
            title=f'Top {top_n} Words by Frequency'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show frequency table
        with st.expander("View frequency table"):
            st.dataframe(word_freq_df.head(top_n))
            
            # Download option
            csv = word_freq_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="word_frequencies.csv">Download word frequency CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        # Additional analysis
        with st.expander("📊 Additional Word Statistics"):
            total_words = len(words)
            unique_words = len(word_freq)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Words", f"{total_words:,}")
            with col2:
                st.metric("Unique Words", f"{unique_words:,}")
            with col3:
                st.metric("Lexical Diversity", f"{(unique_words/total_words if total_words > 0 else 0):.4f}")
            
            # Word length distribution
            word_lengths = [len(word) for word in word_freq.keys()]
            fig = px.histogram(
                x=word_lengths,
                title="Word Length Distribution",
                labels={'x': 'Word Length', 'y': 'Frequency'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def generate_word_cloud(text_data):
        """Generate and display a word cloud from text data"""
        st.markdown("#### ☁️ Word Cloud Generator")
        
        # Preprocessing options
        col1, col2 = st.columns(2)
        with col1:
            remove_stopwords = st.checkbox("Remove stopwords", True, key='wc_stopwords')
            remove_punctuation = st.checkbox("Remove punctuation", True, key='wc_punct')
        with col2:
            min_word_length = st.slider("Minimum word length", 1, 10, 3, key='wc_min_len')
            max_words = st.slider("Maximum words", 50, 500, 200)
        
        # Process the text
        combined_text = " ".join(text_data)
        combined_text = combined_text.lower()
            
        if remove_punctuation:
            combined_text = re.sub(r'[^\w\s]', ' ', combined_text)
        
        # Remove stopwords
        if remove_stopwords and NLTK_AVAILABLE:
            stops = set(stopwords.words('english'))
            words = combined_text.split()
            combined_text = " ".join([word for word in words if word.lower() not in stops and len(word) >= min_word_length])
        
        # WordCloud settings
        width = st.slider("Cloud width", 400, 1200, 800)
        height = st.slider("Cloud height", 300, 1000, 400)
        
        # Color schemes
        color_schemes = {
            'Viridis': 'viridis',
            'Plasma': 'plasma',
            'Inferno': 'inferno',
            'Magma': 'magma',
            'Blues': 'Blues',
            'Reds': 'Reds',
            'YlOrBr': 'YlOrBr',
            'BuPu': 'BuPu',
            'RdPu': 'RdPu'
        }
        
        color_map = st.selectbox("Color scheme", list(color_schemes.keys()))
        
        # Background color
        bg_color = st.color_picker("Background color", "#FFFFFF")
        
        # Generate word cloud
        if st.button("Generate Word Cloud"):
            with st.spinner("Generating word cloud..."):
                try:
                    # Create word cloud
                    wordcloud = WordCloud(
                        width=width,
                        height=height,
                        background_color=bg_color,
                        colormap=color_schemes[color_map],
                        max_words=max_words,
                        min_word_length=min_word_length
                    ).generate(combined_text)
                    
                    # Display the generated image
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis("off")
                    
                    st.pyplot(plt)
                    
                    # Download option
                    img = wordcloud.to_image()
                    img_buffer = BytesIO()
                    img.save(img_buffer, format='PNG')
                    img_b64 = base64.b64encode(img_buffer.getvalue()).decode()
                    href = f'<a href="data:image/png;base64,{img_b64}" download="wordcloud.png">Download Word Cloud Image</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error generating word cloud: {str(e)}")
    
    @staticmethod
    def ngram_analysis(text_data):
        """Analyze n-grams (phrases) in text data"""
        st.markdown("#### 📈 N-gram Analysis")
        
        # Preprocessing options
        col1, col2 = st.columns(2)
        with col1:
            ngram_type = st.radio("N-gram size", ["Bigrams (2)", "Trigrams (3)", "4-grams", "5-grams"])
            
            if ngram_type == "Bigrams (2)":
                n = 2
            elif ngram_type == "Trigrams (3)":
                n = 3
            elif ngram_type == "4-grams":
                n = 4
            else:
                n = 5
                
        with col2:
            remove_stopwords = st.checkbox("Remove stopwords", True, key='ng_stopwords')
            min_count = st.slider("Minimum count", 1, 20, 2)
        
        # Process the text
        combined_text = " ".join(text_data).lower()
        combined_text = re.sub(r'[^\w\s]', ' ', combined_text)
        
        # Tokenize text
        if NLTK_AVAILABLE:
            tokens = nltk.word_tokenize(combined_text)
            
            # Remove stopwords if requested
            if remove_stopwords:
                stops = set(stopwords.words('english'))
                tokens = [token for token in tokens if token.lower() not in stops]
            
            # Generate n-grams
            ngrams = list(nltk.ngrams(tokens, n))
            
            # Count n-gram frequencies
            ngram_freq = {}
            for gram in ngrams:
                gram_str = " ".join(gram)
                if gram_str in ngram_freq:
                    ngram_freq[gram_str] += 1
                else:
                    ngram_freq[gram_str] = 1
            
            # Filter by minimum count
            ngram_freq = {k: v for k, v in ngram_freq.items() if v >= min_count}
            
            # Convert to dataframe
            ngram_df = pd.DataFrame(list(ngram_freq.items()), columns=['N-gram', 'Frequency'])
            ngram_df = ngram_df.sort_values('Frequency', ascending=False)
            
            # Show top n-grams
            top_n = st.slider("Number of top n-grams to show", 10, 50, 20)
            
            if not ngram_df.empty:
                # Bar chart of top n-grams
                fig = px.bar(
                    ngram_df.head(top_n),
                    x='N-gram',
                    y='Frequency',
                    title=f'Top {top_n} {ngram_type}'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show frequency table
                with st.expander("View n-gram frequency table"):
                    st.dataframe(ngram_df.head(top_n))
                    
                    # Download option
                    csv = ngram_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="ngram_frequencies.csv">Download n-gram frequency CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
            else:
                st.warning(f"No {ngram_type} found with frequency >= {min_count}")
        else:
            st.error("NLTK is required for n-gram analysis. Please install with: pip install nltk")
    
    @staticmethod
    def topic_extraction(text_data):
        """Extract topics from text data using various techniques"""
        st.markdown("#### 🔍 Topic Extraction")
        
        # Choose topic extraction method
        method = st.selectbox(
            "Select topic extraction method",
            ["TF-IDF Keywords", "Custom Keyword Extraction"]
        )
        
        if method == "TF-IDF Keywords":
            TextAnalyzer._tfidf_keyword_extraction(text_data)
        elif method == "Custom Keyword Extraction":
            TextAnalyzer._custom_keyword_extraction(text_data)
    
    @staticmethod
    def _tfidf_keyword_extraction(text_data):
        """Extract keywords using TF-IDF"""
        st.write("**TF-IDF Keyword Extraction**")
        
        # Parameters
        col1, col2 = st.columns(2)
        with col1:
            max_features = st.slider("Max features", 10, 1000, 100)
            min_df = st.slider("Minimum document frequency (%)", 1, 50, 2) / 100
        with col2:
            remove_stopwords = st.checkbox("Remove stopwords", True, key='tfidf_stopwords')
            ngram_range = st.radio("N-gram range", ["1-gram only", "1-2 grams", "1-3 grams"])
            
        # Convert n-gram option to tuple
        if ngram_range == "1-gram only":
            ng_range = (1, 1)
        elif ngram_range == "1-2 grams":
            ng_range = (1, 2)
        else:
            ng_range = (1, 3)
        
        # Process text data
        with st.spinner("Extracting keywords..."):
            try:
                # Create TF-IDF vectorizer
                tfidf = TfidfVectorizer(
                    max_features=max_features,
                    min_df=min_df,
                    stop_words='english' if remove_stopwords else None,
                    ngram_range=ng_range
                )
                
                # Fit and transform the data
                tfidf_matrix = tfidf.fit_transform(text_data)
                
                # Get feature names
                feature_names = tfidf.get_feature_names_out()
                
                # Calculate average TF-IDF scores across documents
                avg_scores = tfidf_matrix.mean(axis=0).A1
                
                # Create dataframe with keywords and scores
                keywords_df = pd.DataFrame({
                    'Keyword': feature_names,
                    'TF-IDF Score': avg_scores
                }).sort_values('TF-IDF Score', ascending=False)
                
                # Show top keywords
                top_n = st.slider("Number of top keywords to show", 10, 50, 20, key="tfidf_top_n")
                
                # Bar chart of top keywords
                fig = px.bar(
                    keywords_df.head(top_n),
                    x='Keyword',
                    y='TF-IDF Score',
                    title=f'Top {top_n} Keywords by TF-IDF Score'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show keywords table
                with st.expander("View keywords table"):
                    st.dataframe(keywords_df.head(top_n))
                    
                    # Download option
                    csv = keywords_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="tfidf_keywords.csv">Download TF-IDF Keywords CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
                
                # Show document-keyword heatmap
                with st.expander("Document-Keyword Matrix (Sample)"):
                    # Limit to a sample of documents and keywords for visualization
                    sample_docs = min(10, len(text_data))
                    sample_keywords = min(10, len(feature_names))
                    
                    # Get sample TF-IDF scores
                    sample_matrix = tfidf_matrix[:sample_docs, :sample_keywords].toarray()
                    
                    # Create heatmap
                    fig = go.Figure(data=go.Heatmap(
                        z=sample_matrix,
                        x=feature_names[:sample_keywords],
                        y=[f"Doc {i+1}" for i in range(sample_docs)],
                        colorscale='Viridis'
                    ))
                    
                    fig.update_layout(
                        title='Document-Keyword TF-IDF Heatmap (Sample)',
                        xaxis_title='Keywords',
                        yaxis_title='Documents'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error extracting keywords: {str(e)}")
    
    @staticmethod
    def _custom_keyword_extraction(text_data):
        """Extract keywords using custom extraction technique"""
        st.write("**Custom Keyword Extraction**")
        
        # Keywords to search for
        user_keywords = st.text_area(
            "Enter keywords to search for (one per line):",
            "price\nquality\nservice\nvalue\ndelivery",
            height=100
        )
        
        # Parse keywords
        keywords = [k.strip() for k in user_keywords.split("\n") if k.strip()]
        
        if not keywords:
            st.warning("Please enter at least one keyword")
            return
        
        # Case sensitivity
        case_sensitive = st.checkbox("Case sensitive", False, key='custom_case')
        
        # Process text data
        with st.spinner("Extracting keywords..."):
            # Count occurrences of each keyword
            keyword_counts = {}
            for keyword in keywords:
                count = 0
                for text in text_data:
                    if case_sensitive:
                        count += text.count(keyword)
                    else:
                        count += text.lower().count(keyword.lower())
                keyword_counts[keyword] = count
            
            # Count documents containing each keyword
            keyword_doc_counts = {}
            for keyword in keywords:
                doc_count = 0
                for text in text_data:
                    if case_sensitive:
                        if keyword in text:
                            doc_count += 1
                    else:
                        if keyword.lower() in text.lower():
                            doc_count += 1
                keyword_doc_counts[keyword] = doc_count
            
            # Create dataframe with keywords and counts
            results_df = pd.DataFrame({
                'Keyword': keywords,
                'Total Occurrences': [keyword_counts[k] for k in keywords],
                'Documents Containing': [keyword_doc_counts[k] for k in keywords],
                'Document %': [keyword_doc_counts[k] / len(text_data) * 100 for k in keywords]
            }).sort_values('Total Occurrences', ascending=False)
            
            # Bar chart of keyword occurrences
            fig = px.bar(
                results_df,
                x='Keyword',
                y='Total Occurrences',
                title='Keyword Occurrences',
                color='Document %',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show results table
            st.dataframe(results_df)
            
            # Download option
            csv = results_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="keyword_analysis.csv">Download Keyword Analysis CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            # Keyword context analysis
            st.write("**Keyword in Context Analysis**")
            selected_keyword = st.selectbox("Select keyword to view in context:", keywords)
            
            context_size = st.slider("Context size (characters around keyword)", 10, 100, 50)
            
            if selected_keyword:
                contexts = []
                for i, text in enumerate(text_data):
                    search_key = selected_keyword if case_sensitive else selected_keyword.lower()
                    search_text = text if case_sensitive else text.lower()
                    
                    start_pos = 0
                    while True:
                        pos = search_text.find(search_key, start_pos)
                        if pos == -1:
                            break
                            
                        # Get context around keyword
                        context_start = max(0, pos - context_size)
                        context_end = min(len(text), pos + len(selected_keyword) + context_size)
                        
                        # Get original case text for display
                        context = text[context_start:context_end]
                        
                        # Highlight the keyword in context
                        keyword_start = pos - context_start
                        keyword_end = keyword_start + len(selected_keyword)
                        
                        contexts.append({
                            'Document': i+1,
                            'Context': context,
                            'Keyword_Start': keyword_start,
                            'Keyword_End': keyword_end
                        })
                        
                        start_pos = pos + len(selected_keyword)
                
                # Show contexts
                if contexts:
                    st.write(f"**{len(contexts)}** occurrences of '{selected_keyword}' found.")
                    
                    # Show a sample of contexts
                    max_contexts = min(10, len(contexts))
                    for i, ctx in enumerate(contexts[:max_contexts]):
                        st.write(f"**Document {ctx['Document']}**")
                        
                        # Display context with highlighted keyword
                        context_html = (
                            ctx['Context'][:ctx['Keyword_Start']] +
                            f"<span style='background-color: yellow;'>{ctx['Context'][ctx['Keyword_Start']:ctx['Keyword_End']]}</span>" +
                            ctx['Context'][ctx['Keyword_End']:]
                        )
                        st.markdown(f"...{context_html}...", unsafe_allow_html=True)
                    
                    if len(contexts) > max_contexts:
                        st.info(f"Showing {max_contexts} of {len(contexts)} occurrences")
                else:
                    st.warning(f"No occurrences of '{selected_keyword}' found")
    
    @staticmethod
    def text_similarity_analysis(text_data):
        """Analyze text similarity between documents"""
        st.markdown("#### 🔎 Text Similarity Analysis")
        
        # If too many documents, require sample selection
        max_docs_for_all = 100
        
        if len(text_data) > max_docs_for_all:
            st.warning(f"Dataset contains {len(text_data)} documents. For performance reasons, select a sample or specific documents.")
            
            sample_size = st.slider("Number of documents to sample", 10, min(200, len(text_data)), 50)
            use_sample = st.checkbox("Use random sample", True)
            
            if use_sample:
                # Use random sample
                np.random.seed(42)  # For reproducibility
                sample_indices = np.random.choice(len(text_data), sample_size, replace=False)
                docs_to_analyze = [text_data[i] for i in sample_indices]
                doc_ids = [f"Doc {i+1}" for i in sample_indices]
            else:
                # Use first N documents
                docs_to_analyze = text_data[:sample_size]
                doc_ids = [f"Doc {i+1}" for i in range(sample_size)]
        else:
            # Use all documents
            docs_to_analyze = text_data
            doc_ids = [f"Doc {i+1}" for i in range(len(text_data))]
        
        # Choose similarity method
        similarity_method = st.selectbox(
            "Select similarity method",
            ["Cosine Similarity (TF-IDF)", "Cosine Similarity (Count Vectors)"]
        )
        
        # Vectorization parameters
        col1, col2 = st.columns(2)
        
        with col1:
            remove_stopwords = st.checkbox("Remove stopwords", True, key='sim_stopwords')
        
        with col2:
            ngram_range = st.radio("N-gram range", ["1-gram only", "1-2 grams"], key='sim_ngrams')
            
        # Convert n-gram option to tuple
        if ngram_range == "1-gram only":
            ng_range = (1, 1)
        else:
            ng_range = (1, 2)
        
        # Create and compute similarity matrix
        with st.spinner("Computing document similarity..."):
            try:
                if similarity_method == "Cosine Similarity (TF-IDF)":
                    vectorizer = TfidfVectorizer(
                        stop_words='english' if remove_stopwords else None,
                        ngram_range=ng_range
                    )
                else:  # Count Vectors
                    vectorizer = CountVectorizer(
                        stop_words='english' if remove_stopwords else None,
                        ngram_range=ng_range
                    )
                
                # Compute document vectors
                doc_vectors = vectorizer.fit_transform(docs_to_analyze)
                
                # Compute similarity matrix
                similarity_matrix = cosine_similarity(doc_vectors)
                
                # Display similarity heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=similarity_matrix,
                    x=doc_ids,
                    y=doc_ids,
                    colorscale='Viridis',
                    zmin=0, zmax=1
                ))
                
                fig.update_layout(
                    title='Document Similarity Matrix',
                    xaxis_title='Documents',
                    yaxis_title='Documents'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display similarity statistics
                with st.expander("Similarity Statistics"):
                    # Calculate average similarity
                    np.fill_diagonal(similarity_matrix, 0)  # Ignore self-similarity
                    avg_sim = np.mean(similarity_matrix)
                    max_sim = np.max(similarity_matrix)
                    min_sim = np.min(similarity_matrix[similarity_matrix > 0])  # Minimum non-zero similarity
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Similarity", f"{avg_sim:.3f}")
                    with col2:
                        st.metric("Maximum Similarity", f"{max_sim:.3f}")
                    with col3:
                        st.metric("Minimum Similarity", f"{min_sim:.3f}")
                    
                    # Find most similar documents
                    max_sim_idx = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
                    st.write(f"**Most similar documents:** {doc_ids[max_sim_idx[0]]} and {doc_ids[max_sim_idx[1]]}")
                    
                    # Show most similar pair
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**{doc_ids[max_sim_idx[0]]}**")
                        st.text_area("", docs_to_analyze[max_sim_idx[0]][:500] + "..." if len(docs_to_analyze[max_sim_idx[0]]) > 500 else docs_to_analyze[max_sim_idx[0]], height=200)
                    with col2:
                        st.write(f"**{doc_ids[max_sim_idx[1]]}**")
                        st.text_area("", docs_to_analyze[max_sim_idx[1]][:500] + "..." if len(docs_to_analyze[max_sim_idx[1]]) > 500 else docs_to_analyze[max_sim_idx[1]], height=200)
                
                # Compare specific documents
                st.write("**Compare specific documents**")
                
                col1, col2 = st.columns(2)
                with col1:
                    doc1_idx = st.selectbox("Select first document", range(len(doc_ids)), format_func=lambda i: doc_ids[i])
                with col2:
                    doc2_idx = st.selectbox("Select second document", range(len(doc_ids)), index=min(1, len(doc_ids)-1), format_func=lambda i: doc_ids[i])
                
                # Show similarity score
                sim_score = similarity_matrix[doc1_idx, doc2_idx]
                st.metric("Similarity Score", f"{sim_score:.4f}")
                
                # Visualize document comparison
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**{doc_ids[doc1_idx]}**")
                    st.text_area("", docs_to_analyze[doc1_idx][:500] + "..." if len(docs_to_analyze[doc1_idx]) > 500 else docs_to_analyze[doc1_idx], height=150, key="doc1_text")
                with col2:
                    st.write(f"**{doc_ids[doc2_idx]}**")
                    st.text_area("", docs_to_analyze[doc2_idx][:500] + "..." if len(docs_to_analyze[doc2_idx]) > 500 else docs_to_analyze[doc2_idx], height=150, key="doc2_text")
                
                # Show common words
                if st.checkbox("Show common terms"):
                    # Get document vectors as arrays
                    doc1_vec = doc_vectors[doc1_idx].toarray().flatten()
                    doc2_vec = doc_vectors[doc2_idx].toarray().flatten()
                    
                    # Get feature names
                    feature_names = vectorizer.get_feature_names_out()
                    
                    # Find common terms
                    common_indices = np.logical_and(doc1_vec > 0, doc2_vec > 0)
                    common_terms = [feature_names[i] for i in range(len(feature_names)) if common_indices[i]]
                    
                    if similarity_method == "Cosine Similarity (TF-IDF)":
                        # Get weights for common terms
                        common_weights = [(term, doc1_vec[feature_names.tolist().index(term)], 
                                         doc2_vec[feature_names.tolist().index(term)]) 
                                         for term in common_terms]
                        
                        # Create dataframe for display
                        common_df = pd.DataFrame(common_weights, columns=['Term', f'{doc_ids[doc1_idx]} Weight', f'{doc_ids[doc2_idx]} Weight'])
                        common_df['Average Weight'] = (common_df[f'{doc_ids[doc1_idx]} Weight'] + common_df[f'{doc_ids[doc2_idx]} Weight']) / 2
                        common_df = common_df.sort_values('Average Weight', ascending=False).head(30)
                    else:
                        # For count vectors, show frequency
                        common_weights = [(term, doc1_vec[feature_names.tolist().index(term)], 
                                         doc2_vec[feature_names.tolist().index(term)]) 
                                         for term in common_terms]
                        
                        # Create dataframe for display
                        common_df = pd.DataFrame(common_weights, columns=['Term', f'{doc_ids[doc1_idx]} Count', f'{doc_ids[doc2_idx]} Count'])
                        common_df['Total Count'] = common_df[f'{doc_ids[doc1_idx]} Count'] + common_df[f'{doc_ids[doc2_idx]} Count']
                        common_df = common_df.sort_values('Total Count', ascending=False).head(30)
                    
                    # Display common terms
                    st.write(f"**Common terms between {doc_ids[doc1_idx]} and {doc_ids[doc2_idx]}:**")
                    st.dataframe(common_df)
            
            except Exception as e:
                st.error(f"Error computing similarity: {str(e)}")
