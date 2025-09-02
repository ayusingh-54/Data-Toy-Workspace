import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    r2_score, mean_squared_error, accuracy_score,
    confusion_matrix, precision_score, recall_score, f1_score,
    classification_report, mean_absolute_error
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import math
import time

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Data Toy - AutoML Platform", 
    layout="wide",
    page_icon="🎯",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "data_processed" not in st.session_state:
    st.session_state.data_processed = False
if "model_results" not in st.session_state:
    st.session_state.model_results = None

def go_to_upload():
    st.session_state.page = "upload"

def go_to_landing():
    st.session_state.page = "landing"
    st.session_state.data_processed = False
    st.session_state.model_results = None

# Custom CSS for enhanced styling
def load_css():
    st.markdown("""
    <style>
        .block-container { padding-top: 2rem; }
        
        /* Landing page styles */
        .project-title {
            font-size: 3.5em; 
            font-weight: 900; 
            color: #4CAF50;
            letter-spacing: 0.05em; 
            text-align: center; 
            margin-bottom: 0.2em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .subtitle {
            font-size: 1.8em; 
            font-weight: 500; 
            color: #81C784; 
            text-align: center; 
            margin-bottom: 0.8em;
        }
        .desc {
            text-align: center; 
            font-size: 1.2em; 
            font-weight: 400; 
            color: #E8F5E8; 
            margin-bottom: 2em;
            line-height: 1.6;
        }
        .bullets { 
            color: #C8E6C9; 
            font-size: 1.15em; 
            font-weight: 400; 
            margin: 0 auto 2em auto;
            max-width: 500px; 
            text-align: left; 
            padding-left: 1.5em;
            line-height: 1.8;
        }
        
        /* Enhanced button styles */
        .stButton > button {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 12px 24px;
            font-size: 1.1em;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .stButton > button:hover {
            background: linear-gradient(45deg, #45a049, #4CAF50);
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        }
        
        /* Header styles */
        .main-header {
            color: #4CAF50;
            text-align: center;
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 1em;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        }
        
        /* Card-like containers */
        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            border-left: 4px solid #4CAF50;
        }
        
        /* Progress bar styling */
        .stProgress > div > div > div > div {
            background: linear-gradient(45deg, #4CAF50, #81C784);
        }
        
        /* Sidebar styling */
        .css-1d391kg { background-color: #1E1E1E; }
        
        /* Error and success message styling */
        .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

###########################
#         LANDING PAGE    #
###########################
if st.session_state.page == "landing":
    load_css()
    
    # Hero section with animated elements
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='project-title'>🎯 Data Toy</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Your Intelligent AutoML Playground</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='desc'>Transform your data into insights with our advanced machine learning platform. Upload datasets, explore features, and deploy models—all without writing a single line of code.</div>",
            unsafe_allow_html=True
        )
    
    # Feature highlights
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🚀 **Quick Start**")
        st.markdown("""
        - Drag & drop CSV files
        - Automatic data preprocessing
        - One-click model training
        - Instant results visualization
        """)
    
    with col2:
        st.markdown("### 🧠 **Smart Models**")
        st.markdown("""
        - 8+ machine learning algorithms
        - Automatic problem type detection
        - Cross-validation scoring
        - Hyperparameter optimization
        """)
    
    with col3:
        st.markdown("### 📊 **Rich Analytics**")
        st.markdown("""
        - Interactive visualizations
        - Comprehensive model metrics
        - Confusion matrices
        - Feature importance analysis
        """)
    
    st.markdown("---")
    
    # Enhanced feature list
    st.markdown('''
    <div style="text-align: center; margin: 2em 0;">
        <h3 style="color: #4CAF50; margin-bottom: 1em;">✨ Platform Features</h3>
    </div>
    <ul class="bullets">
      <li>🔄 <strong>Multi-Model Training:</strong> Compare 8+ algorithms simultaneously</li>
      <li>📈 <strong>Interactive Dashboards:</strong> Real-time metrics and visualizations</li>
      <li>🎯 <strong>Smart Preprocessing:</strong> Automatic handling of missing values and encoding</li>
      <li>⚡ <strong>Fast Results:</strong> Get model insights in under 60 seconds</li>
      <li>📱 <strong>Responsive Design:</strong> Works seamlessly across devices</li>
      <li>🔍 <strong>Model Explainability:</strong> Understand how your models make decisions</li>
    </ul>
    ''', unsafe_allow_html=True)

    # Call-to-action buttons
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🚀 Start Building", use_container_width=True, type="primary"):
            go_to_upload()
    
    # Footer with stats
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Models Supported", "8+", "Advanced ML")
    with col2:
        st.metric("Processing Speed", "< 60s", "Ultra Fast")
    with col3:
        st.metric("Data Types", "Tabular", "CSV Support")
    with col4:
        st.metric("Code Required", "0 Lines", "No-Code Platform")

###########################
#      DATA LOAD/ML PAGE  #
###########################
if st.session_state.page == "upload":
    load_css()
    
    # Header with navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Back to Home", type="secondary"):
            go_to_landing()
    with col2:
        st.markdown("<div class='main-header'>Data Toy Workspace</div>", unsafe_allow_html=True)
    
    # Progress indicator
    st.markdown("### 📊 Data Processing Pipeline")
    progress_col1, progress_col2, progress_col3, progress_col4 = st.columns(4)
    with progress_col1:
        st.markdown("**1️⃣ Data Upload** ✅")
    with progress_col2:
        upload_status = "✅" if st.session_state.get('data_loaded', False) else "⏳"
        st.markdown(f"**2️⃣ Data Preview** {upload_status}")
    with progress_col3:
        feature_status = "✅" if st.session_state.get('features_selected', False) else "⏳"
        st.markdown(f"**3️⃣ Feature Selection** {feature_status}")
    with progress_col4:
        model_status = "✅" if st.session_state.get('models_trained', False) else "⏳"
        st.markdown(f"**4️⃣ Model Training** {model_status}")
    
    st.markdown("---")

    # Enhanced dataset upload section
    st.markdown("### 📁 Dataset Upload")
    
    # Create tabs for different upload methods
    upload_tab1, upload_tab2 = st.tabs(["📂 Sample Datasets", "⬆️ Upload Your Data"])
    
    df = None
    
    with upload_tab1:
        st.markdown("**Choose from our curated sample datasets:**")
        dataset_dir = "datasets"
        
        # Check if datasets directory exists
        if os.path.exists(dataset_dir):
            builtin = [f for f in os.listdir(dataset_dir) if f.endswith(".csv")]
            
            if builtin:
                # Create columns for dataset preview
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    choice = st.selectbox("Available datasets:", ["Select a dataset..."] + builtin)
                    
                    if choice and choice != "Select a dataset...":
                        # Dataset descriptions
                        descriptions = {
                            "Iris.csv": "🌸 Classic iris flower classification dataset with 4 features",
                            "Titanic-Dataset.csv": "🚢 Passenger survival prediction with demographic features",
                            "breast-cancer.csv": "🔬 Medical diagnosis dataset for cancer detection"
                        }
                        
                        if choice in descriptions:
                            st.info(descriptions[choice])
                        
                        if st.button(f"Load {choice}", type="primary"):
                            try:
                                df = pd.read_csv(os.path.join(dataset_dir, choice))
                                st.session_state.data_loaded = True
                                st.success(f"✅ Successfully loaded: {choice}")
                                st.balloons()
                            except Exception as e:
                                st.error(f"❌ Error loading dataset: {str(e)}")
                
                with col2:
                    if choice and choice != "Select a dataset...":
                        try:
                            preview_df = pd.read_csv(os.path.join(dataset_dir, choice))
                            st.markdown("**Dataset Preview:**")
                            st.dataframe(preview_df.head(3), use_container_width=True)
                            
                            # Quick stats
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Rows", preview_df.shape[0])
                            with col_b:
                                st.metric("Columns", preview_df.shape[1])
                            with col_c:
                                st.metric("Missing %", f"{(preview_df.isnull().sum().sum() / (preview_df.shape[0] * preview_df.shape[1]) * 100):.1f}")
                        except:
                            pass
            else:
                st.warning("No sample datasets found in the datasets directory.")
        else:
            st.warning("Datasets directory not found.")
    
    with upload_tab2:
        st.markdown("**Upload your own CSV file:**")
        uploaded = st.file_uploader(
            "Choose a CSV file", 
            type=["csv"],
            help="Upload a CSV file with your data. Make sure it has proper column headers."
        )
        
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                st.session_state.data_loaded = True
                st.success("✅ File uploaded successfully!")
                
                # Show upload summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("File Size", f"{uploaded.size / 1024:.1f} KB")
                with col2:
                    st.metric("Rows", df.shape[0])
                with col3:
                    st.metric("Columns", df.shape[1])
                    
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                st.info("💡 Make sure your file is a valid CSV with proper formatting.")

    # --- ML workflow continues only if dataframe loaded ---
    if isinstance(df, pd.DataFrame):
        st.markdown("---")
        st.markdown("### 🔍 Data Exploration & Feature Engineering")
        
        # Data overview in tabs
        overview_tab1, overview_tab2, overview_tab3, overview_tab4 = st.tabs([
            "📊 Dataset Overview", "📈 Statistical Summary", "🔍 Data Quality", "🎯 Feature Selection"
        ])
        
        with overview_tab1:
            st.markdown("**Dataset Preview:**")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Dataset statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", f"{df.shape[0]:,}")
            with col2:
                st.metric("Total Columns", df.shape[1])
            with col3:
                numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
                st.metric("Numeric Columns", numeric_cols)
            with col4:
                categorical_cols = df.select_dtypes(include=['object']).shape[1]
                st.metric("Categorical Columns", categorical_cols)
        
        with overview_tab2:
            st.markdown("**Statistical Summary:**")
            
            # Separate numeric and categorical summaries
            numeric_df = df.select_dtypes(include=[np.number])
            categorical_df = df.select_dtypes(include=['object'])
            
            if not numeric_df.empty:
                st.markdown("*Numeric Features:*")
                st.dataframe(numeric_df.describe(), use_container_width=True)
            
            if not categorical_df.empty:
                st.markdown("*Categorical Features:*")
                cat_summary = pd.DataFrame({
                    'Unique Values': categorical_df.nunique(),
                    'Most Frequent': categorical_df.mode().iloc[0] if len(categorical_df.mode()) > 0 else 'N/A',
                    'Missing Count': categorical_df.isnull().sum()
                })
                st.dataframe(cat_summary, use_container_width=True)
        
        with overview_tab3:
            st.markdown("**Data Quality Assessment:**")
            
            # Missing values analysis
            missing_data = df.isnull().sum()
            missing_percent = (missing_data / len(df)) * 100
            
            quality_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Count': missing_data.values,
                'Missing %': missing_percent.values,
                'Data Type': df.dtypes.values
            })
            quality_df = quality_df[quality_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
            
            if not quality_df.empty:
                st.dataframe(quality_df, use_container_width=True)
                
                # Visualization of missing data
                if len(quality_df) > 0:
                    fig = px.bar(
                        quality_df, 
                        x='Column', 
                        y='Missing %',
                        title='Missing Data by Column',
                        color='Missing %',
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No missing values detected in the dataset!")
            
            # Data type distribution
            dtype_counts = df.dtypes.value_counts()
            fig_dtype = px.pie(
                values=dtype_counts.values,
                names=dtype_counts.index,
                title='Data Type Distribution'
            )
            st.plotly_chart(fig_dtype, use_container_width=True)
        
        with overview_tab4:
            columns = df.columns.tolist()
            
            # Enhanced target selection
            st.markdown("**🎯 Target Variable Selection:**")
            target = st.selectbox(
                "Select your target column (what you want to predict):",
                ["Select target column..."] + columns,
                help="Choose the column you want to predict. This will determine if it's a classification or regression problem."
            )
            
            if target and target != "Select target column...":
                st.session_state.features_selected = True
                
                # Show target variable analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Target Analysis: `{target}`**")
                    target_info = {
                        'Data Type': str(df[target].dtype),
                        'Unique Values': df[target].nunique(),
                        'Missing Values': df[target].isnull().sum(),
                        'Missing %': f"{(df[target].isnull().sum() / len(df)) * 100:.2f}%"
                    }
                    
                    for key, value in target_info.items():
                        st.write(f"- **{key}:** {value}")
                
                with col2:
                    # Target distribution visualization
                    if df[target].dtype == 'object' or df[target].nunique() <= 20:
                        fig = px.histogram(df, x=target, title=f'Distribution of {target}')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = px.histogram(df, x=target, nbins=30, title=f'Distribution of {target}')
                        st.plotly_chart(fig, use_container_width=True)
                
                # Feature selection
                available_features = [col for col in columns if col != target]
                st.markdown("**🔧 Feature Selection:**")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    features = st.multiselect(
                        "Select features for model training:",
                        available_features,
                        default=available_features,
                        help="Choose which columns to use as input features for your model."
                    )
                
                with col2:
                    st.markdown("**Quick Actions:**")
                    if st.button("Select All Features"):
                        st.experimental_rerun()
                    if st.button("Clear Selection"):
                        st.experimental_rerun()
                
                # Feature correlation heatmap for numeric features
                if len([f for f in features if f in df.select_dtypes(include=[np.number]).columns]) > 1:
                    st.markdown("**📊 Feature Correlation Matrix:**")
                    numeric_features = [f for f in features if f in df.select_dtypes(include=[np.number]).columns]
                    if target in df.select_dtypes(include=[np.number]).columns:
                        numeric_features.append(target)
                    
                    if len(numeric_features) > 1:
                        corr_matrix = df[numeric_features].corr()
                        fig = px.imshow(
                            corr_matrix,
                            title="Feature Correlation Heatmap",
                            color_continuous_scale='RdBu',
                            aspect='auto'
                        )
                        st.plotly_chart(fig, use_container_width=True)

        def detect_problem_type(df, target):
            """Enhanced problem type detection with more sophisticated logic"""
            target_series = df[target].dropna()
            dtype = target_series.dtype
            nunique = target_series.nunique()
            
            # Check if it's clearly categorical
            if dtype == "object":
                return "Classification"
            
            # For numeric data, use more sophisticated heuristics
            if nunique <= 2:
                return "Classification"
            elif nunique <= 20 and nunique < len(target_series) * 0.05:
                return "Classification"
            else:
                return "Regression"

        def advanced_preprocess_features(df, features):
            """Enhanced preprocessing with better handling of different data types"""
            X = df[features].copy()
            preprocessing_log = []
            
            # Remove columns with too many missing values
            missing_threshold = 0.8
            drop_cols = [col for col in X.columns if X[col].isnull().mean() > missing_threshold]
            if drop_cols:
                X = X.drop(columns=drop_cols)
                preprocessing_log.append(f"Dropped {len(drop_cols)} columns with >80% missing values")
            
            # Separate numeric and categorical columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
            
            # Handle missing values
            if numeric_cols:
                # Use median for numeric columns
                numeric_imputer = SimpleImputer(strategy='median')
                X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])
                preprocessing_log.append(f"Imputed missing values in {len(numeric_cols)} numeric columns using median")
            
            if categorical_cols:
                # Use most frequent for categorical columns
                categorical_imputer = SimpleImputer(strategy='most_frequent')
                X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])
                preprocessing_log.append(f"Imputed missing values in {len(categorical_cols)} categorical columns using mode")
            
            # Handle categorical variables
            for col in categorical_cols:
                if X[col].nunique() > 50:  # Too many categories
                    X = X.drop(columns=[col])
                    preprocessing_log.append(f"Dropped '{col}' due to high cardinality (>50 categories)")
                else:
                    # Label encoding for categorical variables
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    preprocessing_log.append(f"Label encoded '{col}' ({X[col].nunique()} categories)")
            
            return X, preprocessing_log

        def advanced_preprocess_target(y):
            """Enhanced target preprocessing with better handling"""
            preprocessing_log = []
            
            if y.dtype == 'object':
                # Handle missing values first
                if y.isnull().any():
                    # For categorical targets, use mode or drop missing
                    if y.isnull().mean() > 0.1:  # If >10% missing, it might be problematic
                        preprocessing_log.append(f"Warning: {y.isnull().mean()*100:.1f}% missing values in target")
                    y = y.fillna(y.mode()[0] if not y.mode().empty else 'unknown')
                
                le = LabelEncoder()
                y_enc = le.fit_transform(y.astype(str))
                preprocessing_log.append(f"Label encoded target variable ({len(le.classes_)} classes)")
                return y_enc, le, preprocessing_log
            else:
                # Handle missing values in numeric targets
                if y.isnull().any():
                    y_filled = y.fillna(y.median())
                    preprocessing_log.append(f"Imputed {y.isnull().sum()} missing values in target using median")
                else:
                    y_filled = y
                preprocessing_log.append("Target is numeric - regression problem detected")
                return y_filled, None, preprocessing_log

                if target and target != "Select target column..." and features:
                    if target in features:
                        st.error("❌ Target column cannot be in features list.")
                    else:
                        st.markdown("---")
                        st.markdown("### 🤖 Machine Learning Pipeline")
                        
                        # Model configuration section
                        config_col1, config_col2 = st.columns(2)
                        
                        with config_col1:
                            st.markdown("**🎛️ Training Configuration:**")
                            
                            # Data validation
                            working_df = df[features + [target]].copy()
                            rows_before = working_df.shape[0]
                            working_df_clean = working_df.dropna()
                            rows_after = working_df_clean.shape[0]
                            
                            if rows_after < 10:
                                st.error("❌ Not enough data rows for training after cleaning. Minimum 10 rows required.")
                                st.stop()
                            
                            st.info(f"📊 Data: {rows_after:,} rows available for training (dropped {rows_before - rows_after:,} rows with missing values)")
                            
                            # Advanced preprocessing
                            try:
                                X, feature_log = advanced_preprocess_features(working_df, features)
                                y_processed, target_encoder, target_log = advanced_preprocess_target(working_df[target])
                                problem_type = detect_problem_type(working_df, target)
                                
                                st.success(f"✅ Problem Type Detected: **{problem_type}**")
                                
                                # Show preprocessing summary
                                with st.expander("🔧 Preprocessing Summary", expanded=False):
                                    st.markdown("**Feature Processing:**")
                                    for log in feature_log:
                                        st.write(f"• {log}")
                                    st.markdown("**Target Processing:**")
                                    for log in target_log:
                                        st.write(f"• {log}")
                                
                            except Exception as e:
                                st.error(f"❌ Preprocessing failed: {str(e)}")
                                st.stop()
                        
                        with config_col2:
                            st.markdown("**⚙️ Algorithm Selection:**")
                            
                            # Enhanced model selection based on problem type
                            if problem_type == "Regression":
                                ALL_MODELS = {
                                    "Linear Regression": "Fast, interpretable linear relationships",
                                    "Random Forest": "Robust, handles non-linear patterns",
                                    "Gradient Boosting": "High performance, complex patterns",
                                    "SVM": "Good for complex boundaries",
                                    "KNN": "Instance-based, local patterns",
                                    "Decision Tree": "Interpretable, rule-based"
                                }
                            else:
                                ALL_MODELS = {
                                    "Logistic Regression": "Fast, interpretable classification",
                                    "Random Forest": "Robust, handles overfitting well",
                                    "Gradient Boosting": "High performance ensemble",
                                    "SVM": "Effective for high-dimensional data",
                                    "KNN": "Instance-based classification",
                                    "Decision Tree": "Interpretable decision rules",
                                    "Naive Bayes": "Fast, works well with small data"
                                }
                            
                            # Model selection with descriptions
                            selected_models = []
                            st.markdown("Select algorithms to compare:")
                            
                            for model_name, description in ALL_MODELS.items():
                                if st.checkbox(f"**{model_name}**", value=model_name in list(ALL_MODELS.keys())[:3]):
                                    selected_models.append(model_name)
                                st.caption(description)
                            
                            if not selected_models:
                                st.warning("⚠️ Please select at least one algorithm.")
                                st.stop()
                        
                        # Training configuration
                        st.markdown("**📐 Training Parameters:**")
                        param_col1, param_col2, param_col3 = st.columns(3)
                        
                        with param_col1:
                            test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, step=0.05, format="%.2f")
                        with param_col2:
                            random_state = st.number_input("Random Seed", value=42, min_value=0, max_value=1000)
                        with param_col3:
                            use_scaling = st.checkbox("Feature Scaling", value=True, help="Recommended for SVM and KNN")
                        
                        # Cross-validation option
                        use_cv = st.checkbox("Enable Cross-Validation", value=True, help="5-fold cross-validation for more robust evaluation")
                        
                        # Training button with enhanced styling
                        if st.button("🚀 Train Selected Models", use_container_width=True, type="primary"):
                            start_time = time.time()
                            
                            # Progress bar setup
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            try:
                                # Data splitting
                                status_text.text("Splitting data...")
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y_processed, test_size=test_size, random_state=random_state, stratify=y_processed if problem_type == "Classification" else None
                                )
                                progress_bar.progress(0.1)
                                
                                # Feature scaling if enabled
                                if use_scaling:
                                    status_text.text("Scaling features...")
                                    scaler = StandardScaler()
                                    X_train_scaled = scaler.fit_transform(X_train)
                                    X_test_scaled = scaler.transform(X_test)
                                    progress_bar.progress(0.2)
                                else:
                                    X_train_scaled = X_train
                                    X_test_scaled = X_test
                                
                                # Enhanced model definitions with better parameters
                                model_defs = {
                                    "Linear Regression": LinearRegression(),
                                    "SVM": SVR(kernel='rbf') if problem_type == "Regression" else SVC(kernel='rbf', random_state=random_state, probability=True),
                                    "KNN": KNeighborsRegressor(n_neighbors=5) if problem_type == "Regression" else KNeighborsClassifier(n_neighbors=5),
                                    "Decision Tree": DecisionTreeRegressor(random_state=random_state, max_depth=10) if problem_type == "Regression" else DecisionTreeClassifier(random_state=random_state, max_depth=10),
                                    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state, max_depth=10) if problem_type == "Regression" else RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=10),
                                    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
                                    "Naive Bayes": GaussianNB(),
                                    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=random_state) if problem_type == "Regression" else GradientBoostingClassifier(n_estimators=100, random_state=random_state)
                                }
                                
                                results = {}
                                preds_test_dict = {}
                                cv_scores = {}
                                
                                # Train each selected model
                                for i, mdl in enumerate(selected_models):
                                    status_text.text(f"Training {mdl}... ({i+1}/{len(selected_models)})")
                                    
                                    model = model_defs[mdl]
                                    
                                    # Use scaled data for models that benefit from it
                                    if mdl in ["SVM", "KNN", "Logistic Regression"] and use_scaling:
                                        X_train_use = X_train_scaled
                                        X_test_use = X_test_scaled
                                    else:
                                        X_train_use = X_train
                                        X_test_use = X_test
                                    
                                    # Train model
                                    model.fit(X_train_use, y_train)
                                    
                                    # Predictions
                                    train_preds = model.predict(X_train_use)
                                    test_preds = model.predict(X_test_use)
                                    preds_test_dict[mdl] = test_preds
                                    
                                    # Cross-validation if enabled
                                    if use_cv:
                                        cv_scoring = 'r2' if problem_type == "Regression" else 'accuracy'
                                        cv_scores[mdl] = cross_val_score(model, X_train_use, y_train, cv=5, scoring=cv_scoring)
                                    
                                    # Calculate metrics
                                    if problem_type == "Regression":
                                        results[mdl] = {
                                            "train": [
                                                r2_score(y_train, train_preds), 
                                                mean_squared_error(y_train, train_preds), 
                                                np.sqrt(mean_squared_error(y_train, train_preds)),
                                                mean_absolute_error(y_train, train_preds)
                                            ],
                                            "test": [
                                                r2_score(y_test, test_preds), 
                                                mean_squared_error(y_test, test_preds), 
                                                np.sqrt(mean_squared_error(y_test, test_preds)),
                                                mean_absolute_error(y_test, test_preds)
                                            ],
                                            "metrics": ["R²", "MSE", "RMSE", "MAE"]
                                        }
                                    else:
                                        results[mdl] = {
                                            "train": [
                                                accuracy_score(y_train, train_preds),
                                                precision_score(y_train, train_preds, average="weighted", zero_division=0),
                                                recall_score(y_train, train_preds, average="weighted", zero_division=0),
                                                f1_score(y_train, train_preds, average="weighted", zero_division=0)
                                            ],
                                            "test": [
                                                accuracy_score(y_test, test_preds),
                                                precision_score(y_test, test_preds, average="weighted", zero_division=0),
                                                recall_score(y_test, test_preds, average="weighted", zero_division=0),
                                                f1_score(y_test, test_preds, average="weighted", zero_division=0)
                                            ],
                                            "metrics": ["Accuracy", "Precision", "Recall", "F1-Score"]
                                        }
                                    
                                    progress_bar.progress(0.3 + (i + 1) * 0.6 / len(selected_models))
                                
                                # Training completed
                                end_time = time.time()
                                training_time = end_time - start_time
                                
                                progress_bar.progress(1.0)
                                status_text.text(f"✅ Training completed in {training_time:.2f} seconds!")
                                st.session_state.models_trained = True
                                st.success(f"🎉 Successfully trained {len(selected_models)} models!")
                                
                                # Store results in session state
                                st.session_state.model_results = {
                                    'results': results,
                                    'predictions': preds_test_dict,
                                    'cv_scores': cv_scores if use_cv else None,
                                    'problem_type': problem_type,
                                    'y_test': y_test,
                                    'target_encoder': target_encoder,
                                    'training_time': training_time
                                }
                                
                            except Exception as e:
                                st.error(f"❌ Training failed: {str(e)}")
                                st.error("💡 Try reducing the number of features or check your data quality.")
                
                # Display results if available
                if st.session_state.model_results:
                    self.display_results()
                    
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.error("💡 Please check your data format and try again.")
    else:
        # No data loaded state
        st.markdown("---")
        st.info("👆 Please select or upload a dataset to begin your machine learning journey!")
        
        # Show some tips while waiting
        st.markdown("### 💡 Tips for Better Results:")
        tip_col1, tip_col2 = st.columns(2)
        
        with tip_col1:
            st.markdown("""
            **Data Quality:**
            - Ensure your CSV has proper headers
            - Remove or handle missing values
            - Check for duplicate rows
            - Verify data types are appropriate
            """)
        
        with tip_col2:
            st.markdown("""
            **Feature Selection:**
            - Choose relevant features for your problem
            - Remove highly correlated features
            - Consider feature engineering
            - Start with fewer features for testing
            """)

def display_results():
    """Enhanced results display with interactive visualizations"""
    if not st.session_state.model_results:
        return
    
    results_data = st.session_state.model_results
    results = results_data['results']
    problem_type = results_data['problem_type']
    cv_scores = results_data.get('cv_scores', None)
    
    st.markdown("---")
    st.markdown("### 🏆 Model Performance Dashboard")
    
    # Performance summary
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.metric("Models Trained", len(results))
    with summary_col2:
        st.metric("Training Time", f"{results_data['training_time']:.2f}s")
    with summary_col3:
        primary_metric = "Accuracy" if problem_type == "Classification" else "R²"
        best_score = max([results[model]['test'][0] for model in results.keys()])
        st.metric(f"Best {primary_metric}", f"{best_score:.3f}")
    
    # Model comparison table
    st.markdown("#### 📊 Model Comparison")
    
    comparison_data = []
    for model_name, model_results in results.items():
        row = {'Model': model_name}
        for i, metric in enumerate(model_results['metrics']):
            row[f'{metric} (Train)'] = model_results['train'][i]
            row[f'{metric} (Test)'] = model_results['test'][i]
        
        if cv_scores and model_name in cv_scores:
            row['CV Score'] = cv_scores[model_name].mean()
            row['CV Std'] = cv_scores[model_name].std()
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df.round(4), use_container_width=True)
    
    # Interactive performance charts
    st.markdown("#### 📈 Performance Visualization")
    
    # Create performance comparison chart
    if problem_type == "Classification":
        primary_metric = 'Accuracy (Test)'
    else:
        primary_metric = 'R² (Test)'
    
    fig = px.bar(
        comparison_df,
        x='Model',
        y=primary_metric,
        title=f'Model Comparison - {primary_metric}',
        color=primary_metric,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Cross-validation results if available
    if cv_scores:
        st.markdown("#### 🔄 Cross-Validation Results")
        cv_data = []
        for model, scores in cv_scores.items():
            for fold, score in enumerate(scores):
                cv_data.append({'Model': model, 'Fold': f'Fold {fold+1}', 'Score': score})
        
        cv_df = pd.DataFrame(cv_data)
        fig_cv = px.box(
            cv_df,
            x='Model',
            y='Score',
            title='Cross-Validation Score Distribution',
            color='Model'
        )
        st.plotly_chart(fig_cv, use_container_width=True)

# Call the display function if results exist
if st.session_state.page == "upload" and st.session_state.model_results:
    display_results()
                        if problem_type == "Regression":
                            ALL_MODELS = ["Linear Regression", "SVM", "KNN", "Decision Tree", "Random Forest"]
                        else:
                            ALL_MODELS = ["Logistic Regression", "SVM", "KNN", "Decision Tree", "Random Forest"]

                        selected_models = st.multiselect(
                            "Select one or more algorithms to train",
                            ALL_MODELS,
                            default=ALL_MODELS[:2]
                        )
                        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, step=0.05, format="%.2f")

                        if st.button("Train Selected Models", use_container_width=True):
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y_processed, test_size=test_size, random_state=42
                            )

                            model_defs = {
                                "Linear Regression": LinearRegression(),
                                "SVM": SVR() if problem_type == "Regression" else SVC(random_state=42),
                                "KNN": KNeighborsRegressor() if problem_type == "Regression" else KNeighborsClassifier(),
                                "Decision Tree": DecisionTreeRegressor(random_state=42) if problem_type == "Regression" else DecisionTreeClassifier(random_state=42),
                                "Random Forest": RandomForestRegressor(random_state=42) if problem_type == "Regression" else RandomForestClassifier(random_state=42),
                                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
                            }

                            results = {}
                            preds_test_dict = {}

                            with st.spinner("Training selected models..."):
                                for mdl in selected_models:
                                    model = model_defs[mdl]
                                    model.fit(X_train, y_train)
                                    train_preds = model.predict(X_train)
                                    test_preds = model.predict(X_test)
                                    preds_test_dict[mdl] = test_preds
                                    if problem_type == "Regression":
                                        results[mdl] = {
                                            "train": [r2_score(y_train, train_preds), mean_squared_error(y_train, train_preds), np.sqrt(mean_squared_error(y_train, train_preds))],
                                            "test": [r2_score(y_test, test_preds), mean_squared_error(y_test, test_preds), np.sqrt(mean_squared_error(y_test, test_preds))],
                                            "metrics": ["R2", "MSE", "RMSE"]
                                        }
                                    else:
                                        results[mdl] = {
                                            "train": [
                                                accuracy_score(y_train, train_preds),
                                                precision_score(y_train, train_preds, average="weighted", zero_division=0),
                                                recall_score(y_train, train_preds, average="weighted", zero_division=0),
                                                f1_score(y_train, train_preds, average="weighted", zero_division=0)
                                            ],
                                            "test": [
                                                accuracy_score(y_test, test_preds),
                                                precision_score(y_test, test_preds, average="weighted", zero_division=0),
                                                recall_score(y_test, test_preds, average="weighted", zero_division=0),
                                                f1_score(y_test, test_preds, average="weighted", zero_division=0)
                                            ],
                                            "metrics": ["Accuracy", "Precision", "Recall", "F1"]
                                        }

                            st.subheader("Model Results and Comparison")
                            num_models = len(selected_models)
                            n_per_row = 2

                            for row_idx in range(math.ceil(num_models / n_per_row)):
                                cols = st.columns(n_per_row, gap="large")
                                for i in range(n_per_row):
                                    model_idx = row_idx * n_per_row + i
                                    if model_idx >= num_models:
                                        continue
                                    mdl = selected_models[model_idx]
                                    with cols[i]:
                                        st.markdown(
                                            f"<h4 style='color:#bbbbbb;margin-top:1.3em;font-size:1.13em'>{mdl}</h4>",
                                            unsafe_allow_html=True
                                        )
                                        model_metrics = results[mdl]["metrics"]
                                        metric_arr = np.array([results[mdl]["train"], results[mdl]["test"]])
                                        df_metrics = pd.DataFrame(metric_arr, columns=model_metrics, index=["Train", "Test"])
                                        st.dataframe(df_metrics.style.format("{:.3f}"), use_container_width=True)
                                        
                                        # Line chart
                                        fig, ax = plt.subplots(figsize=(2.9, 1.35))
                                        for j, metric in enumerate(model_metrics):
                                            ax.plot(["Train", "Test"], [results[mdl]["train"][j], results[mdl]["test"][j]],
                                                marker="o", label=metric, linewidth=2)
                                        if problem_type != "Regression":
                                            ax.set_ylim(0, 1.03)
                                        ax.set_xticks(["Train", "Test"])
                                        ax.grid(alpha=0.18, linestyle="--", linewidth=0.7)
                                        box = ax.get_position()
                                        ax.set_position([box.x0, box.y0 + box.height * 0.18, box.width, box.height * 0.82])
                                        ax.legend(fontsize="x-small", ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.2))
                                        ax.set_ylabel("Score")
                                        ax.set_title("Metric Trend", fontsize=9, color="#bbbbbb")
                                        st.pyplot(fig, use_container_width=False)

                                        # Confusion matrix for classification
                                        if problem_type == "Classification":
                                            st.markdown("**Confusion Matrix (Test)**")
                                            class_labels = np.unique(y_test)
                                            n_class = len(class_labels)
                                            fig_cm, ax_cm = plt.subplots(figsize=(max(2.1, 0.55*n_class), max(1.4, 0.45*n_class)))
                                            cm = confusion_matrix(y_test, preds_test_dict[mdl], labels=class_labels)
                                            sns.heatmap(
                                                cm, annot=True, fmt="d",
                                                cmap="Blues", ax=ax_cm, cbar=False,
                                                annot_kws={"size": 10 if n_class <= 4 else 8}
                                            )
                                            ax_cm.set_xlabel("Pred", fontsize=9)
                                            ax_cm.set_ylabel("Actual", fontsize=9)
                                            ax_cm.set_xticks(np.arange(n_class)+0.5)
                                            ax_cm.set_yticks(np.arange(n_class)+0.5)
                                            ax_cm.set_xticklabels(class_labels, rotation=0, fontsize=8)
                                            ax_cm.set_yticklabels(class_labels, rotation=0, fontsize=8)
                                            plt.tight_layout(pad=1.2)
                                            st.pyplot(fig_cm, use_container_width=False)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error("Check your data for missing values, text columns, or insufficient samples.")
    else:
        st.info("Please choose or upload a dataset to continue.")

