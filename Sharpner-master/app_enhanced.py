import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    r2_score, mean_squared_error, accuracy_score,
    confusion_matrix, precision_score, recall_score, f1_score,
    classification_report, mean_absolute_error
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import math

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Data Toy - AutoML Platform", 
    layout="wide",
    page_icon="🎯",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "features_selected" not in st.session_state:
    st.session_state.features_selected = False
if "models_trained" not in st.session_state:
    st.session_state.models_trained = False
if "model_results" not in st.session_state:
    st.session_state.model_results = None

# Navigation functions
def go_to_upload():
    st.session_state.page = "upload"

def go_to_landing():
    st.session_state.page = "landing"
    st.session_state.data_loaded = False
    st.session_state.features_selected = False
    st.session_state.models_trained = False
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
        
        /* Progress bar styling */
        .stProgress > div > div > div > div {
            background: linear-gradient(45deg, #4CAF50, #81C784);
        }
    </style>
    """, unsafe_allow_html=True)

# Machine Learning utility functions
class MLPipeline:
    @staticmethod
    def detect_problem_type(df, target):
        """Enhanced problem type detection"""
        target_series = df[target].dropna()
        dtype = target_series.dtype
        nunique = target_series.nunique()
        
        if dtype == "object":
            return "Classification"
        elif nunique <= 2:
            return "Classification"
        elif nunique <= 20 and nunique < len(target_series) * 0.05:
            return "Classification"
        else:
            return "Regression"
    
    @staticmethod
    def preprocess_features(df, features):
        """Enhanced feature preprocessing"""
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
            numeric_imputer = SimpleImputer(strategy='median')
            X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])
            preprocessing_log.append(f"Imputed missing values in {len(numeric_cols)} numeric columns")
        
        if categorical_cols:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])
            preprocessing_log.append(f"Imputed missing values in {len(categorical_cols)} categorical columns")
        
        # Handle categorical variables
        for col in categorical_cols:
            if X[col].nunique() > 50:
                X = X.drop(columns=[col])
                preprocessing_log.append(f"Dropped '{col}' due to high cardinality")
            else:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                preprocessing_log.append(f"Label encoded '{col}'")
        
        return X, preprocessing_log
    
    @staticmethod
    def preprocess_target(y):
        """Enhanced target preprocessing"""
        preprocessing_log = []
        
        if y.dtype == 'object':
            if y.isnull().any():
                y = y.fillna(y.mode()[0] if not y.mode().empty else 'unknown')
            
            le = LabelEncoder()
            y_enc = le.fit_transform(y.astype(str))
            preprocessing_log.append(f"Label encoded target ({len(le.classes_)} classes)")
            return y_enc, le, preprocessing_log
        else:
            if y.isnull().any():
                y_filled = y.fillna(y.median())
                preprocessing_log.append(f"Imputed missing values in target")
            else:
                y_filled = y
            preprocessing_log.append("Numeric target - regression problem")
            return y_filled, None, preprocessing_log
    
    @staticmethod
    def get_models(problem_type, random_state=42):
        """Get model definitions based on problem type"""
        if problem_type == "Regression":
            return {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state, max_depth=10),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=random_state),
                "SVM": SVR(kernel='rbf'),
                "KNN": KNeighborsRegressor(n_neighbors=5),
                "Decision Tree": DecisionTreeRegressor(random_state=random_state, max_depth=10)
            }
        else:
            return {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=10),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=random_state),
                "SVM": SVC(kernel='rbf', random_state=random_state, probability=True),
                "KNN": KNeighborsClassifier(n_neighbors=5),
                "Decision Tree": DecisionTreeClassifier(random_state=random_state, max_depth=10),
                "Naive Bayes": GaussianNB()
            }
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, problem_type):
        """Calculate metrics based on problem type"""
        if problem_type == "Regression":
            return {
                "R²": r2_score(y_true, y_pred),
                "MSE": mean_squared_error(y_true, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
                "MAE": mean_absolute_error(y_true, y_pred)
            }
        else:
            return {
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
                "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
                "F1-Score": f1_score(y_true, y_pred, average="weighted", zero_division=0)
            }

###########################
#         LANDING PAGE    #
###########################
if st.session_state.page == "landing":
    load_css()
    
    # Hero section
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
        - 7+ machine learning algorithms
        - Automatic problem type detection
        - Cross-validation scoring
        - Feature scaling optimization
        """)
    
    with col3:
        st.markdown("### 📊 **Rich Analytics**")
        st.markdown("""
        - Interactive visualizations
        - Comprehensive model metrics
        - Confusion matrices
        - Performance comparisons
        """)
    
    st.markdown("---")
    
    # Call-to-action
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🚀 Start Building", use_container_width=True, type="primary"):
            go_to_upload()
    
    # Footer stats
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Models Supported", "7+", "Advanced ML")
    with col2:
        st.metric("Processing Speed", "< 60s", "Ultra Fast")
    with col3:
        st.metric("Data Types", "CSV", "Tabular Data")
    with col4:
        st.metric("Code Required", "0 Lines", "No-Code")

###########################
#      DATA UPLOAD PAGE   #
###########################
elif st.session_state.page == "upload":
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
        upload_status = "✅" if st.session_state.data_loaded else "⏳"
        st.markdown(f"**2️⃣ Data Preview** {upload_status}")
    with progress_col3:
        feature_status = "✅" if st.session_state.features_selected else "⏳"
        st.markdown(f"**3️⃣ Feature Selection** {feature_status}")
    with progress_col4:
        model_status = "✅" if st.session_state.models_trained else "⏳"
        st.markdown(f"**4️⃣ Model Training** {model_status}")
    
    st.markdown("---")
    
    # Dataset upload section
    st.markdown("### 📁 Dataset Upload")
    
    upload_tab1, upload_tab2 = st.tabs(["📂 Sample Datasets", "⬆️ Upload Your Data"])
    
    df = None
    
    with upload_tab1:
        st.markdown("**Choose from our curated sample datasets:**")
        dataset_dir = "datasets"
        
        if os.path.exists(dataset_dir):
            builtin = [f for f in os.listdir(dataset_dir) if f.endswith(".csv")]
            
            if builtin:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    choice = st.selectbox("Available datasets:", ["Select a dataset..."] + builtin)
                    
                    if choice and choice != "Select a dataset...":
                        descriptions = {
                            "Iris.csv": "🌸 Classic iris flower classification dataset",
                            "Titanic-Dataset.csv": "🚢 Passenger survival prediction",
                            "breast-cancer.csv": "🔬 Medical diagnosis dataset"
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
                            
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Rows", preview_df.shape[0])
                            with col_b:
                                st.metric("Columns", preview_df.shape[1])
                            with col_c:
                                missing_pct = (preview_df.isnull().sum().sum() / (preview_df.shape[0] * preview_df.shape[1]) * 100)
                                st.metric("Missing %", f"{missing_pct:.1f}")
                        except:
                            pass
            else:
                st.warning("No sample datasets found.")
        else:
            st.warning("Datasets directory not found.")
    
    with upload_tab2:
        st.markdown("**Upload your own CSV file:**")
        uploaded = st.file_uploader(
            "Choose a CSV file", 
            type=["csv"],
            help="Upload a CSV file with proper column headers."
        )
        
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                st.session_state.data_loaded = True
                st.success("✅ File uploaded successfully!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("File Size", f"{uploaded.size / 1024:.1f} KB")
                with col2:
                    st.metric("Rows", df.shape[0])
                with col3:
                    st.metric("Columns", df.shape[1])
                    
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    # Data analysis and ML pipeline
    if isinstance(df, pd.DataFrame):
        st.markdown("---")
        st.markdown("### 🔍 Data Exploration & Analysis")
        
        # Data overview tabs
        overview_tab1, overview_tab2, overview_tab3 = st.tabs([
            "📊 Dataset Overview", "📈 Statistics", "🎯 ML Configuration"
        ])
        
        with overview_tab1:
            st.markdown("**Dataset Preview:**")
            st.dataframe(df.head(10), use_container_width=True)
            
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
            
            numeric_df = df.select_dtypes(include=[np.number])
            categorical_df = df.select_dtypes(include=['object'])
            
            if not numeric_df.empty:
                st.markdown("*Numeric Features:*")
                st.dataframe(numeric_df.describe(), use_container_width=True)
            
            if not categorical_df.empty:
                st.markdown("*Categorical Features:*")
                cat_summary = pd.DataFrame({
                    'Unique Values': categorical_df.nunique(),
                    'Missing Count': categorical_df.isnull().sum()
                })
                st.dataframe(cat_summary, use_container_width=True)
        
        with overview_tab3:
            columns = df.columns.tolist()
            
            # Target selection
            st.markdown("**🎯 Target Variable Selection:**")
            target = st.selectbox(
                "Select your target column:",
                ["Select target column..."] + columns
            )
            
            if target and target != "Select target column...":
                st.session_state.features_selected = True
                
                # Feature selection
                available_features = [col for col in columns if col != target]
                features = st.multiselect(
                    "Select features for training:",
                    available_features,
                    default=available_features[:min(10, len(available_features))]
                )
                
                if target in features:
                    st.error("❌ Target column cannot be in features.")
                elif not features:
                    st.warning("⚠️ Please select at least one feature.")
                else:
                    # ML Pipeline
                    st.markdown("---")
                    st.markdown("### 🤖 Machine Learning Pipeline")
                    
                    try:
                        # Preprocessing
                        working_df = df[features + [target]].copy()
                        
                        X, feature_log = MLPipeline.preprocess_features(working_df, features)
                        y_processed, target_encoder, target_log = MLPipeline.preprocess_target(working_df[target])
                        problem_type = MLPipeline.detect_problem_type(working_df, target)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.success(f"✅ Problem Type: **{problem_type}**")
                            
                            with st.expander("🔧 Preprocessing Summary"):
                                for log in feature_log + target_log:
                                    st.write(f"• {log}")
                        
                        with col2:
                            # Model selection
                            available_models = MLPipeline.get_models(problem_type)
                            selected_models = st.multiselect(
                                "Select algorithms:",
                                list(available_models.keys()),
                                default=list(available_models.keys())[:3]
                            )
                        
                        # Training parameters
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, step=0.05)
                        with col2:
                            random_state = st.number_input("Random Seed", value=42, min_value=0)
                        with col3:
                            use_scaling = st.checkbox("Feature Scaling", value=True)
                        
                        # Train models
                        if selected_models and st.button("🚀 Train Models", type="primary", use_container_width=True):
                            start_time = time.time()
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            try:
                                # Data splitting
                                status_text.text("Splitting data...")
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y_processed, test_size=test_size, random_state=random_state,
                                    stratify=y_processed if problem_type == "Classification" else None
                                )
                                progress_bar.progress(0.1)
                                
                                # Feature scaling
                                if use_scaling:
                                    scaler = StandardScaler()
                                    X_train_scaled = scaler.fit_transform(X_train)
                                    X_test_scaled = scaler.transform(X_test)
                                else:
                                    X_train_scaled = X_train
                                    X_test_scaled = X_test
                                
                                # Train models
                                results = {}
                                model_defs = MLPipeline.get_models(problem_type, random_state)
                                
                                for i, model_name in enumerate(selected_models):
                                    status_text.text(f"Training {model_name}...")
                                    
                                    model = model_defs[model_name]
                                    
                                    # Use scaled data for certain models
                                    if model_name in ["SVM", "KNN", "Logistic Regression"] and use_scaling:
                                        X_train_use = X_train_scaled
                                        X_test_use = X_test_scaled
                                    else:
                                        X_train_use = X_train
                                        X_test_use = X_test
                                    
                                    # Train and predict
                                    model.fit(X_train_use, y_train)
                                    train_preds = model.predict(X_train_use)
                                    test_preds = model.predict(X_test_use)
                                    
                                    # Calculate metrics
                                    train_metrics = MLPipeline.calculate_metrics(y_train, train_preds, problem_type)
                                    test_metrics = MLPipeline.calculate_metrics(y_test, test_preds, problem_type)
                                    
                                    results[model_name] = {
                                        'train_metrics': train_metrics,
                                        'test_metrics': test_metrics,
                                        'test_predictions': test_preds
                                    }
                                    
                                    progress_bar.progress(0.2 + (i + 1) * 0.7 / len(selected_models))
                                
                                # Training completed
                                end_time = time.time()
                                training_time = end_time - start_time
                                
                                progress_bar.progress(1.0)
                                status_text.text(f"✅ Training completed in {training_time:.2f} seconds!")
                                st.session_state.models_trained = True
                                
                                # Store results
                                st.session_state.model_results = {
                                    'results': results,
                                    'problem_type': problem_type,
                                    'y_test': y_test,
                                    'training_time': training_time
                                }
                                
                                st.success(f"🎉 Successfully trained {len(selected_models)} models!")
                                
                            except Exception as e:
                                st.error(f"❌ Training failed: {str(e)}")
                    
                    except Exception as e:
                        st.error(f"❌ Preprocessing failed: {str(e)}")
    
    # Display results
    if st.session_state.model_results:
        st.markdown("---")
        st.markdown("### 🏆 Model Performance Dashboard")
        
        results_data = st.session_state.model_results
        results = results_data['results']
        problem_type = results_data['problem_type']
        
        # Performance summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Models Trained", len(results))
        with col2:
            st.metric("Training Time", f"{results_data['training_time']:.2f}s")
        with col3:
            primary_metric = "Accuracy" if problem_type == "Classification" else "R²"
            best_score = max([list(results[model]['test_metrics'].values())[0] for model in results.keys()])
            st.metric(f"Best {primary_metric}", f"{best_score:.3f}")
        
        # Model comparison table
        st.markdown("#### 📊 Model Comparison")
        
        comparison_data = []
        for model_name, model_results in results.items():
            row = {'Model': model_name}
            
            # Add train metrics
            for metric, value in model_results['train_metrics'].items():
                row[f'{metric} (Train)'] = value
            
            # Add test metrics
            for metric, value in model_results['test_metrics'].items():
                row[f'{metric} (Test)'] = value
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df.round(4), use_container_width=True)
        
        # Performance visualization
        if problem_type == "Classification":
            primary_metric = 'Accuracy (Test)'
        else:
            primary_metric = 'R² (Test)'
        
        fig = px.bar(
            comparison_df,
            x='Model',
            y=primary_metric,
            title=f'Model Performance - {primary_metric}',
            color=primary_metric,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrices for classification
        if problem_type == "Classification":
            st.markdown("#### 🔍 Confusion Matrices")
            
            cols = st.columns(min(3, len(results)))
            for i, (model_name, model_results) in enumerate(results.items()):
                if i < len(cols):
                    with cols[i]:
                        y_test = results_data['y_test']
                        y_pred = model_results['test_predictions']
                        
                        cm = confusion_matrix(y_test, y_pred)
                        
                        fig = px.imshow(
                            cm,
                            title=f'{model_name}',
                            color_continuous_scale='Blues',
                            aspect='auto'
                        )
                        fig.update_layout(
                            width=300,
                            height=250,
                            title_x=0.5
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    else:
        # No data loaded state
        st.markdown("---")
        st.info("👆 Please select or upload a dataset to begin your machine learning journey!")
        
        st.markdown("### 💡 Tips for Better Results:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Data Quality:**
            - Ensure CSV has proper headers
            - Handle missing values appropriately
            - Remove duplicate rows
            - Verify data types
            """)
        
        with col2:
            st.markdown("""
            **Feature Selection:**
            - Choose relevant features
            - Remove highly correlated features
            - Start with fewer features
            - Consider domain knowledge
            """)
