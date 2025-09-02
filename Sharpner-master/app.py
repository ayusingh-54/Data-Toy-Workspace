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
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, chi2
from sklearn.decomposition import PCA
from scipy import stats
from scipy.stats import pearsonr, spearmanr
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
if "current_df" not in st.session_state:
    st.session_state.current_df = None
if "data_analysis_done" not in st.session_state:
    st.session_state.data_analysis_done = False
if "feature_engineering_applied" not in st.session_state:
    st.session_state.feature_engineering_applied = False

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
def safe_convert_for_plotly(data):
    """Safely convert pandas data types to JSON-serializable types for Plotly"""
    if hasattr(data, 'tolist'):
        return data.tolist()
    elif hasattr(data, 'values'):
        return data.values.tolist()
    else:
        return data

class DataAnalyzer:
    @staticmethod
    def generate_data_report(df):
        """Generate comprehensive data analysis report"""
        report = {}
        
        # Basic info
        report['basic_info'] = {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'duplicates': df.duplicated().sum(),
            'numeric_cols': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_cols': df.select_dtypes(include=['object']).columns.tolist(),
            'datetime_cols': df.select_dtypes(include=['datetime64']).columns.tolist()
        }
        
        # Missing values analysis
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        report['missing_data'] = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percentage': missing_percent
        }).sort_values('Missing_Percentage', ascending=False)
        
        # Numeric columns analysis
        if report['basic_info']['numeric_cols']:
            numeric_df = df[report['basic_info']['numeric_cols']]
            report['numeric_analysis'] = {
                'describe': numeric_df.describe(),
                'skewness': numeric_df.skew(),
                'kurtosis': numeric_df.kurtosis(),
                'outliers': {}
            }
            
            # Detect outliers using IQR method
            for col in numeric_df.columns:
                Q1 = numeric_df[col].quantile(0.25)
                Q3 = numeric_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = numeric_df[(numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)][col]
                report['numeric_analysis']['outliers'][col] = len(outliers)
        
        # Categorical columns analysis
        if report['basic_info']['categorical_cols']:
            categorical_df = df[report['basic_info']['categorical_cols']]
            report['categorical_analysis'] = {}
            for col in categorical_df.columns:
                report['categorical_analysis'][col] = {
                    'unique_count': categorical_df[col].nunique(),
                    'top_values': categorical_df[col].value_counts().head(10),
                    'cardinality': 'High' if categorical_df[col].nunique() > 50 else 'Low'
                }
        
        return report
    
    @staticmethod
    def create_correlation_analysis(df):
        """Create correlation analysis for numeric columns"""
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return None
        
        # Pearson correlation
        pearson_corr = numeric_df.corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(pearson_corr.columns)):
            for j in range(i+1, len(pearson_corr.columns)):
                corr_val = pearson_corr.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_corr_pairs.append({
                        'Feature1': pearson_corr.columns[i],
                        'Feature2': pearson_corr.columns[j],
                        'Correlation': corr_val
                    })
        
        return {
            'correlation_matrix': pearson_corr,
            'high_correlations': pd.DataFrame(high_corr_pairs)
        }
    
    @staticmethod
    def detect_data_quality_issues(df):
        """Detect various data quality issues"""
        issues = []
        
        # Check for duplicates
        if df.duplicated().sum() > 0:
            issues.append({
                'type': 'Duplicates',
                'count': df.duplicated().sum(),
                'severity': 'Medium',
                'description': f'Found {df.duplicated().sum()} duplicate rows'
            })
        
        # Check for high missing values
        missing_pct = (df.isnull().sum() / len(df)) * 100
        high_missing = missing_pct[missing_pct > 50]
        if len(high_missing) > 0:
            issues.append({
                'type': 'High Missing Values',
                'count': len(high_missing),
                'severity': 'High',
                'description': f'{len(high_missing)} columns have >50% missing values'
            })
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            issues.append({
                'type': 'Constant Columns',
                'count': len(constant_cols),
                'severity': 'Medium',
                'description': f'Columns with constant values: {constant_cols}'
            })
        
        # Check for high cardinality categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        high_cardinality = [col for col in categorical_cols if df[col].nunique() > 100]
        if high_cardinality:
            issues.append({
                'type': 'High Cardinality',
                'count': len(high_cardinality),
                'severity': 'Low',
                'description': f'High cardinality categorical columns: {high_cardinality}'
            })
        
        return pd.DataFrame(issues) if issues else pd.DataFrame()

class FeatureEngineer:
    @staticmethod
    def create_polynomial_features(df, columns, degree=2):
        """Create polynomial features for specified columns"""
        new_df = df.copy()
        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                for d in range(2, degree + 1):
                    new_col_name = f"{col}_poly_{d}"
                    new_df[new_col_name] = df[col] ** d
        return new_df
    
    @staticmethod
    def create_interaction_features(df, col1, col2):
        """Create interaction features between two columns"""
        new_df = df.copy()
        if col1 in df.columns and col2 in df.columns:
            if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                new_df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
                new_df[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + 1e-8)  # Avoid division by zero
        return new_df
    
    @staticmethod
    def create_binning_features(df, column, bins=5, strategy='equal_width'):
        """Create binned versions of continuous variables"""
        new_df = df.copy()
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            if strategy == 'equal_width':
                new_df[f"{column}_binned"] = pd.cut(df[column], bins=bins, labels=False)
            elif strategy == 'equal_frequency':
                new_df[f"{column}_binned"] = pd.qcut(df[column], q=bins, labels=False, duplicates='drop')
        return new_df
    
    @staticmethod
    def create_log_features(df, columns):
        """Create log-transformed features"""
        new_df = df.copy()
        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # Add small constant to handle zeros and negative values
                new_df[f"{col}_log"] = np.log1p(df[col] - df[col].min() + 1)
        return new_df
    
    @staticmethod
    def create_statistical_features(df, columns):
        """Create statistical features like rolling means, etc."""
        new_df = df.copy()
        numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if len(numeric_cols) >= 2:
            # Create mean and std across selected features
            new_df['features_mean'] = df[numeric_cols].mean(axis=1)
            new_df['features_std'] = df[numeric_cols].std(axis=1)
            new_df['features_min'] = df[numeric_cols].min(axis=1)
            new_df['features_max'] = df[numeric_cols].max(axis=1)
        
        return new_df
    
    @staticmethod
    def apply_scaling(df, columns, method='standard'):
        """Apply different scaling methods"""
        new_df = df.copy()
        numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            return new_df
        
        if numeric_cols:
            new_df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        return new_df

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
        # Store current dataframe in session state
        st.session_state.current_df = df.copy()
        
        st.markdown("---")
        st.markdown("### 🔬 Comprehensive Data Analysis & Feature Engineering")
        
        # Create comprehensive tabs for analysis
        analysis_tabs = st.tabs([
            "📊 Dataset Overview", 
            "🔍 Data Quality Report", 
            "📈 Statistical Analysis", 
            "🔗 Correlation Analysis",
            "📊 Data Visualization",
            "⚙️ Feature Engineering",
            "🎯 ML Configuration"
        ])
        
        with analysis_tabs[0]:  # Dataset Overview
            st.markdown("**📋 Dataset Summary:**")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(df.head(10), use_container_width=True)
                
                # Basic statistics
                st.markdown("**Basic Information:**")
                info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                with info_col1:
                    st.metric("Total Rows", f"{df.shape[0]:,}")
                with info_col2:
                    st.metric("Total Columns", df.shape[1])
                with info_col3:
                    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
                    st.metric("Memory Usage", f"{memory_mb:.2f} MB")
                with info_col4:
                    st.metric("Duplicates", df.duplicated().sum())
            
            with col2:
                st.markdown("**Column Types:**")
                dtype_counts = df.dtypes.value_counts()
                
                # Convert to native Python types for JSON serialization
                try:
                    fig_dtype = px.pie(
                        values=safe_convert_for_plotly(dtype_counts.values),
                        names=[str(name) for name in dtype_counts.index],
                        title='Data Type Distribution'
                    )
                    fig_dtype.update_layout(height=300)
                    st.plotly_chart(fig_dtype, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating data type chart: {str(e)}")
                    st.write("**Data Types:**")
                    st.write(dtype_counts)
                
                # Column breakdown
                st.markdown("**Feature Breakdown:**")
                numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                categorical_cols = len(df.select_dtypes(include=['object']).columns)
                datetime_cols = len(df.select_dtypes(include=['datetime64']).columns)
                
                st.write(f"🔢 Numeric: {numeric_cols}")
                st.write(f"📝 Categorical: {categorical_cols}")
                st.write(f"📅 DateTime: {datetime_cols}")
        
        with analysis_tabs[1]:  # Data Quality Report
            st.markdown("**🔍 Data Quality Assessment:**")
            
            # Generate data quality report
            quality_issues = DataAnalyzer.detect_data_quality_issues(df)
            
            if not quality_issues.empty:
                st.markdown("**⚠️ Identified Issues:**")
                
                for _, issue in quality_issues.iterrows():
                    severity_color = {
                        'High': '🔴',
                        'Medium': '🟡', 
                        'Low': '🟢'
                    }
                    st.warning(f"{severity_color[issue['severity']]} **{issue['type']}**: {issue['description']}")
                
                st.dataframe(quality_issues, use_container_width=True)
            else:
                st.success("✅ No major data quality issues detected!")
            
            # Missing values analysis
            st.markdown("**🕳️ Missing Values Analysis:**")
            missing_data = df.isnull().sum()
            missing_percent = (missing_data / len(df)) * 100
            
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Count': missing_data.values,
                'Missing %': missing_percent.values
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
            
            if not missing_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(missing_df, use_container_width=True)
                
                with col2:
                    fig_missing = px.bar(
                        missing_df,
                        x='Column',
                        y='Missing %',
                        title='Missing Data by Column',
                        color='Missing %',
                        color_continuous_scale='Reds'
                    )
                    fig_missing.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.success("✅ No missing values detected!")
        
        with analysis_tabs[2]:  # Statistical Analysis
            st.markdown("**📊 Statistical Summary:**")
            
            # Numeric analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.markdown("**🔢 Numeric Features Analysis:**")
                
                # Descriptive statistics
                numeric_stats = df[numeric_cols].describe()
                st.dataframe(numeric_stats, use_container_width=True)
                
                # Distribution analysis
                st.markdown("**📈 Distribution Analysis:**")
                selected_numeric = st.selectbox("Select column for distribution analysis:", numeric_cols)
                
                if selected_numeric:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Histogram
                        try:
                            fig_hist = px.histogram(
                                df, 
                                x=selected_numeric, 
                                title=f'Distribution of {selected_numeric}',
                                marginal='box'
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating histogram: {str(e)}")
                            st.write(f"**Basic statistics for {selected_numeric}:**")
                            st.write(df[selected_numeric].describe())
                    
                    with col2:
                        # Q-Q plot
                        try:
                            fig_qq = go.Figure()
                            sample_data = df[selected_numeric].dropna()
                            qq_data = stats.probplot(sample_data, dist="norm")
                            
                            fig_qq.add_trace(go.Scatter(
                                x=safe_convert_for_plotly(qq_data[0][0]),
                                y=safe_convert_for_plotly(qq_data[0][1]),
                                mode='markers',
                                name='Sample Data'
                            ))
                            
                            fig_qq.add_trace(go.Scatter(
                                x=safe_convert_for_plotly(qq_data[0][0]),
                                y=safe_convert_for_plotly(qq_data[1][1] + qq_data[1][0] * qq_data[0][0]),
                                mode='lines',
                                name='Normal Distribution',
                                line=dict(color='red')
                            ))
                            
                            fig_qq.update_layout(
                                title=f'Q-Q Plot: {selected_numeric}',
                                xaxis_title='Theoretical Quantiles',
                                yaxis_title='Sample Quantiles'
                            )
                            st.plotly_chart(fig_qq, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating Q-Q plot: {str(e)}")
                            st.write(f"**Normality test for {selected_numeric}:**")
                            skewness = df[selected_numeric].skew()
                            kurtosis = df[selected_numeric].kurtosis()
                            st.write(f"Skewness: {skewness:.3f}")
                            st.write(f"Kurtosis: {kurtosis:.3f}")
                
                # Skewness and kurtosis
                skew_kurt_df = pd.DataFrame({
                    'Column': numeric_cols,
                    'Skewness': [df[col].skew() for col in numeric_cols],
                    'Kurtosis': [df[col].kurtosis() for col in numeric_cols]
                })
                
                st.markdown("**📐 Skewness & Kurtosis:**")
                st.dataframe(skew_kurt_df, use_container_width=True)
            
            # Categorical analysis
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                st.markdown("**📝 Categorical Features Analysis:**")
                
                selected_categorical = st.selectbox("Select categorical column:", categorical_cols)
                
                if selected_categorical:
                    value_counts = df[selected_categorical].value_counts().head(15)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.dataframe(value_counts, use_container_width=True)
                    
                    with col2:
                        try:
                            fig_cat = px.bar(
                                x=safe_convert_for_plotly(value_counts.index),
                                y=safe_convert_for_plotly(value_counts.values),
                                title=f'Value Counts: {selected_categorical}'
                            )
                            fig_cat.update_xaxes(tickangle=45)
                            st.plotly_chart(fig_cat, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating chart: {str(e)}")
                            st.write("**Value Counts:**")
                            st.write(value_counts)
        
        with analysis_tabs[3]:  # Correlation Analysis
            st.markdown("**🔗 Feature Correlation Analysis:**")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                correlation_analysis = DataAnalyzer.create_correlation_analysis(df)
                
                if correlation_analysis:
                    # Correlation heatmap
                    try:
                        fig_corr = px.imshow(
                            correlation_analysis['correlation_matrix'],
                            title="Feature Correlation Matrix",
                            color_continuous_scale='RdBu',
                            aspect='auto'
                        )
                        fig_corr.update_layout(height=600)
                        st.plotly_chart(fig_corr, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating correlation heatmap: {str(e)}")
                        st.write("**Correlation Matrix:**")
                        st.dataframe(correlation_analysis['correlation_matrix'], use_container_width=True)
                    
                    # High correlation pairs
                    if not correlation_analysis['high_correlations'].empty:
                        st.markdown("**⚠️ Highly Correlated Feature Pairs (|r| > 0.7):**")
                        st.dataframe(correlation_analysis['high_correlations'], use_container_width=True)
                        st.warning("Consider removing one feature from highly correlated pairs to reduce multicollinearity.")
                    else:
                        st.success("✅ No highly correlated feature pairs found.")
            else:
                st.info("Need at least 2 numeric columns for correlation analysis.")
        
        with analysis_tabs[4]:  # Data Visualization
            st.markdown("**📊 Interactive Data Visualization:**")
            
            viz_type = st.selectbox(
                "Select visualization type:",
                ["Scatter Plot", "Box Plot", "Violin Plot", "Pair Plot", "Distribution Comparison"]
            )
            
            if viz_type == "Scatter Plot":
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 2:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        x_axis = st.selectbox("X-axis:", numeric_cols)
                    with col2:
                        y_axis = st.selectbox("Y-axis:", [col for col in numeric_cols if col != x_axis])
                    with col3:
                        color_col = st.selectbox("Color by:", ["None"] + df.columns.tolist())
                    
                    if x_axis and y_axis:
                        try:
                            fig_scatter = px.scatter(
                                df, 
                                x=x_axis, 
                                y=y_axis, 
                                color=color_col if color_col != "None" else None,
                                title=f'{y_axis} vs {x_axis}',
                                trendline="ols" if color_col == "None" else None
                            )
                            st.plotly_chart(fig_scatter, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating scatter plot: {str(e)}")
                            st.write("Please try with different columns or check your data.")
            
            elif viz_type == "Box Plot":
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                
                if numeric_cols:
                    col1, col2 = st.columns(2)
                    with col1:
                        y_col = st.selectbox("Numeric column:", numeric_cols)
                    with col2:
                        x_col = st.selectbox("Group by (optional):", ["None"] + categorical_cols)
                    
                    if y_col:
                        try:
                            fig_box = px.box(
                                df, 
                                y=y_col, 
                                x=x_col if x_col != "None" else None,
                                title=f'Box Plot: {y_col}'
                            )
                            st.plotly_chart(fig_box, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating box plot: {str(e)}")
                            st.write("Please try with different columns.")
            
            elif viz_type == "Distribution Comparison":
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 2:
                    selected_cols = st.multiselect(
                        "Select columns to compare:",
                        numeric_cols,
                        default=numeric_cols[:3]
                    )
                    
                    if selected_cols:
                        try:
                            fig_dist = go.Figure()
                            for col in selected_cols:
                                fig_dist.add_trace(go.Histogram(
                                    x=safe_convert_for_plotly(df[col].dropna()),
                                    name=col,
                                    opacity=0.7
                                ))
                            
                            fig_dist.update_layout(
                                title="Distribution Comparison",
                                barmode='overlay'
                            )
                            st.plotly_chart(fig_dist, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating distribution comparison: {str(e)}")
                            st.write("Please try with different columns.")
        
        with analysis_tabs[5]:  # Feature Engineering
            st.markdown("**⚙️ Advanced Feature Engineering:**")
            
            # Initialize feature engineering dataframe
            if 'engineered_df' not in st.session_state:
                st.session_state.engineered_df = df.copy()
            
            fe_options = st.multiselect(
                "Select feature engineering techniques:",
                [
                    "Polynomial Features",
                    "Interaction Features", 
                    "Binning/Discretization",
                    "Log Transformation",
                    "Statistical Features",
                    "Feature Scaling"
                ]
            )
            
            if "Polynomial Features" in fe_options:
                st.markdown("**🔢 Polynomial Features:**")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                poly_cols = st.multiselect("Select columns for polynomial features:", numeric_cols)
                poly_degree = st.slider("Polynomial degree:", 2, 5, 2)
                
                if poly_cols and st.button("Create Polynomial Features"):
                    st.session_state.engineered_df = FeatureEngineer.create_polynomial_features(
                        st.session_state.engineered_df, poly_cols, poly_degree
                    )
                    st.success(f"✅ Created polynomial features for {len(poly_cols)} columns")
            
            if "Interaction Features" in fe_options:
                st.markdown("**🔗 Interaction Features:**")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        int_col1 = st.selectbox("First column:", numeric_cols, key="int1")
                    with col2:
                        int_col2 = st.selectbox("Second column:", [col for col in numeric_cols if col != int_col1], key="int2")
                    
                    if int_col1 and int_col2 and st.button("Create Interaction Features"):
                        st.session_state.engineered_df = FeatureEngineer.create_interaction_features(
                            st.session_state.engineered_df, int_col1, int_col2
                        )
                        st.success(f"✅ Created interaction features between {int_col1} and {int_col2}")
            
            if "Binning/Discretization" in fe_options:
                st.markdown("**📊 Binning/Discretization:**")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                bin_col = st.selectbox("Select column to bin:", numeric_cols, key="bin_col")
                col1, col2 = st.columns(2)
                with col1:
                    bin_count = st.slider("Number of bins:", 3, 10, 5)
                with col2:
                    bin_strategy = st.selectbox("Binning strategy:", ["equal_width", "equal_frequency"])
                
                if bin_col and st.button("Apply Binning"):
                    st.session_state.engineered_df = FeatureEngineer.create_binning_features(
                        st.session_state.engineered_df, bin_col, bin_count, bin_strategy
                    )
                    st.success(f"✅ Created binned features for {bin_col}")
            
            if "Log Transformation" in fe_options:
                st.markdown("**📈 Log Transformation:**")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                log_cols = st.multiselect("Select columns for log transformation:", numeric_cols, key="log_cols")
                
                if log_cols and st.button("Apply Log Transformation"):
                    st.session_state.engineered_df = FeatureEngineer.create_log_features(
                        st.session_state.engineered_df, log_cols
                    )
                    st.success(f"✅ Applied log transformation to {len(log_cols)} columns")
            
            if "Statistical Features" in fe_options:
                st.markdown("**📊 Statistical Features:**")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                stat_cols = st.multiselect("Select columns for statistical features:", numeric_cols, key="stat_cols")
                
                if len(stat_cols) >= 2 and st.button("Create Statistical Features"):
                    st.session_state.engineered_df = FeatureEngineer.create_statistical_features(
                        st.session_state.engineered_df, stat_cols
                    )
                    st.success(f"✅ Created statistical features from {len(stat_cols)} columns")
            
            if "Feature Scaling" in fe_options:
                st.markdown("**⚖️ Feature Scaling:**")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                scale_cols = st.multiselect("Select columns to scale:", numeric_cols, key="scale_cols")
                scale_method = st.selectbox("Scaling method:", ["standard", "minmax", "robust"])
                
                if scale_cols and st.button("Apply Scaling"):
                    st.session_state.engineered_df = FeatureEngineer.apply_scaling(
                        st.session_state.engineered_df, scale_cols, scale_method
                    )
                    st.success(f"✅ Applied {scale_method} scaling to {len(scale_cols)} columns")
            
            # Show current engineered features
            if len(st.session_state.engineered_df.columns) > len(df.columns):
                st.markdown("**🆕 Engineered Features Summary:**")
                new_features = [col for col in st.session_state.engineered_df.columns if col not in df.columns]
                st.write(f"Created {len(new_features)} new features:")
                st.write(new_features)
                
                if st.button("Reset Feature Engineering"):
                    st.session_state.engineered_df = df.copy()
                    st.success("✅ Reset to original dataset")
                    st.experimental_rerun()
                
                # Option to use engineered features for ML
                use_engineered = st.checkbox("Use engineered features for machine learning", value=True)
                if use_engineered:
                    df = st.session_state.engineered_df.copy()
                    st.session_state.feature_engineering_applied = True
        
        with analysis_tabs[6]:  # ML Configuration
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
