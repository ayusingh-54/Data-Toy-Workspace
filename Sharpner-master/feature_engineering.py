import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineering:
    """Advanced feature engineering toolkit for data transformation and optimization"""
    
    @staticmethod
    def transform_features(df, target_col=None):
        """Apply advanced feature engineering techniques to a dataset"""
        if df is None or df.empty:
            st.error("No data loaded")
            return df
        
        st.markdown("### 🔧 Feature Engineering Toolkit")
        
        # Save original data for comparison
        original_df = df.copy()
        
        # Create tabs for different engineering operations
        tabs = st.tabs([
            "📊 Data Overview", "✂️ Feature Selection", 
            "🔄 Feature Transformation", "🔍 Dimensionality Reduction",
            "🧩 Feature Generation", "📈 Before/After Comparison"
        ])
        
        transformed_df = df.copy()
        transformation_log = []
        
        with tabs[0]:
            FeatureEngineering._show_data_overview(df, target_col)
            
        with tabs[1]:
            transformed_df, new_logs = FeatureEngineering._feature_selection(transformed_df, target_col)
            transformation_log.extend(new_logs)
            
        with tabs[2]:
            transformed_df, new_logs = FeatureEngineering._feature_transformation(transformed_df)
            transformation_log.extend(new_logs)
            
        with tabs[3]:
            transformed_df, new_logs = FeatureEngineering._dimensionality_reduction(transformed_df, target_col)
            transformation_log.extend(new_logs)
            
        with tabs[4]:
            transformed_df, new_logs = FeatureEngineering._feature_generation(transformed_df)
            transformation_log.extend(new_logs)
            
        with tabs[5]:
            FeatureEngineering._compare_before_after(original_df, transformed_df, transformation_log)
        
        return transformed_df
    
    @staticmethod
    def _show_data_overview(df, target_col):
        """Show overview of the data and features"""
        st.markdown("#### 📊 Data Overview")
        
        # Display summary of feature types
        num_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_features = df.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()
        
        # Feature type counts
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Features", len(df.columns))
        with col2:
            st.metric("Numeric Features", len(num_features))
        with col3:
            st.metric("Categorical Features", len(cat_features))
        
        # Missing values overview
        missing_vals = df.isnull().sum()
        missing_cols = missing_vals[missing_vals > 0].sort_values(ascending=False)
        
        if not missing_cols.empty:
            st.write("#### 🔍 Missing Values")
            
            # Calculate missing percentage
            missing_percent = (missing_cols / len(df) * 100).round(2)
            missing_df = pd.DataFrame({
                'Column': missing_cols.index,
                'Missing Count': missing_cols.values,
                'Missing %': missing_percent.values
            })
            
            # Create bar chart for missing values
            fig = px.bar(
                missing_df,
                x='Column',
                y='Missing %',
                title='Missing Values by Column',
                color='Missing %',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show table
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("✅ No missing values in the dataset")
        
        # Feature correlations for numeric data
        if len(num_features) > 1:
            st.write("#### 📈 Feature Correlations")
            
            # Create correlation heatmap
            corr_matrix = df[num_features].corr().round(2)
            
            # Show correlation heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale='RdBu_r',
                title='Feature Correlation Matrix',
                zmin=-1, zmax=1
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show target correlations if target is specified
            if target_col and target_col in num_features:
                target_corrs = corr_matrix[target_col].sort_values(ascending=False)
                target_corrs = target_corrs.drop(target_col)
                
                st.write(f"#### 🎯 Feature Correlations with Target: {target_col}")
                
                fig = px.bar(
                    x=target_corrs.index,
                    y=target_corrs.values,
                    title=f'Feature Correlations with {target_col}',
                    color=target_corrs.values,
                    color_continuous_scale='RdBu_r',
                    labels={'x': 'Feature', 'y': 'Correlation'},
                    range_color=[-1, 1]
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Feature distributions
        st.write("#### 📊 Feature Distributions")
        
        # Select features to visualize
        num_to_show = min(5, len(num_features))
        features_to_visualize = st.multiselect(
            "Select features to visualize:",
            options=num_features,
            default=num_features[:num_to_show] if num_features else []
        )
        
        if features_to_visualize:
            # Create distribution plots
            for feature in features_to_visualize:
                fig = px.histogram(
                    df,
                    x=feature,
                    marginal="box",
                    title=f'Distribution of {feature}',
                    color_discrete_sequence=['#636EFA']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show descriptive statistics
                stats_df = pd.DataFrame({
                    'Statistic': ['Mean', 'Median', 'Std', 'Min', 'Max', 'Skew'],
                    'Value': [
                        df[feature].mean(),
                        df[feature].median(),
                        df[feature].std(),
                        df[feature].min(),
                        df[feature].max(),
                        df[feature].skew()
                    ]
                })
                st.dataframe(stats_df, use_container_width=True)
    
    @staticmethod
    def _feature_selection(df, target_col):
        """Feature selection techniques"""
        st.markdown("#### ✂️ Feature Selection")
        transformation_log = []
        
        # Check if we have valid data for feature selection
        if df is None or df.empty:
            st.warning("No data available for feature selection")
            return df, transformation_log
        
        # Create tabs for different feature selection methods
        method_tabs = st.tabs([
            "🎯 Target-Based", "📊 Variance", "🔗 Correlation", "🔍 Manual"
        ])
        
        # Make a copy of the input dataframe
        selected_df = df.copy()
        
        with method_tabs[0]:
            if target_col:
                st.write("##### Target-Based Feature Selection")
                
                # Separate features and target
                X = selected_df.drop(columns=[target_col])
                y = selected_df[target_col]
                
                # Only keep numeric features
                num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                
                if len(num_features) > 0:
                    X_num = X[num_features]
                    
                    # Choose selection method
                    selection_method = st.selectbox(
                        "Select feature selection method:",
                        ["ANOVA F-value", "Mutual Information", "Chi-squared (for positive data)"]
                    )
                    
                    # Choose number of features to select
                    k = st.slider(
                        "Number of features to select:",
                        min_value=1,
                        max_value=len(num_features),
                        value=min(5, len(num_features))
                    )
                    
                    if st.button("Apply Target-Based Selection"):
                        try:
                            # Apply selected method
                            if selection_method == "ANOVA F-value":
                                selector = SelectKBest(f_classif, k=k)
                                method_name = "ANOVA F-value"
                            elif selection_method == "Mutual Information":
                                selector = SelectKBest(mutual_info_classif, k=k)
                                method_name = "Mutual Information"
                            else:  # Chi-squared
                                # Check if data is non-negative
                                if (X_num < 0).any().any():
                                    st.error("Chi-squared requires non-negative values. Please use data transformation first.")
                                    return selected_df, transformation_log
                                selector = SelectKBest(chi2, k=k)
                                method_name = "Chi-squared"
                            
                            # Fit and transform
                            X_transformed = selector.fit_transform(X_num, y)
                            
                            # Get selected feature names
                            selected_features = [num_features[i] for i in selector.get_support(indices=True)]
                            
                            # Get scores
                            scores = selector.scores_
                            feature_scores = pd.DataFrame({
                                'Feature': num_features,
                                'Score': scores
                            }).sort_values('Score', ascending=False)
                            
                            # Show selected features
                            st.write(f"**Selected {k} features using {method_name}:**")
                            st.write(", ".join(selected_features))
                            
                            # Show feature scores
                            fig = px.bar(
                                feature_scores.head(10),
                                x='Feature',
                                y='Score',
                                title=f'Top Features by {method_name}',
                                color='Score'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Update dataframe to include only selected features and target
                            selected_df = df[selected_features + [target_col]]
                            
                            # Log transformation
                            transformation_log.append(f"Applied {method_name} to select {k} features: {', '.join(selected_features)}")
                            
                            st.success(f"✅ Successfully selected {k} features with {method_name}")
                            
                        except Exception as e:
                            st.error(f"Error in feature selection: {str(e)}")
                else:
                    st.warning("No numeric features available for selection")
            else:
                st.info("Please specify a target column for target-based feature selection")
        
        with method_tabs[1]:
            st.write("##### Variance-Based Feature Selection")
            
            # Only keep numeric features
            num_features = selected_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if len(num_features) > 0:
                # Calculate variance of each feature
                variances = selected_df[num_features].var().sort_values(ascending=False)
                
                # Display variance
                variance_df = pd.DataFrame({
                    'Feature': variances.index,
                    'Variance': variances.values
                })
                
                fig = px.bar(
                    variance_df,
                    x='Feature',
                    y='Variance',
                    title='Feature Variance',
                    color='Variance'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Variance threshold
                threshold = st.slider(
                    "Variance threshold:",
                    min_value=0.0,
                    max_value=float(variance_df['Variance'].max()),
                    value=0.1,
                    format="%.5f"
                )
                
                # Apply threshold
                if st.button("Apply Variance Threshold"):
                    # Select features above threshold
                    selected_features = variance_df[variance_df['Variance'] > threshold]['Feature'].tolist()
                    
                    if target_col and target_col not in selected_features:
                        selected_features.append(target_col)
                    
                    # Update dataframe
                    selected_df = selected_df[selected_features]
                    
                    # Log transformation
                    transformation_log.append(f"Applied variance threshold of {threshold} to select {len(selected_features)} features")
                    
                    st.success(f"✅ Selected {len(selected_features)} features with variance > {threshold}")
            else:
                st.warning("No numeric features available for variance-based selection")
        
        with method_tabs[2]:
            st.write("##### Correlation-Based Feature Selection")
            
            # Only keep numeric features
            num_features = selected_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if len(num_features) > 1:
                # Calculate correlation matrix
                corr_matrix = selected_df[num_features].corr().abs()
                
                # Display correlation heatmap
                fig = px.imshow(
                    corr_matrix,
                    title='Feature Correlation Matrix',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation threshold
                threshold = st.slider(
                    "Correlation threshold:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.8,
                    step=0.05
                )
                
                if st.button("Remove Highly Correlated Features"):
                    # Find highly correlated feature pairs
                    features_to_drop = set()
                    
                    # If we have a target column, prioritize keeping features correlated with the target
                    target_corrs = None
                    if target_col and target_col in num_features:
                        target_corrs = corr_matrix[target_col].sort_values(ascending=False)
                    
                    # Loop through correlation matrix
                    for i in range(len(num_features)):
                        for j in range(i+1, len(num_features)):
                            feature_i = num_features[i]
                            feature_j = num_features[j]
                            
                            # Skip if target column
                            if feature_i == target_col or feature_j == target_col:
                                continue
                                
                            # Check if correlation is above threshold
                            if corr_matrix.loc[feature_i, feature_j] > threshold:
                                # Decide which feature to drop
                                if target_corrs is not None:
                                    # Keep the one more correlated with target
                                    if target_corrs[feature_i] > target_corrs[feature_j]:
                                        features_to_drop.add(feature_j)
                                    else:
                                        features_to_drop.add(feature_i)
                                else:
                                    # Without target, drop the one with more missing values
                                    missing_i = selected_df[feature_i].isnull().sum()
                                    missing_j = selected_df[feature_j].isnull().sum()
                                    
                                    if missing_i > missing_j:
                                        features_to_drop.add(feature_i)
                                    else:
                                        features_to_drop.add(feature_j)
                    
                    # Convert to list
                    drop_list = list(features_to_drop)
                    
                    if drop_list:
                        # Update dataframe
                        selected_df = selected_df.drop(columns=drop_list)
                        
                        # Log transformation
                        transformation_log.append(f"Removed {len(drop_list)} highly correlated features (threshold={threshold}): {', '.join(drop_list)}")
                        
                        st.success(f"✅ Removed {len(drop_list)} highly correlated features")
                    else:
                        st.info("No highly correlated features found with the given threshold")
            else:
                st.warning("Need at least 2 numeric features for correlation analysis")
        
        with method_tabs[3]:
            st.write("##### Manual Feature Selection")
            
            # Display all features for selection
            all_features = selected_df.columns.tolist()
            
            if target_col:
                # Pre-select target
                default_selection = [target_col]
            else:
                default_selection = []
            
            selected_features = st.multiselect(
                "Select features to keep:",
                options=all_features,
                default=default_selection
            )
            
            if st.button("Apply Manual Selection") and selected_features:
                # Update dataframe
                selected_df = selected_df[selected_features]
                
                # Log transformation
                transformation_log.append(f"Manually selected {len(selected_features)} features")
                
                st.success(f"✅ Selected {len(selected_features)} features")
        
        return selected_df, transformation_log
    
    @staticmethod
    def _feature_transformation(df):
        """Apply transformations to features"""
        st.markdown("#### 🔄 Feature Transformation")
        transformation_log = []
        
        # Make a copy of the input dataframe
        transformed_df = df.copy()
        
        # Create tabs for different transformation methods
        transform_tabs = st.tabs([
            "📏 Scaling", "🧹 Missing Values", "🔢 Encoding"
        ])
        
        with transform_tabs[0]:
            st.write("##### Feature Scaling")
            
            # Get numeric features
            num_features = transformed_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if num_features:
                # Select features to scale
                features_to_scale = st.multiselect(
                    "Select features to scale:",
                    options=num_features,
                    default=num_features
                )
                
                # Select scaling method
                scaling_method = st.selectbox(
                    "Select scaling method:",
                    ["StandardScaler (μ=0, σ=1)", "MinMaxScaler (0 to 1)", "RobustScaler (median, IQR)"]
                )
                
                if st.button("Apply Scaling") and features_to_scale:
                    try:
                        # Create scaler
                        if scaling_method == "StandardScaler (μ=0, σ=1)":
                            scaler = StandardScaler()
                            method_name = "StandardScaler"
                        elif scaling_method == "MinMaxScaler (0 to 1)":
                            scaler = MinMaxScaler()
                            method_name = "MinMaxScaler"
                        else:  # RobustScaler
                            scaler = RobustScaler()
                            method_name = "RobustScaler"
                        
                        # Apply scaling
                        transformed_df[features_to_scale] = scaler.fit_transform(transformed_df[features_to_scale])
                        
                        # Log transformation
                        transformation_log.append(f"Applied {method_name} to {len(features_to_scale)} features")
                        
                        st.success(f"✅ Successfully scaled {len(features_to_scale)} features with {method_name}")
                        
                        # Show before/after comparison for first feature
                        if features_to_scale:
                            example_feature = features_to_scale[0]
                            
                            # Get original values
                            original_values = df[example_feature].values
                            
                            # Get transformed values
                            transformed_values = transformed_df[example_feature].values
                            
                            # Create comparison dataframe
                            comparison_df = pd.DataFrame({
                                'Original': original_values,
                                'Transformed': transformed_values
                            }).head(10)
                            
                            st.write(f"**Sample transformation for {example_feature}:**")
                            st.dataframe(comparison_df)
                            
                            # Show statistics
                            st.write("**Statistics before and after:**")
                            stats_df = pd.DataFrame({
                                'Statistic': ['Mean', 'Std', 'Min', '25%', 'Median', '75%', 'Max'],
                                'Before': [
                                    df[example_feature].mean(),
                                    df[example_feature].std(),
                                    df[example_feature].min(),
                                    df[example_feature].quantile(0.25),
                                    df[example_feature].median(),
                                    df[example_feature].quantile(0.75),
                                    df[example_feature].max()
                                ],
                                'After': [
                                    transformed_df[example_feature].mean(),
                                    transformed_df[example_feature].std(),
                                    transformed_df[example_feature].min(),
                                    transformed_df[example_feature].quantile(0.25),
                                    transformed_df[example_feature].median(),
                                    transformed_df[example_feature].quantile(0.75),
                                    transformed_df[example_feature].max()
                                ]
                            })
                            
                            st.dataframe(stats_df)
                            
                    except Exception as e:
                        st.error(f"Error in feature scaling: {str(e)}")
            else:
                st.warning("No numeric features available for scaling")
        
        with transform_tabs[1]:
            st.write("##### Missing Value Imputation")
            
            # Get columns with missing values
            missing_cols = transformed_df.columns[transformed_df.isnull().any()].tolist()
            
            if missing_cols:
                # Select columns for imputation
                cols_to_impute = st.multiselect(
                    "Select columns with missing values to impute:",
                    options=missing_cols,
                    default=missing_cols
                )
                
                if cols_to_impute:
                    # Split by data type
                    num_cols = transformed_df[cols_to_impute].select_dtypes(include=['int64', 'float64']).columns.tolist()
                    cat_cols = transformed_df[cols_to_impute].select_dtypes(include=['object', 'bool', 'category']).columns.tolist()
                    
                    # Numeric imputation method
                    if num_cols:
                        st.write("**Numeric Columns Imputation:**")
                        num_impute_method = st.selectbox(
                            "Select imputation method for numeric columns:",
                            ["Mean", "Median", "Zero", "Constant", "KNN"]
                        )
                        
                        # Get constant value if selected
                        num_const_value = None
                        if num_impute_method == "Constant":
                            num_const_value = st.number_input("Constant value for numeric imputation:", value=0)
                    
                    # Categorical imputation method
                    if cat_cols:
                        st.write("**Categorical Columns Imputation:**")
                        cat_impute_method = st.selectbox(
                            "Select imputation method for categorical columns:",
                            ["Most Frequent", "Constant", "Missing Category"]
                        )
                        
                        # Get constant value if selected
                        cat_const_value = None
                        if cat_impute_method == "Constant":
                            cat_const_value = st.text_input("Constant value for categorical imputation:", value="Missing")
                    
                    if st.button("Apply Imputation"):
                        try:
                            # Process numeric columns
                            if num_cols:
                                # Create imputer
                                if num_impute_method == "Mean":
                                    num_imputer = SimpleImputer(strategy='mean')
                                    method_name = "Mean"
                                elif num_impute_method == "Median":
                                    num_imputer = SimpleImputer(strategy='median')
                                    method_name = "Median"
                                elif num_impute_method == "Zero":
                                    num_imputer = SimpleImputer(strategy='constant', fill_value=0)
                                    method_name = "Zero"
                                elif num_impute_method == "Constant":
                                    num_imputer = SimpleImputer(strategy='constant', fill_value=num_const_value)
                                    method_name = f"Constant ({num_const_value})"
                                else:  # KNN
                                    num_imputer = KNNImputer(n_neighbors=5)
                                    method_name = "KNN (k=5)"
                                
                                # Apply imputation
                                transformed_df[num_cols] = num_imputer.fit_transform(transformed_df[num_cols])
                                
                                # Log transformation
                                transformation_log.append(f"Imputed {len(num_cols)} numeric columns with {method_name}")
                            
                            # Process categorical columns
                            if cat_cols:
                                # Create imputer
                                if cat_impute_method == "Most Frequent":
                                    cat_imputer = SimpleImputer(strategy='most_frequent')
                                    method_name = "Most Frequent"
                                elif cat_impute_method == "Constant":
                                    cat_imputer = SimpleImputer(strategy='constant', fill_value=cat_const_value)
                                    method_name = f"Constant ({cat_const_value})"
                                else:  # Missing Category
                                    # Manual imputation with 'Missing' category
                                    for col in cat_cols:
                                        transformed_df[col].fillna("Missing", inplace=True)
                                    method_name = "Missing Category"
                                
                                # Apply imputation (except for Missing Category which is done above)
                                if cat_impute_method != "Missing Category":
                                    for col in cat_cols:
                                        col_data = transformed_df[col].values.reshape(-1, 1)
                                        transformed_df[col] = cat_imputer.fit_transform(col_data).flatten()
                                
                                # Log transformation
                                transformation_log.append(f"Imputed {len(cat_cols)} categorical columns with {method_name}")
                            
                            st.success(f"✅ Successfully imputed {len(cols_to_impute)} columns")
                            
                            # Show missing values count after imputation
                            missing_after = transformed_df[cols_to_impute].isnull().sum()
                            
                            if missing_after.sum() > 0:
                                st.warning(f"There are still {missing_after.sum()} missing values after imputation")
                            else:
                                st.success("All selected missing values have been imputed")
                                
                        except Exception as e:
                            st.error(f"Error in imputation: {str(e)}")
                else:
                    st.info("Please select columns to impute")
            else:
                st.success("✅ No missing values in the dataset")
        
        with transform_tabs[2]:
            st.write("##### Categorical Encoding")
            
            # Get categorical features
            cat_features = transformed_df.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()
            
            if cat_features:
                # Select features to encode
                features_to_encode = st.multiselect(
                    "Select categorical features to encode:",
                    options=cat_features,
                    default=cat_features
                )
                
                # Select encoding method
                encoding_method = st.selectbox(
                    "Select encoding method:",
                    ["One-hot encoding", "Label encoding", "Ordinal encoding (specify order)"]
                )
                
                # For ordinal encoding
                ordinal_orders = {}
                if encoding_method == "Ordinal encoding (specify order)":
                    st.write("Specify the order for each selected feature:")
                    
                    for feature in features_to_encode:
                        # Get unique values
                        unique_vals = transformed_df[feature].dropna().unique()
                        
                        # Create multiselect for ordering
                        st.write(f"**Order for {feature}:**")
                        order = st.multiselect(
                            f"Order categories from lowest to highest (leave empty for alphabetical):",
                            options=unique_vals,
                            key=f"ordinal_{feature}"
                        )
                        
                        if order:
                            ordinal_orders[feature] = order
                
                if st.button("Apply Encoding") and features_to_encode:
                    try:
                        for feature in features_to_encode:
                            if encoding_method == "One-hot encoding":
                                # Apply one-hot encoding
                                one_hot = pd.get_dummies(transformed_df[feature], prefix=feature)
                                
                                # Drop original column and join encoded columns
                                transformed_df = transformed_df.drop(columns=[feature])
                                transformed_df = pd.concat([transformed_df, one_hot], axis=1)
                                
                                # Log transformation
                                transformation_log.append(f"Applied one-hot encoding to {feature}")
                            
                            elif encoding_method == "Label encoding":
                                # Apply label encoding
                                unique_values = transformed_df[feature].dropna().unique()
                                value_map = {val: idx for idx, val in enumerate(unique_values)}
                                
                                # Handle NaN values
                                transformed_df[f"{feature}_encoded"] = transformed_df[feature].map(value_map)
                                
                                # Log transformation
                                transformation_log.append(f"Applied label encoding to {feature}")
                                
                                # Show mapping
                                st.write(f"**Encoding map for {feature}:**")
                                mapping_df = pd.DataFrame({
                                    'Original': value_map.keys(),
                                    'Encoded': value_map.values()
                                })
                                st.dataframe(mapping_df)
                            
                            else:  # Ordinal encoding
                                if feature in ordinal_orders and ordinal_orders[feature]:
                                    # Use specified order
                                    order = ordinal_orders[feature]
                                    value_map = {val: idx for idx, val in enumerate(order)}
                                    
                                    # For values not in the order, assign NaN
                                    transformed_df[f"{feature}_encoded"] = transformed_df[feature].map(value_map)
                                else:
                                    # Use alphabetical order
                                    unique_values = sorted(transformed_df[feature].dropna().unique())
                                    value_map = {val: idx for idx, val in enumerate(unique_values)}
                                    
                                    # Handle NaN values
                                    transformed_df[f"{feature}_encoded"] = transformed_df[feature].map(value_map)
                                
                                # Log transformation
                                transformation_log.append(f"Applied ordinal encoding to {feature}")
                                
                                # Show mapping
                                st.write(f"**Encoding map for {feature}:**")
                                mapping_df = pd.DataFrame({
                                    'Original': value_map.keys(),
                                    'Encoded': value_map.values()
                                })
                                st.dataframe(mapping_df)
                        
                        st.success(f"✅ Successfully encoded {len(features_to_encode)} features")
                        
                    except Exception as e:
                        st.error(f"Error in encoding: {str(e)}")
            else:
                st.info("No categorical features in the dataset")
        
        return transformed_df, transformation_log
    
    @staticmethod
    def _dimensionality_reduction(df, target_col):
        """Apply dimensionality reduction techniques"""
        st.markdown("#### 🔍 Dimensionality Reduction")
        transformation_log = []
        
        # Make a copy of the input dataframe
        reduced_df = df.copy()
        
        # Get numeric features
        num_features = reduced_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Exclude target column from numeric features if specified
        if target_col and target_col in num_features:
            num_features.remove(target_col)
        
        if len(num_features) > 1:
            # Select dimensionality reduction method
            reduction_method = st.selectbox(
                "Select dimensionality reduction method:",
                ["Principal Component Analysis (PCA)"]
            )
            
            if reduction_method == "Principal Component Analysis (PCA)":
                # Select features for PCA
                features_for_pca = st.multiselect(
                    "Select numeric features for PCA:",
                    options=num_features,
                    default=num_features
                )
                
                if features_for_pca:
                    # Number of components
                    max_components = min(len(features_for_pca), len(reduced_df))
                    n_components = st.slider(
                        "Number of principal components:",
                        min_value=1,
                        max_value=max_components,
                        value=min(2, max_components)
                    )
                    
                    # Standardize data option
                    standardize = st.checkbox("Standardize features before PCA", True)
                    
                    if st.button("Apply PCA"):
                        try:
                            # Prepare data
                            X = reduced_df[features_for_pca].copy()
                            
                            # Handle missing values if any
                            if X.isnull().any().any():
                                imputer = SimpleImputer(strategy='mean')
                                X = pd.DataFrame(
                                    imputer.fit_transform(X),
                                    columns=X.columns
                                )
                            
                            # Standardize if selected
                            if standardize:
                                scaler = StandardScaler()
                                X = pd.DataFrame(
                                    scaler.fit_transform(X),
                                    columns=X.columns
                                )
                            
                            # Apply PCA
                            pca = PCA(n_components=n_components)
                            pca_result = pca.fit_transform(X)
                            
                            # Create dataframe with PCA results
                            pca_df = pd.DataFrame(
                                data=pca_result,
                                columns=[f'PC{i+1}' for i in range(n_components)]
                            )
                            
                            # Add PCA results to original dataframe
                            for col in pca_df.columns:
                                reduced_df[col] = pca_df[col].values
                            
                            # Log transformation
                            transformation_log.append(f"Applied PCA to {len(features_for_pca)} features, reducing to {n_components} components")
                            
                            # Display explained variance
                            explained_var = pca.explained_variance_ratio_ * 100
                            total_var = sum(explained_var)
                            
                            st.success(f"✅ PCA completed. Total variance explained: {total_var:.2f}%")
                            
                            # Plot explained variance
                            explained_var_df = pd.DataFrame({
                                'Principal Component': [f'PC{i+1}' for i in range(n_components)],
                                'Explained Variance (%)': explained_var
                            })
                            
                            fig = px.bar(
                                explained_var_df,
                                x='Principal Component',
                                y='Explained Variance (%)',
                                title='Explained Variance by Principal Component',
                                color='Explained Variance (%)',
                                text_auto='.2f'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show cumulative explained variance
                            cum_explained_var = np.cumsum(explained_var)
                            cum_explained_df = pd.DataFrame({
                                'Principal Component': [f'PC{i+1}' for i in range(n_components)],
                                'Cumulative Explained Variance (%)': cum_explained_var
                            })
                            
                            fig = px.line(
                                cum_explained_df,
                                x='Principal Component',
                                y='Cumulative Explained Variance (%)',
                                title='Cumulative Explained Variance',
                                markers=True
                            )
                            fig.update_layout(yaxis_range=[0, 100])
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show feature contributions to principal components
                            if len(features_for_pca) <= 20:  # Only show for reasonably sized matrices
                                loadings = pca.components_
                                loadings_df = pd.DataFrame(
                                    loadings.T,
                                    columns=[f'PC{i+1}' for i in range(n_components)],
                                    index=features_for_pca
                                )
                                
                                st.write("**Feature contributions to principal components:**")
                                st.dataframe(loadings_df)
                                
                                # Heatmap of feature contributions
                                fig = px.imshow(
                                    loadings_df,
                                    title='Feature Contributions to Principal Components',
                                    color_continuous_scale='RdBu_r',
                                    aspect='auto'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Visualize data in 2D if we have at least 2 components
                            if n_components >= 2:
                                st.write("**Data projection onto first two principal components:**")
                                
                                # Create scatter plot
                                if target_col:
                                    fig = px.scatter(
                                        reduced_df,
                                        x='PC1',
                                        y='PC2',
                                        color=target_col,
                                        title='2D Projection of Data using PCA',
                                        opacity=0.7
                                    )
                                else:
                                    fig = px.scatter(
                                        reduced_df,
                                        x='PC1',
                                        y='PC2',
                                        title='2D Projection of Data using PCA',
                                        opacity=0.7
                                    )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error in PCA: {str(e)}")
        else:
            st.warning("Not enough numeric features for dimensionality reduction (need at least 2)")
        
        return reduced_df, transformation_log
    
    @staticmethod
    def _feature_generation(df):
        """Generate new features from existing ones"""
        st.markdown("#### 🧩 Feature Generation")
        transformation_log = []
        
        # Make a copy of the input dataframe
        generated_df = df.copy()
        
        # Create tabs for different feature generation methods
        gen_tabs = st.tabs([
            "🧮 Mathematical", "🔄 Interactions", "📊 Binning"
        ])
        
        with gen_tabs[0]:
            st.write("##### Mathematical Transformations")
            
            # Get numeric features
            num_features = generated_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if num_features:
                # Select features for transformation
                features_for_math = st.multiselect(
                    "Select features for mathematical transformation:",
                    options=num_features,
                    key="math_features"
                )
                
                if features_for_math:
                    # Select transformation types
                    transformations = st.multiselect(
                        "Select mathematical transformations to apply:",
                        options=["Log", "Square", "Square Root", "Cube", "Cube Root", "Reciprocal"],
                        default=["Log", "Square Root"]
                    )
                    
                    if st.button("Generate Mathematical Features"):
                        for feature in features_for_math:
                            for transform in transformations:
                                try:
                                    if transform == "Log":
                                        # Check for non-positive values
                                        if (generated_df[feature] <= 0).any():
                                            st.warning(f"Skipping Log of {feature} - contains non-positive values")
                                            continue
                                            
                                        generated_df[f"{feature}_log"] = np.log(generated_df[feature])
                                        transformation_log.append(f"Created Log of {feature}")
                                        
                                    elif transform == "Square":
                                        generated_df[f"{feature}_squared"] = generated_df[feature] ** 2
                                        transformation_log.append(f"Created Square of {feature}")
                                        
                                    elif transform == "Square Root":
                                        # Check for negative values
                                        if (generated_df[feature] < 0).any():
                                            st.warning(f"Skipping Square Root of {feature} - contains negative values")
                                            continue
                                            
                                        generated_df[f"{feature}_sqrt"] = np.sqrt(generated_df[feature])
                                        transformation_log.append(f"Created Square Root of {feature}")
                                        
                                    elif transform == "Cube":
                                        generated_df[f"{feature}_cubed"] = generated_df[feature] ** 3
                                        transformation_log.append(f"Created Cube of {feature}")
                                        
                                    elif transform == "Cube Root":
                                        generated_df[f"{feature}_cbrt"] = np.cbrt(generated_df[feature])
                                        transformation_log.append(f"Created Cube Root of {feature}")
                                        
                                    elif transform == "Reciprocal":
                                        # Check for zeros
                                        if (generated_df[feature] == 0).any():
                                            st.warning(f"Skipping Reciprocal of {feature} - contains zeros")
                                            continue
                                            
                                        generated_df[f"{feature}_recip"] = 1 / generated_df[feature]
                                        transformation_log.append(f"Created Reciprocal of {feature}")
                                        
                                except Exception as e:
                                    st.error(f"Error creating {transform} of {feature}: {str(e)}")
                        
                        st.success(f"✅ Generated {len(transformations) * len(features_for_math)} new features")
            else:
                st.warning("No numeric features available for transformation")
        
        with gen_tabs[1]:
            st.write("##### Feature Interactions")
            
            # Get numeric features
            num_features = generated_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if len(num_features) >= 2:
                # Select features for interactions
                st.write("**Select features for interactions:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    features1 = st.multiselect(
                        "First set of features:",
                        options=num_features,
                        key="interact_features1"
                    )
                
                with col2:
                    features2 = st.multiselect(
                        "Second set of features:",
                        options=num_features,
                        key="interact_features2"
                    )
                
                # Select interaction types
                interaction_types = st.multiselect(
                    "Select interaction types to generate:",
                    options=["Multiplication", "Division", "Addition", "Subtraction"],
                    default=["Multiplication"]
                )
                
                if st.button("Generate Interaction Features") and features1 and features2:
                    # Create feature interactions
                    for f1 in features1:
                        for f2 in features2:
                            # Skip if same feature
                            if f1 == f2:
                                continue
                                
                            for interaction in interaction_types:
                                try:
                                    if interaction == "Multiplication":
                                        generated_df[f"{f1}_x_{f2}"] = generated_df[f1] * generated_df[f2]
                                        transformation_log.append(f"Created interaction: {f1} × {f2}")
                                        
                                    elif interaction == "Division":
                                        # Check for zeros in denominator
                                        if (generated_df[f2] == 0).any():
                                            st.warning(f"Skipping {f1} ÷ {f2} - denominator contains zeros")
                                            continue
                                            
                                        generated_df[f"{f1}_div_{f2}"] = generated_df[f1] / generated_df[f2]
                                        transformation_log.append(f"Created interaction: {f1} ÷ {f2}")
                                        
                                    elif interaction == "Addition":
                                        generated_df[f"{f1}_plus_{f2}"] = generated_df[f1] + generated_df[f2]
                                        transformation_log.append(f"Created interaction: {f1} + {f2}")
                                        
                                    elif interaction == "Subtraction":
                                        generated_df[f"{f1}_minus_{f2}"] = generated_df[f1] - generated_df[f2]
                                        transformation_log.append(f"Created interaction: {f1} - {f2}")
                                        
                                except Exception as e:
                                    st.error(f"Error creating {f1} {interaction} {f2}: {str(e)}")
                    
                    st.success(f"✅ Generated interaction features")
            else:
                st.warning("Need at least 2 numeric features for interactions")
        
        with gen_tabs[2]:
            st.write("##### Feature Binning")
            
            # Get numeric features
            num_features = generated_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if num_features:
                # Select features for binning
                features_for_binning = st.multiselect(
                    "Select features for binning:",
                    options=num_features,
                    key="binning_features"
                )
                
                if features_for_binning:
                    # Binning options
                    bin_method = st.radio(
                        "Binning method:",
                        ["Equal width", "Equal frequency", "Custom bins"]
                    )
                    
                    # Number of bins
                    if bin_method in ["Equal width", "Equal frequency"]:
                        num_bins = st.slider(
                            "Number of bins:",
                            min_value=2,
                            max_value=20,
                            value=5
                        )
                    
                    # Set bin labels
                    use_labels = st.checkbox("Use custom bin labels", True)
                    
                    if st.button("Generate Binned Features"):
                        for feature in features_for_binning:
                            try:
                                # Handle missing values
                                feature_data = generated_df[feature].copy()
                                
                                if bin_method == "Equal width":
                                    # Equal width binning
                                    bin_edges = np.linspace(
                                        feature_data.min(),
                                        feature_data.max(),
                                        num_bins + 1
                                    )
                                    
                                    # Create labels if requested
                                    if use_labels:
                                        bin_labels = [f"Bin {i+1}" for i in range(num_bins)]
                                    else:
                                        bin_labels = False
                                    
                                    # Apply binning
                                    binned = pd.cut(
                                        feature_data,
                                        bins=bin_edges,
                                        labels=bin_labels,
                                        include_lowest=True
                                    )
                                    
                                elif bin_method == "Equal frequency":
                                    # Equal frequency binning
                                    bin_edges = pd.qcut(
                                        feature_data,
                                        q=num_bins,
                                        retbins=True,
                                        duplicates='drop'
                                    )[1]
                                    
                                    # Create labels if requested
                                    if use_labels:
                                        bin_labels = [f"Bin {i+1}" for i in range(len(bin_edges)-1)]
                                    else:
                                        bin_labels = False
                                    
                                    # Apply binning
                                    binned = pd.cut(
                                        feature_data,
                                        bins=bin_edges,
                                        labels=bin_labels,
                                        include_lowest=True
                                    )
                                    
                                else:  # Custom bins
                                    # Show histogram to help with bin selection
                                    fig = px.histogram(
                                        generated_df,
                                        x=feature,
                                        nbins=20,
                                        title=f"Distribution of {feature}"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Get custom bin edges
                                    bin_edges_str = st.text_input(
                                        f"Enter custom bin edges for {feature} (comma-separated values):",
                                        value=f"{feature_data.min()},{feature_data.max()}",
                                        key=f"bin_edges_{feature}"
                                    )
                                    
                                    # Parse bin edges
                                    try:
                                        bin_edges = [float(x.strip()) for x in bin_edges_str.split(",")]
                                    except:
                                        st.error("Invalid bin edges. Please enter comma-separated numbers.")
                                        continue
                                    
                                    # Create labels if requested
                                    if use_labels:
                                        bin_labels = [f"Bin {i+1}" for i in range(len(bin_edges)-1)]
                                    else:
                                        bin_labels = False
                                    
                                    # Apply binning
                                    binned = pd.cut(
                                        feature_data,
                                        bins=bin_edges,
                                        labels=bin_labels,
                                        include_lowest=True
                                    )
                                
                                # Add binned feature to dataframe
                                generated_df[f"{feature}_binned"] = binned
                                
                                # Log transformation
                                transformation_log.append(f"Binned {feature} using {bin_method} method")
                                
                                # Show bin distribution
                                st.write(f"**Bin distribution for {feature}:**")
                                bin_counts = binned.value_counts().sort_index()
                                
                                fig = px.bar(
                                    x=bin_counts.index,
                                    y=bin_counts.values,
                                    labels={'x': 'Bin', 'y': 'Count'},
                                    title=f"Bin Distribution for {feature}"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                            except Exception as e:
                                st.error(f"Error binning {feature}: {str(e)}")
                        
                        st.success(f"✅ Created binned features")
            else:
                st.warning("No numeric features available for binning")
        
        return generated_df, transformation_log
    
    @staticmethod
    def _compare_before_after(original_df, transformed_df, transformation_log):
        """Compare data before and after transformations"""
        st.markdown("#### 📈 Before/After Comparison")
        
        # Show transformations applied
        if transformation_log:
            st.write("**Applied Transformations:**")
            for i, transform in enumerate(transformation_log):
                st.write(f"{i+1}. {transform}")
        else:
            st.info("No transformations applied")
        
        # Compare column counts
        orig_cols = set(original_df.columns)
        new_cols = set(transformed_df.columns)
        
        added_cols = new_cols - orig_cols
        removed_cols = orig_cols - new_cols
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Features", len(original_df.columns))
        with col2:
            st.metric("Transformed Features", len(transformed_df.columns))
        with col3:
            st.metric("Feature Difference", len(transformed_df.columns) - len(original_df.columns))
        
        # Show added/removed columns
        if added_cols:
            st.write("**Added Columns:**")
            st.write(", ".join(sorted(added_cols)))
        
        if removed_cols:
            st.write("**Removed Columns:**")
            st.write(", ".join(sorted(removed_cols)))
        
        # Compare summary statistics for common numeric columns
        common_num_cols = list(
            set(original_df.select_dtypes(include=['int64', 'float64']).columns) &
            set(transformed_df.select_dtypes(include=['int64', 'float64']).columns)
        )
        
        if common_num_cols:
            st.write("### 📊 Compare Statistics for Common Numeric Features")
            
            # Select a feature to compare
            feature_to_compare = st.selectbox(
                "Select a feature to compare:",
                options=common_num_cols
            )
            
            # Get statistics for selected feature
            orig_stats = original_df[feature_to_compare].describe()
            trans_stats = transformed_df[feature_to_compare].describe()
            
            # Display side-by-side histograms
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Original", "Transformed")
            )
            
            fig.add_trace(
                go.Histogram(x=original_df[feature_to_compare], name="Original"),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Histogram(x=transformed_df[feature_to_compare], name="Transformed"),
                row=1, col=2
            )
            
            fig.update_layout(
                title=f"Distribution Comparison for {feature_to_compare}",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display statistics comparison
            stats_df = pd.DataFrame({
                'Statistic': orig_stats.index,
                'Original': orig_stats.values,
                'Transformed': trans_stats.values
            })
            
            st.dataframe(stats_df)
        
        # Compare missing values
        st.write("### 🔍 Missing Values Comparison")
        
        orig_missing = original_df.isnull().sum().sum()
        trans_missing = transformed_df.isnull().sum().sum()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Missing Values", orig_missing)
        with col2:
            st.metric("Transformed Missing Values", trans_missing)
        
        # Show details of missing values by column
        missing_comparison = pd.DataFrame({
            'Original': original_df.isnull().sum(),
            'Transformed': pd.Series(0, index=original_df.columns)  # Default to 0 for comparison
        })
        
        # Update transformed column counts for existing columns
        for col in transformed_df.columns:
            if col in missing_comparison.index:
                missing_comparison.loc[col, 'Transformed'] = transformed_df[col].isnull().sum()
        
        # Filter to show only rows with missing values
        missing_comparison = missing_comparison[(missing_comparison['Original'] > 0) | (missing_comparison['Transformed'] > 0)]
        
        if not missing_comparison.empty:
            missing_comparison['Difference'] = missing_comparison['Transformed'] - missing_comparison['Original']
            
            st.dataframe(missing_comparison)
        else:
            st.success("No missing values in either dataset")
        
        # Data shape comparison
        st.write("### 📏 Dataset Dimensions")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Shape", f"{original_df.shape[0]} rows × {original_df.shape[1]} columns")
        with col2:
            st.metric("Transformed Shape", f"{transformed_df.shape[0]} rows × {transformed_df.shape[1]} columns")
        
        # Memory usage comparison
        orig_mem = original_df.memory_usage(deep=True).sum() / (1024 * 1024)
        trans_mem = transformed_df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Memory Usage", f"{orig_mem:.2f} MB")
        with col2:
            st.metric("Transformed Memory Usage", f"{trans_mem:.2f} MB")
