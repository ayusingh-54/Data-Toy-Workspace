import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataDriftMonitor:
    """Advanced data drift monitoring and analysis"""
    
    @staticmethod
    def detect_data_drift(reference_df, current_df, timestamp_col=None):
        """Detect and visualize data drift between two datasets"""
        st.markdown("### 🔄 Data Drift Analysis")
        
        if reference_df is None or current_df is None:
            st.error("Both reference and current datasets are required for drift analysis")
            return
        
        try:
            # Prepare datasets
            if timestamp_col and timestamp_col in reference_df.columns and timestamp_col in current_df.columns:
                # Sort by timestamp if available
                reference_df = reference_df.sort_values(timestamp_col)
                current_df = current_df.sort_values(timestamp_col)
                
                st.info(f"Datasets sorted by timestamp column: {timestamp_col}")
                
                # Show time ranges
                ref_time_range = (reference_df[timestamp_col].min(), reference_df[timestamp_col].max())
                cur_time_range = (current_df[timestamp_col].min(), current_df[timestamp_col].max())
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Reference Time Range", f"{ref_time_range[0]} to {ref_time_range[1]}")
                with col2:
                    st.metric("Current Time Range", f"{cur_time_range[0]} to {cur_time_range[1]}")
            
            # Get common columns
            common_cols = list(set(reference_df.columns) & set(current_df.columns))
            st.write(f"Analyzing drift in {len(common_cols)} common columns")
            
            # Select specific columns for analysis or use all common columns
            cols_to_analyze = st.multiselect(
                "Select columns to analyze for drift",
                common_cols,
                default=common_cols[:min(10, len(common_cols))]
            )
            
            if not cols_to_analyze:
                st.warning("Please select at least one column for drift analysis")
                return
            
            # Analyze drift for each selected column
            drift_results = []
            
            for col in cols_to_analyze:
                ref_data = reference_df[col].dropna()
                cur_data = current_df[col].dropna()
                
                # Skip empty columns
                if len(ref_data) == 0 or len(cur_data) == 0:
                    continue
                
                # Check column type
                if pd.api.types.is_numeric_dtype(ref_data) and pd.api.types.is_numeric_dtype(cur_data):
                    # Numeric column analysis
                    
                    # Basic statistics comparison
                    ref_mean, ref_std = ref_data.mean(), ref_data.std()
                    cur_mean, cur_std = cur_data.mean(), cur_data.std()
                    
                    # Statistical test for drift (KS test for distribution comparison)
                    try:
                        ks_stat, p_value = stats.ks_2samp(ref_data, cur_data)
                    except Exception:
                        ks_stat, p_value = 0, 1.0
                    
                    drift_detected = p_value < 0.05
                    
                    drift_results.append({
                        'column': col,
                        'type': 'numeric',
                        'ref_mean': ref_mean,
                        'cur_mean': cur_mean,
                        'mean_diff_pct': ((cur_mean - ref_mean) / (abs(ref_mean) if ref_mean != 0 else 1)) * 100,
                        'ref_std': ref_std,
                        'cur_std': cur_std,
                        'std_diff_pct': ((cur_std - ref_std) / (ref_std if ref_std != 0 else 1)) * 100,
                        'ks_stat': ks_stat,
                        'p_value': p_value,
                        'drift_detected': drift_detected,
                        'drift_severity': 'High' if p_value < 0.01 else 'Medium' if p_value < 0.05 else 'Low'
                    })
                    
                else:
                    # Categorical column analysis
                    ref_value_counts = ref_data.value_counts(normalize=True)
                    cur_value_counts = cur_data.value_counts(normalize=True)
                    
                    # Calculate chi-square statistic for categorical variables
                    try:
                        # Get all possible values
                        all_values = list(set(ref_value_counts.index) | set(cur_value_counts.index))
                        
                        # Calculate expected and observed frequencies
                        ref_freqs = [ref_value_counts.get(val, 0) for val in all_values]
                        cur_freqs = [cur_value_counts.get(val, 0) for val in all_values]
                        
                        # Chi-square test
                        chi2_stat, p_value = stats.chisquare(cur_freqs, f_exp=ref_freqs)
                    except Exception:
                        chi2_stat, p_value = 0, 1.0
                    
                    # Calculate Jensen-Shannon divergence
                    js_div = DataDriftMonitor._jensen_shannon_divergence(ref_value_counts, cur_value_counts)
                    
                    drift_detected = p_value < 0.05
                    
                    drift_results.append({
                        'column': col,
                        'type': 'categorical',
                        'unique_vals_ref': ref_data.nunique(),
                        'unique_vals_cur': cur_data.nunique(),
                        'chi2_stat': chi2_stat,
                        'p_value': p_value,
                        'js_divergence': js_div,
                        'drift_detected': drift_detected,
                        'drift_severity': 'High' if p_value < 0.01 else 'Medium' if p_value < 0.05 else 'Low'
                    })
            
            # Create drift summary
            drift_df = pd.DataFrame(drift_results)
            
            if drift_df.empty:
                st.warning("No valid columns for drift analysis")
                return
            
            # Summary statistics
            drifted_cols = drift_df[drift_df['drift_detected']]['column'].tolist()
            num_drifted = len(drifted_cols)
            drift_pct = (num_drifted / len(drift_df)) * 100
            
            # Display summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Columns Analyzed", len(drift_df))
            with col2:
                st.metric("Columns with Drift", num_drifted)
            with col3:
                st.metric("Drift Percentage", f"{drift_pct:.1f}%")
            
            # Display drift table
            st.subheader("Drift Analysis Results")
            if 'type' in drift_df.columns and 'drift_severity' in drift_df.columns:
                # Color-code the drift severity
                drift_display = drift_df.copy()
                
                # Format p-value for better readability
                if 'p_value' in drift_display.columns:
                    drift_display['p_value'] = drift_display['p_value'].apply(lambda x: f"{x:.4f}")
                
                # Create display dataframe with relevant columns
                if 'numeric' in drift_display['type'].values:
                    numeric_display = drift_display[drift_display['type'] == 'numeric'].copy()
                    if not numeric_display.empty:
                        st.write("**Numeric Columns:**")
                        numeric_display['mean_diff'] = numeric_display.apply(
                            lambda row: f"{row['mean_diff_pct']:.2f}% ({'↑' if row['mean_diff_pct'] > 0 else '↓'})",
                            axis=1
                        )
                        st.dataframe(numeric_display[['column', 'drift_severity', 'mean_diff', 'p_value']])
                
                if 'categorical' in drift_display['type'].values:
                    cat_display = drift_display[drift_display['type'] == 'categorical'].copy()
                    if not cat_display.empty:
                        st.write("**Categorical Columns:**")
                        st.dataframe(cat_display[['column', 'drift_severity', 'js_divergence', 'p_value']])
            
            # Show alert if drift is detected
            if num_drifted > 0:
                st.warning(f"⚠️ Drift detected in {num_drifted} columns: {', '.join(drifted_cols)}")
            else:
                st.success("✅ No significant drift detected in the analyzed columns")
            
            # Detailed visualizations for drifted columns
            if num_drifted > 0:
                st.subheader("Detailed Drift Visualizations")
                
                for col in drifted_cols:
                    st.write(f"**Column: {col}**")
                    
                    # Get column data
                    ref_data = reference_df[col].dropna()
                    cur_data = current_df[col].dropna()
                    
                    if pd.api.types.is_numeric_dtype(ref_data) and pd.api.types.is_numeric_dtype(cur_data):
                        # Numeric column visualization
                        
                        # Distribution comparison
                        fig = go.Figure()
                        
                        # Add histograms
                        fig.add_trace(go.Histogram(
                            x=ref_data,
                            opacity=0.7,
                            name='Reference',
                            nbinsx=30
                        ))
                        
                        fig.add_trace(go.Histogram(
                            x=cur_data,
                            opacity=0.7,
                            name='Current',
                            nbinsx=30
                        ))
                        
                        fig.update_layout(
                            title=f"Distribution Comparison: {col}",
                            xaxis_title=col,
                            yaxis_title="Frequency",
                            barmode='overlay'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show statistics
                        drift_info = drift_df[drift_df['column'] == col].iloc[0]
                        
                        metric_cols = st.columns(3)
                        with metric_cols[0]:
                            st.metric(
                                "Mean", 
                                f"{drift_info['cur_mean']:.2f}",
                                f"{drift_info['mean_diff_pct']:.2f}%"
                            )
                        with metric_cols[1]:
                            st.metric(
                                "Std Dev", 
                                f"{drift_info['cur_std']:.2f}", 
                                f"{drift_info['std_diff_pct']:.2f}%"
                            )
                        with metric_cols[2]:
                            st.metric("KS Stat", f"{drift_info['ks_stat']:.4f}")
                        
                        # Box plot comparison
                        fig = go.Figure()
                        
                        fig.add_trace(go.Box(
                            y=ref_data,
                            name='Reference',
                            boxmean=True
                        ))
                        
                        fig.add_trace(go.Box(
                            y=cur_data,
                            name='Current',
                            boxmean=True
                        ))
                        
                        fig.update_layout(
                            title=f"Box Plot Comparison: {col}",
                            yaxis_title=col
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        # Categorical column visualization
                        
                        # Get value counts
                        ref_counts = ref_data.value_counts().sort_index()
                        cur_counts = cur_data.value_counts().sort_index()
                        
                        # Normalize to get proportions
                        ref_props = ref_counts / ref_counts.sum()
                        cur_props = cur_counts / cur_counts.sum()
                        
                        # Get all unique values
                        all_values = sorted(list(set(ref_props.index) | set(cur_props.index)))
                        
                        # Create comparison dataframe
                        comp_data = []
                        for val in all_values:
                            comp_data.append({
                                'Value': str(val),
                                'Reference': ref_props.get(val, 0),
                                'Current': cur_props.get(val, 0),
                                'Difference': cur_props.get(val, 0) - ref_props.get(val, 0)
                            })
                        
                        comp_df = pd.DataFrame(comp_data)
                        
                        # Bar chart comparison
                        fig = go.Figure()
                        
                        fig.add_trace(go.Bar(
                            x=comp_df['Value'],
                            y=comp_df['Reference'],
                            name='Reference'
                        ))
                        
                        fig.add_trace(go.Bar(
                            x=comp_df['Value'],
                            y=comp_df['Current'],
                            name='Current'
                        ))
                        
                        fig.update_layout(
                            title=f"Distribution Comparison: {col}",
                            xaxis_title="Value",
                            yaxis_title="Proportion",
                            barmode='group'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show difference heatmap for categories
                        diff_fig = px.bar(
                            comp_df, 
                            x='Value', 
                            y='Difference',
                            title=f"Distribution Difference: {col}",
                            color='Difference',
                            color_continuous_scale='RdBu_r',
                            labels={'Difference': 'Current - Reference'}
                        )
                        
                        st.plotly_chart(diff_fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error in drift analysis: {str(e)}")
            st.info("Try selecting different columns or check for data issues")
    
    @staticmethod
    def _jensen_shannon_divergence(p, q):
        """Calculate Jensen-Shannon divergence between two distributions"""
        # Get all keys from both distributions
        all_keys = set(p.keys()) | set(q.keys())
        
        # Create arrays with values for each key
        p_values = np.array([p.get(k, 0) for k in all_keys])
        q_values = np.array([q.get(k, 0) for k in all_keys])
        
        # Normalize if not already normalized
        p_sum = p_values.sum()
        q_sum = q_values.sum()
        
        if p_sum > 0:
            p_values = p_values / p_sum
        
        if q_sum > 0:
            q_values = q_values / q_sum
        
        # Calculate midpoint distribution
        m = (p_values + q_values) / 2
        
        # Calculate KL divergence for both distributions
        js_div = 0
        
        for i in range(len(p_values)):
            if p_values[i] > 0 and m[i] > 0:
                js_div += 0.5 * p_values[i] * np.log(p_values[i] / m[i])
            
            if q_values[i] > 0 and m[i] > 0:
                js_div += 0.5 * q_values[i] * np.log(q_values[i] / m[i])
        
        return js_div
    
    @staticmethod
    def show_data_quality_dashboard(df):
        """Show comprehensive data quality dashboard"""
        st.markdown("### 📊 Data Quality Dashboard")
        
        if df is None:
            st.error("No dataset provided for quality analysis")
            return
        
        try:
            # Basic dataset info
            row_count = df.shape[0]
            col_count = df.shape[1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Rows", f"{row_count:,}")
            with col2:
                st.metric("Columns", col_count)
            with col3:
                memory_usage = df.memory_usage(deep=True).sum() / (1024**2)
                st.metric("Memory Usage", f"{memory_usage:.2f} MB")
            with col4:
                duplicate_rows = df.duplicated().sum()
                dup_percentage = (duplicate_rows / row_count * 100) if row_count > 0 else 0
                st.metric("Duplicate Rows", f"{duplicate_rows:,} ({dup_percentage:.1f}%)")
            
            # Missing values analysis
            st.subheader("Missing Values Analysis")
            
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Count': df.isnull().sum().values,
                'Missing %': (df.isnull().sum() / len(df) * 100).values,
                'Data Type': df.dtypes.values
            }).sort_values('Missing %', ascending=False)
            
            # Calculate overall missing percentage
            total_missing = df.isnull().sum().sum()
            total_elements = df.size
            overall_missing_pct = (total_missing / total_elements) * 100 if total_elements > 0 else 0
            
            st.metric("Overall Missing Values", f"{total_missing:,} ({overall_missing_pct:.2f}%)")
            
            # Missing values chart
            if missing_df['Missing Count'].sum() > 0:
                fig = px.bar(
                    missing_df.head(15), 
                    x='Column', 
                    y='Missing %',
                    text='Missing Count',
                    color='Missing %',
                    title='Missing Values by Column (Top 15)',
                    color_continuous_scale='Reds'
                )
                
                fig.update_traces(texttemplate='%{text:,}', textposition='outside')
                fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show full missing values table
                with st.expander("Show complete missing values table"):
                    st.dataframe(missing_df)
            else:
                st.success("✅ No missing values detected in the dataset")
            
            # Data type distribution
            st.subheader("Data Type Analysis")
            
            # Count data types
            dtype_counts = df.dtypes.value_counts().reset_index()
            dtype_counts.columns = ['Data Type', 'Count']
            dtype_counts['Data Type'] = dtype_counts['Data Type'].astype(str)
            
            # Create data type distribution chart
            fig = px.pie(
                dtype_counts, 
                values='Count', 
                names='Data Type',
                title='Column Data Types Distribution',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Show data type breakdown
                st.write("**Data Type Breakdown:**")
                for idx, row in dtype_counts.iterrows():
                    st.write(f"• {row['Data Type']}: {row['Count']} columns")
                
                # Count special column types
                numeric_cols = df.select_dtypes(include=['number']).columns
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                date_cols = df.select_dtypes(include=['datetime']).columns
                
                st.write("**Special Column Types:**")
                st.write(f"• Numeric: {len(numeric_cols)}")
                st.write(f"• Categorical: {len(categorical_cols)}")
                st.write(f"• Date/Time: {len(date_cols)}")
            
            # Statistical overview
            st.subheader("Statistical Overview")
            
            # Create tabs for different column types
            num_tab, cat_tab = st.tabs(["Numeric Columns", "Categorical Columns"])
            
            with num_tab:
                if len(numeric_cols) > 0:
                    # Get descriptive statistics
                    desc_stats = df[numeric_cols].describe().T
                    desc_stats['range'] = desc_stats['max'] - desc_stats['min']
                    desc_stats['cv'] = (desc_stats['std'] / desc_stats['mean']).abs()
                    
                    # Add skewness and kurtosis
                    desc_stats['skewness'] = df[numeric_cols].skew()
                    desc_stats['kurtosis'] = df[numeric_cols].kurtosis()
                    
                    # Display stats
                    st.dataframe(desc_stats.round(2))
                    
                    # Distribution analysis
                    st.write("**Distribution Analysis:**")
                    
                    selected_num_col = st.selectbox("Select column for distribution analysis:", numeric_cols)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Histogram
                        fig = px.histogram(
                            df, 
                            x=selected_num_col,
                            title=f"Distribution of {selected_num_col}",
                            marginal='box'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Show outlier analysis
                        q1 = df[selected_num_col].quantile(0.25)
                        q3 = df[selected_num_col].quantile(0.75)
                        iqr = q3 - q1
                        
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        
                        outliers = df[(df[selected_num_col] < lower_bound) | (df[selected_num_col] > upper_bound)][selected_num_col]
                        outlier_pct = len(outliers) / len(df) * 100
                        
                        st.write(f"**Outlier Analysis for {selected_num_col}:**")
                        st.write(f"• IQR: {iqr:.2f}")
                        st.write(f"• Lower bound: {lower_bound:.2f}")
                        st.write(f"• Upper bound: {upper_bound:.2f}")
                        st.write(f"• Outliers: {len(outliers)} ({outlier_pct:.2f}%)")
                        
                        # Box plot for outliers
                        fig = px.box(df, y=selected_num_col, title=f"Boxplot for {selected_num_col}")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No numeric columns found in the dataset")
            
            with cat_tab:
                if len(categorical_cols) > 0:
                    # Get categorical column to analyze
                    selected_cat_col = st.selectbox("Select column for category analysis:", categorical_cols)
                    
                    # Calculate value counts
                    val_counts = df[selected_cat_col].value_counts()
                    val_percent = df[selected_cat_col].value_counts(normalize=True) * 100
                    
                    # Combine counts and percentages
                    cat_stats = pd.DataFrame({
                        'Count': val_counts,
                        'Percentage': val_percent
                    }).reset_index()
                    
                    cat_stats.columns = ['Value', 'Count', 'Percentage']
                    
                    # Show categorical stats
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Basic statistics
                        st.metric("Unique Values", df[selected_cat_col].nunique())
                        
                        # Top categories chart
                        top_n = min(10, len(cat_stats))
                        
                        fig = px.bar(
                            cat_stats.head(top_n), 
                            x='Value', 
                            y='Percentage',
                            text='Count',
                            title=f'Top {top_n} Categories in {selected_cat_col}'
                        )
                        
                        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Show stats table
                        st.write(f"**Value Distribution for {selected_cat_col}:**")
                        st.dataframe(cat_stats.head(15).style.format({'Percentage': '{:.2f}%'}))
                        
                        if len(cat_stats) > 15:
                            st.info(f"Showing 15 out of {len(cat_stats)} categories")
                        
                        # High cardinality warning
                        unique_ratio = df[selected_cat_col].nunique() / len(df)
                        if unique_ratio > 0.5:
                            st.warning(f"⚠️ High cardinality detected: {unique_ratio:.2%} of values are unique")
                else:
                    st.info("No categorical columns found in the dataset")
            
            # Overall data quality score
            st.subheader("Data Quality Score")
            
            # Calculate quality metrics
            completeness_score = 1 - (total_missing / total_elements) if total_elements > 0 else 0
            uniqueness_score = 1 - (duplicate_rows / row_count) if row_count > 0 else 0
            
            # Calculate validity score based on outliers in numeric columns
            validity_score = 0
            if len(numeric_cols) > 0:
                validity_scores = []
                for col in numeric_cols:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                    outlier_pct = len(outliers) / len(df) if len(df) > 0 else 0
                    
                    col_validity = 1 - outlier_pct
                    validity_scores.append(col_validity)
                
                validity_score = sum(validity_scores) / len(validity_scores) if validity_scores else 0
            
            # Overall quality score (weighted average)
            overall_quality = (completeness_score * 0.5) + (uniqueness_score * 0.3) + (validity_score * 0.2)
            overall_quality_pct = overall_quality * 100
            
            # Display quality scores
            quality_cols = st.columns(4)
            
            with quality_cols[0]:
                st.metric("Overall Quality", f"{overall_quality_pct:.1f}%")
            with quality_cols[1]:
                st.metric("Completeness", f"{completeness_score*100:.1f}%")
            with quality_cols[2]:
                st.metric("Uniqueness", f"{uniqueness_score*100:.1f}%")
            with quality_cols[3]:
                st.metric("Validity", f"{validity_score*100:.1f}%")
            
            # Quality gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = overall_quality_pct,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Data Quality Score"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "red"},
                        {'range': [50, 70], 'color': "orange"},
                        {'range': [70, 85], 'color': "yellow"},
                        {'range': [85, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 85
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations based on quality analysis
            st.subheader("Recommendations")
            
            recommendations = []
            
            # Missing data recommendations
            if overall_missing_pct > 5:
                recommendations.append("📊 **Missing Values**: Consider imputing or removing columns with high missing percentages")
                
                # Columns with high missing values
                high_missing_cols = missing_df[missing_df['Missing %'] > 30]['Column'].tolist()
                if high_missing_cols:
                    recommendations.append(f"⚠️ Consider removing these columns with >30% missing values: {', '.join(high_missing_cols[:5])}" + 
                                          (f" and {len(high_missing_cols)-5} more" if len(high_missing_cols) > 5 else ""))
            
            # Duplicate recommendations
            if duplicate_rows > 0:
                recommendations.append(f"🔄 **Duplicate Rows**: {duplicate_rows:,} duplicates detected ({dup_percentage:.1f}%). Consider removing them for better model quality")
            
            # Outlier recommendations
            if validity_score < 0.9 and len(numeric_cols) > 0:
                recommendations.append("🔍 **Outliers**: Several numeric columns contain outliers that may affect model performance")
            
            # Categorical data recommendations
            if len(categorical_cols) > 0:
                high_cardinality_cols = []
                for col in categorical_cols:
                    if df[col].nunique() / len(df) > 0.5:
                        high_cardinality_cols.append(col)
                
                if high_cardinality_cols:
                    recommendations.append(f"📋 **High Cardinality**: These columns have too many unique values: {', '.join(high_cardinality_cols[:3])}" + 
                                          (f" and {len(high_cardinality_cols)-3} more" if len(high_cardinality_cols) > 3 else ""))
            
            # Display recommendations
            if recommendations:
                for rec in recommendations:
                    st.info(rec)
            else:
                st.success("✅ Data quality looks good! No major issues detected.")
            
            # Export report button
            st.subheader("Export Quality Report")
            if st.button("📥 Export Data Quality Report"):
                # Create a more comprehensive report as HTML
                report_html = DataDriftMonitor._generate_quality_report_html(df, missing_df, overall_quality_pct)
                
                # Create download link
                b64 = base64.b64encode(report_html.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="data_quality_report.html">Download Data Quality Report</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error in data quality analysis: {str(e)}")
    
    @staticmethod
    def _generate_quality_report_html(df, missing_df, quality_score):
        """Generate a detailed HTML quality report"""
        # Basic dataset info
        row_count = df.shape[0]
        col_count = df.shape[1]
        memory_usage = df.memory_usage(deep=True).sum() / (1024**2)
        duplicate_rows = df.duplicated().sum()
        dup_percentage = (duplicate_rows / row_count * 100) if row_count > 0 else 0
        
        # Missing values
        total_missing = df.isnull().sum().sum()
        total_elements = df.size
        overall_missing_pct = (total_missing / total_elements) * 100 if total_elements > 0 else 0
        
        # Data types
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        date_cols = df.select_dtypes(include=['datetime']).columns
        
        # Create HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Quality Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; color: #333; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .metric-container {{ display: flex; flex-wrap: wrap; margin: 20px 0; }}
                .metric {{ 
                    background: #f8f9fa; 
                    border-radius: 8px; 
                    padding: 15px; 
                    margin: 10px; 
                    flex: 1; 
                    min-width: 200px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric h3 {{ margin-top: 0; color: #7b8a8b; }}
                .metric p {{ font-size: 24px; font-weight: bold; margin: 5px 0; }}
                .metric small {{ color: #95a5a6; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .warning {{ background-color: #fcf8e3; padding: 10px; border-left: 5px solid #faebcc; margin: 10px 0; }}
                .success {{ background-color: #dff0d8; padding: 10px; border-left: 5px solid #d6e9c6; margin: 10px 0; }}
                .quality-meter {{ 
                    height: 30px;
                    background: #ecf0f1;
                    border-radius: 15px;
                    margin: 15px 0;
                    overflow: hidden;
                }}
                .quality-fill {{ 
                    height: 100%;
                    background: linear-gradient(90deg, #e74c3c 0%, #f39c12 40%, #2ecc71 80%);
                    width: {quality_score}%;
                    position: relative;
                }}
                .quality-label {{ 
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    color: white;
                    font-weight: bold;
                    text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Data Quality Report</h1>
                <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
                
                <h2>Dataset Overview</h2>
                <div class="metric-container">
                    <div class="metric">
                        <h3>Rows</h3>
                        <p>{row_count:,}</p>
                    </div>
                    <div class="metric">
                        <h3>Columns</h3>
                        <p>{col_count}</p>
                    </div>
                    <div class="metric">
                        <h3>Memory Usage</h3>
                        <p>{memory_usage:.2f} MB</p>
                    </div>
                    <div class="metric">
                        <h3>Duplicate Rows</h3>
                        <p>{duplicate_rows:,}</p>
                        <small>{dup_percentage:.1f}% of total rows</small>
                    </div>
                </div>
                
                <h2>Data Quality Score</h2>
                <div class="quality-meter">
                    <div class="quality-fill">
                        <div class="quality-label">{quality_score:.1f}%</div>
                    </div>
                </div>
                
                <div class="metric-container">
                    <div class="metric">
                        <h3>Completeness</h3>
                        <p>{100 - overall_missing_pct:.1f}%</p>
                        <small>Missing values: {total_missing:,} ({overall_missing_pct:.2f}%)</small>
                    </div>
                    <div class="metric">
                        <h3>Uniqueness</h3>
                        <p>{100 - dup_percentage:.1f}%</p>
                        <small>Unique rows: {row_count - duplicate_rows:,}</small>
                    </div>
                </div>
                
                <h2>Column Types</h2>
                <div class="metric-container">
                    <div class="metric">
                        <h3>Numeric</h3>
                        <p>{len(numeric_cols)}</p>
                    </div>
                    <div class="metric">
                        <h3>Categorical</h3>
                        <p>{len(categorical_cols)}</p>
                    </div>
                    <div class="metric">
                        <h3>Date/Time</h3>
                        <p>{len(date_cols)}</p>
                    </div>
                </div>
                
                <h2>Missing Values by Column</h2>
        """
        
        # Add missing values table
        if missing_df['Missing Count'].sum() > 0:
            # Sort by missing percentage
            missing_df_sorted = missing_df.sort_values('Missing %', ascending=False)
            
            # Add table with missing data
            html += "<table>"
            html += "<tr><th>Column</th><th>Data Type</th><th>Missing Count</th><th>Missing %</th></tr>"
            
            for idx, row in missing_df_sorted.iterrows():
                if row['Missing Count'] > 0:
                    html += f"""
                    <tr>
                        <td>{row['Column']}</td>
                        <td>{row['Data Type']}</td>
                        <td>{row['Missing Count']:,}</td>
                        <td>{row['Missing %']:.2f}%</td>
                    </tr>
                    """
            
            html += "</table>"
            
            # Add recommendations for highly missing columns
            high_missing = missing_df_sorted[missing_df_sorted['Missing %'] > 30]
            if not high_missing.empty:
                html += "<div class='warning'>"
                html += f"<p><strong>Warning:</strong> {len(high_missing)} columns have more than 30% missing values:</p>"
                html += "<ul>"
                for idx, row in high_missing.iterrows():
                    html += f"<li>{row['Column']} ({row['Missing %']:.1f}% missing)</li>"
                html += "</ul>"
                html += "</div>"
        else:
            html += "<div class='success'><p>✅ No missing values detected in the dataset</p></div>"
        
        # Close HTML document
        html += """
                <h2>Recommendations</h2>
                <ul>
        """
        
        # Add recommendations based on analysis
        if overall_missing_pct > 5:
            html += "<li>Consider imputing or removing columns with high missing percentages</li>"
        
        if duplicate_rows > 0:
            html += f"<li>Remove {duplicate_rows:,} duplicate rows to improve data quality</li>"
        
        # Add column specific recommendations
        high_cardinality_cols = []
        for col in categorical_cols:
            if df[col].nunique() / len(df) > 0.5:
                high_cardinality_cols.append(col)
        
        if high_cardinality_cols:
            html += f"<li>Consider encoding or feature engineering for high cardinality columns: {', '.join(high_cardinality_cols[:5])}"
            if len(high_cardinality_cols) > 5:
                html += f" and {len(high_cardinality_cols) - 5} more"
            html += "</li>"
        
        # Close the HTML
        html += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html
