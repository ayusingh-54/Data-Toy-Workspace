import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import base64
import shap
import matplotlib.pyplot as plt
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Try importing optional libraries
try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    import eli5
    from eli5.sklearn import PermutationImportance
    ELI5_AVAILABLE = True
except ImportError:
    ELI5_AVAILABLE = False

class ModelExplainer:
    """Advanced model interpretation and explanation tools"""
    
    @staticmethod
    def explain_model(model, X_train, X_test, feature_names, model_type="classification"):
        """Provide comprehensive model explanations"""
        st.markdown("### 🔍 Model Interpretation & Explainability")
        
        explanation_method = st.selectbox(
            "Select explanation method:",
            ["Feature Importance", "SHAP Values", "Partial Dependence Plots", 
             "LIME Explanations", "Permutation Importance"]
        )
        
        if explanation_method == "Feature Importance":
            ModelExplainer.show_feature_importance(model, feature_names)
        
        elif explanation_method == "SHAP Values":
            ModelExplainer.show_shap_analysis(model, X_test, feature_names)
        
        elif explanation_method == "Partial Dependence Plots":
            ModelExplainer.show_partial_dependence(model, X_train, feature_names)
        
        elif explanation_method == "LIME Explanations":
            ModelExplainer.show_lime_explanations(model, X_train, X_test, feature_names, model_type)
            
        elif explanation_method == "Permutation Importance":
            ModelExplainer.show_permutation_importance(model, X_test, X_test.iloc[:, -1], feature_names)
    
    @staticmethod
    def show_feature_importance(model, feature_names):
        """Display feature importance for tree-based models"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # Create feature importance dataframe
                features_df = pd.DataFrame({
                    'Feature': [feature_names[i] for i in indices],
                    'Importance': importances[indices]
                }).head(15)  # Show top 15 features
                
                # Plot feature importance
                fig = px.bar(
                    features_df, 
                    x='Importance', 
                    y='Feature',
                    title='Feature Importance',
                    orientation='h'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Show feature importance table
                st.dataframe(features_df)
                
                # Download feature importance as CSV
                csv = features_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="feature_importance.csv">Download Feature Importance CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
                
            else:
                st.warning("⚠️ This model doesn't support direct feature importance. Try SHAP or Permutation Importance instead.")
        
        except Exception as e:
            st.error(f"❌ Error calculating feature importance: {str(e)}")
            st.info("💡 Try a different explanation method for this model type")
    
    @staticmethod
    def show_shap_analysis(model, X_test, feature_names):
        """Display SHAP value analysis"""
        st.write("Generating SHAP values (this may take a moment)...")
        
        try:
            # Sample data for SHAP analysis (for performance)
            sample_size = min(100, X_test.shape[0])
            X_sample = X_test.sample(sample_size, random_state=42)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create explainer
            status_text.text("Creating SHAP explainer...")
            progress_bar.progress(0.2)
            
            try:
                # Try faster TreeExplainer for tree-based models
                explainer = shap.TreeExplainer(model)
                status_text.text("Tree-based SHAP explainer created")
            except:
                # Fall back to KernelExplainer for other model types
                status_text.text("Creating KernelExplainer (this may take longer)...")
                if hasattr(model, 'predict_proba'):
                    explainer = shap.KernelExplainer(model.predict_proba, X_sample)
                else:
                    explainer = shap.KernelExplainer(model.predict, X_sample)
                status_text.text("KernelExplainer created")
            
            progress_bar.progress(0.4)
            
            # Calculate SHAP values
            status_text.text("Computing SHAP values...")
            shap_values = explainer.shap_values(X_sample)
            progress_bar.progress(0.8)
            
            # Check if we have multiple outputs (for classification)
            if isinstance(shap_values, list):
                st.info("📊 Classification model detected, showing SHAP values for class 0")
                shap_values = shap_values[0]
            
            # Reset matplotlib figure
            plt.clf()
            
            # Create SHAP summary plot
            status_text.text("Generating SHAP summary plot...")
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
            st.pyplot(fig)
            
            progress_bar.progress(0.9)
            
            # Create SHAP bar plot
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
            st.pyplot(fig)
            
            progress_bar.progress(1.0)
            status_text.text("✅ SHAP analysis complete!")
            
            # Show SHAP waterfall plot for a sample instance
            with st.expander("🔍 SHAP Waterfall Plot (Sample Instance)"):
                instance_idx = st.slider("Select instance index", 0, sample_size-1, 0)
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.waterfall_plot(shap.Explanation(values=shap_values[instance_idx], 
                                                    base_values=explainer.expected_value,
                                                    data=X_sample.iloc[instance_idx],
                                                    feature_names=feature_names), show=False)
                st.pyplot(fig)
            
            # Show SHAP dependence plots
            with st.expander("🔄 SHAP Dependence Plots"):
                # Select top features by importance
                shap_importance = np.mean(np.abs(shap_values), axis=0)
                top_indices = np.argsort(shap_importance)[-5:]  # Top 5 features
                top_features = [feature_names[i] for i in top_indices]
                
                selected_feature = st.selectbox("Select feature for dependence plot", top_features)
                feature_idx = list(feature_names).index(selected_feature)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.dependence_plot(feature_idx, shap_values, X_sample, feature_names=feature_names, show=False)
                st.pyplot(fig)
        
        except Exception as e:
            st.error(f"❌ Error in SHAP analysis: {str(e)}")
            st.info("💡 SHAP analysis requires compatible model types and may not work with all algorithms")
    
    @staticmethod
    def show_partial_dependence(model, X_train, feature_names):
        """Show partial dependence plots"""
        try:
            # Import here to avoid loading sklearn when not needed
            from sklearn.inspection import partial_dependence, PartialDependenceDisplay
            
            st.write("Generating partial dependence plots...")
            
            # Select features for partial dependence
            num_features = min(5, len(feature_names))
            
            # Choose top features if we have feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                top_indices = np.argsort(importances)[-num_features:]
                top_features = [feature_names[i] for i in top_indices]
            else:
                # Otherwise let user select
                top_features = st.multiselect(
                    "Select features for partial dependence (max 3 recommended):",
                    feature_names,
                    default=feature_names[:min(3, len(feature_names))]
                )
                
                if not top_features:
                    st.warning("Please select at least one feature")
                    return
            
            # Calculate partial dependence for each selected feature
            for feature in top_features:
                feature_idx = list(feature_names).index(feature)
                
                st.write(f"**Partial Dependence for {feature}**")
                
                # Compute partial dependence
                try:
                    pdp = partial_dependence(
                        model, X_train, [feature_idx], 
                        kind="average", grid_resolution=20
                    )
                    
                    # Create plot using Plotly
                    feature_values = pdp["values"][0]
                    pdp_values = pdp["average"][0]
                    
                    fig = px.line(
                        x=feature_values, y=pdp_values,
                        title=f"Partial Dependence Plot: {feature}",
                        labels={"x": feature, "y": "Partial Dependence"}
                    )
                    
                    # Add scatter points
                    fig.add_trace(go.Scatter(
                        x=feature_values, y=pdp_values,
                        mode='markers',
                        name='PDP Values'
                    ))
                    
                    # Add reference line at y=0
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Could not compute partial dependence for {feature}: {str(e)}")
            
            # 2D Partial Dependence
            if len(top_features) >= 2:
                with st.expander("🔄 2D Partial Dependence Plot"):
                    # Allow user to select features for 2D plot
                    col1, col2 = st.columns(2)
                    with col1:
                        feature1 = st.selectbox("First feature", top_features, index=0)
                    with col2:
                        remaining_features = [f for f in top_features if f != feature1]
                        feature2 = st.selectbox("Second feature", remaining_features, index=0)
                    
                    if feature1 != feature2:
                        feature_idx1 = list(feature_names).index(feature1)
                        feature_idx2 = list(feature_names).index(feature2)
                        
                        try:
                            # Compute 2D partial dependence
                            pdp_2d = partial_dependence(
                                model, X_train, [(feature_idx1, feature_idx2)], 
                                kind="average", grid_resolution=15
                            )
                            
                            # Get values for plot
                            x_values = pdp_2d["values"][0]
                            y_values = pdp_2d["values"][1]
                            pdp_values = pdp_2d["average"][0]
                            
                            # Create a meshgrid for the plot
                            X, Y = np.meshgrid(x_values, y_values)
                            
                            # Create contour plot
                            fig = go.Figure(data=[
                                go.Contour(
                                    z=pdp_values.T,
                                    x=x_values,
                                    y=y_values,
                                    colorscale='Viridis',
                                    contours=dict(showlabels=True),
                                    colorbar=dict(title="Partial Dependence")
                                )
                            ])
                            
                            fig.update_layout(
                                title=f"2D Partial Dependence: {feature1} vs {feature2}",
                                xaxis_title=feature1,
                                yaxis_title=feature2
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.warning(f"Could not compute 2D partial dependence: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Error generating partial dependence: {str(e)}")
    
    @staticmethod
    def show_lime_explanations(model, X_train, X_test, feature_names, model_type):
        """Show LIME explanations for individual predictions"""
        if not LIME_AVAILABLE:
            st.error("❌ LIME package not installed. Please install with: pip install lime")
            st.info("💡 LIME provides local interpretability for individual predictions")
            return
        
        try:
            st.write("Generating LIME explanations...")
            
            # Initialize LIME explainer
            mode = "classification" if model_type == "Classification" else "regression"
            
            # Create the LIME explainer
            explainer = lime_tabular.LimeTabularExplainer(
                X_train.values,
                feature_names=feature_names,
                class_names=['Negative', 'Positive'] if mode == "classification" else None,
                mode=mode
            )
            
            # Let user select instance to explain
            num_instances = min(10, X_test.shape[0])
            instance_idx = st.slider("Select instance to explain", 0, num_instances-1, 0)
            
            # Generate explanation for the selected instance
            st.write(f"Explaining prediction for instance {instance_idx}:")
            
            # Display the actual feature values for the instance
            instance = X_test.iloc[instance_idx]
            st.dataframe(pd.DataFrame([instance.values], columns=feature_names))
            
            # Generate the explanation
            if mode == "classification":
                exp = explainer.explain_instance(
                    instance.values, 
                    model.predict_proba, 
                    num_features=10
                )
                
                # Get prediction probability
                pred_prob = model.predict_proba([instance.values])[0]
                pred_class = model.predict([instance.values])[0]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Class", f"Class {pred_class}")
                with col2:
                    st.metric("Probability", f"{pred_prob[int(pred_class)]:.3f}")
                
            else:  # Regression
                exp = explainer.explain_instance(
                    instance.values, 
                    model.predict, 
                    num_features=10
                )
                
                # Get prediction
                pred_value = model.predict([instance.values])[0]
                st.metric("Predicted Value", f"{pred_value:.3f}")
            
            # Convert LIME explanation to HTML
            lime_html = exp.as_html()
            
            # Display explanation
            st.components.v1.html(lime_html, height=400)
            
            # Extract and plot the feature weights
            feature_weights = {name: weight for name, weight in exp.as_list()}
            
            # Sort by absolute weight
            sorted_weights = sorted(feature_weights.items(), key=lambda x: abs(x[1]), reverse=True)
            features = [item[0] for item in sorted_weights]
            weights = [item[1] for item in sorted_weights]
            
            # Create a color map (green for positive, red for negative weights)
            colors = ['#EF553B' if w < 0 else '#636EFA' for w in weights]
            
            # Plot the weights
            fig = go.Figure([
                go.Bar(
                    x=weights,
                    y=features,
                    orientation='h',
                    marker_color=colors
                )
            ])
            
            fig.update_layout(
                title="Feature Contributions (LIME)",
                xaxis_title="Weight / Contribution",
                yaxis_title="Feature"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error generating LIME explanations: {str(e)}")
    
    @staticmethod
    def show_permutation_importance(model, X, y, feature_names):
        """Show permutation importance for features"""
        if not ELI5_AVAILABLE:
            st.error("❌ ELI5 package not installed. Please install with: pip install eli5")
            return
        
        try:
            st.write("Calculating permutation importance...")
            
            # Create and fit permutation importance
            perm = PermutationImportance(model, random_state=42).fit(X, y)
            
            # Get importance values
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': perm.feature_importances_,
                'StdDev': perm.feature_importances_std_
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Plot importances
            fig = go.Figure([
                go.Bar(
                    x=importance_df['Feature'],
                    y=importance_df['Importance'],
                    error_y=dict(
                        type='data',
                        array=importance_df['StdDev'],
                        visible=True
                    ),
                    name='Importance'
                )
            ])
            
            fig.update_layout(
                title="Permutation Feature Importance",
                xaxis_title="Feature",
                yaxis_title="Importance (mean decrease in model score)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show importance table
            st.dataframe(importance_df)
            
            # Feature importance interpretation
            st.markdown("**Interpretation:**")
            st.markdown("""
            - **Positive values** indicate features that help the model's performance
            - **Larger values** indicate more important features
            - **Zero or negative values** suggest the feature may be noise or redundant
            """)
            
        except Exception as e:
            st.error(f"❌ Error calculating permutation importance: {str(e)}")
