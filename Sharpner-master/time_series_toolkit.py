import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import base64
import uuid
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import time series libraries (with fallbacks)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("Prophet not installed. Install with: pip install prophet")

try:
    import statsmodels.api as sm
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    st.warning("Statsmodels not installed. Install with: pip install statsmodels")

try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False

class TimeSeriesToolkit:
    """Comprehensive time series analysis toolkit"""
    
    @staticmethod
    def detect_seasonality(df, date_col, value_col):
        """Automatic seasonality detection"""
        st.markdown("### 🔍 Seasonality Detection")
        
        try:
            # Ensure datetime column
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
            df.set_index(date_col, inplace=True)
            
            # Resample to handle irregular frequencies
            freq = st.selectbox("Select Frequency", ["D", "W", "M", "Q", "Y"], index=0)
            ts_data = df[value_col].resample(freq).mean().dropna()
            
            if len(ts_data) < 24:  # Need sufficient data for seasonality
                st.warning("Insufficient data for seasonality detection (need at least 24 periods)")
                return None
            
            # Seasonal decomposition
            if STATSMODELS_AVAILABLE:
                col1, col2 = st.columns(2)
                
                with col1:
                    decomposition = seasonal_decompose(ts_data, model='additive', period=min(12, len(ts_data)//2))
                    
                    fig = make_subplots(
                        rows=4, cols=1,
                        subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'],
                        vertical_spacing=0.08
                    )
                    
                    fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data.values, name='Original'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=ts_data.index, y=decomposition.trend, name='Trend'), row=2, col=1)
                    fig.add_trace(go.Scatter(x=ts_data.index, y=decomposition.seasonal, name='Seasonal'), row=3, col=1)
                    fig.add_trace(go.Scatter(x=ts_data.index, y=decomposition.resid, name='Residual'), row=4, col=1)
                    
                    fig.update_layout(height=800, title_text="Time Series Decomposition")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Seasonality statistics
                    seasonal_strength = np.var(decomposition.seasonal) / np.var(decomposition.seasonal + decomposition.resid)
                    trend_strength = np.var(decomposition.trend) / np.var(decomposition.trend + decomposition.resid)
                    
                    st.metric("Seasonal Strength", f"{seasonal_strength:.3f}")
                    st.metric("Trend Strength", f"{trend_strength:.3f}")
                    
                    # Seasonality interpretation
                    if seasonal_strength > 0.3:
                        st.success("🌊 Strong seasonality detected!")
                    elif seasonal_strength > 0.1:
                        st.info("📊 Moderate seasonality detected")
                    else:
                        st.warning("📈 Weak or no seasonality detected")
                    
                    # Additional insights
                    st.markdown("**Insights:**")
                    seasonal_period = len(decomposition.seasonal[decomposition.seasonal == decomposition.seasonal.max()])
                    st.write(f"• Detected seasonal period: {seasonal_period}")
                    st.write(f"• Trend direction: {'Increasing' if decomposition.trend.iloc[-1] > decomposition.trend.iloc[0] else 'Decreasing'}")
                
                return {'seasonal_strength': seasonal_strength, 'trend_strength': trend_strength, 'decomposition': decomposition}
            else:
                st.error("Statsmodels required for seasonality detection")
                return None
                
        except Exception as e:
            st.error(f"Error in seasonality detection: {str(e)}")
            return None
    
    @staticmethod
    def forecast_with_prophet(df, date_col, value_col):
        """Forecasting with Prophet"""
        st.markdown("### 🔮 Prophet Forecasting")
        
        if not PROPHET_AVAILABLE:
            st.error("Prophet not available. Please install: pip install prophet")
            return
        
        try:
            # Prepare data for Prophet
            prophet_df = df[[date_col, value_col]].copy()
            prophet_df.columns = ['ds', 'y']
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
            prophet_df = prophet_df.dropna().sort_values('ds')
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.markdown("**Forecast Settings**")
                periods = st.slider("Forecast Periods", 1, 365, 30)
                freq = st.selectbox("Frequency", ["D", "W", "M"], index=0)
                
                # Advanced settings
                with st.expander("Advanced Settings"):
                    yearly_seasonality = st.checkbox("Yearly Seasonality", True)
                    weekly_seasonality = st.checkbox("Weekly Seasonality", True)
                    daily_seasonality = st.checkbox("Daily Seasonality", False)
                    
                    changepoint_prior_scale = st.slider("Changepoint Prior Scale", 0.001, 0.5, 0.05)
                    seasonality_prior_scale = st.slider("Seasonality Prior Scale", 0.01, 10.0, 10.0)
            
            with col1:
                if st.button("🚀 Run Prophet Forecast"):
                    with st.spinner("Training Prophet model..."):
                        # Initialize and fit Prophet model
                        model = Prophet(
                            yearly_seasonality=yearly_seasonality,
                            weekly_seasonality=weekly_seasonality,
                            daily_seasonality=daily_seasonality,
                            changepoint_prior_scale=changepoint_prior_scale,
                            seasonality_prior_scale=seasonality_prior_scale
                        )
                        
                        model.fit(prophet_df)
                        
                        # Make future dataframe
                        future = model.make_future_dataframe(periods=periods, freq=freq)
                        forecast = model.predict(future)
                        
                        # Plot results
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=prophet_df['ds'],
                            y=prophet_df['y'],
                            mode='markers',
                            name='Historical Data',
                            marker=dict(color='blue')
                        ))
                        
                        # Forecast
                        fig.add_trace(go.Scatter(
                            x=forecast['ds'],
                            y=forecast['yhat'],
                            mode='lines',
                            name='Forecast',
                            line=dict(color='red')
                        ))
                        
                        # Confidence intervals
                        fig.add_trace(go.Scatter(
                            x=forecast['ds'],
                            y=forecast['yhat_upper'],
                            fill=None,
                            mode='lines',
                            line_color='rgba(0,0,0,0)',
                            showlegend=False
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=forecast['ds'],
                            y=forecast['yhat_lower'],
                            fill='tonexty',
                            mode='lines',
                            line_color='rgba(0,0,0,0)',
                            name='Confidence Interval',
                            fillcolor='rgba(255,0,0,0.2)'
                        ))
                        
                        fig.update_layout(
                            title="Prophet Forecast Results",
                            xaxis_title="Date",
                            yaxis_title=value_col,
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Forecast summary
                        last_historical = prophet_df['y'].iloc[-1]
                        first_forecast = forecast[forecast['ds'] > prophet_df['ds'].max()]['yhat'].iloc[0]
                        change_pct = ((first_forecast - last_historical) / last_historical) * 100
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Last Historical Value", f"{last_historical:.2f}")
                        with col_b:
                            st.metric("Next Forecast", f"{first_forecast:.2f}", f"{change_pct:+.1f}%")
                        with col_c:
                            st.metric("Forecast Periods", periods)
                        
                        # Component plots
                        with st.expander("📊 Forecast Components"):
                            components_fig = TimeSeriesToolkit.plot_prophet_components(model, forecast)
                            st.plotly_chart(components_fig, use_container_width=True)
                        
                        # Store results in session state
                        st.session_state['prophet_forecast'] = {
                            'forecast': forecast,
                            'model': model,
                            'historical': prophet_df
                        }
        
        except Exception as e:
            st.error(f"Error in Prophet forecasting: {str(e)}")
    
    @staticmethod
    def arima_forecast(df, date_col, value_col):
        """ARIMA forecasting with auto parameter selection and robust error handling"""
        st.markdown("### 📈 ARIMA Forecasting")
        
        if not STATSMODELS_AVAILABLE:
            st.error("❌ Statsmodels required for ARIMA forecasting. Please install with: pip install statsmodels")
            st.info("💡 You can run this in your terminal: `pip install statsmodels`")
            return
        
        try:
            # Validate inputs
            if date_col not in df.columns:
                st.error(f"❌ Date column '{date_col}' not found in the dataset.")
                return
            
            if value_col not in df.columns:
                st.error(f"❌ Value column '{value_col}' not found in the dataset.")
                return
                
            # Check if we have enough data
            if len(df) < 10:
                st.error("❌ Not enough data for ARIMA forecasting. At least 10 data points are required.")
                return
            
            # Prepare data - with robust date parsing
            try:
                ts_df = df[[date_col, value_col]].copy()
                ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors='coerce')
                
                # Check for parse failures
                invalid_dates = ts_df[ts_df[date_col].isnull()].shape[0]
                if invalid_dates > 0:
                    st.warning(f"⚠️ {invalid_dates} rows had invalid date formats and were removed.")
                
                # Remove null dates
                ts_df = ts_df.dropna(subset=[date_col])
                if len(ts_df) < 10:
                    st.error("❌ After removing invalid dates, not enough data remains. Please check your date column format.")
                    return
                    
                ts_df = ts_df.sort_values(date_col).dropna()
                ts_df.set_index(date_col, inplace=True)
            except Exception as e:
                st.error(f"❌ Error processing date column: {str(e)}")
                st.info("💡 Make sure the date column contains valid date formats.")
                return
            
            # Resample if needed - with robust frequency handling
            freq_options = {
                "Daily (D)": "D", 
                "Weekly (W)": "W", 
                "Monthly (M)": "M", 
                "Quarterly (Q)": "Q"
            }
            freq_choice = st.selectbox("Data Frequency", list(freq_options.keys()), index=0, key="arima_freq")
            freq = freq_options[freq_choice]
            
            try:
                ts_data = ts_df[value_col].resample(freq).mean().dropna()
                
                # Check if resampling gave us enough data
                if len(ts_data) < 8:
                    st.warning(f"⚠️ After resampling to {freq_choice}, only {len(ts_data)} data points remain. Results may be unreliable.")
                    if len(ts_data) < 3:
                        st.error("❌ Not enough data points after resampling. Try a different frequency.")
                        return
            except Exception as e:
                st.error(f"❌ Error resampling data: {str(e)}")
                return
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.markdown("**ARIMA Settings**")
                forecast_steps = st.slider("Forecast Steps", 1, min(50, len(ts_data)), min(10, len(ts_data)//3))
                
                # Manual or auto parameter selection
                param_method = st.radio("Parameter Selection", ["Auto", "Manual"])
                
                if param_method == "Manual":
                    p = st.slider("AR order (p)", 0, 5, 1)
                    d = st.slider("Differencing (d)", 0, 2, 1)
                    q = st.slider("MA order (q)", 0, 5, 1)
                    
                    # Warning for high order models with little data
                    if p + d + q > len(ts_data) // 3:
                        st.warning(f"⚠️ Model order (p+d+q={p+d+q}) is high relative to data length ({len(ts_data)}). Consider reducing parameters.")
                else:
                    p, d, q = None, None, None
            
            with col1:
                if st.button("🚀 Run ARIMA Forecast"):
                    with st.spinner("Training ARIMA model..."):
                        try:
                            # Try to check stationarity - with error handling
                            try:
                                # Only run on data with sufficient length
                                if len(ts_data) >= 8:
                                    adf_result = adfuller(ts_data.dropna())
                                    is_stationary = adf_result[1] < 0.05
                                    st.info(f"🔍 ADF Test p-value: {adf_result[1]:.4f} - {'Stationary' if is_stationary else 'Non-stationary'}")
                                    
                                    # If auto mode and data is not stationary, suggest differencing
                                    if not is_stationary and param_method == "Auto":
                                        st.info("💡 Data appears non-stationary. Differencing will be applied.")
                                else:
                                    st.info("🔍 Skipping stationarity test due to insufficient data.")
                            except Exception as e:
                                st.warning(f"⚠️ Could not perform stationarity test: {str(e)}")
                            
                            # Determine model parameters
                            if param_method == "Auto":
                                if PMDARIMA_AVAILABLE:
                                    # Auto ARIMA with error handling
                                    try:
                                        auto_model = auto_arima(
                                            ts_data, 
                                            seasonal=False, 
                                            stepwise=True, 
                                            suppress_warnings=True,
                                            max_p=min(5, len(ts_data)//4),
                                            max_d=2,
                                            max_q=min(5, len(ts_data)//4),
                                            error_action='ignore'
                                        )
                                        p, d, q = auto_model.order
                                        st.success(f"✅ Auto-selected ARIMA({p},{d},{q})")
                                    except Exception as auto_error:
                                        st.warning(f"⚠️ Auto ARIMA failed: {str(auto_error)}")
                                        p, d, q = 1, 1, 1
                                        st.info(f"💡 Using fallback ARIMA({p},{d},{q})")
                                else:
                                    # Fallback with sensible defaults based on stationarity
                                    d = 0 if (is_stationary if 'is_stationary' in locals() else False) else 1
                                    p, q = 1, 1
                                    st.info(f"💡 Using default ARIMA({p},{d},{q}) - install pmdarima for auto selection")
                            
                            # Fit ARIMA model with robust error handling
                            try:
                                model = ARIMA(ts_data, order=(p, d, q))
                                fitted_model = model.fit()
                                
                                # Generate forecast
                                forecast = fitted_model.forecast(steps=forecast_steps)
                                forecast_ci = fitted_model.get_forecast(steps=forecast_steps).conf_int()
                            except Exception as model_error:
                                st.error(f"❌ Error fitting ARIMA model: {str(model_error)}")
                                
                                # Try simpler model as fallback
                                if p > 0 or q > 0:
                                    st.info("💡 Trying a simpler model...")
                                    try:
                                        simple_model = ARIMA(ts_data, order=(0, 1, 0))  # Simple random walk
                                        fitted_model = simple_model.fit()
                                        forecast = fitted_model.forecast(steps=forecast_steps)
                                        forecast_ci = fitted_model.get_forecast(steps=forecast_steps).conf_int()
                                        st.success("✅ Fallback model fitted successfully.")
                                        p, d, q = 0, 1, 0
                                    except Exception:
                                        st.error("❌ Could not fit any ARIMA model to this data.")
                                        return
                                else:
                                    st.error("❌ Could not fit the ARIMA model. Try different parameters or check your data.")
                                    return
                            
                            # Create forecast dates with robust error handling
                            try:
                                last_date = ts_data.index[-1]
                                
                                # Handle different frequency types safely
                                if freq == 'D':
                                    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_steps, freq='D')
                                elif freq == 'W':
                                    forecast_dates = pd.date_range(start=last_date + timedelta(days=7), periods=forecast_steps, freq='W')
                                elif freq == 'M':
                                    # Handle month-end correctly
                                    next_month = last_date + pd.DateOffset(months=1)
                                    forecast_dates = pd.date_range(start=next_month, periods=forecast_steps, freq='M')
                                elif freq == 'Q':
                                    next_quarter = last_date + pd.DateOffset(months=3)
                                    forecast_dates = pd.date_range(start=next_quarter, periods=forecast_steps, freq='Q')
                                else:
                                    # Fallback for any other frequency
                                    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_steps, freq=freq)
                                
                            except Exception as date_error:
                                st.warning(f"⚠️ Error creating forecast dates: {str(date_error)}")
                                # Create simple numeric index as fallback
                                st.info("💡 Using numeric forecast periods instead of dates")
                                last_idx = len(ts_data)
                                forecast_dates = range(last_idx + 1, last_idx + forecast_steps + 1)
                            
                            # Plot results with error handling
                            try:
                                fig = go.Figure()
                                
                                # Historical data
                                fig.add_trace(go.Scatter(
                                    x=ts_data.index,
                                    y=ts_data.values,
                                    mode='lines+markers',
                                    name='Historical Data',
                                    line=dict(color='blue')
                                ))
                                
                                # Forecast
                                fig.add_trace(go.Scatter(
                                    x=forecast_dates,
                                    y=forecast,
                                    mode='lines+markers',
                                    name='ARIMA Forecast',
                                    line=dict(color='red', dash='dash')
                                ))
                                
                                # Confidence intervals if available
                                try:
                                    fig.add_trace(go.Scatter(
                                        x=forecast_dates,
                                        y=forecast_ci.iloc[:, 1],
                                        fill=None,
                                        mode='lines',
                                        line_color='rgba(0,0,0,0)',
                                        showlegend=False
                                    ))
                                    
                                    fig.add_trace(go.Scatter(
                                        x=forecast_dates,
                                        y=forecast_ci.iloc[:, 0],
                                        fill='tonexty',
                                        mode='lines',
                                        line_color='rgba(0,0,0,0)',
                                        name='95% Confidence Interval',
                                        fillcolor='rgba(255,0,0,0.2)'
                                    ))
                                except Exception:
                                    st.info("💡 Confidence intervals not available for this model")
                                
                                # Layout
                                fig.update_layout(
                                    title=f"ARIMA({p},{d},{q}) Forecast",
                                    xaxis_title="Date",
                                    yaxis_title=value_col,
                                    height=500,
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    )
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Forecast summary metrics
                                cols = st.columns(3)
                                with cols[0]:
                                    last_val = ts_data.iloc[-1]
                                    st.metric("Last Actual Value", f"{last_val:.2f}")
                                with cols[1]:
                                    first_forecast_val = forecast[0]
                                    change = ((first_forecast_val - last_val) / abs(last_val)) * 100 if last_val != 0 else 0
                                    st.metric("First Forecast", f"{first_forecast_val:.2f}", f"{change:+.2f}%")
                                with cols[2]:
                                    if len(forecast) > 1:
                                        trend = "Up" if forecast[-1] > forecast[0] else "Down" if forecast[-1] < forecast[0] else "Flat"
                                        forecast_trend = ((forecast[-1] - forecast[0]) / abs(forecast[0])) * 100 if forecast[0] != 0 else 0
                                        st.metric("Forecast Trend", trend, f"{forecast_trend:+.2f}%")
                                    else:
                                        st.metric("Forecast Length", f"{len(forecast)} steps")
                            
                            except Exception as plot_error:
                                st.error(f"❌ Error plotting forecast: {str(plot_error)}")
                            
                            # Model diagnostics with error handling
                            with st.expander("📊 Model Diagnostics"):
                                try:
                                    # Model fit metrics
                                    col_a, col_b, col_c = st.columns(3)
                                    with col_a:
                                        try:
                                            st.metric("AIC", f"{fitted_model.aic:.2f}")
                                        except:
                                            st.metric("AIC", "N/A")
                                    with col_b:
                                        try:
                                            st.metric("BIC", f"{fitted_model.bic:.2f}")
                                        except:
                                            st.metric("BIC", "N/A")
                                    with col_c:
                                        try:
                                            st.metric("Log Likelihood", f"{fitted_model.llf:.2f}")
                                        except:
                                            st.metric("Log Likelihood", "N/A")
                                    
                                    # Residuals plot
                                    try:
                                        residuals = fitted_model.resid
                                        residuals_fig = go.Figure()
                                        residuals_fig.add_trace(go.Scatter(
                                            x=ts_data.index,
                                            y=residuals,
                                            mode='lines+markers',
                                            name='Residuals'
                                        ))
                                        residuals_fig.update_layout(title="Model Residuals", height=300)
                                        st.plotly_chart(residuals_fig, use_container_width=True)
                                        
                                        # Add residual statistics
                                        st.markdown("**Residuals Statistics:**")
                                        res_mean = residuals.mean()
                                        res_std = residuals.std()
                                        st.write(f"• Mean: {res_mean:.4f}")
                                        st.write(f"• Std Dev: {res_std:.4f}")
                                        
                                        # Simple test for residual normality
                                        if abs(res_mean) < 0.1 * res_std:
                                            st.success("✅ Residuals appear to have zero mean (good)")
                                        else:
                                            st.warning("⚠️ Residuals may be biased (mean not close to zero)")
                                    except Exception:
                                        st.warning("⚠️ Could not compute residual analysis")
                                except Exception as diag_error:
                                    st.error(f"❌ Error computing model diagnostics: {str(diag_error)}")
                                    
                                # Save forecast to session state for reference
                                st.session_state['arima_forecast'] = {
                                    'forecast_values': forecast,
                                    'forecast_dates': forecast_dates,
                                    'model_order': (p, d, q),
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                }
                                
                                # Show download button for forecast data
                                try:
                                    forecast_df = pd.DataFrame({
                                        'date': forecast_dates,
                                        'forecast': forecast
                                    })
                                    csv = forecast_df.to_csv(index=False)
                                    b64 = base64.b64encode(csv.encode()).decode()
                                    href = f'<a href="data:file/csv;base64,{b64}" download="arima_forecast.csv">Download Forecast CSV</a>'
                                    st.markdown(href, unsafe_allow_html=True)
                                except Exception:
                                    pass
                        
                        except Exception as e:
                            st.error(f"Error fitting ARIMA model: {str(e)}")
        
        except Exception as e:
            st.error(f"Error in ARIMA forecasting: {str(e)}")
    
    @staticmethod
    def detect_change_points(df, date_col, value_col):
        """Change point detection in time series"""
        st.markdown("### 🎯 Change Point Detection")
        
        try:
            # Prepare data
            ts_df = df[[date_col, value_col]].copy()
            ts_df[date_col] = pd.to_datetime(ts_df[date_col])
            ts_df = ts_df.sort_values(date_col).dropna()
            
            # Simple change point detection using rolling statistics
            window_size = st.slider("Detection Window Size", 5, 50, 20)
            threshold = st.slider("Change Threshold (std)", 1.0, 5.0, 2.0)
            
            # Calculate rolling mean and std
            ts_df['rolling_mean'] = ts_df[value_col].rolling(window=window_size, center=True).mean()
            ts_df['rolling_std'] = ts_df[value_col].rolling(window=window_size, center=True).std()
            
            # Detect changes
            ts_df['mean_change'] = abs(ts_df['rolling_mean'].diff())
            ts_df['std_change'] = abs(ts_df['rolling_std'].diff())
            
            mean_threshold = threshold * ts_df['mean_change'].std()
            change_points = ts_df[ts_df['mean_change'] > mean_threshold]
            
            # Plot results
            fig = go.Figure()
            
            # Original data
            fig.add_trace(go.Scatter(
                x=ts_df[date_col],
                y=ts_df[value_col],
                mode='lines',
                name='Original Data',
                line=dict(color='blue')
            ))
            
            # Rolling mean
            fig.add_trace(go.Scatter(
                x=ts_df[date_col],
                y=ts_df['rolling_mean'],
                mode='lines',
                name='Rolling Mean',
                line=dict(color='green', dash='dash')
            ))
            
            # Change points
            if len(change_points) > 0:
                fig.add_trace(go.Scatter(
                    x=change_points[date_col],
                    y=change_points[value_col],
                    mode='markers',
                    name='Change Points',
                    marker=dict(color='red', size=10, symbol='x')
                ))
            
            fig.update_layout(
                title="Change Point Detection",
                xaxis_title="Date",
                yaxis_title=value_col,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary
            st.markdown(f"**Detected {len(change_points)} change points:**")
            if len(change_points) > 0:
                for idx, cp in change_points.iterrows():
                    st.write(f"• {cp[date_col].strftime('%Y-%m-%d')}: Value = {cp[value_col]:.2f}")
            else:
                st.info("No significant change points detected with current settings.")
        
        except Exception as e:
            st.error(f"Error in change point detection: {str(e)}")
    
    @staticmethod
    def plot_prophet_components(model, forecast):
        """Plot Prophet forecast components"""
        components = ['trend']
        if 'yearly' in forecast.columns:
            components.append('yearly')
        if 'weekly' in forecast.columns:
            components.append('weekly')
        if 'daily' in forecast.columns:
            components.append('daily')
        
        fig = make_subplots(
            rows=len(components), cols=1,
            subplot_titles=components,
            vertical_spacing=0.1
        )
        
        for i, component in enumerate(components):
            if component in forecast.columns:
                fig.add_trace(
                    go.Scatter(
                        x=forecast['ds'],
                        y=forecast[component],
                        mode='lines',
                        name=component.title()
                    ),
                    row=i+1, col=1
                )
        
        fig.update_layout(height=200*len(components), title_text="Forecast Components")
        return fig
    
    @staticmethod
    def create_monitoring_dashboard():
        """Real-time monitoring dashboard for time series"""
        st.markdown("### 📊 Real-time Monitoring Dashboard")
        st.markdown("---")
        
        # Simulate real-time time series data
        if st.button("🚀 Start Monitoring"):
            placeholder = st.empty()
            
            # Initialize data
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=30, freq='D')
            base_trend = np.linspace(100, 120, 30)
            seasonal = 10 * np.sin(np.arange(30) * 2 * np.pi / 7)  # Weekly seasonality
            noise = np.random.normal(0, 2, 30)
            values = base_trend + seasonal + noise
            
            for i in range(20):  # 20 updates
                # Add new data point
                new_date = dates[-1] + timedelta(days=i+1)
                new_trend = 120 + i * 0.5
                new_seasonal = 10 * np.sin((30 + i) * 2 * np.pi / 7)
                new_noise = np.random.normal(0, 2)
                new_value = new_trend + new_seasonal + new_noise
                
                # Update arrays
                current_dates = np.append(dates, new_date)[-30:]  # Keep last 30 days
                current_values = np.append(values, new_value)[-30:]
                
                with placeholder.container():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        # Main time series plot
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=current_dates,
                            y=current_values,
                            mode='lines+markers',
                            name='Live Data',
                            line=dict(color='blue')
                        ))
                        
                        # Add anomaly detection
                        mean_val = np.mean(current_values)
                        std_val = np.std(current_values)
                        if abs(new_value - mean_val) > 2 * std_val:
                            fig.add_trace(go.Scatter(
                                x=[new_date],
                                y=[new_value],
                                mode='markers',
                                name='Anomaly',
                                marker=dict(color='red', size=15, symbol='x')
                            ))
                        
                        fig.update_layout(
                            title=f"Live Time Series Monitor (Update #{i+1})",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Current metrics
                        st.metric("Current Value", f"{new_value:.2f}")
                        change = new_value - current_values[-2] if len(current_values) > 1 else 0
                        st.metric("Change", f"{change:+.2f}")
                        st.metric("Mean (30d)", f"{np.mean(current_values):.2f}")
                        
                        # Anomaly alert
                        if abs(new_value - mean_val) > 2 * std_val:
                            st.error("🚨 ANOMALY DETECTED!")
                    
                    with col3:
                        # Quick stats
                        st.markdown("**Quick Stats:**")
                        st.write(f"Min: {np.min(current_values):.2f}")
                        st.write(f"Max: {np.max(current_values):.2f}")
                        st.write(f"Std: {np.std(current_values):.2f}")
                        
                        # Trend indicator
                        recent_trend = np.polyfit(range(10), current_values[-10:], 1)[0]
                        trend_emoji = "📈" if recent_trend > 0 else "📉"
                        st.write(f"Trend: {trend_emoji}")
                
                time.sleep(1)  # Real-time delay
            
            st.success("✅ Monitoring session completed!")
    
    @staticmethod
    def trend_analysis(df, date_col, value_col):
        """Advanced trend analysis and decomposition"""
        st.markdown("### 📊 Trend Analysis & Decomposition")
        
        try:
            # Prepare data
            ts_df = df[[date_col, value_col]].copy()
            ts_df[date_col] = pd.to_datetime(ts_df[date_col])
            ts_df = ts_df.sort_values(date_col).dropna()
            ts_df.set_index(date_col, inplace=True)
            
            # Resample data
            freq = st.selectbox("Analysis Frequency", ["D", "W", "M"], index=1, key="trend_freq")
            ts_data = ts_df[value_col].resample(freq).mean().dropna()
            
            if len(ts_data) < 10:
                st.warning("Need at least 10 data points for trend analysis")
                return
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Linear trend analysis
                x = np.arange(len(ts_data))
                coeffs = np.polyfit(x, ts_data.values, 1)
                trend_line = np.poly1d(coeffs)(x)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ts_data.index,
                    y=ts_data.values,
                    mode='lines+markers',
                    name='Data',
                    line=dict(color='blue')
                ))
                
                fig.add_trace(go.Scatter(
                    x=ts_data.index,
                    y=trend_line,
                    mode='lines',
                    name='Linear Trend',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(title="Linear Trend Analysis", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Polynomial trend
                degree = st.slider("Polynomial Degree", 1, 5, 2)
                poly_coeffs = np.polyfit(x, ts_data.values, degree)
                poly_trend = np.poly1d(poly_coeffs)(x)
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=ts_data.index,
                    y=ts_data.values,
                    mode='lines+markers',
                    name='Data',
                    line=dict(color='blue')
                ))
                
                fig2.add_trace(go.Scatter(
                    x=ts_data.index,
                    y=poly_trend,
                    mode='lines',
                    name=f'Polynomial Trend (degree {degree})',
                    line=dict(color='green', dash='dash')
                ))
                
                fig2.update_layout(title=f"Polynomial Trend (Degree {degree})", height=400)
                st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                # Trend statistics
                st.markdown("**Trend Statistics:**")
                
                # Linear trend slope
                slope = coeffs[0]
                if freq == 'D':
                    freq_text = "per day"
                elif freq == 'W':
                    freq_text = "per week"
                else:
                    freq_text = "per month"
                
                st.metric("Linear Trend Slope", f"{slope:.4f} {freq_text}")
                
                # R-squared for linear fit
                ss_res = np.sum((ts_data.values - trend_line) ** 2)
                ss_tot = np.sum((ts_data.values - np.mean(ts_data.values)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                st.metric("Linear Fit R²", f"{r_squared:.4f}")
                
                # Trend direction
                if slope > 0:
                    st.success("📈 Upward Trend")
                elif slope < 0:
                    st.error("📉 Downward Trend")
                else:
                    st.info("➡️ No Clear Trend")
                
                # Recent vs overall trend
                recent_data = ts_data.tail(len(ts_data)//4)  # Last quarter
                recent_x = np.arange(len(recent_data))
                recent_slope = np.polyfit(recent_x, recent_data.values, 1)[0]
                
                st.markdown("**Recent Trend:**")
                st.metric("Recent Slope", f"{recent_slope:.4f} {freq_text}", 
                         f"{((recent_slope - slope) / slope * 100):+.1f}%" if slope != 0 else "N/A")
                
                # Volatility analysis
                volatility = ts_data.std()
                st.metric("Volatility (Std Dev)", f"{volatility:.2f}")
                
                # Growth rate (if all positive values)
                if (ts_data > 0).all():
                    first_val = ts_data.iloc[0]
                    last_val = ts_data.iloc[-1]
                    periods = len(ts_data) - 1
                    growth_rate = ((last_val / first_val) ** (1/periods) - 1) * 100
                    st.metric("Compound Growth Rate", f"{growth_rate:.2f}% {freq_text}")
        
        except Exception as e:
            st.error(f"Error in trend analysis: {str(e)}")
