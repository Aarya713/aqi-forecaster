
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from utils import predict_next_day, fetch_pollution_last_30_days, simulate_aqi

st.set_page_config(page_title="AQI Forecaster", page_icon="🌫️")
st.title("🌫️ Delhi Air Quality Forecaster")
st.markdown("Predict tomorrow's AQI using deep learning and real‑time data.")

def get_latest_pollution():
    try:
        poll_df = fetch_pollution_last_30_days()
        if len(poll_df) > 0:
            return poll_df.iloc[-1]
    except:
        pass
    return None

if st.button("🔮 Get Tomorrow's AQI"):
    with st.spinner("Fetching data and running model..."):
        try:
            pred = predict_next_day()
            st.success(f"**Predicted AQI for tomorrow:** {pred:.2f}")

            # Category and health advisory
            if pred <= 50:
                category, color, health = "Good", "green", "Air quality is acceptable. Enjoy outdoor activities."
            elif pred <= 100:
                category, color, health = "Satisfactory", "lightgreen", "Sensitive individuals should reduce prolonged outdoor exertion."
            elif pred <= 200:
                category, color, health = "Moderate", "yellow", "Everyone may experience health effects; limit outdoor activities."
            elif pred <= 300:
                category, color, health = "Poor", "orange", "Health alert: avoid outdoor activities, wear masks."
            elif pred <= 400:
                category, color, health = "Very Poor", "red", "Serious health risk. Stay indoors, use air purifiers."
            else:
                category, color, health = "Severe", "darkred", "Health emergency. Everyone should remain indoors."

            st.markdown(f"**Category:** <span style='color:{color}; font-weight:bold;'>{category}</span>", unsafe_allow_html=True)
            st.info(f"**Health Advisory:** {health}")

            # Pollution reduction suggestions based on current highest pollutant
            st.subheader("💡 Pollution Reduction Suggestions")
            latest = get_latest_pollution()
            if latest is not None:
                pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone']
                values = {p: latest.get(p, 0) for p in pollutants}
                max_poll = max(values, key=values.get)
                if max_poll == 'PM2.5':
                    st.write("• Limit vehicle use, avoid burning waste, use public transport.")
                elif max_poll == 'PM10':
                    st.write("• Control dust from construction and roads; water sprinkling.")
                elif max_poll == 'NO2':
                    st.write("• Reduce traffic congestion, promote electric vehicles.")
                elif max_poll == 'SO2':
                    st.write("• Use cleaner fuels in industries, enforce emission standards.")
                elif max_poll == 'CO':
                    st.write("• Improve vehicle efficiency, avoid idling.")
                elif max_poll == 'Ozone':
                    st.write("• Reduce volatile organic compounds (VOCs) from paints and solvents.")
                else:
                    st.write("No dominant pollutant identified.")
            else:
                st.write("Unable to fetch current pollution data for suggestions.")

            # ---- What‑If Simulator ----
            with st.expander("⚙️ What‑If Simulator (Adjust pollution levels)"):
                st.write("Reduce each pollutant by a percentage to see the impact on tomorrow's AQI.")
                col1, col2 = st.columns(2)
                with col1:
                    red_pm25 = st.slider("PM2.5 reduction (%)", 0, 100, 0) / 100
                    red_pm10 = st.slider("PM10 reduction (%)", 0, 100, 0) / 100
                    red_no2 = st.slider("NO2 reduction (%)", 0, 100, 0) / 100
                with col2:
                    red_so2 = st.slider("SO2 reduction (%)", 0, 100, 0) / 100
                    red_co = st.slider("CO reduction (%)", 0, 100, 0) / 100
                    red_o3 = st.slider("Ozone reduction (%)", 0, 100, 0) / 100

                if st.button("Simulate"):
                    changes = {}
                    if red_pm25 > 0: changes['PM2.5'] = 1 - red_pm25
                    if red_pm10 > 0: changes['PM10'] = 1 - red_pm10
                    if red_no2 > 0: changes['NO2'] = 1 - red_no2
                    if red_so2 > 0: changes['SO2'] = 1 - red_so2
                    if red_co > 0: changes['CO'] = 1 - red_co
                    if red_o3 > 0: changes['Ozone'] = 1 - red_o3
                    if changes:
                        try:
                            new_aqi = simulate_aqi(changes)
                            st.info(f"**Simulated AQI after reductions:** {new_aqi:.2f}")
                            if new_aqi < pred:
                                st.success(f"🎉 That's a {pred - new_aqi:.2f} point improvement!")
                            else:
                                st.warning("No significant improvement.")
                        except Exception as e:
                            st.error(f"Simulation failed: {e}")
                    else:
                        st.write("No reductions selected.")

        except Exception as e:
            st.error(f"Error: {e}")

# Show recent trend if data available
with st.expander("📈 Recent AQI Trend (last 30 days)"):
    try:
        poll_df = fetch_pollution_last_30_days()
        if len(poll_df) > 0 and 'AQI' in poll_df.columns and not poll_df['AQI'].isnull().all():
            fig, ax = plt.subplots()
            ax.plot(poll_df.index, poll_df['AQI'], marker='o', linestyle='-')
            ax.set_title("Last 30 Days AQI (real historical values)")
            ax.set_xlabel("Date")
            ax.set_ylabel("AQI")
            st.pyplot(fig)
        else:
            st.write("No AQI data available for trend.")
    except Exception as e:
        st.write(f"Could not load trend data: {e}")
