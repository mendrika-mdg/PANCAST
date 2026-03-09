import streamlit as st
from datetime import datetime, timedelta
from pathlib import Path
import time

st.set_page_config(layout="wide")

data_dir = Path("/home/users/mendrika/PANCAST/outputs/archive")
placeholder = Path("/home/users/mendrika/PANCAST/outputs/no_nowcast.png")

st.title("Thunderstorm Nowcast")

left, _, right = st.columns([1, 1, 2])

with left:

    st.subheader("Controls")

    col_date = st.columns(2)

    years = list(range(2020, 2027))
    months = list(range(1, 13))
    days = list(range(1, 32))
    hours = list(range(24))
    minutes = [0, 15, 30, 45]

    with col_date[0]:
        year = st.selectbox("Year", years, index=years.index(2024))
        day = st.selectbox("Day", days, index=days.index(30))
        hour = st.selectbox("Hour", hours, index=hours.index(13))

    with col_date[1]:
        month = st.selectbox("Month", months, index=months.index(9))
        minute = st.selectbox("Minute", minutes, index=minutes.index(0))

    t0 = datetime(year, month, day, hour, minute)

    loop = st.toggle("Loop animation")

    lead_times = [30, 60, 90, 120]

    lead_time = st.radio(
        "Lead time (minutes)",
        options=lead_times
    )

    valid_time_placeholder = st.empty()

with right:

    viewer = st.empty()

timestamp = t0.strftime("%Y%m%d_%H%M")

def show_frame(lead):

    file_path = data_dir / f"{timestamp}_{lead}.png"
    img_path = file_path if file_path.exists() else placeholder

    valid_time = t0 + timedelta(minutes=lead)

    valid_time_placeholder.markdown(
        f"### **Valid time: {valid_time:%Y-%m-%d %H:%M UTC}**"
    )

    viewer.image(str(img_path), width=700)

if loop:

    while True:
        for lead in lead_times:
            show_frame(lead)
            time.sleep(0.8)

else:

    show_frame(lead_time)