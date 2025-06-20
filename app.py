import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import json

st.set_page_config(
    page_title="Udemy Revenue Simulator",
    page_icon="📊",
    layout="centered"
)

model = joblib.load('pipeline_lgb.pkl')

with open("course_mapping.json", "r") as f:
    course_mapping = json.load(f)

course_types = list(course_mapping.keys())

st.title("Udemy Course Price Simulator")

st.write("Fill in the course details to simulate expected revenue.")

placeholder_text = ["Please select"]
course_type_options = ['Development', 'Business', 'Design', 'Marketing']
course_type_options = placeholder_text + sorted(course_type_options)
course_type = st.selectbox("Course Type", course_type_options, index=0)

subcategory_options = []
if course_type != "Please select":
    subcategory_options = course_mapping.get(course_type, [])
    subcategory_options = sorted([s for s in subcategory_options if s != "Unknown"])
    if "Unknown" in course_mapping[course_type]:
        subcategory_options.append("Unknown")

subcategory_options = placeholder_text + subcategory_options
subcategory = st.selectbox("Sub-category", subcategory_options, index=0)

instructional_level_options = ['All', 'Beginner', 'Intermediate', 'Expert']
instructional_level_options = placeholder_text + sorted(instructional_level_options)
instructional_level = st.selectbox("Instructional Level", instructional_level_options, index=0)

total_hours = st.number_input("Total Hours", min_value=0.0, value=10.0)
num_lectures = st.number_input("Number of Published Lectures", min_value=1, value=20)
title = st.text_input("Course Title", "Master Python Fast")
headline = st.text_input("Course Headline", "Build real-world apps with Python")

rating = st.number_input("Rating", min_value=0.0, max_value=5.0, value=4.0)
num_reviews = st.number_input("Number of Reviews", min_value=0, value=100)

if (
    course_type != "Please select"
    and subcategory != "Please select"
    and instructional_level != "Please select"
):
    price_range = np.linspace(10, 200, 50)
    data_rows = []

    for price in price_range:
        row = {
            'course_type': course_type,
            'subcategory': subcategory,
            'instructional_level': instructional_level,
            'total_hours': total_hours,
            'num_published_lectures': num_lectures,
            'amount': price,
            'title': title,
            'headline': headline,
            'rating': rating,
            'num_reviews': num_reviews
        }
        data_rows.append(row)

    sim_df = pd.DataFrame(data_rows)

    log_revenue = model.predict(sim_df)
    predicted_revenue = np.expm1(log_revenue)

    st.subheader("📈 Predicted Revenue vs. Course Price")
    fig, ax = plt.subplots()
    ax.plot(price_range, predicted_revenue)
    ax.set_xlabel("Course Price (RM)")
    ax.set_ylabel("Predicted Revenue (RM)")
    ax.set_title("Revenue Simulation")
    st.pyplot(fig)

    best_idx = np.argmax(predicted_revenue)
    best_price = price_range[best_idx]
    best_rev = predicted_revenue[best_idx]
    st.success(f"💡 Optimal Price Suggestion: **RM {best_price:.2f}**, expected revenue: **RM {best_rev:,.2f}**")

else:
    st.warning("⚠️ Please select values for **Course Type**, **Sub-ategory**, and **Instructional Level** to see the simulation.")