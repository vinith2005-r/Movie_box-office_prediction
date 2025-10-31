import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import io

# Set page configuration
st.set_page_config(
    page_title="Movie Box-Office Prediction",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #ff4b4b;
        text-shadow: 2px 2px 4px #000000;
        margin-bottom: 2rem;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        color: #ff4b4b;
        text-align: center;
        padding: 1rem;
        background-color: #ffffff;
        border-radius: 10px;
        border: 2px solid #ff4b4b;
    }
    .section-header {
        color: #ff4b4b;
        font-size: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Create dataset for Indian cinema
def create_dataset():
    data = {
        'budget': [50000000, 80000000, 120000000, 30000000, 150000000,
                   70000000, 100000000, 60000000, 90000000, 200000000,
                   40000000, 250000000, 180000000, 75000000, 95000000],

        'genre': ['Action', 'Drama', 'Romance', 'Comedy', 'Action',
                  'Thriller', 'Drama', 'Comedy', 'Romance', 'Action',
                  'Drama', 'Action', 'Thriller', 'Comedy', 'Romance'],

        'language': ['Hindi', 'Tamil', 'Telugu', 'Hindi', 'Telugu',
                     'Tamil', 'Hindi', 'Telugu', 'Tamil', 'Hindi',
                     'Telugu', 'Tamil', 'Hindi', 'Telugu', 'Tamil'],

        'industry': ['Bollywood', 'Kollywood', 'Tollywood', 'Bollywood', 'Tollywood',
                     'Kollywood', 'Bollywood', 'Tollywood', 'Kollywood', 'Bollywood',
                     'Tollywood', 'Kollywood', 'Bollywood', 'Tollywood', 'Kollywood'],

        'releasing_time': ['Festival', 'Normal', 'Holiday', 'Normal', 'Festival',
                           'Holiday', 'Normal', 'Festival', 'Holiday', 'Festival',
                           'Normal', 'Holiday', 'Festival', 'Normal', 'Holiday'],

        'category': ['U', 'UA', 'A', 'U', 'UA', 'A', 'U', 'UA', 'A', 'U', 'UA', 'A', 'U', 'UA', 'A'],

        'casts': ['vijay,samantha', 'prabhas,anushka', 'alluarjun,pooja', 'akshaykumar,kareena',
                  'rajinikanth,nayanthara', 'vijay,samantha', 'amitabh,deepika', 'maheshbabu,shriya',
                  'surya,trisha', 'hrithik,tiger', 'ntr,janhvi', 'ajith,shraddha',
                  'ranveer,alia', 'ramcharan,kiara', 'dhanush,sonam'],

        'director': ['sanjayleela', 'rajamouli', 'trivikram', 'rohitshetty', 'shankar',
                     'atlee', 'karanjohar', 'sukumar', 'maniratnam', 'vishnuvardhan',
                     'koratala', 'lokesh', 'anuraag', 'prashanth', 'vetrimaran'],

        'music_director': ['anirudh', 'keeravani', 'devi', 'prasad', 'anirudh',
                           'harris', 'arrahman', 'thaman', 'arrahman', 'vishal',
                           'keeravani', 'anirudh', 'amit', 'devi', 'gvp'],

        'box_office': [200000000, 500000000, 300000000, 80000000, 800000000,
                       150000000, 120000000, 250000000, 180000000, 400000000,
                       90000000, 600000000, 350000000, 120000000, 160000000]
    }
    return pd.DataFrame(data)


# Create the dataset
df = create_dataset()

# Define mappings for categorical features
industry_mapping = {'Bollywood': 1, 'Tollywood': 2, 'Kollywood': 3}
genre_mapping = {'Action': 1, 'Drama': 2, 'Romance': 3, 'Comedy': 4, 'Thriller': 5}
language_mapping = {'Hindi': 1, 'Tamil': 2, 'Telugu': 3}
time_mapping = {'Festival': 1, 'Holiday': 2, 'Normal': 3}
category_mapping = {'U': 1, 'UA': 2, 'A': 3}

# Cast and crew mappings with their impact on box office
cast_mapping = {
    'vijay': 200, 'samantha': 150, 'prabhas': 300, 'anushka': 100,
    'alluarjun': 250, 'pooja': 80, 'akshaykumar': 180, 'kareena': 160,
    'rajinikanth': 400, 'nayanthara': 140, 'amitabh': 350, 'deepika': 200,
    'maheshbabu': 220, 'shriya': 90, 'surya': 190, 'trisha': 130,
    'hrithik': 280, 'tiger': 120, 'ntr': 260, 'janhvi': 70,
    'ajith': 210, 'shraddha': 110, 'ranveer': 170, 'alia': 180,
    'ramcharan': 240, 'kiara': 100, 'dhanush': 160, 'sonam': 90
}

director_mapping = {
    'sanjayleela': 180, 'rajamouli': 400, 'trivikram': 220, 'rohitshetty': 200,
    'shankar': 350, 'atlee': 280, 'karanjohar': 250, 'sukumar': 230,
    'maniratnam': 300, 'vishnuvardhan': 190, 'koratala': 210, 'lokesh': 270,
    'anuraag': 160, 'prashanth': 240, 'vetrimaran': 200
}

music_director_mapping = {
    'anirudh': 250, 'keeravani': 300, 'devi': 180, 'prasad': 160,
    'harris': 170, 'arrahman': 350, 'thaman': 190, 'vishal': 200,
    'amit': 150, 'gvp': 140
}


# Simple prediction model
def predict_box_office(input_data):
    # Base prediction from budget
    base_prediction = input_data['budget'] * 2.5

    # Adjust for genre
    genre_factor = {'Action': 1.3, 'Drama': 1.0, 'Romance': 0.9, 'Comedy': 1.1, 'Thriller': 1.2}
    base_prediction *= genre_factor.get(input_data['genre'], 1.0)

    # Adjust for industry
    industry_factor = {'Bollywood': 1.2, 'Tollywood': 1.1, 'Kollywood': 1.0}
    base_prediction *= industry_factor.get(input_data['industry'], 1.0)

    # Adjust for releasing time
    time_factor = {'Festival': 1.4, 'Holiday': 1.2, 'Normal': 1.0}
    base_prediction *= time_factor.get(input_data['releasing_time'], 1.0)

    # Adjust for category
    category_factor = {'U': 1.3, 'UA': 1.1, 'A': 1.0}
    base_prediction *= category_factor.get(input_data['category'], 1.0)

    # Add cast value
    cast_value = 0
    for cast_member in input_data['casts'].split(','):
        cast_value += cast_mapping.get(cast_member.strip().lower(), 0) * 10000

    # Add director value
    director_value = director_mapping.get(input_data['director'].lower(), 150) * 50000

    # Add music director value
    music_value = music_director_mapping.get(input_data['music_director'].lower(), 150) * 40000

    total_prediction = base_prediction + cast_value + director_value + music_value

    # Add some random variation for realism
    variation = np.random.uniform(0.8, 1.2)
    total_prediction *= variation

    return int(total_prediction)


# Generate report
def generate_report(prediction, input_data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Bar chart
    components = ['Budget', 'Cast Impact', 'Director Impact', 'Music Impact', 'Other Factors']
    values = [
        input_data['budget'],
        sum([cast_mapping.get(cast.strip().lower(), 0) * 10000 for cast in input_data['casts'].split(',')]),
        director_mapping.get(input_data['director'].lower(), 150) * 50000,
        music_director_mapping.get(input_data['music_director'].lower(), 150) * 40000,
        prediction - input_data['budget'] - sum([cast_mapping.get(cast.strip().lower(), 0) * 10000 for cast in
                                                 input_data['casts'].split(',')]) - director_mapping.get(
            input_data['director'].lower(), 150) * 50000 - music_director_mapping.get(
            input_data['music_director'].lower(), 150) * 40000
    ]

    ax1.bar(components, values, color=['#ff4b4b', '#ff6b6b', '#ff8e8e', '#ffb6b6', '#ffd6d6'])
    ax1.set_title('Revenue Components Breakdown', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)

    # Line graph (simulated performance over time)
    weeks = ['Week 1', 'Week 2', 'Week 3', 'Week 4']
    revenue = [
        prediction * 0.6,
        prediction * 0.8,
        prediction * 0.95,
        prediction
    ]

    ax2.plot(weeks, revenue, marker='o', linewidth=2, color='#ff4b4b')
    ax2.fill_between(weeks, revenue, alpha=0.3, color='#ff6b6b')
    ax2.set_title('Projected Revenue Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# Main application
def main():
    # Header
    st.markdown('<div class="main-header">🎬 Movie Box-Office Prediction</div>', unsafe_allow_html=True)

    # Create two columns for layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-header">Movie Details</div>', unsafe_allow_html=True)

        # Input form
        with st.form("prediction_form"):
            col1_1, col1_2 = st.columns(2)

            with col1_1:
                budget = st.number_input("Budget (INR)", min_value=1000000, max_value=1000000000, value=50000000,
                                         step=1000000)
                genre = st.selectbox("Genre", ["Action", "Drama", "Romance", "Comedy", "Thriller"])
                language = st.selectbox("Language", ["Hindi", "Tamil", "Telugu"])
                industry = st.radio("Industry", ["Bollywood", "Tollywood", "Kollywood"], horizontal=True)

            with col1_2:
                releasing_time = st.selectbox("Releasing Time", ["Festival", "Holiday", "Normal"])
                category = st.selectbox("Category", ["U", "UA", "A"])
                casts = st.text_input("Casts (comma separated)", "vijay, samantha")
                director = st.text_input("Director", "rajamouli")
                music_director = st.text_input("Music Director", "anirudh")

            # File uploaders
            st.markdown('<div class="section-header">Media Upload (For Preview Only)</div>', unsafe_allow_html=True)

            col2_1, col2_2 = st.columns(2)

            with col2_1:
                poster = st.file_uploader("First Look Poster", type=['jpg', 'jpeg', 'png'])
                if poster:
                    st.image(poster, caption="First Look Poster Preview", use_container_width=True)

            with col2_2:
                trailer = st.file_uploader("Trailer", type=['mp4', 'mov', 'avi'])
                if trailer:
                    st.video(trailer)

            submit_button = st.form_submit_button("Predict Box Office")

    with col2:
        st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)

        if submit_button:
            # Prepare input data
            input_data = {
                'budget': budget,
                'genre': genre,
                'language': language,
                'industry': industry,
                'releasing_time': releasing_time,
                'category': category,
                'casts': casts,
                'director': director,
                'music_director': music_director
            }

            # Make prediction
            prediction = predict_box_office(input_data)

            # Display result
            st.markdown(f'<div class="prediction-result">₹ {prediction:,.0f}</div>', unsafe_allow_html=True)

            # Display input summary
            st.subheader("Input Summary")
            st.write(f"**Budget:** ₹ {budget:,.0f}")
            st.write(f"**Genre:** {genre}")
            st.write(f"**Industry:** {industry}")
            st.write(f"**Director:** {director}")
            st.write(f"**Music Director:** {music_director}")
            st.write(f"**Casts:** {casts}")

            # Generate and display report
            st.subheader("Analysis Report")
            fig = generate_report(prediction, input_data)
            st.pyplot(fig)

            # Download report
            st.subheader("Download Report")

            # Create report text
            report_text = f"""
            MOVIE BOX OFFICE PREDICTION REPORT
            ==================================

            PREDICTION RESULT:
            - Estimated Box Office: ₹ {prediction:,.0f}

            MOVIE DETAILS:
            - Budget: ₹ {budget:,.0f}
            - Genre: {genre}
            - Language: {language}
            - Industry: {industry}
            - Releasing Time: {releasing_time}
            - Category: {category}
            - Director: {director}
            - Music Director: {music_director}
            - Casts: {casts}

            REVENUE BREAKDOWN:
            - Budget Impact: ₹ {budget:,.0f}
            - Cast Impact: ₹ {sum([cast_mapping.get(cast.strip().lower(), 0) * 10000 for cast in casts.split(',')]):,.0f}
            - Director Impact: ₹ {director_mapping.get(director.lower(), 150) * 50000:,.0f}
            - Music Director Impact: ₹ {music_director_mapping.get(music_director.lower(), 150) * 40000:,.0f}

            Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            """

            # Convert figure to bytes for download
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)

            col_d1, col_d2 = st.columns(2)

            with col_d1:
                st.download_button(
                    label="Download Text Report",
                    data=report_text,
                    file_name=f"box_office_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

            with col_d2:
                st.download_button(
                    label="Download Charts",
                    data=buf,
                    file_name=f"box_office_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )

    # Dataset info in sidebar
    with st.sidebar:
        st.markdown('<div class="section-header">Dataset Information</div>', unsafe_allow_html=True)
        st.write(f"Total movies in dataset: {len(df)}")
        st.write("Sample data:")
        st.dataframe(df.head(), use_container_width=True)

        st.markdown("---")
        st.subheader("About")
        st.write(
            "This model predicts box office collection for Indian movies based on various factors including budget, cast, director, and industry.")


if __name__ == "__main__":
    main()