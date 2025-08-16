🎬 Netflix Series & Movie Recommendation Chatbot

A smart Netflix recommendation chatbot built with Python, Streamlit, Pandas, and the TMDb API.
It takes user input in natural language (e.g., "recommend me a murder mystery") and returns the most relevant Netflix shows/movies, complete with title, description, genres, ratings, and posters.

📌 Features

Genre-based filtering – Suggests titles matching specific genres like Comedy, Drama, Sci-Fi, etc.

Description-based search – Finds shows/movies from partial descriptions.

Live TMDb API integration – Fetches up-to-date ratings, popularity, and poster images.

Ranking by quality – Sorts results by relevance and TMDb ratings.

Interactive UI – Built with Streamlit for an easy-to-use interface.

🛠 Tech Stack

Python – Core programming language.

Streamlit – For the interactive web UI.

Pandas – To manage and search the Netflix dataset.

Requests – To call the TMDb API.

TMDb API – For live ratings, genres, and posters.

📂 Dataset

We are using the public Netflix titles dataset from Kaggle:
Netflix Movies and TV Shows Dataset

Save it as netflix_titles.csv in your project folder.

🚀 Getting Started
1️⃣ Install Dependencies
pip install streamlit pandas requests

2️⃣ Get a TMDb API Key

Sign up at The Movie Database

Go to Settings → API and create a key

Store the key in a .env file as:

TMDB_API_KEY=your_api_key_here

3️⃣ Run the App
streamlit run app.py

📌 Example Query

User:

recommend me a sci-fi thriller with a good rating

Chatbot Output:

Stranger Things (Rating: 8.6)
A group of kids uncover supernatural mysteries in their town.


📄 Future Plans

Ask user about past watched shows to improve personalization.

Integrate collaborative filtering for smarter recommendations.

Support for multiple streaming platforms.
