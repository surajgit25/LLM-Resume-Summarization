import streamlit as st
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Sample data (replace with your own data loading code)
resume_text = "Experienced Data Scientist with a demonstrated history of working in the research industry. Skilled in Machine Learning, Natural Language Processing, and Deep Learning. Strong engineering professional with a Master's degree focused on Computer Science."
job_description = "We are looking for a Data Scientist with expertise in Machine Learning and Natural Language Processing. The ideal candidate should have a Master's degree in Computer Science and strong programming skills."

# Function to create word cloud
def create_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot()

# Function to analyze text
def analyze_text(text):
    st.header("Text Analytics")
    
    # Word count
    words = text.split()
    st.write("Word Count:", len(words))
    
    # Most common words
    word_counter = Counter(words)
    st.subheader("Most Common Words")
    st.write(word_counter.most_common(5))
    
    # Word cloud
    st.subheader("Word Cloud")
    create_wordcloud(text)

# Main
def main():
    st.title("Resume and Job Description Analytics")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page", ["Resume Analysis", "Job Description Analysis"])
    
    if page == "Resume Analysis":
        st.header("Resume Analysis")
        analyze_text(resume_text)
    elif page == "Job Description Analysis":
        st.header("Job Description Analysis")
        analyze_text(job_description)

if __name__ == "__main__":
    main()