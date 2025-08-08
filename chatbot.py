import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Hospital FAQ data
faq_data = [
    {"question": "Hi", "answer": "How can I help you?"},
    {"question": "What are your visiting hours?", "answer": "Visiting hours are from 10 AM to 8 PM every day."},
    {"question": "How can I book an appointment?", "answer": "You can book an appointment through our website or call our front desk."},
    {"question": "Do you have emergency services?", "answer": "Yes, we offer 24/7 emergency services."},
    {"question": "Which insurance plans do you accept?", "answer": "We accept most major insurance plans. Please contact support for a detailed list."},
    {"question": "How do I get my medical test results?", "answer": "Test results are available through our patient portal or can be collected from reception."},
    {"question": "Do you provide ambulance service?", "answer": "Yes, we have ambulance services available 24/7. Call our helpline to request one."},
    {"question": "Are there specialists available?", "answer": "Yes, we have specialists in cardiology, neurology, orthopedics, and more."},
    {"question": "What documents do I need for admission?", "answer": "You need your ID card, insurance documents, and doctor’s referral if available."}
]

# Prepare questions and answers
questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]

# Vectorize the questions
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# Chatbot response logic
def respond(message, history):
    if not message.strip():
        return "Please enter a question."

    user_vec = vectorizer.transform([message])
    similarities = cosine_similarity(user_vec, X)
    best_match_index = similarities.argmax()
    best_score = similarities[0][best_match_index]

    if best_score < 0.3:
        return "Sorry, I couldn't find an answer to your question."
    
    return answers[best_match_index]

# Chat interface using Gradio
chat_interface = gr.ChatInterface(
    fn=respond,
    title="Hospital FAQ Chatbot",
    description="Ask any question related to our hospital services."
)

chat_interface.launch()
