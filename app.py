import os
import re
import string
import random
from datetime import datetime
from functools import lru_cache

from flask import Flask, render_template, request, jsonify

# NLTK / ML imports are kept in one place so the backend stays easy to maintain.
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# -----------------------------
# Fast, one-time NLTK setup
# -----------------------------
NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}


def ensure_nltk_resources():
    for package, locator in NLTK_RESOURCES.items():
        try:
            nltk.data.find(locator)
        except LookupError:
            nltk.download(package, quiet=True)


ensure_nltk_resources()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# -----------------------------
# Knowledge base
# -----------------------------
FAQ_DATA = [
    {"question": "What is AyuQ?", "answer": "AyuQ is an intelligent conversational AI platform designed to deliver instant, contextual responses for technical queries, FAQs, and learning assistance."},
    {"question": "What is AyuQSpark?", "answer": "AyuQ Chatbot is the official AI assistant persona powering AyuQ. It guides users through queries using NLP, semantic matching, and interactive assistance."},
    {"question": "Who created AyuQ?", "answer": "AyuQ was designed and engineered by Ayush Aryan, a Computer Science Engineering student specializing in Artificial Intelligence (C.S.E. [A.I.]) at Accurate Institute of Management and Technology."},
    {"question": "What is NLP?", "answer": "Natural Language Processing (NLP) is a subfield of Artificial Intelligence that enables computers to understand, interpret, preprocess, and generate human language meaningfully."},
    {"question": "What is Natural Language Processing?", "answer": "Natural Language Processing (NLP) combines computational linguistics with statistical, machine learning, and deep learning models to process human textual and spoken interactions."},
    {"question": "What is NLTK?", "answer": "NLTK (Natural Language Toolkit) is a comprehensive Python library providing interfaces and lexical resources for text processing, tokenization, stemming, lemmatization, and tagging."},
    {"question": "What is TF-IDF?", "answer": "TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistical measure evaluating how relevant a word is to a document in a collection or corpus."},
    {"question": "What is cosine similarity?", "answer": "Cosine similarity measures the metric cosine angle between two non-zero vectors in an inner product space, determining semantic orientation independent of vector magnitude."},
    {"question": "What is machine learning?", "answer": "Machine Learning (ML) is a branch of AI focusing on data-driven statistical algorithms that generalize and improve performance iteratively without explicit hardcoded rules."},
    {"question": "What is artificial intelligence?", "answer": "Artificial Intelligence (AI) is the simulation of human intelligence processes by computational machines, including perception, reasoning, decision-making, and self-correction."},
    {"question": "What is a chatbot?", "answer": "A chatbot is an automated software interface engineered to simulate text or voice conversations with human users via rule-based heuristics or machine learning."},
    {"question": "What is Flask?", "answer": "Flask is a lightweight, extensible Python WSGI web microframework designed for fast API development, web services, and minimal server footprints."},
    {"question": "How does AyuQ work?", "answer": "AyuQ receives user prompts, executes an NLTK preprocessing pipeline, transforms queries via TF-IDF vectorization, computes cosine similarity against stored vectors, and surfaces the optimal match."},
    {"question": "What preprocessing does AyuQ use?", "answer": "AyuQ applies lowercasing, punctuation stripping, word tokenization via NLTK, English stop-word filtering, and morphological lemmatization via WordNet."},
    {"question": "What technology does AyuQ use?", "answer": "AyuQ is built on a clean Python & Flask backend, utilizing NLTK, Scikit-Learn (TF-IDF & Cosine Similarity), along with an aesthetic HTML5/CSS3/Vanilla JS frontend."},
    {"question": "How do I run AyuQ?", "answer": "Install dependencies using 'pip install -r requirements.txt', then execute 'python app.py' in your terminal and open 'http://127.0.0.1:5000' in your web browser."},
    {"question": "What are the features of AyuQ?", "answer": "Key features include automated NLTK tokenization & lemmatization, TF-IDF cosine matching, real-time confidence scores, quick question triggers, study tips, and a responsive glassmorphic UI."},
    {"question": "What is an FAQ?", "answer": "An FAQ (Frequently Asked Questions) is an organized compendium of queries and authoritative solutions addressing recurring inquiries within a domain."},
    {"question": "Give me a study tip", "answer": "Try the Pomodoro Technique: 25 minutes of high-intensity, distraction-free study followed by a strict 5-minute break to maximize neuro-retention."},
    {"question": "How can I study better?", "answer": "Employ Active Recall and Spaced Repetition: test yourself from memory rather than passively rereading notes, and review tricky concepts at increasing intervals."},
]

# Added from the supplied FAQ list.
ADDITIONAL_FAQ = [
    ("What is Python?", "Python is a high-level, interpreted and general-purpose programming language. It is popular because its syntax is simple, readable and easy to learn."),
    ("Why is Python popular?", "Python is popular because it is easy to learn, has a large collection of libraries, supports multiple programming styles and is widely used in web development, AI, data science, automation and software development."),
    ("What are the features of Python?", "Important Python features include simple syntax, dynamic typing, interpreted execution, object-oriented programming, extensive libraries, portability and open-source availability."),
    ("What is a variable in Python?", "A variable is a name used to store a value in Python. For example, age = 20 stores the value 20 in the variable age."),
    ("What are data types in Python?", "Common Python data types include int, float, string, boolean, list, tuple, set and dictionary."),
    ("What is a list in Python?", "A list is an ordered and mutable collection in Python. For example, numbers = [10, 20, 30]."),
    ("What is a tuple in Python?", "A tuple is an ordered collection that cannot normally be changed after creation. For example, point = (10, 20)."),
    ("What is a dictionary in Python?", "A dictionary stores data as key-value pairs. For example, student = {'name': 'Ayush', 'age': 20}."),
    ("What is a function in Python?", "A function is a reusable block of code designed to perform a specific task. Functions are created using the def keyword."),
    ("What is a Python module?", "A module is a Python file containing reusable code such as functions, classes or variables. Modules can be imported into other Python programs."),
    ("What is a data structure?", "A data structure is a method of organizing and storing data so that it can be accessed and modified efficiently."),
    ("Why are data structures important?", "Data structures help organize data efficiently and make operations such as searching, insertion, deletion and sorting easier and faster."),
    ("What is an array?", "An array is a data structure that stores elements in an ordered sequence, usually using contiguous memory locations. Elements can typically be accessed using an index."),
    ("What is a stack?", "A stack is a linear data structure that follows LIFO, meaning Last In, First Out. Common operations are push and pop."),
    ("What is a queue?", "A queue is a linear data structure that follows FIFO, meaning First In, First Out. Elements are generally inserted at the rear and removed from the front."),
    ("What is a linked list?", "A linked list is a data structure made of nodes where each node contains data and a reference to another node."),
    ("What is a tree in data structure?", "A tree is a hierarchical data structure consisting of nodes connected by edges. A common example is a binary tree."),
    ("What is a binary tree?", "A binary tree is a tree data structure in which each node can have at most two children, usually called the left child and right child."),
    ("What is a graph?", "A graph is a data structure consisting of vertices and edges. It can represent relationships such as roads, social networks or computer networks."),
    ("What is a hash table?", "A hash table is a data structure that stores key-value pairs and uses a hash function to provide efficient average-time lookup."),
    ("What is an algorithm?", "An algorithm is a step-by-step procedure used to solve a problem or perform a computation."),
    ("What is searching?", "Searching is the process of finding a particular element or value in a collection of data."),
    ("What is linear search?", "Linear search checks elements one by one until the required element is found or the collection ends. Its worst-case time complexity is O(n)."),
    ("What is binary search?", "Binary search repeatedly divides a sorted collection into two parts to find an element. Its time complexity is O(log n)."),
    ("What is sorting?", "Sorting is the process of arranging data in a particular order, such as ascending or descending order."),
    ("What is time complexity?", "Time complexity describes how the running time of an algorithm grows as the input size increases."),
    ("What is Big O notation?", "Big O notation describes the upper-bound growth rate of an algorithm's time or space requirements."),
    ("What is Artificial Intelligence?", "Artificial Intelligence, or AI, is a field of computer science focused on creating systems that can perform tasks that normally require human-like intelligence."),
    ("What are the applications of AI?", "AI is used in recommendation systems, chatbots, computer vision, speech recognition, fraud detection, healthcare, robotics, autonomous systems and many other areas."),
    ("What is Machine Learning?", "Machine Learning is a branch of AI in which computers learn patterns from data and use those patterns to make predictions or decisions."),
    ("What are the types of Machine Learning?", "The main types are supervised learning, unsupervised learning and reinforcement learning."),
    ("What is supervised learning?", "Supervised learning trains a model using labeled data where the expected output is known."),
    ("What is unsupervised learning?", "Unsupervised learning works with data without predefined labels and tries to discover patterns, groups or structures in the data."),
    ("What is reinforcement learning?", "Reinforcement learning is a machine learning approach where an agent learns by interacting with an environment and receiving rewards or penalties."),
    ("What is deep learning?", "Deep learning is a branch of machine learning that uses neural networks with multiple layers to learn complex patterns from data."),
    ("What is a neural network?", "A neural network is a machine learning model inspired by the structure of biological neural systems. It consists of interconnected computational units called neurons."),
    ("What is Generative AI?", "Generative AI refers to AI systems that can generate new content such as text, images, audio, video or code based on learned patterns."),
    ("What is text preprocessing?", "Text preprocessing prepares raw text for analysis. Common steps include lowercasing, tokenization, punctuation removal, stop-word removal and lemmatization."),
    ("What is tokenization?", "Tokenization is the process of breaking text into smaller units called tokens, such as words or sentences."),
    ("What are stop words?", "Stop words are common words that may provide limited information for some NLP tasks, such as 'the', 'is', 'a' and 'and'."),
    ("What is lemmatization?", "Lemmatization converts words into their base or dictionary form using linguistic information. For example, 'running' may be converted to 'run'."),
    ("What is Data Science?", "Data Science combines programming, statistics, mathematics and domain knowledge to extract useful insights from data."),
    ("What is a database?", "A database is an organized collection of data that can be stored, managed and retrieved efficiently."),
    ("What is SQL?", "SQL stands for Structured Query Language. It is used to create, retrieve, update and manage data in relational databases."),
    ("What is an API?", "API stands for Application Programming Interface. It provides a defined way for different software applications or services to communicate with each other."),
    ("What is cloud computing?", "Cloud computing provides computing resources such as servers, storage and databases over the internet instead of requiring all resources to be hosted locally."),
    ("What is Git?", "Git is a distributed version-control system used to track changes in source code and collaborate with other developers."),
    ("What is GitHub?", "GitHub is a platform for hosting Git repositories and collaborating on software projects."),
    ("What is an operating system?", "An operating system is system software that manages computer hardware and provides services for applications. Examples include Windows, Linux, macOS and Android."),
]
FAQ_DATA.extend({"question": q, "answer": a} for q, a in ADDITIONAL_FAQ)

STUDY_TIPS = [
    "Practice Active Recall: Close your reference notes and summarize the core algorithm or concept on blank paper purely from memory.",
    "Use the Feynman Technique: Try explaining the topic in elementary terms as if teaching it to a novice. This highlights conceptual blind spots.",
    "Interleaving Technique: Alternate between two different subjects during long revision blocks.",
    "Implement the 25/5 Pomodoro Cycle: Focus intensely for 25 minutes, then disengage completely for 5 minutes to prevent cognitive fatigue.",
    "Spaced Repetition: Review freshly learned topics after 24 hours, then 3 days, and finally after 1 week to lock them into long-term memory.",
]

CONVERSATIONAL_INTENTS = {
    "greetings": {
        "patterns": ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "greetings", "namaste"],
        "responses": [
            "Hello! I'm AyuQ Chatbot ✨ How can I help you learn today?",
            "Hey there! 🌸 AyuQ Chatbot is ready. What shall we explore?",
            "Hi! 💫 Ask me about Python, AI, NLP, data structures, or your study routine.",
        ],
    },
    "gratitude": {
        "patterns": ["thank you", "thanks", "thx", "appreciate it"],
        "responses": ["You're very welcome! 🌷", "Glad I could help! Keep learning and keep asking. ✨", "Anytime! 💙"],
    },
    "farewell": {
        "patterns": ["bye", "goodbye", "see you", "exit", "quit"],
        "responses": ["Goodbye! Have a productive study session. 🌸", "See you soon! Keep learning. ✨"],
    },
    "stressed": {
        "patterns": ["stressed", "i am stressed", "anxious", "overwhelmed", "exam tension"],
        "responses": ["Take a small breath 🌿. Break the topic into one focused sub-problem at a time.", "Pause for five minutes, hydrate, relax your shoulders, then restart with one tiny task."],
    },
    "bored": {
        "patterns": ["bored", "i am bored", "feeling lazy", "nothing to do"],
        "responses": ["Let's make it fun ✨ Try a quick Python or AI question, or tap Study Tip.", "Boredom can become a mini learning session 🌱. Ask me about TF-IDF, stacks, Python, or AI."],
    },
}

# Simple timetable/routine assistant for common schedule questions.
TIMETABLE = [
    ("09:00", "10:00", "Computer Science / Core Class"),
    ("10:00", "11:00", "Artificial Intelligence"),
    ("11:15", "12:15", "Data Structures"),
    ("12:15", "13:00", "Lunch / Break"),
    ("14:00", "15:00", "Python / Programming Practice"),
    ("15:00", "16:00", "Self Study / Revision"),
]


def preprocess_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    return " ".join(
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and token.isalnum()
    )


# Precompute once at startup: queries only transform, they do not refit the model.
faq_questions_clean = [preprocess_text(item["question"]) for item in FAQ_DATA]
vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
tfidf_matrix = vectorizer.fit_transform(faq_questions_clean)


def check_conversational(text: str):
    clean = re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()
    for _, data in CONVERSATIONAL_INTENTS.items():
        for pattern in data["patterns"]:
            if pattern == clean or pattern in clean.split() or clean.startswith(pattern + " "):
                return random.choice(data["responses"])
    return None


def suggestions_for(query: str):
    q = query.lower()
    if any(x in q for x in ["python", "programming"]):
        return ["What is a Python variable?", "What is a Python function?", "What is a list in Python?"]
    if any(x in q for x in ["data structure", "stack", "queue", "tree", "graph"]):
        return ["What is a stack?", "What is a queue?", "What is a binary tree?"]
    if any(x in q for x in ["ai", "machine learning", "deep learning"]):
        return ["What is Machine Learning?", "What is deep learning?", "What is Generative AI?"]
    return ["What is NLP?", "What is TF-IDF?", "Give me a study tip"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    user_message = payload.get("message", "").strip()

    if not user_message:
        return jsonify({"assistant_name": "AyuQ Chatbot", "reply": "Please enter a question or choose a quick prompt ✨", "confidence": 0, "suggestions": suggestions_for("")})

    # Time-aware greeting.
    if user_message.lower().strip() in {"good morning", "good afternoon", "good evening"}:
        hour = datetime.now().hour
        greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
        return jsonify({"assistant_name": "AyuQ Chatbot", "reply": f"{greeting}! 🌸 I'm ready to help. What would you like to learn?", "confidence": 100, "suggestions": suggestions_for("")})

    # Lightweight conversational/sentiment layer before semantic FAQ matching.
    conv_reply = check_conversational(user_message)
    if conv_reply:
        return jsonify({"assistant_name": "AyuQ Chatbot", "reply": conv_reply, "confidence": 100, "suggestions": suggestions_for(user_message)})

    # Routine/timetable intent.
    q_lower = user_message.lower()
    if any(k in q_lower for k in ["timetable", "routine", "schedule", "class timing", "today's class"]):
        lines = [f"🕘 {start}–{end} — {subject}" for start, end, subject in TIMETABLE]
        return jsonify({"assistant_name": "AyuQ Chatbot", "reply": "Here is the sample AyuQ study routine:\n\n" + "\n".join(lines), "confidence": 100, "suggestions": ["What is Python?", "Give me a study tip", "What is AI?"]})

    processed_user_query = preprocess_text(user_message)
    if not processed_user_query.strip():
        return jsonify({"assistant_name": "AyuQ Chatbot", "reply": "I couldn't extract enough useful keywords. Could you rephrase it? 🌷", "confidence": 0, "suggestions": suggestions_for(user_message)})

    user_vec = vectorizer.transform([processed_user_query])
    similarity_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    best_index = int(similarity_scores.argmax())
    highest_score = float(similarity_scores[best_index])
    confidence_pct = round(highest_score * 100, 1)

    if highest_score >= 0.25:
        matched_faq = FAQ_DATA[best_index]
        return jsonify({
            "assistant_name": "AyuQ Chatbot",
            "reply": matched_faq["answer"],
            "matched_question": matched_faq["question"],
            "confidence": confidence_pct,
            "suggestions": suggestions_for(matched_faq["question"]),
        })

    # Smart fallback / did-you-mean: surface the nearest known question when confidence is modest.
    if highest_score >= 0.12:
        nearest = FAQ_DATA[best_index]["question"]
        reply = f"I couldn't find a strong match yet. Did you mean: “{nearest}”?\n\nTry asking that, or choose a suggestion below."
    else:
        reply = "I don't have a close FAQ match for that yet. Try asking about AyuQ, Python, AI, NLP, data structures, TF-IDF, or study tips."

    return jsonify({"assistant_name": "AyuQ Chatbot", "reply": reply, "confidence": confidence_pct, "suggestions": suggestions_for(user_message)})


@app.route("/study-tip", methods=["GET"])
def get_study_tip():
    return jsonify({"assistant_name": "AyuQ Chatbot", "reply": f"📘 Study Tip: {random.choice(STUDY_TIPS)}", "confidence": 100})


if __name__ == "__main__":
    # Debug reloader is off so NLTK/model initialization happens only once.
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
