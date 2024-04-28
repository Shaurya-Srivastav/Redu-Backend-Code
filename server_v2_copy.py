import os
import sqlite3
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
from tqdm import tqdm
from uuid import uuid4
import base64
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import jwt_required, get_jwt_identity
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from googleapiclient.discovery import build

app = Flask(__name__)
CORS(app, origins="*", allow_headers=[
    "Content-Type", "Authorization", "Access-Control-Allow-Credentials"],
     supports_credentials=True)

app.config['SECRET_KEY'] = b'\xb5\x12\xe8{\x91\xf4\x8e]h2\x81\x96\xb0\xc7RF\x12\xa0\x08\xe1\x1e\x9d\x00I'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////home/ubuntu/redu/chat_history.db'
app.config['JWT_SECRET_KEY'] = '0hMEaKgmGVKWswVxc5toFdJP9Z28uuKm-2_S7JtG6lE'
db = SQLAlchemy(app)
jwt = JWTManager(app)


@app.route('/validate_token', methods=['GET'])
@jwt_required()
def validate_token():
    try:
        # Get the identity of the current user, which you set when creating the access token
        current_user = get_jwt_identity()
        # Optionally, load more user information from the database if needed
        return jsonify({'user': current_user, 'valid': True}), 200
    except Exception as e:
        return jsonify({'message': 'Invalid token', 'errors': str(e)}), 401

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    country = db.Column(db.String(100), nullable=True)
    language = db.Column(db.String(50), nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)



@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    name = data['name']
    email = data['email']
    password = data['password']

    if User.query.filter_by(email=email).first():
        return jsonify({'message': 'Email already exists'}), 400

    new_user = User(name=name, email=email)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()

    access_token = create_access_token(identity=email)
    return jsonify({'message': 'User created successfully', 'access_token': access_token}), 201



@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data['email']
    password = data['password']
    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password):
        access_token = create_access_token(identity=email)
        return jsonify({'message': 'Logged in successfully', 'access_token': access_token}), 200
    return jsonify({'message': 'Invalid credentials'}), 401

@app.route('/update_profile', methods=['POST'])
def update_profile():
    data = request.get_json()
    user_id = get_jwt_identity()  # Ensure you're using JWT to get the user's identity

    user = User.query.get(user_id)
    if not user:
        return jsonify({'message': 'User not found'}), 404

    user.country = data.get('country', user.country)  # Default to existing value if not provided
    user.language = data.get('language', user.language)

    db.session.commit()
    return jsonify({'message': 'Profile updated successfully'}), 200

# Set up your OpenAI API key

# Get the database file path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'chat_history.db')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')


# INITIALIZE MODEL
# Load the model and tokenizer
device = 'cpu'
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to('cuda' if torch.cuda.is_available() else 'cpu')

# Load the FAISS index
index = faiss.read_index('embeddings/final_page_text_embeddings.faiss')

# Load the original CSV file
df = pd.read_csv('dataset/cleaned_textbook_content.csv', dtype=str, quotechar='"', delimiter=',')


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Users Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            country TEXT,
            language TEXT
        )
    ''')

    # Categories Table (no changes here)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT
        )
    ''')

    # Chat Sessions Table (now includes user_id)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            session_name TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')

    # Chat History Table (no changes needed for user_id here)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            user_message TEXT,
            bot_response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES chat_sessions(id)
        )
    ''')

    # Discussions Table (now includes user_id)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS discussions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            title TEXT,
            category_id INTEGER,
            content TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(category_id) REFERENCES categories(id)
        )
    ''')

    # Discussion Images Table (no user-specific changes needed)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS discussion_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            discussion_id INTEGER,
            image_data TEXT,
            FOREIGN KEY(discussion_id) REFERENCES discussions(id)
        )
    ''')

    # Replies Table (now includes user_id)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS replies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            discussion_id INTEGER,
            content TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(discussion_id) REFERENCES discussions(id)
        )
        ''')
    conn.commit()

# Insert initial categories
    categories = [
        'Geography',
        'Political Science',
        'Art',
        'Education',
        'Economics',
        'Science',
        'Mathematics',
        'History',
        'Philosophy',
        'Biology',
        'Chemistry',
        'Physics',
        'Computer Science',
        'Engineering',
        'Medicine',
        'Business',
        'Finance',
        'Literature',
        'Psychology',
        'Sociology',
    ]

    for category in categories:
        cursor.execute("INSERT INTO categories (name) VALUES (?)", (category,))

    conn.commit()
    conn.close()

def generate_response(conversation_history):
    start_time = time.time()
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=conversation_history,
        max_tokens=500,  # Adjust as needed
        n=1,
        stop=None,
        temperature=0.7,
    )
    end_time = time.time()
    if response.choices[0].message.content is not None:
        response_content = response.choices[0].message.content.strip()
    else:
        response_content = ""
    response_time = end_time - start_time
    return response_content, response_time

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    user_message = data['message']
    selected_chat_id = data.get('selectedChatId')

    print("Selected chat ID in /api/chatbot:", selected_chat_id)

    # Generate response using OpenAI API
    response_content, response_time = generate_response([{"role": "user", "content": user_message}])

    # Save the user message and bot response to the database
    conn = get_db_connection()
    cursor = conn.cursor()

    # Insert chat history based on selectedChatId
    cursor.execute("INSERT INTO chat_history (session_id, user_message, bot_response) VALUES (?, ?, ?)",
                   (selected_chat_id, user_message, response_content))
    conn.commit()
    conn.close()

    return jsonify({'response': response_content, 'response_time': response_time})


@app.route('/api/delete-chat-session/<int:chat_id>', methods=['DELETE'])
@jwt_required()
def delete_chat_session(chat_id):
    user_id = get_jwt_identity()  # Assuming user ID is used as identity
    conn = get_db_connection()
    cursor = conn.cursor()
    # Ensure the session belongs to the user
    cursor.execute("SELECT user_id FROM chat_sessions WHERE id = ?", (chat_id,))
    result = cursor.fetchone()
    if not result or result['user_id'] != user_id:
        return jsonify({'message': 'Unauthorized or session not found'}), 403

    cursor.execute("DELETE FROM chat_history WHERE session_id = ?", (chat_id,))
    cursor.execute("DELETE FROM chat_sessions WHERE id = ?", (chat_id,))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Chat session deleted successfully'})

@app.route('/api/clear-all-history', methods=['DELETE'])
def clear_all_history():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_history")
    cursor.execute("DELETE FROM chat_sessions")
    conn.commit()
    conn.close()

    return jsonify({'message': 'All chat history cleared successfully'})



@app.route('/api/chat-sessions', methods=['GET'])
@jwt_required()
def get_chat_sessions():
    user_id = get_jwt_identity()  # Assuming user ID is used as identity
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT cs.id, cs.session_name, cs.timestamp, ch.user_message, ch.bot_response
        FROM chat_sessions cs
        LEFT JOIN chat_history ch ON cs.id = ch.session_id
        WHERE cs.user_id = ?
        ORDER BY cs.timestamp DESC, ch.timestamp ASC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()

    chat_sessions = []
    current_session_id = None
    current_session = None

    for row in rows:
        if row['id'] != current_session_id:
            if current_session:
                chat_sessions.append(current_session)
            current_session_id = row['id']
            current_session = {
                'id': row['id'],
                'session_name': row['session_name'],
                'timestamp': row['timestamp'],
                'chat_history': []
            }
        current_session['chat_history'].append({
            'user_message': row['user_message'],
            'bot_response': row['bot_response']
        })

    if current_session:
        chat_sessions.append(current_session)

    return jsonify(chat_sessions)


@app.route('/api/new-chat', methods=['POST'])
@jwt_required()
def create_new_chat():
    user_id = get_jwt_identity()
    data = request.get_json()
    session_name = data.get('session_name', '')

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_sessions (user_id, session_name) VALUES (?, ?)", (user_id, session_name))
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return jsonify({'id': session_id, 'session_name': session_name, 'timestamp': time.time()})


@app.route('/api/update-chat-history', methods=['POST'])
def update_chat_history():
    data = request.get_json()
    selected_chat_id = data['selectedChatId']
    user_message = data['userMessage']
    bot_response = data['botResponse']

    print("Selected chat ID in /api/update-chat-history:", selected_chat_id)
    print("User message in /api/update-chat-history:", user_message)
    print("Bot response in /api/update-chat-history:", bot_response)

    conn = get_db_connection()
    cursor = conn.cursor()

    # Check if the user message already exists in the chat history
    cursor.execute("SELECT COUNT(*) FROM chat_history WHERE session_id = ? AND user_message = ?",
                   (selected_chat_id, user_message))
    count = cursor.fetchone()[0]

    if count == 0:
        # Insert the new user message and bot response into the chat history
        cursor.execute("INSERT INTO chat_history (session_id, user_message, bot_response) VALUES (?, ?, ?)",
                       (selected_chat_id, user_message, bot_response))
        conn.commit()

    conn.close()

    return jsonify({'message': 'Chat history updated successfully'})

@app.route('/api/chat-history/<int:chat_id>', methods=['GET'])
def get_chat_history(chat_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT user_message, bot_response FROM chat_history WHERE session_id = ? ORDER BY timestamp",
                   (chat_id,))
    chat_history = cursor.fetchall()
    conn.close()

    print("Chat ID:", chat_id)
    print("Chat history:", chat_history)

    chat_messages = [{'user_message': row['user_message'], 'bot_response': row['bot_response']} for row in chat_history]

    print("Chat messages:", chat_messages)

    return jsonify(chat_messages)

@app.route('/api/discussions', methods=['GET'])
def get_discussions():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT d.id, d.title, d.content, di.image_data, c.name AS category
            FROM discussions d
            LEFT JOIN discussion_images di ON d.id = di.discussion_id
            LEFT JOIN categories c ON d.category_id = c.id
        """)
        discussions = cursor.fetchall()
        discussions = [{
            'id': row[0],
            'title': row[1],
            'content': row[2],
            'images': [row[3]],
            'category': row[4]
        } for row in discussions]
        return jsonify(discussions)
    finally:
        conn.close()


@app.route('/api/discussions', methods=['POST'])
@jwt_required()
def create_discussion():
    user_id = get_jwt_identity()
    data = request.get_json()
    title = data['title']
    category_id = data['category_id']
    content = data['content']
    image_data = data['images']  # This should be a list of Base64 strings

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("INSERT INTO discussions (user_id, title, category_id, content) VALUES (?, ?, ?, ?)",
                   (user_id, title, category_id, content))
    discussion_id = cursor.lastrowid

    for base64_string in image_data:
        cursor.execute("INSERT INTO discussion_images (discussion_id, image_data) VALUES (?, ?)",
                       (discussion_id, base64_string))

    conn.commit()
    conn.close()


    print('Discussion created successfully:', {
        'title': title,
        'category_id': category_id,
        'content': content,
        'images': [f"Image {i+1}" for i in range(len(image_data))]
    })
    return jsonify({'message': 'Discussion created successfully', 'discussion_id': discussion_id})


@app.route('/api/discussions/<int:discussion_id>/replies', methods=['POST'])
@jwt_required()
def create_reply(discussion_id):
    user_id = get_jwt_identity()
    data = request.get_json()
    content = data['content']

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("INSERT INTO replies (user_id, discussion_id, content) VALUES (?, ?, ?)", (user_id, discussion_id, content))
    conn.commit()
    conn.close()

    return jsonify({'message': 'Reply created successfully'})


@app.route('/api/discussions/<int:discussion_id>/replies', methods=['GET'])
def get_replies(discussion_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, content, created_at, updated_at FROM replies WHERE discussion_id = ?", (discussion_id,))
    replies = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return jsonify(replies)

@app.route('/api/clear-discussions', methods=['DELETE'])
@jwt_required()
def clear_discussions():
    user_id = get_jwt_identity()  # Use this to add further logic if needed
    conn = get_db_connection()
    cursor = conn.cursor()

    # Restrict clearing to specific conditions, e.g., user role
    cursor.execute("DELETE FROM discussion_images WHERE discussion_id IN (SELECT id FROM discussions WHERE user_id = ?)", (user_id,))
    cursor.execute("DELETE FROM replies WHERE discussion_id IN (SELECT id FROM discussions WHERE user_id = ?)", (user_id,))
    cursor.execute("DELETE FROM discussions WHERE user_id = ?", (user_id,))

    conn.commit()
    conn.close()

    return jsonify({'message': 'All discussions cleared successfully'})


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # Device operations assumed handled externally
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def embed_text(text, tokenizer, model, max_length=512):
    tokens = tokenizer.tokenize(text)
    token_chunks = []

    while tokens:
        if len(tokens) > max_length:
            chunk, tokens = tokens[:max_length], tokens[max_length:]
        else:
            chunk, tokens = tokens, []
        token_chunks.append(chunk)

    embeddings = []
    for chunk in token_chunks:
        input_ids = tokenizer.convert_tokens_to_ids(chunk)
        attention_mask = [1] * len(input_ids)

        while len(input_ids) < max_length:
            input_ids.append(0)
            attention_mask.append(0)

        input_tensor = torch.tensor([input_ids]).to(device)
        attention_mask = torch.tensor([attention_mask]).to(device)

        with torch.no_grad():
            outputs = model(input_tensor, attention_mask=attention_mask)
            chunk_embedding = mean_pooling(outputs, attention_mask).cpu().numpy()
            embeddings.append(chunk_embedding)

    avg_embedding = np.mean(embeddings, axis=0)
    return avg_embedding

def check_meaning(input_string):
    start_time = time.time()
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant, skilled at checking if the query has meaning or if its gibberish. If you feel that the input is gibberish or does not mean anything, only output the word 'dontsearch', if the input does have a meaning then only output the word 'okay'"},
            {"role": "user", "content": input_string}
        ],
        max_tokens=100,  # Adjust as needed
        stop=["\n"]
    )

    end_time = time.time()

    if response.choices[0].message.content is not None:
        response_content = response.choices[0].message.content.strip()
    else:
        response_content = ""

    response_time = end_time - start_time
    return response_content, response_time



def expand_query_for_search(input_string):
    start_time = time.time()
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant, skilled at expanding brief ideas into detailed, search-friendly descriptions for semantic searching."},
            {"role": "user", "content": input_string}
        ],
        max_tokens=100,  # Adjust as needed
        stop=["\n"]
    )

    end_time = time.time()

    if response.choices[0].message.content is not None:
        response_content = response.choices[0].message.content.strip()
    else:
        response_content = ""

    response_time = end_time - start_time
    return response_content, response_time


@app.route('/search', methods=['POST'])
def search():
    content = request.get_json()
    query = content.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    res, expansion_time = check_meaning(query)
    if res == 'okay':
        if len(query.split()) < 3:
            expanded_idea, expansion_time = expand_query_for_search(query)
            print(f"Expanded idea: {expanded_idea} (expansion took {expansion_time} seconds)")
            if expanded_idea:
                query = expanded_idea
    elif res == 'dontsearch':
        return jsonify({"status": "error", "message": "Search resulted in no similar patents."}), 400

    # Embed the query
    query_embedding = embed_text(query, tokenizer, model).astype('float32').reshape(1, -1)


    # Search in the FAISS index
    D, I = index.search(query_embedding, k=10)  # Search for the top 10 results

    # Prepare results
    results = []
    for idx, distance in zip(I[0], D[0]):
        row = df.iloc[idx]
        result = {
            "Title": row['Title'] if 'Title' in df.columns else 'Title not found',
            "PDF_Title": row['PDF_Title'] if 'PDF_Title' in df.columns else 'PDF title not found',
            "Author": str(row['Author'] if 'Author' in df.columns else 'Author not found'),
            "Pages": row['Pages'] if 'Pages' in df.columns else 'Page count not found',
            "URL": row['URL'] if 'URL' in df.columns else 'URL not found',
            "Genre": row['Genre'] if 'Genre' in df.columns else 'Genre not found',
            "Page_Number": row['Page_Number'] if 'Page_Number' in df.columns else 'Page number not found',
            "Page_Text": row['Page_Text'] if 'Page_Text' in df.columns else 'Page text not found',
            "Distance": float(distance)
        }
        results.append(result) 
    return jsonify(results)



@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    response = requests.post('http://localhost:5001/translate', json={
        'q': data['text'],
        'source': data['source'],
        'target': data['target']
    })
    return jsonify(response.json())


@app.route('/api/generate-practice-questions', methods=['POST'])
def generate_practice_questions():
    data = request.get_json()
    page_text = data['pageText']

    try:
        # Generate practice questions using OpenAI API
        questions = generate_questions(page_text)
        return jsonify({'questions': questions})
    except Exception as e:
        print(f"Error generating practice questions: {str(e)}")
        return jsonify({'error': 'Failed to generate practice questions'}), 500

@app.route('/api/submit-answer', methods=['POST'])
def submit_answer():
    data = request.get_json()
    question_id = data['questionId']
    answer = data['answer']
    page_text = data['pageText']

    try:
        # Generate feedback using OpenAI API
        feedback = generate_feedback(question_id, answer, page_text)
        return jsonify({'feedback': feedback})
    except Exception as e:
        print(f"Error generating feedback: {str(e)}")
        return jsonify({'error': 'Failed to generate feedback'}), 500


def generate_questions(page_text):
    try:
        messages = [
            {"role": "system", "content": "You are a question generation assistant."},
            {"role": "user", "content": f"Generate 10 practice questions based on the following text:\n\n{page_text}\n\nQuestions:"}
        ]

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1000,
            n=1,
            stop=None,
            temperature=0.7,
        )

        questions_text = response.choices[0].message.content.strip()
        questions = [{"id": str(uuid4()), "text": q.strip()} for q in questions_text.split("\n")]
        return questions[:10]

    except Exception as e:
        print(f"Error in generate_questions: {str(e)}")
        raise

def generate_feedback(question_id, answer, page_text):
    try:
        messages = [
            {"role": "system", "content": "You are a feedback generation assistant."},
            {"role": "user", "content": f"Question ID: {question_id}\nAnswer: {answer}\n\nBased on the following text, provide feedback on the answer:\n\n{page_text}\n\nFeedback:"}
        ]

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.7,
        )

        feedback = response.choices[0].message.content.strip()
        return feedback

    except Exception as e:
        print(f"Error in generate_feedback: {str(e)}")
        raise

@app.route('/api/generate-video-search-query', methods=['POST'])
def generate_video_search_query():
    data = request.get_json()
    search_query = data['searchQuery']
    page_text = data['pageText']

    messages = [
        {"role": "system", "content": "You are a helpful assistant skilled at generating video search queries based on a given search query and page text. Focus on generating queries that yield the most relevant educational videos related to the book content."},
        {"role": "user", "content": f"Search Query: {search_query}\nPage Text: {page_text}\n\nGenerate a video search query based on the given search query and page text, optimized for finding relevant educational videos, if the query does not seem education, just make it education and do not worry about to the page context:"}
    ]

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7,
    )

    video_search_query = response.choices[0].message.content.strip()
    return jsonify({'videoSearchQuery': video_search_query})

@app.route('/api/search-related-videos', methods=['POST'])
def search_related_videos():
    data = request.get_json()
    video_search_query = data['videoSearchQuery']

    youtube = build('youtube', 'v3', developerKey='AIzaSyDCPs8Bbux92m9cMZTFcOGSEJe2y0b_kxI')
    search_response = youtube.search().list(
        q=video_search_query,
        type='video',
        part='id,snippet',
        maxResults=25
    ).execute()

    related_videos = []
    for search_result in search_response.get('items', []):
        video_id = search_result['id']['videoId']
        video_title = search_result['snippet']['title']
        video_thumbnail = search_result['snippet']['thumbnails']['default']['url']
        video_url = f"https://www.youtube.com/embed/{video_id}"
        related_videos.append({
            'title': video_title,
            'thumbnail': video_thumbnail,
            'url': video_url
        })

    return jsonify({'videos': related_videos})


def generate_response_page(conversation_history, page_text):
    start_time = time.time()

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant trained on the following page text:"},
        {"role": "system", "content": page_text},
    ]
    messages.extend(conversation_history)

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=500,  # Adjust as needed
        n=1,
        stop=None,
        temperature=0.7,
    )

    end_time = time.time()

    if response.choices[0].message.content is not None:
        response_content = response.choices[0].message.content.strip()
    else:
        response_content = ""

    response_time = end_time - start_time

    return response_content, response_time

@app.route('/api/chatbot-page', methods=['POST'])
def chatbot_page():
    data = request.get_json()
    user_message = data['message']
    page_text = data['pageText']

    conversation_history = [{"role": "user", "content": user_message}]

    # Generate response using OpenAI API
    response_content, response_time = generate_response_page(conversation_history, page_text)

    # Save the user message, page text, and bot response to the database (optional)
    # ...

    return jsonify({'response': response_content, 'response_time': response_time})

if __name__ == '__main__':
    # Initialize the database
    initialize_database()
    app.run(debug=False, host='0.0.0.0', port=5000)
