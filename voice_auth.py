# Suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings('ignore')

import re
import shutil
import pickle
import numpy as np
import librosa
import soundfile as sf
import speech_recognition as sr
import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore
from sklearn.model_selection import train_test_split
import torch
from speechbrain.pretrained import SpeakerRecognition  # type: ignore
from difflib import SequenceMatcher
import sqlite3
from sqlite3 import Error

# Set up logging
logging.basicConfig(filename='voice_auth.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Base directory for all user data
BASE_DIR = "users"
DATABASE = "voice_auth.db"

# Ensure directories exist
if not os.path.exists(BASE_DIR):
    print(f"Creating directory: {BASE_DIR}")
    os.makedirs(BASE_DIR)

# Initialize SpeechBrain model for speaker identification
speaker_recognizer = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

# Audio processing parameters
SAMPLE_RATE = 22050
DURATION = 5  # seconds
N_MELS = 64
HOP_LENGTH = 512
N_FFT = 1024
SAMPLES_PER_QUESTION = 10
ENERGY_THRESHOLD = 0.0001

# Database setup
def create_connection():
    try:
        conn = sqlite3.connect(DATABASE)
        return conn
    except Error as e:
        print(f"Error connecting to database: {e}")
    return None

def create_tables():
    conn = create_connection()
    if conn is None:
        print("Failed to create database connection.")
        return

    try:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                password TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                question TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER,
                answer TEXT NOT NULL,
                FOREIGN KEY (question_id) REFERENCES questions (id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audio_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                question_id INTEGER,
                sample_path TEXT NOT NULL,
                sample_type TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (user_id),
                FOREIGN KEY (question_id) REFERENCES questions (id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                user_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_mapping (
                user_id TEXT PRIMARY KEY,
                label INTEGER NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        conn.commit()
    except Error as e:
        print(f"Error creating tables: {e}")
    finally:
        conn.close()

# Only create tables if they don't exist (no reset)
create_tables()

def clean_filename(text):
    return re.sub(r'[<>:"/\\|?*]', '', text)

def add_noise(audio_data, noise_factor=0.015):
    noise = np.random.randn(len(audio_data))
    augmented_data = audio_data + noise_factor * noise
    return augmented_data

def pitch_shift(audio_data, sr, n_steps=3):
    return librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=n_steps)

def audio_to_spectrogram(audio_file, augment=False):
    try:
        audio_data, sr = sf.read(audio_file)
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        audio_data = librosa.util.fix_length(audio_data, size=SAMPLE_RATE * DURATION)

        energy = np.mean(np.abs(audio_data))
        if energy < ENERGY_THRESHOLD:
            print(f"No speech detected (energy {energy:.6f} < {ENERGY_THRESHOLD}).")
            return None

        audio_data = librosa.util.normalize(audio_data)
        if augment:
            audio_data = add_noise(audio_data)
            audio_data = pitch_shift(audio_data, SAMPLE_RATE)

        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_data, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT
        )
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrogram_db = (mel_spectrogram_db - mel_spectrogram_db.min()) / (mel_spectrogram_db.max() - mel_spectrogram_db.min() + 1e-10)
        mel_spectrogram_db = np.expand_dims(mel_spectrogram_db, axis=-1)
        return mel_spectrogram_db
    except Exception as e:
        print(f"Error converting to spectrogram for {audio_file}: {e}")
        return None

def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_speech_cnn(user_id, spectrograms, answer_labels):
    user_dir = os.path.join(BASE_DIR, user_id)
    valid_indices = [i for i, label in enumerate(answer_labels) if label != -1]
    spectrograms = [spectrograms[i] for i in valid_indices]
    answer_labels = [answer_labels[i] for i in valid_indices]

    spectrograms = np.array(spectrograms)
    answer_labels = np.array(answer_labels)

    if len(spectrograms) == 0:
        print("No valid samples to train speech CNN. Skipping training.")
        return None

    print(f"Training speech CNN with {len(spectrograms)} samples, shape: {spectrograms.shape}")
    X_train, X_val, y_train, y_val = train_test_split(spectrograms, answer_labels, test_size=0.1, random_state=42)
    input_shape = (spectrograms.shape[1], spectrograms.shape[2], 1)
    model = build_cnn_model(input_shape, num_classes=3)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=16, verbose=1)
    model.save(os.path.join(user_dir, "speech_cnn_model.keras"))
    return model

def resample_audio(audio_data, original_sr, target_sr=16000):
    if original_sr != target_sr:
        audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sr)
    return audio_data

def load_all_users_samples():
    user_mapping = {}
    user_embeddings = {}
    label = 0

    conn = create_connection()
    if conn is None:
        return user_mapping, user_embeddings

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM users")
        users = cursor.fetchall()

        for user_id_tuple in users:
            user_id = user_id_tuple[0]
            user_mapping[user_id] = label
            embeddings = []

            cursor.execute("SELECT id, question FROM questions WHERE user_id = ?", (user_id,))
            questions = cursor.fetchall()

            for q_id, question in questions:
                cursor.execute("SELECT sample_path FROM audio_samples WHERE user_id = ? AND question_id = ? AND sample_type = 'registration'", (user_id, q_id))
                samples = cursor.fetchall()
                for sample_path_tuple in samples:
                    filename = sample_path_tuple[0]
                    if os.path.exists(filename):
                        try:
                            audio_data, sr = sf.read(filename)
                            if len(audio_data.shape) > 1:
                                audio_data = np.mean(audio_data, axis=1)
                            audio_data = resample_audio(audio_data, sr)
                            audio_tensor = torch.tensor(audio_data).float()
                            embedding = speaker_recognizer.encode_batch(audio_tensor)
                            embeddings.append(embedding.squeeze().numpy())
                        except Exception as e:
                            print(f"Error computing embedding for {filename}: {e}")

            cursor.execute("SELECT sample_path FROM audio_samples WHERE user_id = ? AND sample_type = 'login'", (user_id,))
            login_samples = cursor.fetchall()
            for sample_path_tuple in login_samples:
                filename = sample_path_tuple[0]
                if os.path.exists(filename):
                    try:
                        audio_data, sr = sf.read(filename)
                        if len(audio_data.shape) > 1:
                            audio_data = np.mean(audio_data, axis=1)
                        audio_data = resample_audio(audio_data, sr)
                        audio_tensor = torch.tensor(audio_data).float()
                        embedding = speaker_recognizer.encode_batch(audio_tensor)
                        embeddings.append(embedding.squeeze().numpy())
                    except Exception as e:
                        print(f"Error computing embedding for {filename}: {e}")

            if embeddings:
                user_embeddings[user_id] = np.mean(embeddings, axis=0)
                embedding_blob = pickle.dumps(user_embeddings[user_id])
                cursor.execute("INSERT OR REPLACE INTO embeddings (user_id, embedding) VALUES (?, ?)", (user_id, embedding_blob))
            label += 1

        for user_id, user_label in user_mapping.items():
            cursor.execute("INSERT OR REPLACE INTO user_mapping (user_id, label) VALUES (?, ?)", (user_id, user_label))

        conn.commit()
    except Error as e:
        print(f"Error loading user samples: {e}")
    finally:
        conn.close()

    return user_mapping, user_embeddings

def load_user_samples(user_id):
    spectrograms = []
    answer_labels = []

    conn = create_connection()
    if conn is None:
        return spectrograms, answer_labels

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, question FROM questions WHERE user_id = ?", (user_id,))
        questions = cursor.fetchall()

        for q_idx, (q_id, question) in enumerate(questions[:3]):
            cursor.execute("SELECT sample_path FROM audio_samples WHERE user_id = ? AND question_id = ? AND sample_type = 'registration'", (user_id, q_id))
            samples = cursor.fetchall()
            for sample_path_tuple in samples:
                filename = sample_path_tuple[0]
                if os.path.exists(filename):
                    spectrogram = audio_to_spectrogram(filename, augment=False)
                    if spectrogram is not None:
                        spectrograms.append(spectrogram)
                        answer_labels.append(q_idx)

        cursor.execute("SELECT sample_path FROM audio_samples WHERE user_id = ? AND sample_type = 'login'", (user_id,))
        login_samples = cursor.fetchall()
        for sample_path_tuple in login_samples:
            filename = sample_path_tuple[0]
            if os.path.exists(filename):
                spectrogram = audio_to_spectrogram(filename, augment=False)
                if spectrogram is not None:
                    spectrograms.append(spectrogram)
                    answer_labels.append(-1)

    except Error as e:
        print(f"Error loading user samples for {user_id}: {e}")
    finally:
        conn.close()

    return spectrograms, answer_labels

def get_unique_filename(base_path, user_id, idx):
    base_name = os.path.join(base_path, user_id, f"login_sample_{idx}.wav")
    counter = idx
    while os.path.exists(base_name):
        counter += 1
        base_name = os.path.join(base_path, user_id, f"login_sample_{counter}.wav")
    return base_name, counter

class VoiceAuth:
    def __init__(self):
        self.SAMPLES_PER_QUESTION = 10
        self.recognizer = sr.Recognizer()  # Already initialized globally, but kept for clarity

    def check_user_exists(self, user_id):
        """Check if a user_id already exists in the database."""
        conn = create_connection()
        if conn is None:
            print("Database connection failed.")
            return False
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
            return cursor.fetchone() is not None
        except Error as e:
            print(f"Error checking user existence: {e}")
            return False
        finally:
            conn.close()

    def validate_password(self, password):
        if len(password) < 8:
            return False, "Password must be at least 8 characters long."
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter."
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter."
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False, "Password must contain at least one special character (!@#$%^&*(),.?\":{}|<>)."
        if not re.search(r'\d', password):
            return False, "Password must contain at least one number."
        return True, "Password is valid."

    def register(self, user_id, password, selected_questions, user_answers, audio_files):
        user_dir = os.path.join(BASE_DIR, user_id)
        conn = create_connection()
        if conn is None:
            return False, "Database connection failed."

        try:
            cursor = conn.cursor()
            # Check for duplicate user_id
            if self.check_user_exists(user_id):
                return False, "User ID already exists. Please choose a different ID."

            # Validate password
            success, message = self.validate_password(password)
            if not success:
                return False, message

            os.makedirs(user_dir, exist_ok=True)
            cursor.execute("INSERT INTO users (user_id, password) VALUES (?, ?)", (user_id, password))

            spectrograms = []
            answer_labels = []
            embeddings = []

            print(f"Processing audio files for user {user_id}: {audio_files}")
            for q_idx, question in enumerate(selected_questions):
                answers = user_answers[question]
                cursor.execute("INSERT INTO questions (user_id, question) VALUES (?, ?)", (user_id, question))
                question_id = cursor.lastrowid
                cursor.execute("INSERT INTO answers (question_id, answer) VALUES (?, ?)", (question_id, answers[0]))

                question_audio_files = audio_files[q_idx]
                print(f"Question {q_idx+1}: {len(question_audio_files)} audio files received")
                for sample_idx in range(self.SAMPLES_PER_QUESTION):
                    audio_file = question_audio_files[sample_idx]
                    if not os.path.exists(audio_file):
                        print(f"Audio file {audio_file} does not exist.")
                        continue

                    audio_data, sr_rate = sf.read(audio_file)
                    if len(audio_data.shape) > 1:
                        audio_data = np.mean(audio_data, axis=1)
                    energy = np.mean(np.abs(audio_data))
                    if energy < ENERGY_THRESHOLD:
                        print(f"No speech detected in {audio_file} (energy {energy:.6f} < {ENERGY_THRESHOLD}).")
                        continue

                    audio_data_resampled = resample_audio(audio_data, sr_rate)
                    audio_tensor = torch.tensor(audio_data_resampled).float()
                    embedding = speaker_recognizer.encode_batch(audio_tensor)
                    embeddings.append(embedding.squeeze().numpy())

                    spectrogram = audio_to_spectrogram(audio_file, augment=True)
                    if spectrogram is None:
                        print(f"Failed to process audio sample for {audio_file}.")
                        continue

                    filename = os.path.join(user_dir, f"{clean_filename(question)}_sample{sample_idx}.wav")
                    os.rename(audio_file, filename)

                    cursor.execute("INSERT INTO audio_samples (user_id, question_id, sample_path, sample_type) VALUES (?, ?, ?, ?)",
                                   (user_id, question_id, filename, 'registration'))

                    spectrograms.append(spectrogram)
                    answer_labels.append(q_idx)

            print(f"Total embeddings collected: {len(embeddings)}")
            print(f"Total spectrograms collected: {len(spectrograms)}")
            if len(embeddings) >= self.SAMPLES_PER_QUESTION * 3 and len(spectrograms) >= self.SAMPLES_PER_QUESTION * 3:
                train_speech_cnn(user_id, spectrograms, answer_labels)

                user_embedding = np.mean(embeddings, axis=0)
                user_mapping, user_embeddings = load_all_users_samples()
                new_label = len(user_mapping)
                user_mapping[user_id] = new_label
                user_embeddings[user_id] = user_embedding

                embedding_blob = pickle.dumps(user_embedding)
                cursor.execute("INSERT INTO embeddings (user_id, embedding) VALUES (?, ?)", (user_id, embedding_blob))
                cursor.execute("INSERT INTO user_mapping (user_id, label) VALUES (?, ?)", (user_id, new_label))

                conn.commit()
                print(f"User {user_id} registered with password {password}")
                return True, "Registration Successful! Data stored in database and embeddings updated."
            else:
                print(f"Not enough valid audio samples to register. Collected {len(embeddings)} embeddings and {len(spectrograms)} spectrograms/{self.SAMPLES_PER_QUESTION * 3} samples.")
                shutil.rmtree(user_dir)
                cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
                conn.commit()
                return False, "Not enough valid audio samples to register."

        except Error as e:
            print(f"Error during registration: {e}")
            if os.path.exists(user_dir):
                shutil.rmtree(user_dir)
            return False, f"Error during registration: {e}"
        finally:
            conn.close()

    def get_security_questions(self, user_id):
        conn = create_connection()
        if conn is None:
            return False, "Database connection failed.", []
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
            if not cursor.fetchone():
                return False, "User not found.", []
            cursor.execute("SELECT question FROM questions WHERE user_id = ?", (user_id,))
            questions = [row[0] for row in cursor.fetchall()]
            if len(questions) != 3:
                return False, "Security questions not found.", []
            return True, "Questions retrieved successfully.", questions
        finally:
            conn.close()

    def login(self, user_id, password):
        conn = create_connection()
        if conn is None:
            return False, "Database connection failed.", []
        try:
            print(f"Login attempt for user_id: '{user_id}'")
            cursor = conn.cursor()
            cursor.execute("SELECT password FROM users WHERE user_id = ?", (user_id,))
            result = cursor.fetchone()
            print(f"Database result for {user_id}: {result}")
            if not result:
                return False, "User not found.", []
            stored_password = result[0]
            if stored_password != password:
                return False, "Incorrect password.", []
            success, message, questions = self.get_security_questions(user_id)
            if not success:
                return False, message, []
            return True, "Password verified. Please answer security questions.", questions
        finally:
            conn.close()

    def forgot_password(self, user_id):
        print(f"Forgot password attempt for user_id: '{user_id}'")
        return self.get_security_questions(user_id)

    def text_similarity(self, text1, text2):
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def verify(self, user_id, answers, audio_files, is_login=True):
        conn = create_connection()
        if conn is None:
            return False, "Database connection failed."
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id, question FROM questions WHERE user_id = ?", (user_id,))
            questions = cursor.fetchall()
            if len(questions) != 3:
                return False, "Security questions not found."

            user_dir = os.path.join(BASE_DIR, user_id)
            try:
                speech_cnn = models.load_model(os.path.join(user_dir, "speech_cnn_model.keras"))
                cursor.execute("SELECT embedding FROM embeddings WHERE user_id = ?", (user_id,))
                result = cursor.fetchone()
                if not result:
                    return False, "User embedding not found."
                user_embedding = pickle.loads(result[0])
            except Exception as e:
                print(f"Error loading models or embeddings: {e}")
                return False, f"Error loading models or embeddings: {e}"

            successful_matches = 0
            new_spectrograms = []
            new_answer_labels = []
            new_embeddings = []

            cursor.execute("SELECT COUNT(*) FROM audio_samples WHERE user_id = ? AND sample_type = 'login'", (user_id,))
            login_sample_idx = cursor.fetchone()[0]

            question_ids = {q[1]: q[0] for q in questions}

            for q_idx, (question_id, question) in enumerate(questions):
                cursor.execute("SELECT answer FROM answers WHERE question_id = ?", (question_id,))
                result = cursor.fetchone()
                if not result:
                    print(f"No answer found for question_id {question_id}")
                    continue
                stored_answer = result[0]
                typed_answer = answers[q_idx]
                text_sim = self.text_similarity(typed_answer, stored_answer)
                text_pass = text_sim >= 0.9

                if not text_pass:
                    print(f"Text similarity too low for question {q_idx+1}: {text_sim:.2%}")
                    continue

                # Check if audio file exists for this question
                if q_idx >= len(audio_files):
                    print(f"No audio file provided for question {q_idx+1} ({question})")
                    continue

                audio_file = audio_files[q_idx]
                if not os.path.exists(audio_file):
                    print(f"Audio file {audio_file} does not exist for question {q_idx+1}")
                    continue

                audio_data, sr_rate = sf.read(audio_file)
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                energy = np.mean(np.abs(audio_data))
                if energy < ENERGY_THRESHOLD:
                    print(f"No speech detected in {audio_file} (energy {energy:.6f} < {ENERGY_THRESHOLD}).")
                    continue

                audio_data_resampled = resample_audio(audio_data, sr_rate)
                audio_tensor = torch.tensor(audio_data_resampled).float()
                login_embedding = speaker_recognizer.encode_batch(audio_tensor).squeeze().numpy()

                cosine_sim = np.dot(login_embedding, user_embedding) / (np.linalg.norm(login_embedding) * np.linalg.norm(user_embedding))
                voice_pass = cosine_sim >= 0.55

                spectrogram = audio_to_spectrogram(audio_file, augment=False)
                if spectrogram is None:
                    print("Failed to process audio.")
                    continue

                spectrogram = np.expand_dims(spectrogram, axis=0)
                speech_pred = speech_cnn.predict(spectrogram, verbose=0)
                speech_confidence = np.max(speech_pred)
                speech_label = np.argmax(speech_pred)
                speech_pass = speech_confidence >= 0.7 and speech_label == q_idx

                spoken_text_pass = False
                try:
                    with sr.AudioFile(audio_file) as source:
                        audio = self.recognizer.record(source)
                    spoken_text = self.recognizer.recognize_google(audio, language='en-US').lower()
                    spoken_text_sim = self.text_similarity(spoken_text, stored_answer)
                    spoken_text_pass = spoken_text_sim >= 0.9
                except sr.UnknownValueError:
                    print("Google Speech-to-Text could not understand the audio.")
                except sr.RequestError as e:
                    print(f"Google Speech-to-Text request failed: {e}")
                except Exception as e:
                    print(f"Error with Google Speech-to-Text: {e}")

                if (voice_pass or speech_pass) and text_pass and spoken_text_pass:
                    successful_matches += 1
                    new_filename, new_idx = get_unique_filename(BASE_DIR, user_id, login_sample_idx)
                    logging.info(f"Renaming {audio_file} to {new_filename}")
                    os.rename(audio_file, new_filename)
                    cursor.execute("INSERT INTO audio_samples (user_id, question_id, sample_path, sample_type) VALUES (?, ?, ?, ?)",
                                   (user_id, question_ids[question], new_filename, 'login'))
                    new_spectrograms.append(spectrogram[0])
                    new_answer_labels.append(q_idx)
                    new_embeddings.append(login_embedding)
                    login_sample_idx = new_idx + 1  # Update for the next iteration
                    print(voice_pass, speech_pass, text_pass, spoken_text_pass)

            if successful_matches >= 2 if is_login else successful_matches == 3:
                if is_login and new_spectrograms:
                    user_spectrograms, user_answer_labels = load_user_samples(user_id)
                    combined_spectrograms = user_spectrograms + new_spectrograms
                    combined_answer_labels = user_answer_labels + new_answer_labels
                    train_speech_cnn(user_id, combined_spectrograms, combined_answer_labels)

                    user_embedding = np.mean([user_embedding] + new_embeddings, axis=0)
                    embedding_blob = pickle.dumps(user_embedding)
                    cursor.execute("INSERT OR REPLACE INTO embeddings (user_id, embedding) VALUES (?, ?)", (user_id, embedding_blob))

                conn.commit()
                return True, "Verification successful."
            else:
                return False, f"Verification failed. Successful matches: {successful_matches}/3"

        except Error as e:
            print(f"Error during verification: {e}")
            return False, f"Error during verification: {e}"
        finally:
            conn.close()

    def reset_password(self, user_id, new_password):
        conn = create_connection()
        if conn is None:
            return False, "Database connection failed."
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
            if not cursor.fetchone():
                return False, "User not found."
            success, message = self.validate_password(new_password)
            if not success:
                return False, message
            cursor.execute("UPDATE users SET password = ? WHERE user_id = ?", (new_password, user_id))
            conn.commit()
            return True, "Password reset successful."
        finally:
            conn.close()