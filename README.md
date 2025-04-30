
# Voice Authentication System

A secure, web-based Voice Authentication System that combines password-based authentication, voice biometrics, and security question verification to provide a robust user authentication mechanism. Users register with their voice samples, answer security questions, and authenticate using voice recognition powered by machine learning. This project was developed as a final-year project to demonstrate advanced concepts in web development, audio processing, and machine learning.

---

## Project Overview

The Voice Authentication System enhances traditional password-based login systems by incorporating voice biometrics. Users register by providing a user ID, password, and recording voice samples for three security questions. During login, users verify their identity using a password and voice input. A forgot-password feature allows password reset via voice verification.

### Highlights:

- **Web Development**: Flask-based responsive web application.  
- **Audio Processing**: Real-time audio handling via `SpeechRecognition` and `Librosa`.  
- **Machine Learning**: Voice verification using SpeechBrain’s ECAPA-TDNN and a custom CNN.  
- **Database**: User and voice data stored in SQLite.  
- **Security**: Password strength validation, session management, and safe file handling.

---

## Features

### 🔐 User Registration:
- Unique user ID and strong password.
- Select 3 security questions and answer them with 10 voice samples each.

### 🔊 Voice-Based Login:
- Authenticate via password + voice answers to security questions.
- One voice sample per question for verification.

### 🔁 Forgot Password:
- Reset password by answering 3 security questions with voice.

### 🎨 Responsive UI:
- Dark/light theme toggle.
- Mobile-friendly, gradient-based design.

### 🔒 Security:
- Password strength indicator.
- Session-based access control.
- Temporary audio files auto-deleted post-verification.

### 🤖 Machine Learning:
- Pretrained ECAPA-TDNN for voice embeddings.
- Custom CNN on mel-spectrograms for answer verification.

### 🔄 Feedback:
- Real-time sample status and processing spinners.

---

## Tech Stack

### Backend:
- Python 3.8  
- Flask 2.0.1  
- SQLite  

### Frontend:
- HTML5  
- CSS3  
- JavaScript  

### Machine Learning & Audio:
- SpeechBrain 0.5.10  
- TensorFlow 2.8.0  
- Librosa 0.8.1  
- SpeechRecognition 3.8.1  
- SoundFile 0.10.3  
- NumPy 1.21.0  
- PyTorch 1.10.0  

---

## Installation

### Prerequisites
- Python 3.8+  
- Git  
- Microphone  
- (Optional) GPU for model acceleration  

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/[YourUsername]/voice-authentication-system.git
   cd voice-authentication-system
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   `requirements.txt` should include:
   ```
   flask==2.0.1
   speechrecognition==3.8.1
   soundfile==0.10.3
   numpy==1.21.0
   librosa==0.8.1
   tensorflow==2.8.0
   torch==1.10.0
   speechbrain==0.5.10
   python-dotenv==0.20.0
   ```

   **Note:**  
   Install `PyAudio` separately:
   ```bash
   pip install pyaudio
   ```

3. **Set Environment Variables**:
   Create a `.env` file:
   ```
   FLASK_SECRET_KEY=your_secure_key
   ```

   Generate a key using:
   ```python
   import secrets
   print(secrets.token_hex(16))
   ```

4. **Run the App**:
   ```bash
   python app.py
   ```

5. **Access**:  
   Go to [http://localhost:5000](http://localhost:5000)

---

## Usage

### 🏠 Home Page
- Choose **Register**, **Login**, or **Forgot Password**.

### 📝 Registration
- Provide user ID and password.
- Choose 3 questions and record 10 samples each.
- Submit and complete setup.

### 🔐 Login
- Enter user ID and password.
- Answer 3 questions with voice.
- Get redirected to the dashboard on success.

### 🔁 Forgot Password
- Verify identity with voice.
- Set a new password after successful verification.

### 📋 Dashboard
- View user ID and logout option.

### 🌗 Theme Toggle
- Switch between dark and light modes.

---

## Screenshots

_Replace placeholders with real screenshots:_

- Home Page  
- Registration  
- Login  
- Forgot Password  
- Dashboard  

---

## Challenges and Solutions

## Challenges and Solutions

**Challenge:** Accurate voice recognition in noisy environments.  
**Solution:** Implemented ambient noise adjustment using SpeechRecognition’s `adjust_for_ambient_noise` and an energy threshold check (`ENERGY_THRESHOLD = 0.0001`) to filter out silent recordings.

**Challenge:** Handling large audio files and model training.  
**Solution:** Used SpeechBrain’s pretrained ECAPA-TDNN model to reduce training time and a lightweight CNN for question-specific verification. Audio files are resampled to 16kHz for efficiency.

**Challenge:** Ensuring text and voice answer alignment.  
**Solution:** Integrated Google Speech-to-Text for spoken answer verification and used `SequenceMatcher` for text similarity (threshold ≥ 0.9).

**Challenge:** Secure storage of sensitive data.  
**Solution:** Stored passwords in plaintext (to be improved with hashing) and excluded `temp/`, `users/`, and `voice_auth.db` from Git using `.gitignore`.

**Challenge:** Model adaptability and up-to-date embeddings.  
**Solution:** The system retrains the custom CNN model and updates the user’s speaker embeddings each time the user logs in successfully. This helps keep the model in sync with recent voice patterns, improving authentication accuracy over time.

---

## Future Improvements

- 🔐 Password hashing with bcrypt  
- 🔑 Add OTP-based MFA (email/SMS)  
- 🎧 Advanced noise cancellation  
- ☁️ Deploy to Heroku, Render, or AWS  
- 🧪 Add unit testing with `pytest`  
- ♿ Improve accessibility with ARIA labels  
- 📊 Track login attempts with analytics  
- 🚀 Optimize ECAPA-TDNN or use lightweight alternatives  
- 🛠️ Admin panel for user logs  
- 🎙️ Real-time audio feedback with progress indicators  

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Contact

**Author:** K. Lokesh Kumar
**Email:** lokeshkumarkona07@gmail.com  
**LinkedIn:** (https://www.linkedin.com/in/kona-lokesh-kumar-57b658344)

_For feedback or questions, raise an issue or reach out via email._

Thanks for checking out the Voice Authentication System! 🚀  
