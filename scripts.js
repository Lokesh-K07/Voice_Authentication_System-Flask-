let currentQuestionIndex = -1;
let currentSampleIndex = 0;
const samplesPerQuestion = 10;
const loginSamplesPerQuestion = 1;
let recordedAudios = Array(3).fill().map(() => Array(samplesPerQuestion).fill(null));

// Function to check password strength
function checkPasswordStrength(password) {
    let strength = 0;
    if (password.length >= 8) strength += 1;
    if (/[A-Z]/.test(password)) strength += 1;
    if (/[a-z]/.test(password)) strength += 1;
    if (/[!@#$%^&*(),.?":{}|<>]/.test(password)) strength += 1;
    if (/\d/.test(password)) strength += 1;

    const strengthElement = document.getElementById('password-strength');
    if (strengthElement) {
        strengthElement.classList.remove('weak', 'medium', 'strong', 'error');

        if (password.length === 0) {
            strengthElement.textContent = '';
            return;
        }

        if (strength <= 2) {
            strengthElement.textContent = 'Weak Password';
            strengthElement.classList.add('weak');
        } else if (strength <= 4) {
            strengthElement.textContent = 'Medium Password';
            strengthElement.classList.add('medium');
        } else {
            strengthElement.textContent = 'Strong Password';
            strengthElement.classList.add('strong');
        }
    }
}

function showLoadingScreen() {
    const loadingScreen = document.getElementById('loading-screen');
    if (loadingScreen) {
        loadingScreen.style.display = 'flex';
    }
}

function hideLoadingScreen() {
    const loadingScreen = document.getElementById('loading-screen');
    if (loadingScreen) {
        loadingScreen.style.display = 'none';
    }
}

async function startRecording(questionIndex, userId, isLoginOrForgot = false) {
    if (currentQuestionIndex !== -1) {
        alert("Please wait until the current recording is finished.");
        return;
    }

    currentQuestionIndex = questionIndex;
    if (!recordedAudios[currentQuestionIndex]) {
        recordedAudios[currentQuestionIndex] = Array(samplesPerQuestion).fill(null);
    }

    const maxSamples = isLoginOrForgot ? loginSamplesPerQuestion : samplesPerQuestion;
    const statusElement = document.getElementById(`recordingStatus_${currentQuestionIndex}`);
    const startButton = document.querySelector(`button.start-recording[data-index="${currentQuestionIndex}"]`);

    for (currentSampleIndex = 0; currentSampleIndex < maxSamples;) {
        startButton.classList.add('recording');
        statusElement.textContent = `Recording sample ${currentSampleIndex + 1}... Speak now.`;
        startButton.disabled = true;

        try {
            const isForgotPassword = document.getElementById('forgotVerifyForm') !== null;
            const endpoint = isForgotPassword
                ? `/forgot_password/record/${userId}/${currentQuestionIndex}`
                : isLoginOrForgot
                ? `/login/record/${userId}/${currentQuestionIndex}`
                : `/record_audio/${userId}/${currentQuestionIndex}/${currentSampleIndex}`;
            console.log(`Recording endpoint: ${endpoint}`);
            const response = await fetch(endpoint, {
                method: 'POST'
            });
            const result = await response.json();

            if (result.success) {
                recordedAudios[currentQuestionIndex][currentSampleIndex] = result.audio_path;
                if (!isLoginOrForgot) {
                    document.getElementById(`sample-counter-${currentQuestionIndex}`).textContent = `Samples Recorded: ${currentSampleIndex + 1}/${samplesPerQuestion}`;
                } else {
                    document.getElementById(`sample-counter-${currentQuestionIndex}`).textContent = `Sample ${currentSampleIndex + 1}/${loginSamplesPerQuestion}`;
                }
                statusElement.textContent = `Sample ${currentSampleIndex + 1} recorded successfully.`;
                statusElement.classList.add('success');
                currentSampleIndex++;
            } else {
                statusElement.textContent = result.message;
                statusElement.classList.add('error');
                if (result.retry) {
                    await new Promise(resolve => setTimeout(resolve, 1000));
                    continue;
                }
                startButton.disabled = false;
                startButton.classList.remove('recording');
                return;
            }
        } catch (err) {
            console.error("Error recording audio:", err);
            statusElement.textContent = "Error recording audio.";
            statusElement.classList.add('error');
            startButton.disabled = false;
            startButton.classList.remove('recording');
            return;
        }

        if (currentSampleIndex < maxSamples) {
            statusElement.textContent = `Preparing for sample ${currentSampleIndex + 1}...`;
            await new Promise(resolve => setTimeout(resolve, 2000));
        }
    }

    statusElement.textContent = "All samples recorded.";
    statusElement.classList.add('success');
    startButton.disabled = true;
    startButton.classList.remove('recording');

    const nextQuestionIndex = currentQuestionIndex + 1;
    if (nextQuestionIndex < 3 && !isLoginOrForgot) {
        const nextContainer = document.getElementById(`question-container-${nextQuestionIndex}`);
        if (nextContainer) {
            nextContainer.classList.remove('hidden');
            nextContainer.classList.add('visible');
        }
    }

    currentQuestionIndex = -1;
    currentSampleIndex = 0;
    updateButtonStates();
}

function resetAudio(questionIndex) {
    recordedAudios[questionIndex] = Array(samplesPerQuestion).fill(null);
    const sampleCounter = document.getElementById(`sample-counter-${questionIndex}`);
    const statusElement = document.getElementById(`recordingStatus_${questionIndex}`);
    const startButton = document.querySelector(`button.start-recording[data-index="${questionIndex}"]`);
    const isLoginOrForgot = document.getElementById('verifyForm') || document.getElementById('forgotVerifyForm');

    if (isLoginOrForgot) {
        sampleCounter.textContent = `Sample 0/${loginSamplesPerQuestion}`;
    } else {
        sampleCounter.textContent = `Samples Recorded: 0/${samplesPerQuestion}`;
    }
    statusElement.textContent = "Not recording";
    statusElement.classList.remove('success', 'error');
    startButton.disabled = false;
    updateButtonStates();
}

function updateButtonStates() {
    const userDetailsForm = document.getElementById('userDetailsForm');
    if (userDetailsForm) {
        const userId = document.getElementById('user_id').value.trim();
        const password = document.getElementById('password').value.trim();
        const confirmPassword = document.getElementById('confirm_password').value.trim();
        const nextButton = userDetailsForm.querySelector('.btn');
        nextButton.disabled = !(userId && password && confirmPassword);
    }

    const registerForm = document.getElementById('registerForm');
    if (registerForm) {
        const answers = Array.from({ length: 3 }, (_, i) => document.getElementById(`answer_${i+1}`).value.trim());
        const allAnswersFilled = answers.every(answer => answer);
        const allRecordingsDone = recordedAudios.every((questionAudios, i) => 
            questionAudios.slice(0, samplesPerQuestion).every(audio => audio !== null)
        );
        const registerButton = registerForm.querySelector('.btn');
        registerButton.disabled = !(allAnswersFilled && allRecordingsDone);
    }

    const loginForm = document.getElementById('loginForm');
    if (loginForm) {
        const userId = document.getElementById('user_id').value.trim();
        const password = document.getElementById('password').value.trim();
        const loginButton = loginForm.querySelector('.btn');
        loginButton.disabled = !(userId && password);
    }

    const verifyForm = document.getElementById('verifyForm');
    if (verifyForm) {
        const answers = Array.from({ length: 3 }, (_, i) => document.querySelector(`input[name="answer_${i}"]`)?.value.trim() || '');
        const allAnswersFilled = answers.every(answer => answer);
        const allRecordingsDone = recordedAudios.every((questionAudios, i) => 
            questionAudios[0] !== null
        );
        const verifyButton = verifyForm.querySelector('.btn');
        verifyButton.disabled = !(allAnswersFilled && allRecordingsDone);
    }

    const forgotPasswordForm = document.getElementById('forgotPasswordForm');
    if (forgotPasswordForm) {
        const userId = document.getElementById('user_id').value.trim();
        const forgotButton = forgotPasswordForm.querySelector('.btn');
        forgotButton.disabled = !userId;
    }

    const forgotVerifyForm = document.getElementById('forgotVerifyForm');
    if (forgotVerifyForm) {
        const answers = Array.from({ length: 3 }, (_, i) => document.querySelector(`input[name="answer_${i}"]`)?.value.trim() || '');
        const allAnswersFilled = answers.every(answer => answer);
        const allRecordingsDone = recordedAudios.every((questionAudios, i) => 
            questionAudios[0] !== null
        );
        const verifyButton = forgotVerifyForm.querySelector('.btn');
        verifyButton.disabled = !(allAnswersFilled && allRecordingsDone);
    }

    const resetForm = document.getElementById('resetForm');
    if (resetForm) {
        const newPassword = document.getElementById('new_password').value.trim();
        const confirmNewPassword = document.getElementById('confirm_new_password').value.trim();
        const resetButton = resetForm.querySelector('.btn');
        resetButton.disabled = !(newPassword && confirmNewPassword);
    }
}

async function handleCancel(context) {
    const formData = new FormData();
    formData.append('context', context);
    const response = await fetch('/cancel', {
        method: 'POST',
        body: formData
    });
    const result = await response.json();
    if (result.success) {
        window.location.href = result.redirect;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.body.className = savedTheme;
        themeToggle.textContent = savedTheme === 'light' ? '🌙' : '🌞';

        themeToggle.addEventListener('click', () => {
            const newTheme = document.body.classList.contains('dark') ? 'light' : 'dark';
            document.body.className = newTheme;
            localStorage.setItem('theme', newTheme);
            themeToggle.textContent = newTheme === 'light' ? '🌙' : '🌞';
        });
    }

    const passwordInput = document.getElementById('password');
    if (passwordInput) {
        passwordInput.addEventListener('input', () => {
            const password = passwordInput.value.trim();
            checkPasswordStrength(password);
            updateButtonStates();
        });
    }

    const questionItems = document.querySelectorAll('.question-item');
    const selectedQuestions = [];
    questionItems.forEach(item => {
        item.addEventListener('click', () => {
            if (selectedQuestions.includes(item.dataset.question)) {
                selectedQuestions.splice(selectedQuestions.indexOf(item.dataset.question), 1);
                item.classList.remove('selected');
            } else if (selectedQuestions.length < 3) {
                selectedQuestions.push(item.dataset.question);
                item.classList.add('selected');
            } else {
                alert('You can only select 3 questions.');
                return;
            }

            for (let i = 0; i < 3; i++) {
                const container = document.getElementById(`question-container-${i}`);
                const questionInput = document.getElementById(`question_${i+1}`);
                const questionText = document.getElementById(`question-text-${i}`);
                if (i < selectedQuestions.length) {
                    container.classList.remove(i === 0 ? 'hidden' : 'visible');
                    container.classList.add(i === 0 ? 'visible' : 'hidden');
                    questionInput.value = selectedQuestions[i];
                    questionText.textContent = selectedQuestions[i];
                } else {
                    container.classList.remove('visible');
                    container.classList.add('hidden');
                    questionInput.value = '';
                    questionText.textContent = '';
                }
            }

            recordedAudios = Array(3).fill().map(() => Array(samplesPerQuestion).fill(null));
            updateButtonStates();
        });
    });

    document.querySelectorAll('.start-recording').forEach(button => {
        button.addEventListener('click', () => {
            const index = parseInt(button.dataset.index);
            const userId = document.getElementById('reg_user_id')?.value.trim() || document.getElementById('verify_user_id')?.value.trim();
            const isLoginOrForgot = button.closest('#questionsSection') !== null;
            console.log(`Starting recording for question ${index}, user ${userId}, isLoginOrForgot: ${isLoginOrForgot}`);
            startRecording(index, userId, isLoginOrForgot);
        });
    });

    document.querySelectorAll('.reset-audio').forEach(button => {
        button.addEventListener('click', () => {
            const index = parseInt(button.dataset.index);
            console.log(`Resetting audio for question ${index}`);
            resetAudio(index);
        });
    });

    document.querySelectorAll('input').forEach(input => {
        input.addEventListener('input', updateButtonStates);
    });

    const userDetailsForm = document.getElementById('userDetailsForm');
    if (userDetailsForm) {
        userDetailsForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const userId = document.getElementById('user_id').value.trim();
            const password = document.getElementById('password').value.trim();
            const confirmPassword = document.getElementById('confirm_password').value.trim();

            if (password !== confirmPassword) {
                document.getElementById('reg-message').textContent = 'Passwords do not match.';
                document.getElementById('reg-message').classList.remove('success', 'error');
                document.getElementById('reg-message').classList.add('error');
                return;
            }

            const formData = new FormData();
            formData.append('user_id', userId);
            formData.append('password', password);
            formData.append('confirm_password', confirmPassword);

            const response = await fetch('/validate_user_details', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            const regMessage = document.getElementById('reg-message');
            regMessage.textContent = result.message;
            regMessage.classList.remove('success', 'error');
            regMessage.classList.add(result.success ? 'success' : 'error');

            if (result.success) {
                document.getElementById('userDetailsForm').style.display = 'none';
                document.getElementById('securityQuestionsSection').style.display = 'block';
                document.getElementById('reg_user_id').value = userId;
                document.getElementById('reg_password').value = password;
                updateButtonStates();
            }
        });
    }

    const registerForm = document.getElementById('registerForm');
    if (registerForm) {
        registerForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData();
            formData.append('user_id', document.getElementById('reg_user_id').value.trim());
            formData.append('password', document.getElementById('reg_password').value.trim());

            for (let i = 0; i < 3; i++) {
                formData.append(`question_${i+1}`, document.getElementById(`question_${i+1}`).value);
                formData.append(`answer_${i+1}`, document.getElementById(`answer_${i+1}`).value.trim());
                for (let j = 0; j < samplesPerQuestion; j++) {
                    if (!recordedAudios[i][j]) {
                        document.getElementById('reg-message').textContent = `Please record all samples for question ${i+1}.`;
                        document.getElementById('reg-message').classList.add('error');
                        return;
                    }
                }
            }

            console.log('Register form data:', Object.fromEntries(formData));
            showLoadingScreen();

            const response = await fetch('/register', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();

            hideLoadingScreen();

            document.getElementById('reg-message').textContent = result.message;
            document.getElementById('reg-message').classList.add(result.success ? 'success' : 'error');
            if (result.success) {
                setTimeout(() => window.location.href = '/', 2000);
            }
        });
    }

    const loginForm = document.getElementById('loginForm');
    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData();
            formData.append('user_id', document.getElementById('user_id').value.trim());
            formData.append('password', document.getElementById('password').value.trim());

            console.log('Login form data:', Object.fromEntries(formData));
            const response = await fetch('/login', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            document.getElementById('login-message').textContent = result.message;
            document.getElementById('login-message').classList.add(result.success ? 'success' : 'error');

            if (result.success) {
                document.getElementById('loginForm').style.display = 'none';
                document.getElementById('questionsSection').style.display = 'block';
                document.getElementById('verify_user_id').value = document.getElementById('user_id').value.trim();
                document.getElementById('verify_password').value = document.getElementById('password').value.trim();

                const questionsDiv = document.getElementById('questions');
                result.questions.forEach((question, index) => {
                    questionsDiv.innerHTML += `
                        <div class="question-container">
                            <p>${question}</p>
                            <input type="password" name="answer_${index}" placeholder="Your answer" required>
                            <div class="recording-wrapper">
                                <button type="button" class="start-recording" data-index="${index}">Start Recording</button>
                                <button type="button" class="reset-audio" data-index="${index}">Reset Audio</button>
                                <span class="recording-indicator">🔴</span>
                            </div>
                            <div class="sample-counter" id="sample-counter-${index}">Sample 0/1</div>
                            <div class="status" id="recordingStatus_${index}"></div>
                        </div>
                    `;
                });
                updateButtonStates();

                document.querySelectorAll('.start-recording').forEach(button => {
                    button.addEventListener('click', () => {
                        const index = parseInt(button.dataset.index);
                        const userId = document.getElementById('verify_user_id').value.trim();
                        console.log(`Starting recording for login question ${index}, user ${userId}`);
                        startRecording(index, userId, true);
                    });
                });

                document.querySelectorAll('.reset-audio').forEach(button => {
                    button.addEventListener('click', () => {
                        const index = parseInt(button.dataset.index);
                        console.log(`Resetting audio for login question ${index}`);
                        resetAudio(index);
                    });
                });
            }
        });
    }

    const verifyForm = document.getElementById('verifyForm');
    if (verifyForm) {
        verifyForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(verifyForm);
            console.log('Verify form data:', Object.fromEntries(formData));
            showLoadingScreen();

            const response = await fetch('/login/verify', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();

            hideLoadingScreen();

            document.getElementById('verify-message').textContent = result.message;
            document.getElementById('verify-message').classList.add(result.success ? 'success' : 'error');
            if (result.success) {
                setTimeout(() => window.location.href = result.redirect, 1000);
            }
        });
    }

    const forgotPasswordForm = document.getElementById('forgotPasswordForm');
    if (forgotPasswordForm) {
        forgotPasswordForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(forgotPasswordForm);
            console.log('Forgot password form data:', Object.fromEntries(formData));
            const response = await fetch('/forgot_password', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            document.getElementById('forgot-message').textContent = result.message;
            document.getElementById('forgot-message').classList.add(result.success ? 'success' : 'error');

            if (result.success) {
                document.getElementById('forgotPasswordForm').style.display = 'none';
                document.getElementById('questionsSection').style.display = 'block';
                document.getElementById('verify_user_id').value = document.getElementById('user_id').value.trim();

                const questionsDiv = document.getElementById('questions');
                result.questions.forEach((question, index) => {
                    questionsDiv.innerHTML += `
                        <div class="question-container">
                            <p>${question}</p>
                            <input type="password" name="answer_${index}" placeholder="Your answer" required>
                            <div class="recording-wrapper">
                                <button type="button" class="start-recording" data-index="${index}">Start Recording</button>
                                <button type="button" class="reset-audio" data-index="${index}">Reset Audio</button>
                                <span class="recording-indicator">🔴</span>
                            </div>
                            <div class="sample-counter" id="sample-counter-${index}">Sample 0/1</div>
                            <div class="status" id="recordingStatus_${index}"></div>
                        </div>
                    `;
                });
                updateButtonStates();

                document.querySelectorAll('.start-recording').forEach(button => {
                    button.addEventListener('click', () => {
                        const index = parseInt(button.dataset.index);
                        const userId = document.getElementById('verify_user_id').value.trim();
                        console.log(`Starting recording for forgot password question ${index}, user ${userId}`);
                        startRecording(index, userId, true);
                    });
                });

                document.querySelectorAll('.reset-audio').forEach(button => {
                    button.addEventListener('click', () => {
                        const index = parseInt(button.dataset.index);
                        console.log(`Resetting audio for forgot password question ${index}`);
                        resetAudio(index);
                    });
                });
            }
        });
    }

    const forgotVerifyForm = document.getElementById('forgotVerifyForm');
    if (forgotVerifyForm) {
        forgotVerifyForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(forgotVerifyForm);
            console.log('Forgot verify form data:', Object.fromEntries(formData));
            showLoadingScreen();

            const response = await fetch('/forgot_password/verify', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();

            hideLoadingScreen();

            document.getElementById('verify-message').textContent = result.message;
            document.getElementById('verify-message').classList.add(result.success ? 'success' : 'error');
            if (result.success) {
                document.getElementById('questionsSection').style.display = 'none';
                document.getElementById('resetPasswordSection').style.display = 'block';
                document.getElementById('reset_user_id').value = document.getElementById('verify_user_id').value.trim();
            }
        });
    }

    const resetForm = document.getElementById('resetForm');
    if (resetForm) {
        resetForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const newPassword = document.getElementById('new_password').value.trim();
            const confirmNewPassword = document.getElementById('confirm_new_password').value.trim();

            if (newPassword !== confirmNewPassword) {
                document.getElementById('reset-message').textContent = 'Passwords do not match.';
                document.getElementById('reset-message').classList.remove('success', 'error');
                document.getElementById('reset-message').classList.add('error');
                return;
            }

            const formData = new FormData(resetForm);
            console.log('Reset form data:', Object.fromEntries(formData));
            showLoadingScreen();

            const response = await fetch('/forgot_password/reset', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();

            hideLoadingScreen();

            document.getElementById('reset-message').textContent = result.message;
            document.getElementById('reset-message').classList.add(result.success ? 'success' : 'error');
            if (result.success) {
                setTimeout(() => window.location.href = result.redirect, 1000);
            }
        });
    }
});