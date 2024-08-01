import base64
import datetime
import os
import re
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for, session
import pyrebase
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import firebase_admin
from firebase_admin import credentials, firestore, storage
import string
import cv2
import numpy as np
from flask import Response

app = Flask(__name__)

# Set secret key
secret_key = secrets.token_hex(16)
app.secret_key = secret_key

# Firebase Admin SDK initialization
cred = credentials.Certificate(r'D:\Project\trip_with_me\tripwithme-db6792-firebase-adminsdk-2itoq-0feea0f265.json')
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'tripwithme-db6792.appspot.com'
    })

bucket = storage.bucket()  # Access the bucket
db = firestore.client()    # Access Firestore

# Firebase configuration for Pyrebase
config = {
    "apiKey": "firebase _api",
    "authDomain": "firebase_authDomain",
    "projectId": "firebase_projectId",
    "storageBucket": "firebase_storageBucket",
    "messagingSenderId": "firebase_senderId",
    "appId": "firebase_appId",
    "measurementId": "firebase_mesurementId",
    "databaseURL": "firebase_databaseURL"
}

# Initialize Firebase with Pyrebase
firebase = pyrebase.initialize_app(config)
auth = firebase.auth()
db = firebase.database()

# SMTP email configuration
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SMTP_USERNAME = 'your_mail'
SMTP_PASSWORD = 'your_password'

def send_otp_email(to_email, otp):
    subject = 'Your OTP Code'
    body = f'Your OTP code is: {otp}'
    
    msg = MIMEMultipart()
    msg['From'] = SMTP_USERNAME
    msg['To'] = to_email
    msg['Subject'] = subject
    
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        text = msg.as_string()
        server.sendmail(SMTP_USERNAME, to_email, text)
        server.quit()
        print(f'Email sent to {to_email}')
    except Exception as e:
        print(f'Error sending email: {str(e)}')

@app.route('/')
def index():
    if 'user' in session:
        user_id = session['user_id']
        # Fetch user image filename from pers_det
        pers_det = db.child("per_det").child(user_id).get().val()
        user_image = pers_det.get('image', 'default_image.jpg')  # Default image if not found
        return render_template('index.html', user_image=user_image)
    return render_template('landing.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        otp = secrets.token_hex(3)  # Generate a 6-character OTP
        user_id = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(6))  # Generate a 6-character user ID

        try:
            user = auth.create_user_with_email_and_password(email, password)
            user_data = {
                "user_id": user_id,
                "name": name,
                "email": email,
                "otp": otp,
                "otpVerified": False,
                "status": 0  # Default value for status
            }
            db.child("users").child(user['localId']).set(user_data)
            send_otp_email(email, otp)
            flash('User registered successfully! Please verify your email using the OTP sent to your email.', 'success')
            return redirect(url_for('verify_otp', user_id=user['localId']))
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
            return redirect(url_for('register'))

    return render_template('register.html')

# Face detection and image capture
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, "Face detected!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/check_faces')
def check_faces():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return jsonify({'success': False, 'error': 'Failed to capture frame'})

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    print(f'Faces detected: {faces}')  # Log detected faces

    return jsonify({'success': True, 'face_count': len(faces)})

@app.route('/capture', methods=['POST'])
def capture():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'User not logged in'})

    data = request.json
    img_data = data['image']

    # Remove the data URL prefix
    img_data = re.sub('^data:image/jpeg;base64,', '', img_data)
    img_bytes = base64.b64decode(img_data)

    # Save image to a file on the server
    user_id = session['user_id']
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    img_name = f"captured_{timestamp}.jpg"
    img_path = os.path.join('static', 'images', img_name)
    
    with open(img_path, 'wb') as f:
        f.write(img_bytes)

    # Save image name to Firestore and update user status
    try:
        db.child("per_det").child(user_id).update({"image": img_name})
        db.child("users").child(user_id).update({"status": 2})

        return jsonify({'success': True})
    except Exception as e:
        print(f'Error saving image data to Firestore: {str(e)}')
        return jsonify({'success': False, 'message': 'Failed to save image data to Firestore'})

@app.route('/photopik')
def photopik():
    return render_template('photopik.html')

# pers_det page route
@app.route('/pers_det')
def pers_det():
    if 'user' in session:
        return render_template('pers_det.html')
    else:
        return redirect(url_for('login'))

# saving personal details to database
@app.route('/saveDetails', methods=['POST'])
def save_details():
    if 'user' in session:
        user_id = session['user']['localId']
        form_data = request.json

        try:
            # Add user_id to form_data
            form_data['user_id'] = user_id

            # Save the form data under a new node
            db.child("per_det").child(user_id).set(form_data)
            
            # Update user status to 1
            db.child("users").child(user_id).update({"status": 1})

            return jsonify({'success': True}), 200
        except Exception as e:
            print(f'Error: {str(e)}')
            return jsonify({'success': False, 'error': str(e)}), 500
    else:
        return jsonify({'success': False, 'error': 'User not logged in'}), 401

# OTP verification
@app.route('/verify_otp/<user_id>', methods=['GET', 'POST'])
def verify_otp(user_id):
    error_message = None
    if request.method == 'POST':
        entered_otp = request.form['otp']
        user_data = db.child("users").child(user_id).get().val()
        actual_otp = user_data.get('otp')
        
        if entered_otp == actual_otp:
            # Mark OTP as verified
            db.child("users").child(user_id).update({"otpVerified": True})
            flash('OTP verified successfully!', 'success')
            return redirect(url_for('pers_det'))
        else:
            error_message = 'Invalid OTP. Please try again.'
    
    return render_template('verify_otp.html', error_message=error_message)

# Password reset request
@app.route('/password_reset_request', methods=['GET', 'POST'])
def password_reset_request():
    if request.method == 'POST':
        email = request.form['email']
        otp = secrets.token_hex(3)
        user = auth.get_user_by_email(email)
        user_id = user.uid

        db.child("password_resets").child(user_id).set({"otp": otp, "email": email})

        send_otp_email(email, otp)
        flash('OTP sent to your email address!', 'success')
        return redirect(url_for('verify_reset_otp', user_id=user_id))

    return render_template('password_reset_request.html')

# OTP verification for password reset
@app.route('/verify_reset_otp/<user_id>', methods=['GET', 'POST'])
def verify_reset_otp(user_id):
    error_message = None
    if request.method == 'POST':
        entered_otp = request.form['otp']
        password_reset_data = db.child("password_resets").child(user_id).get().val()
        actual_otp = password_reset_data.get('otp')

        if entered_otp == actual_otp:
            flash('OTP verified. Please enter your new password.', 'success')
            return redirect(url_for('reset_password', user_id=user_id))
        else:
            error_message = 'Invalid OTP. Please try again.'

    return render_template('verify_reset_otp.html', error_message=error_message)

# Password reset
@app.route('/reset_password/<user_id>', methods=['GET', 'POST'])
def reset_password(user_id):
    if request.method == 'POST':
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']

        if new_password == confirm_password:
            try:
                auth.update_user(user_id, password=new_password)
                db.child("password_resets").child(user_id).remove()
                flash('Password reset successfully!', 'success')
                return redirect(url_for('login'))
            except Exception as e:
                flash(f'Error: {str(e)}', 'danger')
        else:
            flash('Passwords do not match. Please try again.', 'danger')

    return render_template('reset_password.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        try:
            user = auth.sign_in_with_email_and_password(email, password)
            user_data = db.child("users").child(user['localId']).get().val()
            status = user_data.get('status')

            session['user'] = user
            session['user_id'] = user['localId']

            if status == 0:
                return redirect(url_for('pers_det'))
            elif status == 1:
                return redirect(url_for('photopik'))
            elif status == 2:
                return redirect(url_for('index'))
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('user_id', None)
    flash('You have been logged out!', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
