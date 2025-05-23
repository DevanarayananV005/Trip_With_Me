import base64
import datetime
import os
import re
from flask import Flask, flash, jsonify, make_response, redirect, render_template, request, url_for, session
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
from datetime import datetime, timedelta
import itertools
import random
import qrcode
from jinja2 import Environment, FileSystemLoader
import math
import stripe
import logging
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO
from flask import send_file
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak
from reportlab.lib import colors
from sklearn.metrics.pairwise import cosine_similarity

#ml code
# Download NLTK stopwords
nltk.download('stopwords', quiet=True)

# Load and prepare the ML model
def load_csv_with_different_encodings(file_path):
    encodings = ['utf-8', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to read the file {file_path} with any of the attempted encodings.")

try:
    df = load_csv_with_different_encodings('static/data.csv')
    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['label'], test_size=0.2, random_state=42)

    custom_stop_words = list(set(stopwords.words('english')))
    tfidf = TfidfVectorizer(max_features=1000, stop_words=custom_stop_words)

    X_train_tfidf = tfidf.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    def classify_review(review_text):
        review_tfidf = tfidf.transform([review_text])
        prediction = model.predict(review_tfidf)
        return "Authentic" if prediction[0] == 1 else "Fake"

except Exception as e:
    print(f"Error loading or processing the CSV file: {str(e)}")
    # Set up a dummy classifier function that always returns "Authentic"
    def classify_review(review_text):
        print("Warning: Using dummy classifier due to CSV loading error.")
        return "Authentic"

#rest code

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__, static_folder='static', static_url_path='/static')

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
db = firestore.client()  
db_firestore = firestore.client()   # Access Firestore

# Firebase configuration for Pyrebase
config = {
    "apiKey": "AIzaSyBR1xj4iWIRS1YuD_5Cwta7QN00-1UtLN4",
    "authDomain": "tripwithme-db6792.firebaseapp.com",
    "projectId": "tripwithme-db6792",
    "storageBucket": "tripwithme-db6792.appspot.com",
    "messagingSenderId": "567401440169",
    "appId": "1:567401440169:web:96885bb3f0859cbff09ab8",
    "measurementId": "G-WLDG1QFZH3",
    "databaseURL": "https://tripwithme-db6792-default-rtdb.firebaseio.com/"
}

# Initialize Firebase with Pyrebase
firebase = pyrebase.initialize_app(config)
auth = firebase.auth()
db = firebase.database()

#razorpay secret key
stripe.api_key = 'sk_test_51OCqKJSDxrkb8ke1nAyVrBC0a8GHM73ixdczaY1acuJs23nGxr0eamRSLHTBxb2O6UeM8LU9vsiUHpKD67R8BwxN00VoItYGe6'

# SMTP email configuration
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SMTP_USERNAME = 'tripwithme5505@gmail.com'
SMTP_PASSWORD = 'znvy zjtj pgac brwd'

def send_otp_email(to_email, otp):
    subject = 'OTP code for Registration'
    body = f'Thank you for exploring our website Trip With Me. Try to find your Travel mate or Explore our wide range of roue packages. Your One time password for registration is: {otp}'
    
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

#enumerate filter
@app.template_filter('enumerate')
def jinja2_enumerate(iterable, start=0):
    return enumerate(iterable, start)


app.jinja_env.filters['enumerate'] = jinja2_enumerate

@app.route('/')
def index():
    if 'user' in session:
        user_id = session['user_id']
        pers_det = db.child("per_det").child(user_id).get().val()
        users = db.child("users").child(user_id).get().val()
        user_image = pers_det.get('image', 'default_image.jpg')
        user_name = users.get('name')
        user_email = users.get('email')
        user_phone = pers_det.get('phone', '')
        
        session['user'] = {
            'name': user_name,
            'email': user_email,
            'phone': user_phone
        }
        
        response = make_response(render_template('index.html', 
                                                 user_name=user_name, 
                                                 user_image=user_image, 
                                                 user_logged_in=True))
    else:
        response = make_response(render_template('landing.html', user_logged_in=False))
    
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

@app.route('/base')
def base():
    user_logged_in = 'user' in session
    return render_template('base.html', user_logged_in=user_logged_in)

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
                "status": 0,
                "role" : 0,
                "c_status" : 0   # Default value for status
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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

#update user
@app.route('/updateuser', methods=['GET', 'POST'])
def updateuser():
    if 'user' in session:
        user_id = session['user_id']
        if request.method == 'POST':
            # Handle form data to update user info
            housename = request.form.get('housename')
            pincode = request.form.get('pincode')
            state = request.form.get('state')
            district = request.form.get('district')
            phone = request.form.get('phone')
            dob = request.form.get('dob')

            try:
                # Update the pers_det node
                db.child("per_det").child(user_id).update({
                    "adname": housename,
                    "pin": pincode,
                    "state": state,
                    "district": district,
                    "phone": phone,
                    "dob": dob
                })
                flash('Details updated successfully!', 'success')
                return redirect(url_for('updateuser'))  # Stay on the same page to show the alert
            except Exception as e:
                flash(f'Error updating details: {str(e)}', 'danger')
                return redirect(url_for('updateuser'))
        
        # Fetch user details
        try:
            user_data = db.child("users").child(user_id).get().val()
            pers_det = db.child("per_det").child(user_id).get().val()

            email = user_data.get('email', '')
            name = user_data.get('name', '')
            role = user_data.get('role', '')
            pincode = pers_det.get('pin', '')
            state = pers_det.get('state', '')
            district = pers_det.get('district', '')
            housename = pers_det.get('adname', '')
            phone = pers_det.get('phone', '')
            dob = pers_det.get('dob', '')
            image_name = pers_det.get('image', 'default_image.jpg')

            return render_template('updateuser.html',
                email=email,
                name=name,
                pincode=pincode,
                state=state,
                district=district,
                housename=housename,
                phone=phone,
                dob=dob,
                image_name=image_name,
                role=role
            )
        except Exception as e:
            flash(f'Error fetching user details: {str(e)}', 'danger')
            return redirect(url_for('index'))
    else:
        flash('User not logged in.', 'danger')
        return redirect(url_for('login'))



#password_reset
@app.route('/password_reset_request', methods=['GET', 'POST'])
def password_reset_request():
    if request.method == 'POST':
        email = request.form['email']

        try:
            # Check if the email exists in the users database
            users = db.child("users").get().val()
            email_exists = any(user.get('email') == email for user in users.values())

            if email_exists:
                auth.send_password_reset_email(email)
                flash('Password reset email sent! Please check your email.', 'success')
            else:
                flash('Email not available. Please check again.', 'danger')
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')

    return render_template('password_reset_request.html')
@app.route('/reset_password_confirm', methods=['GET', 'POST'])
def reset_password_confirm():
    if request.method == 'POST':
        oob_code = request.form['oob_code']
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']

        if new_password == confirm_password:
            try:
                auth.confirm_password_reset(oob_code, new_password)
                flash('Password reset successfully!', 'success')
                return redirect(url_for('login'))
            except Exception as e:
                flash(f'Error: {str(e)}', 'danger')
        else:
            flash('Passwords do not match. Please try again.', 'danger')

    oob_code = request.args.get('oobCode')
    return render_template('reset_password_confirm.html', oob_code=oob_code)

#find trip and schedule trip
@app.route('/scheduletrip', methods=['GET', 'POST'])
def scheduletrip():
    if 'user' in session:
        user_id = session['user_id']
        # Fetch user data from the database
        pers_det = db.child("per_det").child(user_id).get().val()
        user_image = pers_det.get('image', 'default_image.jpg') 
        if request.method == 'POST':
            user_id = session['user_id']
            destination = request.form.get('destination')
            tripdate = request.form.get('tripdate')
            budget = request.form.get('budget')

            if not destination or not tripdate or not budget:
                flash('Please fill in all fields.', 'warning')
                return redirect(url_for('scheduletrip'))

            # Prepare data to be saved
            trip_data = {
                "user_id": user_id,
                "destination": destination,
                "tripdate": tripdate,
                "budget": budget,
                "tripstatus": 0
            }

            # Save data to Firebase under 'scheduletrip' node
            try:
                db.child("scheduletrip").push(trip_data)
                flash('Trip Scheduled successfully. If any request arrives will be notified.', 'success')
                return redirect(url_for('scheduletrip'))
            except Exception as e:
                flash(f'Error saving trip data: {str(e)}', 'danger')
                return redirect(url_for('scheduletrip'))
        else:
            # Render the schedule trip form on GET request
            return render_template('scheduletrip.html', user_image=user_image)
    else:
        flash('User not logged in.', 'danger')
        return redirect(url_for('login'))


from datetime import datetime

@app.route('/findtrip')
def findtrip():
    if 'user' in session:
        try:
            # Fetch all schedule trip data
            scheduletrips = db.child("scheduletrip").get().val()
            print("Fetched schedule trip data:", scheduletrips)

            if scheduletrips is None:
                # No data found, set trips_data to an empty list
                trips_data = []
                print("No schedule trips found.")
            else:
                trips_data = []
                today = datetime.now().date()
                for key, trip in scheduletrips.items():
                    # Check if tripstatus is 0 and tripdate is after today
                    tripdate = datetime.strptime(trip.get('tripdate'), '%Y-%m-%d').date()
                    if trip.get('tripstatus') != 0 or tripdate <= today:
                        continue  # Skip this trip if tripstatus is not 0 or tripdate is not after today

                    user_id = trip.get('user_id')
                    print(f"Processing trip for user_id: {user_id}")

                    user_data = db.child("users").child(user_id).get().val()
                    if not user_data:
                        print(f"No user data found for user_id: {user_id}")
                        continue  # Skip if no user data found
                    print(f"User data for {user_id}:", user_data)

                    # Use user_id from the trip directly to fetch personal data
                    pers_data = db.child("per_det").child(user_id).get().val()
                    print(f"Personal data for {user_id}:", pers_data)

                    if pers_data is None:
                        print(f"No personal data found for user_id: {user_id}")
                        continue  # Skip this trip if no personal data is found

                    # Calculate age from dob
                    dob = pers_data.get('dob')
                    dob_date = datetime.strptime(dob, '%Y-%m-%d')
                    age = (datetime.now() - dob_date).days // 365

                    trip_info = {
                        "id": key,  # Include the trip ID
                        "destination": trip.get('destination'),
                        "tripdate": trip.get('tripdate'),
                        "budget": trip.get('budget'),
                        "name": user_data.get('name'),
                        "state": pers_data.get('state'),
                        "dob": dob,
                        "age": age,
                        "image": pers_data.get('image'),
                        "sender_id": session['user_id'],  # Add sender_id to trip_info
                        "receiver_id": trip.get('user_id')  # Add receiver_id to trip_info
                    }
                    trips_data.append(trip_info)

                print("Trips data:", trips_data)

            # Fetch the logged-in user's image
            logged_in_user_id = session['user_id']
            logged_in_user_data = db.child("per_det").child(logged_in_user_id).get().val()
            user_image = logged_in_user_data.get('image', 'default_image.jpg')  # Use a default image if not found

            return render_template('findtrip.html', trips_data=trips_data, user_image=user_image)
        except Exception as e:
            print(f"Error: {str(e)}")
            flash(f'Error fetching trip data: {str(e)}', 'danger')
            return redirect(url_for('index'))
    else:
        flash('User not logged in.', 'danger')
        return redirect(url_for('login'))

#triprequest
@app.route('/sendrequest', methods=['POST'])
def sendrequest():
    if 'user' in session:
        try:
            message = request.form.get('message')
            sender_id = request.form.get('sender_id')
            receiver_id = request.form.get('receiver_id')
            request_status = int(request.form.get('request_status'))
            scheduletrip_id = request.form.get('scheduletrip_id')

            if sender_id == receiver_id:
                return jsonify({'status': 'error', 'message': 'Cannot send request to yourself.'})

            # Check for existing requests
            existing_requests = db.child("trip_requests").get().val()
            if existing_requests is None:
                existing_requests = {}

            for req in existing_requests.values():
                if req['sender_id'] == sender_id and req['scheduletrip_id'] == scheduletrip_id:
                    return jsonify({'status': 'error', 'message': 'Sorry, you have already sent a request.'})

            # Add data to the database
            db.child("trip_requests").push({
                "message": message,
                "sender_id": sender_id,
                "receiver_id": receiver_id,
                "request_status": request_status,
                "scheduletrip_id": scheduletrip_id
            })

            # Fetch receiver's email address
            receiver_data = db.child("users").child(receiver_id).get().val()
            if receiver_data:
                receiver_email = receiver_data.get('email')

                if receiver_email:
                    # Compose email
                    subject = 'New Trip Request Received'
                    email_body = f'You have received a new request for your trip.\n\nMessage: {message}'

                    # Send email
                    send_email(receiver_email, subject, email_body)

            return jsonify({'status': 'success'})
        except Exception as e:
            print(f"Error: {str(e)}")
            return jsonify({'status': 'error', 'message': 'Error saving request.'})
    else:
        return jsonify({'status': 'error', 'message': 'User not logged in.'})

def send_email(to_email, subject, body):
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


#checkmessages
@app.route('/check_messages')
def check_messages():
    if 'user' in session:
        try:
            user_id = session['user_id']
            trip_requests = db.child("trip_requests").get().val()
            
            # Check if there are any trip requests that match the conditions
            has_new_messages = any(
                (req.get('receiver_id') == user_id and req.get('request_status') == 0) or
                (req.get('sender_id') == user_id and req.get('request_status') == 1)
                for req in trip_requests.values()
            )
            
            return jsonify({'has_new_messages': has_new_messages})
        except Exception as e:
            print(f"Error: {str(e)}")
            return jsonify({'has_new_messages': False})
    else:
        return jsonify({'has_new_messages': False})


#notifications
@app.route('/notifications')
def notifications():
    if 'user' in session:
        user_id = session['user_id']
        notifications = []

        trip_requests = db.child("trip_requests").get().val()
        scheduletrips = db.child("scheduletrip").get().val()
        users = db.child("users").get().val()
        per_det = db.child("per_det").get().val()

        if trip_requests and scheduletrips and users and per_det:
            for request_id, request_data in trip_requests.items():
                if request_data['receiver_id'] == user_id:
                    sender_id = request_data['sender_id']
                    scheduletrip_id = request_data['scheduletrip_id']

                    sender_data = users.get(sender_id, {})
                    sender_pers_det = per_det.get(sender_id, {})
                    trip_data = scheduletrips.get(scheduletrip_id, {})

                    if sender_data and sender_pers_det and trip_data:
                        dob = sender_pers_det.get('dob', '')
                        dob_date = datetime.strptime(dob, '%Y-%m-%d') if dob else None
                        age = (datetime.now() - dob_date).days // 365 if dob_date else 'N/A'

                        notification = {
                            'name': sender_data.get('name'),
                            'age': age,
                            'state': sender_pers_det.get('state'),
                            'image': sender_pers_det.get('image'),
                            'destination': trip_data.get('destination'),
                            'tripdate': trip_data.get('tripdate'),
                            'budget': trip_data.get('budget'),
                            'request_id': request_id,
                            'scheduletrip_id': scheduletrip_id,
                            'sender_id': sender_id,
                            'receiver_id': user_id, 
                            'request_status': request_data.get('request_status'),
                            'tripstatus': trip_data.get('tripstatus')
                        }
                        notifications.append(notification)

                # Additional notifications where logged-in user is the sender and request_status is 1
                elif request_data['sender_id'] == user_id and request_data['request_status'] == 1:
                    receiver_id = request_data['receiver_id']
                    scheduletrip_id = request_data['scheduletrip_id']

                    receiver_data = users.get(receiver_id, {})
                    receiver_pers_det = per_det.get(receiver_id, {})
                    trip_data = scheduletrips.get(scheduletrip_id, {})

                    if receiver_data and receiver_pers_det and trip_data:
                        dob = receiver_pers_det.get('dob', '')
                        dob_date = datetime.strptime(dob, '%Y-%m-%d') if dob else None
                        age = (datetime.now() - dob_date).days // 365 if dob_date else 'N/A'

                        notification = {
                            'name': receiver_data.get('name'),
                            'age': age,
                            'state': receiver_pers_det.get('state'),
                            'image': receiver_pers_det.get('image'),
                            'destination': trip_data.get('destination'),
                            'tripdate': trip_data.get('tripdate'),
                            'budget': trip_data.get('budget'),
                            'request_id': request_id,
                            'scheduletrip_id': scheduletrip_id,
                            'request_status': request_data.get('request_status'),
                            'tripstatus': trip_data.get('tripstatus'),
                            'accepted': True  # Indicator for accepted requests by logged-in user
                        }
                        notifications.append(notification)

        # Fetch the logged-in user's image
        logged_in_user_data = db.child("per_det").child(user_id).get().val()
        user_image = logged_in_user_data.get('image', 'default_image.jpg')  # Use a default image if not found

        current_date = datetime.now().strftime("%Y-%m-%d")
        return render_template('notifications.html', notifications=notifications, user_image=user_image, current_date=current_date)
    else:
        flash('User not logged in.', 'danger')
        return redirect(url_for('login'))

@app.route('/accept_request', methods=['POST'])
def accept_request():
    if 'user' in session:
        request_id = request.form.get('request_id')
        scheduletrip_id = request.form.get('scheduletrip_id')

        if request_id and scheduletrip_id:
            try:
                # Update request_status in trip_requests
                db.child("trip_requests").child(request_id).update({"request_status": 1})

                # Update tripstatus in scheduletrip
                db.child("scheduletrip").child(scheduletrip_id).update({"tripstatus": 1})

                flash('Request accepted successfully!', 'success')
            except Exception as e:
                flash(f'Error accepting request: {str(e)}', 'danger')
        else:
            flash('Invalid request data.', 'danger')

        return redirect(url_for('notifications'))
    else:
        flash('User not logged in.', 'danger')
        return redirect(url_for('login'))
    
#upgrade_account
@app.route('/upgrade_account', methods=['GET', 'POST'])
def upgrade_account():
    if 'user' in session:
        user_id = session['user_id']
        if request.method == 'POST':
            if 'driver_name' in request.form:  # Driver form submission
                try:
                    driver_name = request.form['driver_name']
                    vehicle_state = request.form['vehicle_state']
                    rto = request.form['rto']
                    licence_number = request.form['licence_number']
                    vehicle_number = request.form['vehicle_number']
                    
                    # Save images
                    rc_image = request.files['rc_image']
                    licence_image = request.files['licence_image']
                    rc_filename = f"{user_id}_rc_{rc_image.filename}"
                    licence_filename = f"{user_id}_licence_{licence_image.filename}"
                    rc_image.save(os.path.join('static', 'images', rc_filename))
                    licence_image.save(os.path.join('static', 'images', licence_filename))
                    
                    # Generate OTP
                    otp = random.randint(100000, 999999)
                    
                    # Store driver data
                    driver_data = {
                        "name": driver_name,
                        "vehicle_state": vehicle_state,
                        "rto": rto,
                        "licence_number": licence_number,
                        "vehicle_number": vehicle_number,
                        "rc_image": rc_filename,
                        "licence_image": licence_filename,
                        "status": 0,
                        "e_status": 0,
                        "f_status":0,
                        "user_id": user_id,
                        "otp": otp
                    }
                    
                    # Save to Firebase
                    db.child("driver").child(user_id).set(driver_data)
                    
                    # Update user role
                    db.child("users").child(user_id).update({"role": 2})  # 2 for driver role
                    
                    # Send OTP email
                    user = db.child("users").child(user_id).get().val()
                    send_otp_email(user.get('email'), otp)
                    
                    # Return success response for SweetAlert
                    return jsonify({
                        'status': 'success',
                        'message': 'Driver Request Successful! You will be able to login after admin approval.',
                        'redirect': url_for('login')
                    })

                except Exception as e:
                    return jsonify({
                        'status': 'error',
                        'message': f'Error: {str(e)}'
                    })

            else:  # Package manager form submission
                company_name = request.form['company_name']
                company_number = request.form['company_number']
                company_email = request.form['company_email']
                state = request.form['state']
                district = request.form['district']
                company_begin_date = request.form['company_begin_date']
                ownership_certificate = request.files['ownership_certificate']
                
                # Save the ownership certificate
                certificate_filename = f"{user_id}_{ownership_certificate.filename}"
                certificate_path = os.path.join('static', 'images', certificate_filename)
                ownership_certificate.save(certificate_path)
                
                # Generate OTP
                otp = random.randint(100000, 999999)
                
                # Store data in the database
                package_manager_data = {
                    "c_name": company_name,
                    "c_phone": company_number,
                    "c_email": company_email,
                    "state": state,
                    "district": district,
                    "c_startdate": company_begin_date,
                    "c_ownership": certificate_filename,
                    "status": 0,
                    "e_status": 0,
                    "user_id": user_id,
                    "otp": otp
                }
                db.child("package_manager").child(user_id).set(package_manager_data)
                
                # Send OTP email
                send_otp_email(company_email, otp)
                
                flash('Data entered successfully to database. An OTP is sent to the given email. Verify the email.', 'success')
                return redirect(url_for('verify_otp_page'))
        
        return render_template('upgrade_account.html')
    else:
        flash('User not logged in.', 'danger')
        return redirect(url_for('login'))

@app.route('/verify_otp_page', methods=['GET', 'POST'])
def verify_otp_page():
    if 'user' in session:
        user_id = session['user_id']
        if request.method == 'POST':
            entered_otp = request.form['otp']
            package_manager_data = db.child("package_manager").child(user_id).get().val()
            actual_otp = package_manager_data.get('otp')
            
            if str(entered_otp) == str(actual_otp):
                # Update email verification status
                db.child("package_manager").child(user_id).update({"e_status": 1})
                
                # Update user role
                db.child("users").child(user_id).update({"role": 1})
                
                # Show success alert and logout
                flash('Email verified successfully. After approval, you will be notified.', 'success')
                return redirect(url_for('logout'))
            else:
                flash('Invalid OTP. Please try again.', 'danger')
        
        return render_template('verify_otp_page.html')
    else:
        flash('User not logged in.', 'danger')
        return redirect(url_for('login'))
CAR_RATE_CONFIG = {
    "hatchback": {"base_rate": 70, "extra_km_rate": 30},
    "sedan": {"base_rate": 80, "extra_km_rate": 35},
    "premium_sedan": {"base_rate": 90, "extra_km_rate": 45},
    "suv": {"base_rate": 95, "extra_km_rate": 45},
    "premium_suv": {"base_rate": 100, "extra_km_rate": 50},
    "crossover": {"base_rate": 70, "extra_km_rate": 30},
    "compact_suv": {"base_rate": 80, "extra_km_rate": 35},
    "micro_suv": {"base_rate": 70, "extra_km_rate": 30},
    "mpv": {"base_rate": 95, "extra_km_rate": 45},
    "premium_mpv": {"base_rate": 110, "extra_km_rate": 60}
}

@app.route('/submit_review', methods=['POST'])
def submit_review():
    if 'user' in session:
        reviewer_id = session['user_id']
        reviewee_id = request.form.get('reviewee_id')
        review = request.form.get('review')
        review_date = datetime.now().strftime("%Y-%m-%d")
        review_time = datetime.now().strftime("%H:%M:%S")

        try:
            # Check if a review already exists
            existing_reviews = db.child("reviews").get().val()
            if existing_reviews:
                for review_id, review_data in existing_reviews.items():
                    if review_data['reviewer_id'] == reviewer_id and review_data['reviewee_id'] == reviewee_id:
                        return jsonify({'success': False, 'message': 'Review already provided'})

            # Verify the review using ML
            review_classification = classify_review(review)
            
            if review_classification == "Authentic":
                # If the review is authentic, submit it to the database
                db.child("reviews").push({
                    'reviewer_id': reviewer_id,
                    'reviewee_id': reviewee_id,
                    'review': review,
                    'review_date': review_date,
                    'review_time': review_time
                })
                return jsonify({'success': True, 'message': 'Review submitted successfully'})
            else:
                # If the review is classified as fake, return an error message
                return jsonify({'success': False, 'message': 'Please enter an authentic review'})

        except Exception as e:
            print(f"Error submitting review: {str(e)}")
            return jsonify({'success': False, 'message': 'An error occurred'})
    else:
        return jsonify({'success': False, 'message': 'User not logged in'})
    

#add package
@app.route('/pack_manager', methods=['GET', 'POST'])
def pack_manager():
    if 'user' in session and session['user_id']:
        user_id = session['user_id']
        user_image = db.child("per_det").child(user_id).child("image").get().val()
        if not user_image:
            user_image = 'default_image.jpg' 
        package_manager_data = db.child("package_manager").child(user_id).get().val()
        c_name = package_manager_data.get('c_name', 'PACKAGE MANAGER TERMINAL') if package_manager_data else 'PACKAGE MANAGER TERMINAL' # Default image if not found

        
        if request.method == 'POST':
            location = request.form['location']
            start_date = request.form['startDate']
            end_date = request.form['endDate']
            basic_amount = request.form['basicAmount']
            tax_percentage = request.form['taxPercentage']
            discount = request.form.get('discount', 0)
            participants = request.form['participants']
            description = request.form['description']

            # Save images
            image1 = request.files['image1']
            image2 = request.files['image2']
            image3 = request.files['image3']
            image1_filename = f"{user_id}_{image1.filename}"
            image2_filename = f"{user_id}_{image2.filename}"
            image3_filename = f"{user_id}_{image3.filename}"
            image1.save(os.path.join('static/images', image1_filename))
            image2.save(os.path.join('static/images', image2_filename))
            image3.save(os.path.join('static/images', image3_filename))

            # Prepare data for database
            trip_data = {
                "user_id": user_id,
                "location": location,
                "start_date": start_date,
                "end_date": end_date,
                "basic_amount": basic_amount,
                "tax_percentage": tax_percentage,
                "discount": discount,
                "participants": participants,
                "description": description,
                "image1": image1_filename,
                "image2": image2_filename,
                "image3": image3_filename,
                "status": 0,
                "bookings": 0
            }

            # Debug print to verify data
            print("Trip Data:", trip_data)

            try:
                # Save data to database
                db.child("trips").push(trip_data)
                flash('Package added successfully!', 'success')
            except Exception as e:
                print("Error saving to database:", str(e))
                flash('Error adding data to the database.', 'danger')

            return redirect(url_for('pack_manager'))

        # Fetch trips from the database
        trips = db.child("trips").get().val()
        if trips:
            trips = [{**trip, 'id': trip_id} for trip_id, trip in trips.items() if trip.get('user_id') == user_id]
        else:
            trips = []

        return render_template('pack_manager.html', trips=trips, user_image=user_image, enumerate=enumerate, c_name=c_name)
    else:
        flash('User not logged in.', 'danger')
        return redirect(url_for('login'))

#package manager booking display
@app.route('/get_locations')
def get_locations():
    user_id = request.args.get('user_id')
    trips = db.child("trips").order_by_child("user_id").equal_to(user_id).get().val()
    locations = []
    if trips:
        for trip_id, trip in trips.items():
            locations.append({
                'trip_id': trip_id,
                'location': trip['location']
            })
    return jsonify(locations)

@app.route('/get_bookings')
def get_bookings():
    trip_id = request.args.get('trip_id')
    bookings = db.child("bookings").order_by_child("trip_id").equal_to(trip_id).get().val()
    bookings_list = []
    if bookings:
        for booking_id, booking in bookings.items():
            user_id = booking['user_id']
            user = db.child("users").child(user_id).get().val()
            per_det = db.child("per_det").child(user_id).get().val()
            bookings_list.append({
                'booking_id': booking_id,  # Add this line
                'name': user['name'],
                'email': user['email'],
                'phone': per_det['phone'],
                'booking_date': booking['booking_date'],
                'booking_time': booking['booking_time'],
                'status': booking['status'],
            })
    return jsonify(bookings_list)

#pack_status update
@app.route('/toggle_status/<trip_id>', methods=['POST'])
def toggle_status(trip_id):
    if 'user' in session and session['user_id']:
        try:
            trip = db.child("trips").child(trip_id).get().val()
            new_status = 1 if trip['status'] == 0 else 0
            db.child("trips").child(trip_id).update({"status": new_status})
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})
    else:
        return jsonify({"success": False, "error": "Unauthorized access"})

#pack_data update
@app.route('/update_trip/<trip_id>', methods=['POST'])
def update_trip(trip_id):
    if 'user' in session and session['user_id']:
        try:
            trip_data = request.json
            db.child("trips").child(trip_id).update(trip_data)
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})
    else:
        return jsonify({"success": False, "error": "Unauthorized access"})


#admin_route
@app.route('/admin')
def admin():
    # Ensure only the admin can access this route
    if 'user_id' in session and session['user_id'] == 'admin':
        # Fetch log data
        log_data = db.child("log_data").get().val()

        # Fetch users data
        users = db.child("users").get().val()
        per_det = db.child("per_det").get().val()
        package_managers = db.child("package_manager").get().val()

        # Debug prints
        print("Users:", users)
        print("Personal Details:", per_det)
        print("Package Managers:", package_managers)

        # Process and combine the data
        processed_log_data = []
        if log_data and users:
            for log_id, log in log_data.items():
                if 'login_date' in log and 'login_time':
                    user_id = log['user_id']
                    user = users.get(user_id, {})
                    processed_log_data.append({
                        'name': user.get('name', 'Unknown'),
                        'email': user.get('email', 'Unknown'),
                        'login_date': log['login_date'],
                        'login_time': log['login_time']
                    })

        # Sort the data by login date and time (most recent first)
        processed_log_data.sort(key=lambda x: (x['login_date'], x['login_time']), reverse=True)

        # Process user data for display
        user_data = []
        if users and per_det:
            for idx, (user_id, user) in enumerate(users.items(), start=1):
                personal_details = per_det.get(user_id, {})
                user_data.append({
                    'sl_no': idx,
                    'user_id': user_id,
                    'name': user.get('name', 'Not Available'),
                    'email': user.get('email', 'Not Available'),
                    'image': personal_details.get('image', 'default_image.jpg'),
                    'adname': personal_details.get('adname', 'Not Available'),
                    'district': personal_details.get('district', 'Not Available'),
                    'dob': personal_details.get('dob', 'Not Available'),
                    'pin': personal_details.get('pin', 'Not Available'),
                    'phone': personal_details.get('phone', 'Not Available'),
                    'status': user.get('status', 'Not Available'),
                    'role': user.get('role', 'Not Available')
                })

        # Fetch scheduletrip data
        scheduletrips = db.child("scheduletrip").get().val()
        scheduled_trips = []
        if scheduletrips and users:
            for idx, (trip_id, trip) in enumerate(scheduletrips.items(), start=1):
                if trip.get('tripstatus') == 0:
                    user_id = trip.get('user_id')
                    user = users.get(user_id, {})
                    scheduled_trips.append({
                        'sl_no': idx,
                        'scheduler': user.get('name', 'Unknown'),
                        'email': user.get('email', 'Unknown'),
                        'destination': trip.get('destination', 'Unknown'),
                        'tripdate': trip.get('tripdate', 'Unknown'),
                        'budget': trip.get('budget', 'Unknown')
                    })

        # Fetch trip_requests data
        trip_requests = db.child("trip_requests").get().val()
        confirmed_trips = []
        if trip_requests and users and scheduletrips:
            for idx, (request_id, request) in enumerate(trip_requests.items(), start=1):
                if request.get('request_status') == 1:
                    receiver_id = request.get('receiver_id')
                    sender_id = request.get('sender_id')
                    scheduletrip_id = request.get('scheduletrip_id')
                    receiver = users.get(receiver_id, {})
                    sender = users.get(sender_id, {})
                    trip = scheduletrips.get(scheduletrip_id, {})
                    confirmed_trips.append({
                        'sl_no': idx,
                        'scheduler': receiver.get('name', 'Unknown'),
                        'companion': sender.get('name', 'Unknown'),
                        'destination': trip.get('destination', 'Unknown'),
                        'tripdate': trip.get('tripdate', 'Unknown'),
                        'budget': trip.get('budget', 'Unknown')
                    })

        # Process package manager data
        package_manager_data = []
        if package_managers and users:
            for idx, (pm_id, pm) in enumerate(package_managers.items(), start=1):
                user_id = pm.get('user_id')
                user = users.get(user_id, {})
                personal_details = per_det.get(user_id, {})
                package_manager_data.append({
                    'sl_no': idx,
                    'c_name': pm.get('c_name', 'Unknown'),
                    'c_email': pm.get('c_email', 'Unknown'),
                    'c_ownership': pm.get('c_ownership', 'Unknown'),
                    'user_id': user_id,
                    'status': pm.get('status', 0),
                    'c_status': user.get('status', 0),
                    'owner_name': user.get('name', 'Unknown'),
                    'owner_email': user.get('email', 'Unknown'),
                    'company_number': pm.get('c_phone', 'Unknown'),
                    'owner_number': personal_details.get('phone', 'Unknown'),
                    'c_startdate': pm.get('c_startdate', 'Unknown'),
                    'state': pm.get('state', 'Unknown'),
                    'district': pm.get('district', 'Unknown')
                })

        # Debug print for package_manager_data
        print("Package Manager Data:", package_manager_data)

        # Fetch driver data
        drivers = db.child("driver").get().val()
        driver_data = []
        if drivers:
            for user_id, driver in drivers.items():
                driver_data.append({
                    'user_id': user_id,
                    'name': driver.get('name', 'Unknown'),
                    'vehicle_state': driver.get('vehicle_state', 'Unknown'),
                    'rto': driver.get('rto', 'Unknown'),
                    'licence_number': driver.get('licence_number', 'Unknown'),
                    'vehicle_number': driver.get('vehicle_number', 'Unknown'),
                    'licence_image': driver.get('licence_image', ''),
                    'rc_image': driver.get('rc_image', ''),
                    'status': driver.get('status', 0)
                })

        return render_template('admin.html', log_data=processed_log_data, user_data=user_data, scheduled_trips=scheduled_trips, confirmed_trips=confirmed_trips, package_manager_data=package_manager_data, driver_data=driver_data, enumerate=enumerate)
    else:
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('login'))
    

#deactivate user
@app.route('/deactivate_user/<user_id>', methods=['POST'])
def deactivate_user(user_id):
    if 'user_id' in session and session['user_id'] == 'admin':
        reason = request.form.get('reason')
        try:
            # Update the user's status to 4 (deactivated)
            db.child("users").child(user_id).update({"status": 4})

            # Store the reason in the database
            db.child("users").child(user_id).update({"deactivation_reason": reason})

            # Send an email to the user with the reason
            user_data = db.child("users").child(user_id).get().val()
            if user_data:
                user_email = user_data.get('email')
                if user_email:
                    subject = 'Account Deactivated'
                    body = f'Your account has been deactivated. Reason: {reason}'
                    send_email(user_email, subject, body)

            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})
    else:
        return jsonify({"success": False, "error": "Unauthorized access"})
def send_email(to_email, subject, body):
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

@app.route('/get_user_details/<user_id>', methods=['GET'])
def get_user_details(user_id):
    user = db.child("users").child(user_id).get().val()
    package_manager = db.child("package_manager").child(user_id).get().val()
    personal_details = db.child("per_det").child(user_id).get().val()

    if user and package_manager and personal_details:
        return jsonify({
            'name': user.get('name', 'Unknown'),
            'email': user.get('email', 'Unknown'),
            'c_phone': package_manager.get('c_phone', 'Unknown'),
            'phone': personal_details.get('phone', 'Unknown'),
            'c_startdate': package_manager.get('c_startdate', 'Unknown'),
            'state': package_manager.get('state', 'Unknown'),
            'district': package_manager.get('district', 'Unknown')
        })
    else:
        return jsonify({'error': 'User not found'}), 404
    


@app.route('/update_status/<user_id>/<int:new_status>', methods=['POST'])
def update_status(user_id, new_status):
    try:
        db.child("package_manager").child(user_id).update({"status": new_status})
        db.child("users").child(user_id).update({"c_status": new_status})

        if new_status == 1:
            user = db.child("users").child(user_id).get().val()
            package_manager = db.child("package_manager").child(user_id).get().val()
            send_approval_email(package_manager.get('c_email'))

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def send_approval_email(to_email):
    subject = 'Company Approved'
    body = '<p style="color:green;font-weight:bold;">Congratulations your company is Approved!!!! <span>Welcome to Trip With Me</span>Hope a great business for the future now you can login to your package_manager terminal</p>'
    
    msg = MIMEMultipart()
    msg['From'] = SMTP_USERNAME
    msg['To'] = to_email
    msg['Subject'] = subject
    
    msg.attach(MIMEText(body, 'html'))
    
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

# Add this new function to send driver activation email
def send_driver_activation_email(to_email):
    subject = 'Driver Account Activated'
    body = '''
    Congratulations! Your driver account has been successfully activated.
    
    You can now log in to your driver platform and start accepting ride requests.
    
    Welcome to Trip With Me - We look forward to having you as part of our driver community!
    '''
    
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
        print(f'Driver activation email sent to {to_email}')
    except Exception as e:
        print(f'Error sending email: {str(e)}')

# Update the existing route to include email notification
@app.route('/update_driver_status/<user_id>/<int:new_status>', methods=['POST'])
def update_driver_status(user_id, new_status):
    if 'user_id' in session and session['user_id'] == 'admin':
        try:
            # Update driver status
            db.child("driver").child(user_id).update({"status": new_status})
            
            # If status is being set to active (1), send activation email
            if new_status == 1:
                # Get user email from users collection
                user_data = db.child("users").child(user_id).get().val()
                if user_data and 'email' in user_data:
                    send_driver_activation_email(user_data['email'])
            
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})
    return jsonify({"success": False, "error": "Unauthorized access"})

#all_trips page
@app.route('/all_trips')
def all_trips():
    if 'user' in session:
        user_id = session['user_id']
        user_image = db.child("per_det").child(user_id).child("image").get().val()
        if not user_image:
            user_image = 'default_image.jpg'  # Default image if not found
        return render_template('all_trips.html', user_image=user_image)
    else:
        flash('User not logged in.', 'danger')
        return redirect(url_for('login'))


#trip search and filter
@app.route('/search_trips', methods=['POST'])
def search_trips():
    if 'user' not in session:
        return jsonify({'error': 'User not logged in'}), 401
    
    search_query = request.json.get('search_query', '')
    filter_options = request.json.get('filter_options', {})
    
    # Fetch all trips
    trips = db.child("trips").get().val()
    if not trips:
        return jsonify([])
    
    # Convert trips to a list of dictionaries
    trips_list = [{**trip, 'id': trip_id} for trip_id, trip in trips.items()]
    
    # Apply search query
    if search_query:
        trips_list = [trip for trip in trips_list if search_query.lower() in trip['location'].lower()]
    
    # Apply filter options
    if filter_options:
        if 'start_date' in filter_options:
            trips_list = [trip for trip in trips_list if trip['start_date'] >= filter_options['start_date']]
        if 'end_date' in filter_options:
            trips_list = [trip for trip in trips_list if trip['end_date'] <= filter_options['end_date']]
        if 'price_min' in filter_options and 'price_max' in filter_options:
            trips_list = [trip for trip in trips_list if filter_options['price_min'] <= float(trip['basic_amount']) <= filter_options['price_max']]
    
    return jsonify(trips_list)

# Additional route to get min_start_date and min_end_date
@app.route('/get_min_dates', methods=['GET'])
def get_min_dates():
    min_start_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    min_end_date = (datetime.now() + timedelta(days=4)).strftime('%Y-%m-%d')
    return jsonify({
        'min_start_date': min_start_date,
        'min_end_date': min_end_date
    })

#packages
@app.route('/get_trips')
def get_trips():
    trips = db.child("trips").get().val()
    if trips:
        available_trips = {}
        for key, trip in trips.items():
            bookings = trip.get('bookings', 0)
            participants = int(trip.get('participants', 0))
            if bookings < participants:
                available_trips[key] = {**trip, 'id': key}
        return jsonify(available_trips)
    else:
        return jsonify({})

@app.route('/trip_details/<trip_id>')
def trip_details(trip_id):
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Fetch the trip details
    trip = db.child("trips").child(trip_id).get().val()
    if not trip:
        return "Trip not found", 404
    
    # Fetch the specific package manager details using the user_id from the trip
    package_manager = db.child("package_manager").child(trip['user_id']).get().val()
    
    # Fetch the user_image for the logged-in user
    user_id = session['user_id']
    user_image = db.child("per_det").child(user_id).child("image").get().val()
    if not user_image:
        user_image = 'default_image.jpg'  # Default image if not found
    
    return render_template('trip_details.html', trip=trip, package_manager=package_manager, user_image=user_image, trip_id=trip_id)

@app.route('/create_booking', methods=['POST'])
def create_booking():
    if 'user' not in session:
        return jsonify({'success': False, 'message': 'User not logged in'})

    booking_data = request.form.to_dict()
    user_id = booking_data.get('user_id')
    trip_id = booking_data.get('trip_id')

    if not user_id or not trip_id:
        return jsonify({'success': False, 'message': 'Missing user_id or trip_id'})

    try:
        # Check if booking already exists
        existing_bookings = db.child("bookings").get().val()
        if existing_bookings:
            for booking_id, booking in existing_bookings.items():
                if booking['user_id'] == user_id and booking['trip_id'] == trip_id:
                    return jsonify({'success': False, 'message': 'Booking already exists'})

        # Get current date and time
        now = datetime.now()
        booking_data['booking_date'] = now.strftime("%Y-%m-%d")
        booking_data['booking_time'] = now.strftime("%H:%M:%S")
        booking_data['status'] = 0

        new_booking = db.child("bookings").push(booking_data)
        booking_id = new_booking['name']  # This is the unique ID generated by Firebase
        print(f"New booking created with ID: {booking_id}")  # Log the new booking ID
        
        return jsonify({'success': True, 'booking_id': booking_id})
    except Exception as e:
        print(f"Error creating booking: {str(e)}")
        return jsonify({'success': False, 'message': 'Failed to create booking'})
    
#payment gateway
@app.route('/payment/<booking_id>')
def payment(booking_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    return render_template('payment.html', booking_id=booking_id, user_id=session['user_id'])

@app.route('/create-checkout-session', methods=['POST'])
def create_checkout_session():
    try:
        booking_id = request.form['booking_id']
        user_id = request.form['user_id']

        # Fetch booking details from Realtime Database
        booking = db.child("bookings").child(booking_id).get().val()
        if not booking:
            return jsonify({"error": "Booking not found"}), 404

        # Fetch trip details
        trip = db.child("trips").child(booking['trip_id']).get().val()
        if not trip:
            return jsonify({"error": "Trip not found"}), 404

        # Fetch user details (assuming you have a 'users' node in your database)
        user = db.child("users").child(user_id).get().val()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Calculate the amount
        amount = int(float(trip['basic_amount']) * 100)  # Convert to cents

        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'inr',
                    'unit_amount': amount,
                    'product_data': {
                        'name': f"Booking for {trip['location']}",
                    },
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url=url_for('payment_success', booking_id=booking_id, _external=True),
            cancel_url=url_for('payment_cancel', booking_id=booking_id, _external=True),
            customer_email=user.get('email'),
            billing_address_collection='required',
            metadata={
                'booking_id': booking_id,
                'trip_id': booking['trip_id']
            }
        )
        return jsonify({"session_id": checkout_session.id})
    except stripe.error.StripeError as e:
        app.logger.error(f"Stripe error: {str(e)}")
        return jsonify({"error": str(e)}), 403
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/payment_success/<booking_id>')
def payment_success(booking_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    current_time = datetime.now()

    try:
        # Update the booking status
        db.child("bookings").child(booking_id).update({"status": 1})
        
        # Create a new payment entry
        payment_data = {
            "user_id": user_id,
            "booking_id": booking_id,
            "payment_date": current_time.strftime("%Y-%m-%d"),
            "payment_time": current_time.strftime("%H:%M:%S")
        }
        db.child("payments").push(payment_data)

        # Fetch the booking details to get the trip_id
        booking = db.child("bookings").child(booking_id).get().val()
        if booking:
            trip_id = booking.get('trip_id')
            if trip_id:
                # Fetch the current bookings count for the trip
                trip = db.child("trips").child(trip_id).get().val()
                if trip:
                    current_bookings = trip.get('bookings', 0)
                    # Increment the bookings count
                    db.child("trips").child(trip_id).update({"bookings": current_bookings + 1})

                    # Fetch user details
                    user = db.child("users").child(user_id).get().val()
                    user_email = user.get('email')

                    # Fetch trip details
                    trip_location = trip.get('location')
                    trip_start_date = trip.get('start_date')
                    trip_end_date = trip.get('end_date')
                    trip_basic_amount = float(trip.get('basic_amount', 0))
                    trip_tax_percentage = float(trip.get('tax_percentage', 0))
                    trip_discount = float(trip.get('discount', 0))

                    # Calculate total amount
                    total_amount = (trip_basic_amount + (trip_basic_amount * trip_tax_percentage / 100)) - (trip_basic_amount * trip_discount / 100)

                    # Fetch package manager details
                    package_manager_user_id = trip.get('user_id')
                    package_manager = db.child("package_manager").child(package_manager_user_id).get().val()
                    package_manager_name = package_manager.get('c_name')
                    package_manager_email = package_manager.get('c_email')
                    package_manager_phone = package_manager.get('c_phone')

                    # Send confirmation email
                    subject = 'Booking Confirmation'
                    body = f"""
                    Thank You for booking the tour package to {trip_location} on {trip_start_date} to {trip_end_date} at a rate of {total_amount}.
                    The company details for contact is given below:
                    Company: {package_manager_name}
                    Email: {package_manager_email}
                    Phone: {package_manager_phone}
                    """
                    snd_email(user_email, subject, body)

        flash('Payment successful!', 'success')
    except Exception as e:
        app.logger.error(f"Error in payment_success: {str(e)}")
        flash('There was an error processing your payment. Please contact support.', 'danger')

    return redirect(url_for('index'))

def snd_email(to_email, subject, body):
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

@app.route('/payment_cancel/<booking_id>')
def payment_cancel(booking_id):
    return render_template('payment_cancel.html', booking_id=booking_id)

@app.route('/cancel')
def cancel():
    return "Payment Cancelled", 400


#login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Check for admin credentials
        if email == 'admin@admin.com' and password == 'admin@123':
            session['user'] = {'email': email, 'localId': 'admin'}
            session['user_id'] = 'admin'
            return redirect(url_for('admin'))

        try:
            user = auth.sign_in_with_email_and_password(email, password)
            user_data = db.child("users").child(user['localId']).get().val()
            
            status = user_data.get('status')
            role = user_data.get('role')
            
            session['user'] = user
            session['user_id'] = user['localId']

            # Log entry
            now = datetime.now()
            log_entry = {
                "user_id": user['localId'],
                "login_time": now.strftime("%H:%M:%S"),
                "login_date": now.strftime("%Y-%m-%d")
            }
            db.child("log_data").push(log_entry)

            # Simple driver check
            driver_data = db.child("driver").child(user['localId']).get().val()
            # Rest of the conditions remain same
            if status == 0 and role == 0:
                return redirect(url_for('pers_det'))
            elif status == 1 and role == 0:
                return redirect(url_for('photopik'))
            elif status == 2 and role == 0:
                return redirect(url_for('index'))
            elif role == 1 and user_data.get('c_status') == 1:
                return redirect(url_for('pack_manager'))
            elif role == 1 and user_data.get('c_status') == 0:
                flash('Sorry, your account is not approved yet.', 'warning')
                return redirect(url_for('login'))
            elif role==2 and driver_data.get('status')==1:
                return redirect(url_for('driver_dashboard'))
            elif role==2 and driver_data.get('status')==0:
                flash('Sorry, your account is not approved yet.', 'warning')
                return redirect(url_for('login'))
            else:
                return redirect(url_for('index'))

        except Exception as e:
            flash('Invalid email or password', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')

#bookings route
from datetime import datetime

@app.route('/bookings')
def bookings():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    current_date = datetime.now().date()
    
    try:
        # Fetch user's image from per_det
        user_image = db.child("per_det").child(user_id).child("image").get().val()
        # If user_image is None or empty, set it to None (will use default in template)
        user_image = user_image if user_image else None

        # Fetch all bookings for the current user
        all_bookings = db.child("bookings").order_by_child("user_id").equal_to(user_id).get()
        print(f"All bookings: {all_bookings.val()}")  # Debug print
        
        bookings_data = []
        if all_bookings is not None:
            for booking in all_bookings.each():
                booking_id = booking.key()
                booking_info = booking.val()
                trip_id = booking_info.get('trip_id')
                
                print(f"Processing booking: {booking_id}")  # Debug print
                print(f"Trip ID: {trip_id}")  # Debug print
                
                trip_data = db.child("trips").child(trip_id).get().val()
                
                if trip_data:
                    package_manager_id = trip_data.get('user_id')
                    package_manager_data = db.child("package_manager").child(package_manager_id).get().val()
                    
                    if package_manager_data:
                        booking_info = {
                            'booking_id': booking_id,
                            'booking_date': booking_info.get('booking_date'),
                            'booking_time': booking_info.get('booking_time'),
                            'status': 'Return payment completed' if booking_info.get('status') == 3 else (
                                'Cancelled' if booking_info.get('status') == 2 else (
                                'Confirmed' if booking_info.get('status') == 1 else 'Pending'
                            )),
                            'location': trip_data.get('location'),
                            'start_date': trip_data.get('start_date'),
                            'end_date': trip_data.get('end_date'),
                            'basic_amount': trip_data.get('basic_amount'),
                            'discount': trip_data.get('discount'),
                            'tax_percentage': trip_data.get('tax_percentage'),
                            'company_name': package_manager_data.get('c_name'),
                            'company_email': package_manager_data.get('c_email'),
                            'company_phone': package_manager_data.get('c_phone')
                        }
                        
                        # Calculate total amount
                        basic_amount = float(trip_data.get('basic_amount', 0))
                        discount = float(trip_data.get('discount', 0))
                        tax_percentage = float(trip_data.get('tax_percentage', 0))
                        total_amount = basic_amount * (1 - discount/100) * (1 + tax_percentage/100)
                        booking_info['total_amount'] = f"{total_amount:.2f}"
                        
                        # Calculate days until start
                        start_date = datetime.strptime(booking_info['start_date'], '%Y-%m-%d').date()
                        days_until_start = (start_date - current_date).days
                        booking_info['days_until_start'] = days_until_start
                        
                        bookings_data.append(booking_info)
                    else:
                        print(f"No package manager data found for package_manager_id: {package_manager_id}")  # Debug print
                else:
                    print(f"No trip data found for trip_id: {trip_id}")  # Debug print

        print(f"Processed bookings data: {bookings_data}")  # Debug print

        return render_template('bookings.html', bookings=bookings_data, user_image=user_image, current_date=current_date)

    except Exception as e:
        app.logger.error(f"Error fetching bookings: {str(e)}")
        return render_template('bookings.html', bookings=None, user_image=None, current_date=current_date, error="An error occurred while fetching your bookings.")

#invoice generation

@app.route('/generate_invoice/<booking_id>')
def generate_invoice(booking_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    
    try:
        # Fetch booking details
        booking = db.child("bookings").child(booking_id).get().val()
        trip_id = booking['trip_id']
        trip = db.child("trips").child(trip_id).get().val()
        package_manager = db.child("package_manager").child(trip['user_id']).get().val()
        user = db.child("users").child(user_id).get().val()
        per_det = db.child("per_det").child(user_id).get().val()

        # Create a buffer for the PDF
        buffer = BytesIO()

        # Create the PDF object, using the buffer as its "file."
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)

        # Container for the 'Flowable' objects
        elements = []

        # Styles
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='CustomTitle', fontSize=36, alignment=TA_CENTER, spaceAfter=0.5*inch))
        styles.add(ParagraphStyle(name='CompanyName', fontSize=20, alignment=TA_CENTER, spaceAfter=0.3*inch))
        styles.add(ParagraphStyle(name='CustomNormal', fontSize=12, alignment=TA_LEFT, spaceAfter=0.1*inch))
        styles.add(ParagraphStyle(name='CustomBold', fontSize=12, alignment=TA_LEFT, fontName='Helvetica-Bold', spaceAfter=0.1*inch))
        styles.add(ParagraphStyle(name='CustomRightAlign', fontSize=12, alignment=TA_RIGHT))
        styles.add(ParagraphStyle(name='CustomLeftAlign', fontSize=12, alignment=TA_LEFT))
        styles.add(ParagraphStyle(name='CustomCenterBold', fontSize=20, alignment=TA_CENTER, spaceAfter=0.2*inch, fontName='Helvetica-Bold'))
        styles.add(ParagraphStyle(name='DestinationStyle', fontSize=18, alignment=TA_CENTER, spaceAfter=0.2*inch))

        # Title
        elements.append(Paragraph("TRIP WITH ME", styles['CustomTitle']))

        # Company Name
        elements.append(Paragraph(package_manager['c_name'], styles['CompanyName']))

        # Add Date and Time
        current_datetime = datetime.now()
        date_time = Table([
            [Paragraph(f"<b>Date:</b> {current_datetime.strftime('%Y-%m-%d')}", styles['CustomLeftAlign']),
             Paragraph(f"<b>Time:</b> {current_datetime.strftime('%H:%M:%S')}", styles['CustomRightAlign'])]
        ], colWidths=[4*inch, 4*inch])
        elements.append(date_time)
        elements.append(Spacer(1, 0.2*inch))

        # Horizontal line
        elements.append(Table([['']], colWidths=[7*inch], style=[('LINEABOVE', (0,0), (-1,0), 1, colors.black)]))
        elements.append(Spacer(1, 0.2*inch))

        # To Section
        elements.append(Paragraph("To,", styles['CustomBold']))
        elements.append(Paragraph(f"      {user['name']}", styles['CustomNormal']))
        elements.append(Paragraph(f"      {per_det['adname']}", styles['CustomNormal']))
        elements.append(Paragraph(f"      {per_det['district']}", styles['CustomNormal']))
        elements.append(Paragraph(f"      {per_det['state']}", styles['CustomNormal']))
        elements.append(Spacer(1, 0.3*inch))

        # Invoice Section
        elements.append(Paragraph("INVOICE", styles['CustomCenterBold']))
        elements.append(Table([['']], colWidths=[7*inch], style=[('LINEABOVE', (0,0), (-1,0), 1, colors.black)]))
        elements.append(Spacer(1, 0.2*inch))

        # Destination - Centered
        elements.append(Paragraph(f"Destination: {trip['location']}", styles['DestinationStyle']))

        # Calculate amounts
        basic_amount = float(trip['basic_amount'])
        tax_percentage = float(trip['tax_percentage'])
        discount = float(trip['discount'])
        tax = basic_amount * (tax_percentage / 100)
        discount_amount = basic_amount * (discount / 100)
        total_amount = basic_amount + tax - discount_amount

        # Add Amount Details
        amount_data = [
            ['Basic Amount:', f"{basic_amount:.2f}"],
            ['Tax:', f"{tax:.2f}"],
            ['Discount:', f"{discount_amount:.2f}"],
            ['', ''],
            ['Total Amount:', f"{total_amount:.2f}"]
        ]
        amount_table = Table(amount_data, colWidths=[3.5*inch, 3.5*inch])
        amount_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (0,-1), 'LEFT'),
            ('ALIGN', (1,0), (1,-1), 'RIGHT'),
            ('FONTNAME', (0,-1), (1,-1), 'Helvetica-Bold'),
            ('LINEABOVE', (0,-2), (-1,-2), 1, colors.black),
        ]))
        elements.append(amount_table)

        # Add Company Seal at the bottom of the page
        elements.append(Spacer(1, 1*inch))  # Add some space before the seal
        elements.append(Paragraph("Company Seal", ParagraphStyle(name='CompanySeal', fontSize=14, alignment=TA_RIGHT)))

        # Build the PDF
        doc.build(elements)

        # Move the buffer position to the beginning
        buffer.seek(0)

        # Send the PDF as a downloadable file
        return send_file(buffer, as_attachment=True, download_name='invoice.pdf', mimetype='application/pdf')

    except Exception as e:
        app.logger.error(f"Error generating invoice: {str(e)}")
        flash('An error occurred while generating the invoice.', 'error')
        return redirect(url_for('bookings'))
        
#reviewpage route
@app.route('/reviews/<user_id>')
def reviews(user_id):
    if 'user' in session:
        try:
            # Fetch reviews for the given user_id
            all_reviews = db.child("reviews").get().val()
            user_reviews = []
            if all_reviews:
                for review_id, review_data in all_reviews.items():
                    if review_data['reviewee_id'] == user_id:
                        # Fetch reviewer's name and image
                        reviewer_data = db.child("users").child(review_data['reviewer_id']).get().val()
                        reviewer_per_det = db.child("per_det").child(review_data['reviewer_id']).get().val()
                        
                        review_info = {
                            'reviewer_name': reviewer_data.get('name', 'Unknown'),
                            'reviewer_image': reviewer_per_det.get('image', 'default_image.jpg'),
                            'review': review_data['review'],
                            'review_date': review_data['review_date'],
                            'review_time': review_data['review_time']
                        }
                        user_reviews.append(review_info)

            # Fetch the logged-in user's image
            logged_in_user_id = session['user_id']
            logged_in_user_data = db.child("per_det").child(logged_in_user_id).get().val()
            user_image = logged_in_user_data.get('image', 'default_image.jpg')

            return render_template('reviews.html', reviews=user_reviews, user_image=user_image)
        except Exception as e:
            print(f"Error: {str(e)}")
            flash(f'Error fetching review data: {str(e)}', 'danger')
            return redirect(url_for('index'))
    else:
        flash('User not logged in.', 'danger')
        return redirect(url_for('login'))
    
#contact route
@app.route('/submit_contact', methods=['POST'])
def submit_contact():
    if request.method == 'POST':
        data = request.json
        user_id = session.get('user_id', 'anonymous')

        if user_id != 'anonymous':
            # User is logged in, fetch details from database
            user_data = db.child("users").child(user_id).get().val()
            per_det = db.child("per_det").child(user_id).get().val()
            
            name = user_data.get('name', data['name'])
            email = user_data.get('email', data['email'])
            phone = per_det.get('phone', data['phone'])
        else:
            # Use data provided in the form
            name = data['name']
            email = data['email']
            phone = data['phone']

        contact_data = {
            'name': name,
            'email': email,
            'phone': phone,
            'comment': data['comment'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        try:
            db.child("contact").push(contact_data)
            return jsonify({'success': True, 'message': 'Contact form submitted successfully'})
        except Exception as e:
            print(f"Error submitting contact form: {str(e)}")
            return jsonify({'success': False, 'message': 'An error occurred while submitting the form'})

    return jsonify({'success': False, 'message': 'Invalid request method'})


@app.route('/cancel_booking', methods=['POST'])
def cancel_booking():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'User not logged in'})

    booking_id = request.form.get('booking_id')
    if not booking_id:
        return jsonify({'success': False, 'message': 'Missing booking_id'})

    try:
        # Update the booking status to 2 (Cancelled)
        db.child("bookings").child(booking_id).update({"status": 2})

        # You may want to add additional logic here, such as:
        # - Updating the trip's booking count
        # - Initiating a refund process
        # - Sending a cancellation confirmation email

        return jsonify({'success': True, 'message': 'Booking cancelled successfully'})
    except Exception as e:
        app.logger.error(f"Error cancelling booking: {str(e)}")
        return jsonify({'success': False, 'message': 'Failed to cancel booking'})


#comments and reply for admin
@app.route('/get_comments')
def get_comments():
    if 'user_id' in session and session['user_id'] == 'admin':
        comments = db.child("contact").get().val()
        return jsonify(list(comments.values()) if comments else [])
    else:
        return jsonify({'error': 'Unauthorized'}), 401

@app.route('/send_reply', methods=['POST'])
def send_reply():
    if 'user_id' in session and session['user_id'] == 'admin':
        data = request.json
        email = data['email']
        reply = data['reply']
        
        try:
            send_email(email, "Reply from Trip With Me for enquiry", reply)
            return jsonify({'success': True})
        except Exception as e:
            print(f"Error sending email: {str(e)}")
            return jsonify({'success': False})
    else:
        return jsonify({'error': 'Unauthorized'}), 401

def send_email(to_email, subject, body):
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
        return True
    except Exception as e:
        print(f'Error sending email: {str(e)}')
        return False
    
#cancellation payment
@app.route('/get_cancellation_details/<booking_id>')
def get_cancellation_details(booking_id):
    booking = db.child("bookings").child(booking_id).get().val()
    
    if not booking:
        return jsonify({'error': 'Booking not found'}), 404

    trip = db.child("trips").child(booking.get('trip_id')).get().val()
    
    if not trip:
        return jsonify({'error': 'Trip not found'}), 404

    user = db.child("users").child(booking.get('user_id')).get().val()
    
    if not user:
        return jsonify({'error': 'User not found'}), 404

    basic_amount = float(trip.get('basic_amount', 0))
    tax_percentage = float(trip.get('tax_percentage', 0))
    discount = float(trip.get('discount', 0))
    total_amount = basic_amount * (1 + tax_percentage/100) * (1 - discount/100)

    start_date = datetime.strptime(trip.get('start_date', ''), '%Y-%m-%d').date()
    days_until_start = (start_date - datetime.now().date()).days

    if days_until_start > 10:
        return_amount = total_amount
    elif 7 <= days_until_start <= 10:
        return_amount = total_amount * 0.75
    elif 5 <= days_until_start < 7:
        return_amount = total_amount * 0.50
    else:
        return_amount = total_amount * 0.25

    return jsonify({
        'name': user.get('name', ''),
        'email': user.get('email', ''),
        'return_amount': return_amount
    })

@app.route('/return_payment/<booking_id>')
def return_payment(booking_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    return render_template('return_payment.html', booking_id=booking_id, user_id=session['user_id'])

@app.route('/create-return-checkout-session', methods=['POST'])
def create_return_checkout_session():
    try:
        booking_id = request.form['booking_id']
        user_id = request.form['user_id']

        # Fetch booking details
        booking = db.child("bookings").child(booking_id).get().val()
        if not booking:
            return jsonify({"error": "Booking not found"}), 404

        # Fetch trip details
        trip = db.child("trips").child(booking['trip_id']).get().val()
        if not trip:
            return jsonify({"error": "Trip not found"}), 404

        # Fetch user details
        user = db.child("users").child(user_id).get().val()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Fetch personal details
        pers_det = db.child("per_det").child(user_id).get().val()
        if not pers_det:
            return jsonify({"error": "Personal details not found"}), 404

        # Calculate the return amount
        return_amount = calculate_return_amount(booking, trip)
        amount = int(return_amount * 100)  # Convert to cents

        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'inr',
                    'unit_amount': amount,
                    'product_data': {
                        'name': f"Return Payment for Booking {booking_id}",
                    },
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url=url_for('return_payment_success', booking_id=booking_id, _external=True),
            cancel_url=url_for('return_payment_cancel', booking_id=booking_id, _external=True),
            customer_email=user.get('email'),
            metadata={
                'booking_id': booking_id,
                'trip_id': booking['trip_id']
            },
            payment_intent_data={
                'shipping': {
                    'name': user.get('name', ''),
                    'address': {
                        'line1': pers_det.get('adname', ''),
                        'city': pers_det.get('district', ''),
                        'state': pers_det.get('state', ''),
                        'postal_code': pers_det.get('pin', ''),
                        'country': 'IN',  # Assuming the country is India
                    },
                },
            },
        )
        return jsonify({"session_id": checkout_session.id})
    except stripe.error.StripeError as e:
        app.logger.error(f"Stripe error: {str(e)}")
        return jsonify({"error": str(e)}), 403
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/return_payment_success/<booking_id>')
def return_payment_success(booking_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    current_time = datetime.now()

    try:
        # Update the booking status
        db.child("bookings").child(booking_id).update({"status": 3})  # 3 for "Return payment completed"
        
        # Create a new return payment entry
        payment_data = {
            "user_id": user_id,
            "booking_id": booking_id,
            "payment_date": current_time.strftime("%Y-%m-%d"),
            "payment_time": current_time.strftime("%H:%M:%S")
        }
        db.child("return_payments").push(payment_data)

        # Fetch user details
        user = db.child("users").child(user_id).get().val()
        user_email = user.get('email')

        # Send confirmation email
        subject = "Return Payment Confirmation"
        body = f"""
        Dear {user.get('name')},
        We have processed your return payment for booking {booking_id} on {current_time.strftime("%Y-%m-%d")} after a small deduction. Thank you for your patience.
        """
        send_email(user_email, subject, body)

        flash('Return payment processed successfully!', 'success')
    except Exception as e:
        app.logger.error(f"Error in return_payment_success: {str(e)}")
        flash('There was an error processing your return payment. Please contact support.', 'danger')

    # Redirect to pack_manager.html instead of index
    return redirect(url_for('pack_manager'))

@app.route('/return_payment_cancel/<booking_id>')
def return_payment_cancel(booking_id):
    flash('Return payment was cancelled.', 'info')
    return redirect(url_for('index'))

def calculate_return_amount(booking, trip):
    # Implement your return amount calculation logic here
    # This is just a placeholder implementation
    basic_amount = float(trip.get('basic_amount', 0))
    tax_percentage = float(trip.get('tax_percentage', 0))
    discount = float(trip.get('discount', 0))
    total_amount = basic_amount * (1 + tax_percentage/100) * (1 - discount/100)

    start_date = datetime.strptime(trip.get('start_date', ''), '%Y-%m-%d').date()
    days_until_start = (start_date - datetime.now().date()).days

    if days_until_start > 10:
        return_amount = total_amount
    elif 7 <= days_until_start <= 10:
        return_amount = total_amount * 0.75
    elif 5 <= days_until_start < 7:
        return_amount = total_amount * 0.50
    else:
        return_amount = total_amount * 0.25

    return return_amount


@app.route('/logout')
def logout():
    if 'user_id' in session:
        user_id = session['user_id']
        try:
            # Update driver status to 0 when logging out
            db.child("driver").child(user_id).update({
                "e_status": 0,
                "last_status_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        except Exception as e:
            print(f"Error updating status on logout: {str(e)}")
    
    session.clear()
    return redirect(url_for('login'))

@app.route('/driver_dashboard')
def driver_dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    
    try:
        # Check driver's f_status
        driver = db.child("driver").child(user_id).get().val()
        f_status = driver.get('f_status', 0) if driver else 0
        
        # Get current year for manufacturing year dropdown
        current_year = datetime.now().year
        
        # Car companies list
        car_companies = [
            "Maruti Suzuki", "Hyundai", "Tata Motors", "Mahindra", "Toyota", 
            "Honda", "Kia", "MG", "Volkswagen", "Skoda", "Ford", "Renault",
            "Nissan", "BMW", "Mercedes-Benz", "Audi", "Volvo", "Jeep",
            "Land Rover", "Porsche", "Jaguar", "Mini", "Lexus"
        ]

        # Create structured driver data
        driver_data = {
            'name': driver.get('name', 'Driver'),
            'vehicle': {
                'model': driver.get('car_model', 'Not Set'),
                'number': driver.get('car_number', 'Not Set')
            },
            'stats': {
                'earnings': driver.get('today_earnings', 0),
                'total_earnings': driver.get('total_earnings', 0),
                'trips': driver.get('today_trips', 0),
                'rating': driver.get('rating', 4.5)
            },
            'earnings_chart': {
                'labels': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                'data': [2100, 1800, 2400, 1900, 2800, 3100, 2500]  # Dummy data
            },
            'trips_chart': {
                'labels': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                'data': [5, 4, 6, 3, 7, 8, 5]  # Dummy data
            },
            'current_location': {
                'lat': driver.get('current_lat', 10.0261),
                'lng': driver.get('current_lng', 76.3125)
            }
        }
        
        return render_template('driver_dashboard.html', 
                             f_status=f_status,
                             car_companies=car_companies,
                             current_year=current_year,
                             driver_data=driver_data)
    except Exception as e:
        print(f"Error in driver_dashboard: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route('/submit_car_details', methods=['POST'])
def submit_car_details():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    try:
        user_id = session['user_id']
        car_type = request.form.get('car_type')
        
        # Get rate configuration for the car type
        rate_config = CAR_RATE_CONFIG.get(car_type.lower())
        if not rate_config:
            return jsonify({'success': False, 'message': 'Invalid car type'}), 400
        
        # Get form data
        car_company = request.form.get('car_company')
        car_model = request.form.get('car_model')
        manufacturing_year = request.form.get('manufacturing_year')
        seating_capacity = request.form.get('seating_capacity')
        
        # Handle file uploads
        car_photos = []
        for i in range(1, 4):
            photo = request.files.get(f'car_photo_{i}')
            if photo:
                filename = f"{user_id}_car_{i}_{photo.filename}"
                photo.save(os.path.join('static/images/cars', filename))
                car_photos.append(filename)
        
        # Save to database
        car_data = {
            "user_id": user_id,
            "car_company": car_company,
            "car_model": car_model,
            "car_type": car_type,
            "manufacturing_year": manufacturing_year,
            "seating_capacity": seating_capacity,
            "car_photos": car_photos,
            "base_rate": rate_config['base_rate'],
            "extra_km_rate": rate_config['extra_km_rate'],
            "updated_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status":0
        }
        
        db.child("cars").child(user_id).set(car_data)
        
        # Update f_status in driver collection
        db.child("driver").child(user_id).update({"f_status": 1})
        
        return jsonify({'success': True, 'message': 'Car details saved successfully'})
    except Exception as e:
        print(f"Error submitting car details: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500



@app.route('/get_vehicle_info')
def get_vehicle_info():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        user_id = session['user_id']
        car_data = db.child("cars").child(user_id).get().val()
        
        if car_data:
            return jsonify({
                'success': True,
                'data': car_data
            })
        return jsonify({'error': 'No vehicle information found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/update_vehicle_info', methods=['POST'])
def update_vehicle_info():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        user_id = session['user_id']
        car_type = request.form.get('car_type')
        
        # Get rate configuration for the car type
        rate_config = CAR_RATE_CONFIG.get(car_type.lower())
        if not rate_config:
            return jsonify({'success': False, 'message': 'Invalid car type'}), 400
        
        # Get form data
        car_company = request.form.get('car_company')
        car_model = request.form.get('car_model')
        manufacturing_year = request.form.get('manufacturing_year')
        seating_capacity = request.form.get('seating_capacity')
        
        # Get current car data to preserve existing photos if no new ones are uploaded
        current_car_data = db.child("cars").child(user_id).get().val()
        car_photos = current_car_data.get('car_photos', []) if current_car_data else []
        
        # Handle new photo uploads
        for i in range(1, 4):
            if f'car_photo_{i}' in request.files:
                photo = request.files[f'car_photo_{i}']
                if photo and photo.filename:
                    # Generate unique filename
                    filename = f"{user_id}_car_{i}_{secure_filename(photo.filename)}"
                    photo.save(os.path.join('static/images/cars', filename))
                    # Update the photo in the list
                    if len(car_photos) >= i:
                        car_photos[i-1] = filename
                    else:
                        car_photos.append(filename)
        
        # Update car data
        update_data = {
            "car_company": car_company,
            "car_model": car_model,
            "car_type": car_type,
            "manufacturing_year": manufacturing_year,
            "seating_capacity": seating_capacity,
            "car_photos": car_photos,
            "base_rate": rate_config['base_rate'],
            "extra_km_rate": rate_config['extra_km_rate'],
            "updated_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Update in database
        db.child("cars").child(user_id).update(update_data)
        
        return jsonify({
            'success': True,
            'message': 'Vehicle information updated successfully',
            'data': update_data
        })
    except Exception as e:
        print(f"Error updating vehicle info: {str(e)}")  # For debugging
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/update_driver_location', methods=['POST'])
def update_driver_location():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    try:
        user_id = session['user_id']
        data = request.json
        current_time = datetime.now()
        
        # Update location in database
        update_data = {
            "current_lat": data['lat'],
            "current_lng": data['lng'],
            "last_location_update": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "is_online": True  # Add online status
        }
        
        db.child("driver").child(user_id).update(update_data)
        
        print(f"Location updated for driver {user_id} at {current_time}")
        
        return jsonify({
            'success': True,
            'message': 'Location updated successfully',
            'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        error_message = f"Error updating location: {str(e)}"
        print(error_message)  # For debugging
        return jsonify({'success': False, 'message': error_message}), 500

@app.route('/update_driver_live_status', methods=['POST'])
def update_driver_live_status():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    try:
        user_id = session['user_id']
        data = request.json
        status = data.get('status')
        
        # Update status in database
        db.child("driver").child(user_id).update({
            "e_status": status,
            "last_status_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        return jsonify({
            'success': True,
            'message': 'Status updated successfully'
        })
        
    except Exception as e:
        print(f"Error updating status: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

# Add this function to calculate average rating
def get_driver_rating(driver_id):
    try:
        # Get all ratings for the driver
        ratings = db.child("ratings").child(driver_id).get().val()
        
        if not ratings:
            # If no ratings exist, create default 5-star rating from Admin
            default_rating = {
                "rating": 5,
                "rater_id": "Admin",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            db.child("ratings").child(driver_id).push(default_rating)
            return 5.0
        
        # Calculate average rating
        total_rating = 0
        count = 0
        for rating in ratings.values():
            total_rating += float(rating['rating'])
            count += 1
        
        return round(total_rating / count, 1)
    except Exception as e:
        print(f"Error calculating rating: {str(e)}")
        return 5.0  # Return default rating on error

@app.route('/get_driver_ratings', methods=['GET'])
def get_driver_ratings():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    try:
        driver_id = session['user_id']
        ratings = db.child("ratings").child(driver_id).get().val()
        average_rating = get_driver_rating(driver_id)
        
        return jsonify({
            'success': True,
            'ratings': ratings if ratings else {},
            'average_rating': average_rating
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/available_cabs')
def available_cabs():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    try:
        # Get user data from database
        user_id = session['user']
        user_data = db.child("users").child(user_id).get().val()
        
        # Get user image, use default if not found
        user_image = user_data.get('photo', 'default.jpg') if user_data else 'default.jpg'
        
        return render_template('available_cabs.html', user_image=user_image)
    except Exception as e:
        print(f"Error loading available cabs: {str(e)}")
        # Use default image if there's an error
        return render_template('available_cabs.html', user_image='default.jpg')

@app.route('/get_nearby_drivers')
def get_nearby_drivers():
    if 'user' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
        
    try:
        lat = float(request.args.get('lat'))
        lng = float(request.args.get('lng'))
        
        # Get all drivers
        drivers = db.child("driver").get().val()
        nearby_drivers = []
        
        if drivers:
            for driver_id, driver in drivers.items():
                # Check if driver is active and available
                if driver.get('e_status') == 1:
                    # Calculate distance
                    driver_lat = float(driver.get('current_lat', 0))
                    driver_lng = float(driver.get('current_lng', 0))
                    distance = calculate_distance(lat, lng, driver_lat, driver_lng)
                    
                    if distance <= 2:  # Within 2km radius
                        # Get car and rating info
                        car_info = db.child("cars").child(driver_id).get().val()
                        rating = get_driver_rating(driver_id)
                        
                        driver_data = {
                            'id': driver_id,
                            'name': driver.get('name', 'Unknown'),
                            'photo': driver.get('photo', 'default.jpg'),
                            'rating': rating,
                            'car_company': car_info.get('car_company'),
                            'car_model': car_info.get('car_model'),
                            'car_type': car_info.get('car_type'),
                            'car_photo': car_info.get('car_photos', [])[0],
                            'distance': round(distance, 1),
                            'base_rate': car_info.get('base_rate'),
                            'extra_km_rate': car_info.get('extra_km_rate')
                        }
                        nearby_drivers.append(driver_data)
        
        return jsonify({
            'success': True,
            'drivers': nearby_drivers
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

def calculate_distance(lat1, lon1, lat2, lon2):
    # Haversine formula to calculate distance between two points
    R = 6371  # Earth's radius in kilometers
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat/2) * math.sin(dlat/2) + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.sin(dlon/2) * math.sin(dlon/2)
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return distance

@app.route('/book_trip', methods=['POST'])
def book_trip():
    if 'user' not in session:
        return jsonify({'success': False, 'message': 'Please login first'})
    
    try:
        data = request.json
        
        # Create a new booking entry in the database
        booking_ref = db.child("cab_bookings").push({
            'user_id': session['user_id'],
            'driver_id': data['driver_id'],
            'from_location': data['from_location'],
            'to_location': data['to_location'],
            'booking_time': data['booking_time'],
            'trip_amount': data['trip_amount'],
            'booking_status': 0,  # "0" for pending
            'trip_status': 0, # "0" for not started
            'payment_status': 0
        })
        
        if booking_ref:
            return jsonify({'success': True, 'booking_id': booking_ref['name']})
        else:
            return jsonify({'success': False, 'message': 'Failed to create booking'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/get_active_trips')
def get_active_trips():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    try:
        driver_id = session['user_id']
        
        # Get active bookings for this driver
        bookings = db.child("cab_bookings").order_by_child("driver_id").equal_to(driver_id).get().val()
        active_trips = []
        
        if bookings:
            for booking_id, booking in bookings.items():
                # Check if booking is active (status 0 or 1)
                if booking.get('booking_status') in [0, 1] and booking.get('trip_status') in [0, 1]:
                    # Get user details
                    user_id = booking.get('user_id')
                    user = db.child("users").child(user_id).get().val()
                    user_details = db.child("per_det").child(user_id).get().val()
                    
                    trip_data = {
                        'booking_id': booking_id,
                        'user_name': user.get('name', 'Unknown'),
                        'user_image': user_details.get('image', 'default.jpg') if user_details else 'default.jpg',
                        'from_location': booking.get('from_location'),
                        'to_location': booking.get('to_location'),
                        'booking_time': booking.get('booking_time'),
                        'trip_amount': booking.get('trip_amount'),
                        'booking_status': booking.get('booking_status'),
                        'trip_status': booking.get('trip_status'),
                        'payment_status': booking.get('payment_status', 0)
                    }
                    active_trips.append(trip_data)
        
        return jsonify({
            'success': True,
            'trips': active_trips
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
@app.route('/get_driver_trips')
def get_driver_trips():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'})
    
    try:
        driver_id = session['user_id']
        
        # Get all cab bookings
        all_bookings = db.child("cab_bookings").get().val()
        if not all_bookings:
            return jsonify({'success': True, 'trips': []})
        
        # Filter bookings for this driver
        trips = []
        for booking_id, booking in all_bookings.items():
            if booking.get('driver_id') == driver_id:
                # Get user details
                user_id = booking.get('user_id')
                user_data = db.child("users").child(user_id).get().val()
                user_details = db.child("per_det").child(user_id).get().val()
                
                trip_data = {
                    'booking_id': booking_id,
                    'user_id': user_id,
                    'user_name': user_data.get('name', 'Unknown User') if user_data else 'Unknown User',
                    'user_image': user_details.get('image', 'default.jpg') if user_details else 'default.jpg',
                    'from_location': booking.get('from_location'),
                    'to_location': booking.get('to_location'),
                    'trip_amount': booking.get('trip_amount'),
                    'booking_time': booking.get('booking_time'),
                    'booking_status': booking.get('booking_status', 0),
                    'trip_status': booking.get('trip_status', 0),
                    'payment_status': booking.get('payment_status', 0)
                }
                trips.append(trip_data)
        
        # Sort trips by booking time, most recent first
        trips.sort(key=lambda x: x.get('booking_time', ''), reverse=True)
        
        return jsonify({
            'success': True,
            'trips': trips
        })
        
    except Exception as e:
        print(f"Error fetching trips: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })
@app.route('/update_trip_status', methods=['POST'])
def update_trip_status():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'})
    
    try:
        data = request.json
        booking_id = data.get('booking_id')
        new_status = data.get('status')
        
        if not booking_id or new_status is None:
            return jsonify({'success': False, 'error': 'Missing required data'})
        
        # Update the booking status in the database
        db.child("cab_bookings").child(booking_id).update({
            'trip_status': new_status
        })
        
        return jsonify({
            'success': True,
            'message': 'Trip status updated successfully'
        })
        
    except Exception as e:
        print(f"Error updating trip status: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/update_payment_status', methods=['POST'])
def update_payment_status():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    try:
        data = request.json
        booking_id = data.get('booking_id')
        payment_status = data.get('payment_status')
        
        if not booking_id:
            return jsonify({'success': False, 'message': 'Missing booking ID'}), 400
        
        # Update the payment status in the database
        db.child("cab_bookings").child(booking_id).update({
            'payment_status': payment_status
        })
        
        return jsonify({
            'success': True,
            'message': 'Payment status updated successfully'
        })
        
    except Exception as e:
        print(f"Error updating payment status: {str(e)}")
        return jsonify({
            'success': False, 
            'message': f'Error updating payment status: {str(e)}'
        }), 500
@app.route('/get_user_trips')
def get_user_trips():
    if 'user_id' not in session:  # Change this according to your session key
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    try:
        user_id = session['user_id']  # Get user_id from session
        print(f"Fetching trips for user: {user_id}")  # Debug print
        
        # Get all bookings for this user
        bookings = db.child("cab_bookings").get().val()
        active_trips = []
        
        if bookings:
            for booking_id, booking_data in bookings.items():
                # Check if this booking belongs to the current user
                if booking_data.get('user_id') == user_id:
                    # Check if it's an active trip
                    if (booking_data.get('booking_status') == 0 or 
                        booking_data.get('trip_status') in [0, 1] or 
                        (booking_data.get('trip_status') == 2 and 
                         booking_data.get('payment_status', 0) == 0)):
                        
                        # Get user details
                        try:
                            user = db.child("users").child(user_id).get().val()
                            user_details = db.child("per_det").child(user_id).get().val()
                        except Exception as e:
                            print(f"Error fetching user details: {e}")
                            user = {}
                            user_details = {}
                        
                        # Add all required data to trip_data
                        trip_data = {
                            'booking_id': booking_id,
                            'user_name': user.get('name', 'Unknown'),
                            'user_image': user_details.get('image', 'default.jpg'),
                            'from_location': booking_data.get('from_location'),
                            'to_location': booking_data.get('to_location'),
                            'booking_time': booking_data.get('booking_time'),
                            'trip_amount': booking_data.get('trip_amount'),
                            'booking_status': booking_data.get('booking_status'),
                            'trip_status': booking_data.get('trip_status'),
                            'payment_status': booking_data.get('payment_status', 0)
                        }
                        active_trips.append(trip_data)
        
        print(f"Found {len(active_trips)} active trips")  # Debug print
        return jsonify({
            'success': True,
            'trips': active_trips
        })
        
    except Exception as e:
        print(f"Error in get_user_trips: {str(e)}")  # Debug print
        return jsonify({
            'success': False,
            'message': f"Server error: {str(e)}"
        }), 500

@app.route('/complete_payment', methods=['POST'])
def complete_payment():
    if 'user' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    try:
        data = request.json
        booking_id = data.get('booking_id')
        token = data.get('token')
        
        # Here you would normally process the payment with Stripe
        # For demo purposes, we'll just update the payment status
        
        db.child("cab_bookings").child(booking_id).update({
            'payment_status': 1
        })
        
        return jsonify({
            'success': True,
            'message': 'Payment completed successfully'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/get_driver_location')
def get_driver_location():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'})
    
    try:
        driver_id = session['user_id']
        driver_data = db.child("driver").child(driver_id).get().val()
        
        if driver_data and 'current_lat' in driver_data and 'current_lng' in driver_data:
            return jsonify({
                'success': True,
                'location': {
                    'lat': float(driver_data['current_lat']),
                    'lng': float(driver_data['current_lng'])
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Driver location not found'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })
@app.route('/get_trip_amount/<booking_id>')
def get_trip_amount(booking_id):
    try:
        booking = db.child("cab_bookings").child(booking_id).get().val()
        if booking:
            return jsonify({
                'success': True,
                'amount': booking.get('trip_amount', 0)
            })
        return jsonify({'success': False, 'error': 'Booking not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/create_payment_intent', methods=['POST'])
def create_payment_intent():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'})
        
    try:
        data = request.json
        booking_id = data.get('booking_id')
        amount = int(float(data.get('amount')) * 100)  # Convert to cents
        
        if not booking_id or not amount:
            return jsonify({'success': False, 'error': 'Missing required data'})
        
        # Create a PaymentIntent with Stripe
        intent = stripe.PaymentIntent.create(
            amount=amount,
            currency='inr',
            payment_method_types=['card'],
            metadata={'booking_id': booking_id}
        )
        
        # Generate QR code data URL containing the payment URL
        payment_url = f"https://checkout.stripe.com/pay/{intent.client_secret}"
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(payment_url)
        qr.make(fit=True)
        
        # Create QR code image
        qr_image = qr.make_image(fill_color="black", back_color="white")
        
        # Convert QR image to base64 string
        buffered = BytesIO()
        qr_image.save(buffered, format="PNG")
        qr_code_base64 = base64.b64encode(buffered.getvalue()).decode()
        qr_code_url = f"data:image/png;base64,{qr_code_base64}"
        
        return jsonify({
            'success': True,
            'qr_code_url': qr_code_url,
            'client_secret': intent.client_secret
        })
        
    except Exception as e:
        print(f"Error creating payment intent: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/recommendations')
def recommendations():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    user_image = db.child("per_det").child(user_id).child("image").get().val()
    return render_template('recommendations.html', user_image=user_image)
@app.route('/get_trip_history')
def get_trip_history():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'})
    
    try:
        driver_id = session['user_id']
        # Get all bookings for the driver where all statuses are 1
        bookings = db.child("cab_bookings").get().val()
        completed_trips = []
        
        if bookings:
            for booking_id, booking in bookings.items():
                if (booking.get('driver_id') == driver_id and 
                    booking.get('booking_status') == 1 and 
                    booking.get('trip_status') == 1 and 
                    booking.get('payment_status') == 1):
                    
                    # Get user details
                    user_id = booking.get('user_id')
                    user_data = db.child("users").child(user_id).get().val()
                    user_details = db.child("per_det").child(user_id).get().val()
                    
                    trip_data = {
                        'booking_id': booking_id,
                        'user_id': user_id,
                        'user_name': user_data.get('name', 'Unknown User') if user_data else 'Unknown User',
                        'user_image': user_details.get('image', 'default.jpg') if user_details else 'default.jpg',
                        'from_location': booking.get('from_location'),
                        'to_location': booking.get('to_location'),
                        'trip_amount': booking.get('trip_amount'),
                        'booking_time': booking.get('booking_time')
                    }
                    completed_trips.append(trip_data)
        
        # Sort trips by booking time, most recent first
        completed_trips.sort(key=lambda x: x.get('booking_time', ''), reverse=True)
        
        return jsonify({
            'success': True,
            'trips': completed_trips
        })
        
    except Exception as e:
        print(f"Error fetching trip history: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })
@app.route('/get_recommendations')
def get_recommendations():
    if 'user' not in session:
        return jsonify({'error': 'Not logged in'})

    user_id = session['user_id']
    
    try:
        # Get user's past trips and reviews
        user_bookings = db.child("bookings").order_by_child("user_id").equal_to(user_id).get().val()
        user_reviews = db.child("reviews").order_by_child("reviewer_id").equal_to(user_id).get().val()
        
        # Get all available trips
        all_trips = db.child("trips").get().val()
        
        if not all_trips:
            return jsonify({'recommendations': []})

        # Create text corpus for TF-IDF
        trip_texts = []
        trip_ids = []
        
        for trip_id, trip in all_trips.items():
            text = f"{trip.get('location', '')} {trip.get('description', '')}"
            trip_texts.append(text)
            trip_ids.append(trip_id)

        # Create user profile based on past interactions
        user_profile = ""
        if user_bookings:
            for booking in user_bookings.values():
                trip_id = booking.get('trip_id')
                if trip_id in all_trips:
                    user_profile += f" {all_trips[trip_id].get('location', '')}"
                    user_profile += f" {all_trips[trip_id].get('description', '')}"
        
        if user_reviews:
            for review in user_reviews.values():
                user_profile += f" {review.get('review', '')}"

        # If no user profile, use a default one
        if not user_profile.strip():
            user_profile = "travel vacation tourism"

        # Add user profile to corpus
        all_texts = trip_texts + [user_profile]
        
        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(all_texts)
        
        # Calculate similarity scores
        user_vector = tfidf_matrix[-1]
        similarity_scores = cosine_similarity(user_vector, tfidf_matrix[:-1])
        
        # Get top recommendations
        top_indices = similarity_scores[0].argsort()[::-1][:6]  # Get top 6 recommendations
        
        recommendations = []
        for idx in top_indices:
            trip_id = trip_ids[idx]
            trip = all_trips[trip_id]
            recommendations.append({
                'trip_id': trip_id,
                'location': trip.get('location'),
                'description': trip.get('description'),
                'image': trip.get('image1'),
                'score': similarity_scores[0][idx]
            })
        
        return jsonify({'recommendations': recommendations})
    
    except Exception as e:
        print(f"Error generating recommendations: {str(e)}")
        return jsonify({'error': str(e)})

if __name__ == '__main__': 
    app.run(debug=True)
