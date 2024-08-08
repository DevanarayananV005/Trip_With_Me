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
    "apiKey": "#",
    "authDomain": "#",
    "projectId": "#",
    "storageBucket": "#",
    "messagingSenderId": "#",
    "appId": "#",
    "measurementId": "W",
    "databaseURL": "#"
}

# Initialize Firebase with Pyrebase
firebase = pyrebase.initialize_app(config)
auth = firebase.auth()
db = firebase.database()

# SMTP email configuration
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SMTP_USERNAME = '#'
SMTP_PASSWORD = '#'

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

@app.route('/')
def index():
    if 'user' in session:
        user_id = session['user_id']
        pers_det = db.child("per_det").child(user_id).get().val()
        user_image = pers_det.get('image', 'default_image.jpg')  # Default image if not found
        response = make_response(render_template('index.html', user_image=user_image, user_logged_in=True))
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
                "role" : 0  # Default value for status
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
                image_name=image_name
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
            return render_template('scheduletrip.html')
    else:
        flash('User not logged in.', 'danger')
        return redirect(url_for('login'))


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
                for key, trip in scheduletrips.items():
                    # Check if tripstatus is 0
                    if trip.get('tripstatus') != 0:
                        continue  # Skip this trip if tripstatus is not 0

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
                    from datetime import datetime
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
            
            return render_template('findtrip.html', trips_data=trips_data)
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

        return render_template('notifications.html', notifications=notifications)
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

        return render_template('admin.html', log_data=processed_log_data, user_data=user_data, scheduled_trips=scheduled_trips, confirmed_trips=confirmed_trips, enumerate=enumerate)
    else:
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('login'))

#deactivate user
@app.route('/deactivate_user/<user_id>', methods=['POST'])
def deactivate_user(user_id):
    if 'user_id' in session and session['user_id'] == 'admin':
        try:
            # Update the user's status to 4 (deactivated)
            db.child("users").child(user_id).update({"status": 4})
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})
    else:
        return jsonify({"success": False, "error": "Unauthorized access"})


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

            if status == 4:
                flash('Your account is temporarily deactivated.', 'danger')
                return redirect(url_for('login'))

            session['user'] = user
            session['user_id'] = user['localId']

            # Capture current date and time
            now = datetime.now()
            login_date = now.strftime("%Y-%m-%d")
            login_time = now.strftime("%H:%M:%S")

            # Create a log entry in Firebase
            log_entry = {
                "user_id": user['localId'],  # Use actual user_id
                "login_time": login_time,
                "login_date": login_date
            }

            # Push the log entry and get the log entry ID
            db.child("log_data").push(log_entry)

            if status == 0:
                return redirect(url_for('pers_det'))
            elif status == 1:
                return redirect(url_for('photopik'))
            elif status == 2:
                return redirect(url_for('index'))
        except Exception as e:
            error_message = str(e)
            if "INVALID_PASSWORD" in error_message or "EMAIL_NOT_FOUND" in error_message:
                flash('Wrong credentials, please try again.', 'danger')
            else:
                flash(f'Wrong credentials, please try again.', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()  # Clear the session
    flash('You have been logged out!', 'info')
    return redirect(url_for('index')) 

if __name__ == '__main__':
    app.run(debug=True)

    app.run(debug=True)