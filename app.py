import streamlit as st
import pandas as pd
import os
import random
import string
from PIL import Image
import numpy as np
import tensorflow_hub as hub
import json
import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart 
import base64
import re 

# Set page title
st.set_page_config(page_title="Plant Disease Prediction", layout="wide")

# Session state for authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "page" not in st.session_state:
    st.session_state.page = "home"  # Default page

# Paths
user_data_path = "pages/user_data.xlsx"
model_path = "models/plant_disease_model.tflite"

json_path = "data/disease_info.json"


# Load or create user data
def load_users():
    if os.path.exists(user_data_path):
        return pd.read_excel(user_data_path)
    else:
        return pd.DataFrame(columns=["Username", "Email", "Gender", "Place", "Phone", "Password"])

def save_users(users_df):
    users_df.to_excel(user_data_path, index=False, engine="openpyxl")


def load_plant_disease_model_tflite():
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return interpreter, input_details, output_details
    except Exception as e:
        st.error(f"❌ Error loading TFLite model: {e}")
        st.stop()

# Load TFLite model
interpreter, input_details, output_details = load_plant_disease_model_tflite()


# ✅ Load and ensure `disease_info` keys are strings
with open("D:/Individual Project39/data/disease_info.json", "r") as f:
    disease_info = {str(key): value for key, value in json.load(f).items()}  # Convert keys to strings


# List of class labels (Ensure this follows the dataset order)
class_labels = [
    "Apple___Apple_scab",
    "Apple___Black_rot", 
    "Apple___Cedar_apple_rust", 
    "Apple___healthy",
    "Blueberry___healthy", 
    "Cherry_(including_sour)___healthy", 
    "Cherry_(including_sour)___Powdery_mildew",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot", 
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy", 
    "Corn_(maize)___Northern_Leaf_Blight", 
    "Corn___Gray_Leaf_Spot",
    "Grape___Black_rot", 
    "Grape___Esca_(Black_Measles)", 
    "Grape___healthy", 
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Orange___Haunglongbing_(Citrus_greening)", 
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper__bell___Bacterial_spot", 
    "Pepper__bell___healthy", "Potato___Early_blight",
    "Potato___healthy", 
    "Potato___Late_blight", 
    "Raspberry___healthy", 
    "Rice___Brown_Spot",
    "Rice___Healthy", 
    "Rice___Leaf_Blast", 
    "Rice___Neck_Blast", 
    "Soybean___healthy",
    "Squash___Powdery_mildew", 
    "Strawberry___healthy", 
    "Strawberry___Leaf_scorch",
    "Sugarcane_Bacterial_Blight", 
    "Sugarcane_Healthy", 
    "Sugarcane_Red_Rot",
    "Tomato__Target_Spot", 
    "Tomato__Tomato_mosaic_virus", 
    "Tomato__Tomato_YellowLeaf_Curl_Virus",
    "Tomato_Bacterial_spot", 
    "Tomato_Early_blight", 
    "Tomato_healthy", 
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold", 
    "Tomato_Septoria_leaf_spot", 
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Wheat___Brown_Rust", 
    "Wheat___Healthy", 
    "Wheat___Yellow_Rust"
]

# Ensure the JSON file has matching class labels
if len(disease_info) != len(class_labels):
    st.error("❌ JSON file does not have the correct number of classes. Check the order!")
    st.stop()

# Map predicted class index to label
def get_predicted_label(index):
    if 0 <= index < len(class_labels):
        return class_labels[index]
    return "Unknown"

def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_image}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )



def show_home():
    set_background("bc2.png")
    
    st.markdown("""
    <style>
    @keyframes glow {
        0% { text-shadow: 0 0 5px #ffffff, 0 0 10px #66ff66, 0 0 15px #66ff66; }
        50% { text-shadow: 0 0 10px #ffffff, 0 0 20px #33ff33, 0 0 30px #33ff33; }
        100% { text-shadow: 0 0 5px #ffffff, 0 0 10px #66ff66, 0 0 15px #66ff66; }
    }

    .animated-title {
        text-align: center;
        color: white;
        font-size: 65px;
        font-weight: bold;
        animation: glow 2s infinite alternate;
    }
    </style>

    <h1 class="animated-title"><br>🌱 Welcome to Plant Disease Prediction<br></h1>
    """, unsafe_allow_html=True)

    # Description about Plant Disease Prediction
    st.markdown("""
    <p style='text-align: center; color: white; font-size: 24px; font-weight: bold; line-height: 1.8;'>
        Our Plant Disease Prediction system helps farmers and plant enthusiasts diagnose plant diseases using Machine Learning. <br>
        Simply upload a leaf image, and our system will predict the disease, its cause, and suggest possible solutions.<br>
    </p>
    """, unsafe_allow_html=True)

    # Motivational Quote
    st.markdown("""
    <p style='text-align: center; color: white; font-size: 20px; font-style: Cascadia Code SemiBold;'>
       <br> "Every plant is a wish of nature. Keep them healthy, keep them growing." 🌿
    </p>
    """, unsafe_allow_html=True)
    
    
def is_valid_email(email):
    return bool(re.match(r"^[a-zA-Z0-9_.+-]+@gmail\.com$", email))

def is_valid_phone(phone):
    return phone.isdigit()

# ✅ Function to Show Signup Page
def show_signup():
    set_background("sc1.png")

    # ✅ Ensure session state is initialized
    if "page" not in st.session_state:
        st.session_state["page"] = "signup"

    # ✅ Add CSS for styling
    st.markdown(
        """
        <style>
        .title-container {
            text-align: center;
            font-size: 60px;
            font-weight: bold;
            color: #333;
            animation: fadeIn 2s ease-in-out;
        }
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(-20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        .stTextInput > div > div > input, 
        .stSelectbox > div > div > select {
            border: 2px solid black !important;
            border-radius: 8px;
            padding: 10px;
            width: 100%;
        }
        .stButton > button {
            border: 2px solid black !important;
            border-radius: 8px;
            padding: 8px 20px;
            background-color: white;
            color: black;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s ease-in-out;
        }
        .stButton > button:hover {
            background-color: black;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ✅ Centered Title
    st.markdown('<div class="title-container">📝 Sign Up</div>', unsafe_allow_html=True)

    # ✅ Centered input fields
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        username = st.text_input("Username")
        email = st.text_input("Email")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        place = st.text_input("Place")
        phone = st.text_input("Phone No")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")

        # ✅ Button Handling
        signup_btn = st.button("📝 Sign Up")
        home_btn = st.button("🏠 Go to Home")

    # ✅ Process Sign Up
    if signup_btn:
        if not username or not email or not password or not confirm_password:
            st.error("❌ All fields are required!")
        elif not is_valid_email(email):
            st.error("❌ Invalid email format! Please enter a valid email (e.g., example@gmail.com).")
        elif not is_valid_phone(phone):
            st.error("❌ Phone number should contain only digits!")
        elif password != confirm_password:
            st.error("❌ Passwords do not match!")
        else:
            users = load_users()
            if username in users["Username"].values:
                st.error("❌ Username already exists!")
            else:
                new_user = pd.DataFrame([[username, email, gender, place, phone, password]], 
                                        columns=["Username", "Email", "Gender", "Place", "Phone", "Password"])
                users = pd.concat([users, new_user], ignore_index=True)
                save_users(users)
                st.success("✅ Account created successfully! Redirecting to login...")

                # ✅ Redirect to login
                st.session_state["page"] = "login"
                st.rerun()

    # ✅ Process Home Button Click
    if home_btn:
        st.session_state["page"] = "home"
        st.rerun()

def generate_captcha():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

def show_login():
    set_background("lc.png")

    # ✅ Include CSS for Styling
    st.markdown(
        """
        <style>
        /* Slide-in Title Animation */
        .title-container {
            text-align: center;
            font-size: 60px;
            font-weight: bold;
            color: #79ED39;
            animation: slideIn 1.5s ease-in-out;
        }

        @keyframes slideIn {
            0% { opacity: 0; transform: translateX(-100px); }
            100% { opacity: 1; transform: translateX(0); }
        }

        /* Black Border Input Fields */
        .stTextInput > div > div > input {
            border: 2px solid black !important;
            border-radius: 8px;
            padding: 10px;
            width: 100%;
        }

        /* Styled Buttons with Borders */
        .stButton > button {
            border: 2px solid black !important;
            border-radius: 8px;
            padding: 8px 20px;
            background-color: white;
            color: black;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s ease-in-out;
        }
        
        /* Hover Effect for Buttons */
        .stButton > button:hover {
            background-color: #86F9B3;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ✅ Title with Slide-in Animation
    st.markdown('<div class="title-container">🔐 Login</div>', unsafe_allow_html=True)

    # ✅ Use Columns to Center Input Fields
    col1, col2, col3 = st.columns([1, 2, 1])  # Centering layout

    with col2:  # Place elements in the middle column
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        # ✅ Ensure Captcha is Generated Only Once per Session
        if "captcha" not in st.session_state:
            st.session_state.captcha = generate_captcha()

        st.write(f"Captcha: **{st.session_state.captcha}**")
        user_captcha = st.text_input("Enter Captcha")

        # ✅ Login Button (Redirects to Main Page)
        if st.button("🔑 Login"):
            users = load_users()

            # ✅ Check if User Exists
            user_row = users[(users["Username"] == username) & (users["Password"] == password)]  

            if not user_row.empty and user_captcha.strip().lower() == st.session_state.captcha.strip().lower():
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.page = "main"  # ✅ Set to Main Page
                st.session_state.pop("captcha", None)  # ✅ Remove Captcha After Successful Login
                st.rerun()  # ✅ Ensures immediate redirection
            else:
                st.error("❌ Invalid credentials or captcha!")
                st.session_state.captcha = generate_captcha()  # ✅ Refresh Captcha Only When Login Fails
                st.rerun()  # ✅ Ensures immediate redirection

        # ✅ "Go to Home" Button
        if st.button("🏠 Go to Home"):
            st.session_state.page = "home"
            st.rerun()  # ✅ Ensures immediate redirection

def display_box(title, content):
    st.markdown(f"""
        <div class='details-box'>
            <h4>{title}</h4>
            <p>{content}</p>
        </div>
    """, unsafe_allow_html=True)

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)

    # Convert grayscale to RGB
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    # Remove alpha channel if present
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    img_array = img_array.astype('float32') / 255.0
    return np.expand_dims(img_array, axis=0)

def show_dashboard():
    set_background("l1.png")
    st.markdown(
        """
        <style>
        @keyframes glow {
            0% { text-shadow: 0 0 5px #ffffff, 0 0 10px #66ff66; }
            50% { text-shadow: 0 0 20px #33ff33, 0 0 30px #33ff33; }
            100% { text-shadow: 0 0 5px #ffffff, 0 0 10px #66ff66; }
        }
        .animated-title {
            text-align: center;
            color: #46B523;
            font-size: 65px;
            font-weight: bold;
            animation: glow 2s infinite alternate;
        }
        .uploaded-image-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 20px;
            margin-left: 150px;
        }
        .uploaded-image-container img {
            border: 6px solid #2E8B57;
            border-radius: 15px;
            padding: 10px;
            max-width: 300px;
            max-height: 300px;
        }
        div[data-testid="stFileUploader"] {
            width: 100% !important;
            margin-left: auto;
            margin-right: auto;
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 10px;
        }
        .details-box {
            border: 2px solid #2F7A18;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 40px;
            width: 98%;
            margin-left: auto;
            margin-right: auto;
            transition: all 0.3s ease;
            background-color: transparent;
            min-height: 40px;
            font-size: 28px;
        }
        .details-box:hover {
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            transform: translateY(-5px);
            background-color: rgba(70, 181, 35, 0.1);
            border-color: #46B523;
        }
        .details-box h4 {
            color: #FF5733;
            font-size: 24px;
        }
        .hover-button {
            background-color: #46B523;
            color: white;
            padding: 10px 20px;
            font-size: 18px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .hover-button:hover {
            background-color: #2E8B57;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if "username" not in st.session_state or not st.session_state["username"]:
        st.warning("⚠️ Please log in first.")
        return

    st.markdown(f'<div class="animated-title">🌿 Plant Disease Predictor - Welcome, {st.session_state["username"]}</div><br><br>', unsafe_allow_html=True)

    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("🏠 Home"):
            st.success("🟢 Navigating to Home Page...")

    with col2:
        if st.button("🚪 Logout"):
            logout()

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        uploaded_file = st.file_uploader("📤 Upload Your Plant Image Here", type=["jpg", "jpeg", "png"], label_visibility="visible")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.markdown('<div class="uploaded-image-container"><br>', unsafe_allow_html=True)
            st.image(image, caption="🖼 Uploaded Image", width=300)
            st.markdown('</div><br>', unsafe_allow_html=True)

            if st.button("🔍 Predict"):
                img_array = preprocess_image(image)
                try:
                    # Set input
                    interpreter.set_tensor(input_details[0]['index'], img_array)
                    interpreter.invoke()
                    predictions = interpreter.get_tensor(output_details[0]['index'])
                    predicted_class = int(np.argmax(predictions))

                    class_name = class_labels[predicted_class] if predicted_class < len(class_labels) else "Unknown"
                    st.subheader(f"🔢 Predicted Class: {predicted_class} ({class_name})")

                    predicted_class_str = str(predicted_class)

                    if predicted_class_str in disease_info:
                        info = disease_info[predicted_class_str]
                        display_box("🌿 Plant", info.get('plant', "N/A"))
                        display_box("🦠 Disease", info.get('disease', "N/A"))
                        display_box("⚠️ Cause", info.get('cause', "N/A"))
                        display_box("🛠 Recovery", info.get('recovery', "N/A"))
                        display_box("🛡 Protection", info.get('protection', "N/A"))
                        display_box("💡 Health Tips", info.get('health_tips', "N/A"))
                    else:
                        st.error(f"⚠️ No data available for class **{predicted_class} ({class_name})**.")
                        st.warning("🔍 Debug: Check if this class exists in `disease_info.json`.")
                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")

def logout():
    st.session_state.username = "" 
    st.rerun()


def show_about():
    set_background("u.png")

    # Animated Title
    st.markdown("""
    <style>
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes moveLeft {
        from {
            transform: translateX(0);
        }
        to {
            transform: translateX(-20px);
        }
    }

    .animated-title {
        animation: slideIn 1.5s ease-in-out;
        color: #4CAF50;
        font-size: 48px;
        text-align: center;
        margin-bottom: 30px;
    }

    .info-box {
        background-color: rgba(255, 255, 255, 0); /* Transparent Background */
        padding: 30px;
        border-radius: 15px;
        color: #C576E0;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        transition: transform 0.3s, box-shadow 0.3s;
        border: 2px solid transparent;
        min-height: 300px;
        max-height: 300px; /* Fixed Height */
        overflow: auto; /* Scroll if content exceeds */
    }


    .info-box:hover {
        animation: moveLeft 0.3s forwards;
        transform: translateY(-10px);
        box-shadow: 0 16px 32px rgba(0,0,0,0.3);
        border-color: #FFD700;
    }

    .info-title {
        color: #429654;
        font-size: 32px;
        margin-bottom: 20px;
    }

    .center-section {
        text-align: center;
        margin-top: 60px;
        color: #FFD700;
    }
    
    .back-button {
        background-color: #4CAF50;
        color: white;
        padding: 12px 24px;
        font-size: 16px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        margin-top: 20px;
    }

    .back-button:hover {
        background-color: #45a049;
    }

    </style>
    <h1 class="animated-title">🌱 About Plant Disease Prediction App</h1><br><br>
    """, unsafe_allow_html=True)

    # First Row of Columns with Transparent Boxes
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="info-box">
            <h3 class="info-title">🚀 Key Features</h3>
            <p>✅ Uses <b>Deep Learning</b> to detect over <b>50+ plant diseases</b>.</p>
            <p>✅ Provides detailed information on <b>plant health</b>.</p>
            <p>✅ Supports <b>multiple image formats</b>.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-box">
            <h3 class="info-title">🛠 How it Works</h3>
            <p>1. <b>Upload a leaf image</b>.</p>
            <p>2. Model predicts <b>disease class</b>.</p>
            <p>3. View <b>disease details and solutions</b>.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-box">
            <h3 class="info-title">🌿 Why Use This App?</h3>
            <p>🌾 Assists <b>farmers and researchers</b>.</p>
            <p>🌿 Prevents <b>crop loss</b> with early detection.</p>
            <p>🌱 Promotes <b>sustainable farming</b>.</p>
        </div><br><br>
        """, unsafe_allow_html=True)

    # Second Row of Columns with Transparent Boxes
    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("""
        <div class="info-box">
            <h3 class="info-title">🌱 Plant Health Tips</h3>
            <p>✅ Monitor leaves for <b>early symptoms</b>.</p>
            <p>✅ Use <b>organic fertilizers</b>.</p>
            <p>✅ Ensure <b>proper watering</b>.</p>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown("""
        <div class="info-box">
            <h3 class="info-title">🔍 Disease Detection</h3>
            <p>✅ <b>AI-powered</b> disease detection.</p>
            <p>✅ Instant results with <b>high accuracy</b>.</p>
            <p>✅ Suitable for <b>multiple plant types</b>.</p>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown("""
        <div class="info-box">
            <h3 class="info-title">📊 Data Insights</h3>
            <p>✅ Track <b>disease occurrence</b>.</p>
            <p>✅ Generate <b>reports for analysis</b>.</p>
            <p>✅ Visualize <b>plant health trends</b>.</p>
        </div>
        """, unsafe_allow_html=True)

    # Developed By Section
    st.markdown("""
    <div class="center-section">
        <p><b>🔍 Developed By:</b> Sobhika</p>
        <p><b>🏫 Final Year CS Project</b></p>
    </div>
    """, unsafe_allow_html=True)

    # Streamlit state management for navigating back
    if st.button("🏠 Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    
def send_email(user_email, user_name):
    try:
        sender_email = "sobhikadevisolaikumar125@gmail.com"  # Update this
        sender_password = "tixl ozzv ecqk essv"  # Use App Password for security
        subject = "Thank You for Your Feedback!"

        sender_email = "sobhikadevisolaikumar125@gmail.com"
        sender_password = "tixl ozzv ecqk essv"  # App Password for security
        subject = "🌱 Thank You for Your Feedback! 🌿"

        # Enhanced Email to User
        body = f"""
        🎉 Hi {user_name}, 

        Thank you for your valuable feedback on our **Plant Disease Prediction App**! 🌾
        We appreciate your effort in helping us improve and provide a better experience. 😊

        
        💡 *Your feedback is like sunshine to our plants — it helps us grow!* ☀️

        Thank you once again for supporting us! 🌱

        Best Regards,  
        **The Plant Disease Prediction Team** 🌿
        """
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = user_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, user_email, msg.as_string())
        server.quit()
        st.success("✅ Thank You email sent to user.")
    except Exception as e:
        st.warning(f"⚠️ Unable to send Thank You email. Error: {e}")

def save_feedback_to_excel(name, email, message):
    data = {'Name': [name], 'Email': [email], 'Message': [message]}
    df = pd.DataFrame(data)

    try:
        # Check if the Excel file exists
        try:
            existing_df = pd.read_excel('feedback.xlsx')
            df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            pass  # File doesn't exist, will create a new one

        # Save the feedback to Excel
        df.to_excel('feedback.xlsx', index=False)
        st.success("✅ Feedback saved to Excel successfully!")
    except Exception as e:
        st.error(f"⚠️ Error saving feedback to Excel: {e}")

def show_feedback():
    set_background("s2.jpg")

    st.markdown(
        f"""
        <style>
        @keyframes bounce {{
            0%, 100% {{ transform: translateY(0); }}
            20% {{ transform: translateY(-10px); }}
            50% {{ transform: translateY(-20px); }}
            80% {{ transform: translateY(-10px); }}
        }}
        
        .animated-title {{
            animation: bounce 2s infinite;
            color: #4CAF50;
            font-size: 3em;
            font-weight: bold;
            text-align: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="animated-title">💬 Feedback Form</div>', unsafe_allow_html=True)
    st.write("We value your feedback! Please share your thoughts about our Plant Disease Prediction App.")

    # Contact Details
    st.markdown("""
    #### 📞 **Contact Us**
    - **Email:** support@plantdiseaseapp.com
    - **Phone:** +1-234-567-8901
    - **Website:** [www.plantdiseaseapp.com](http://www.plantdiseaseapp.com)
    """)

    # Feedback Form with Custom Styling
    name = st.text_input("👤 Your Name", key="name")
    email = st.text_input("📧 Your Email", key="email")
    message = st.text_area("✍️ Your Feedback", key="message")
    
    st.markdown("""
    <style>
    input, textarea {
        border: 2px solid #4CAF50;
        border-radius: 8px;
        padding: 10px;
        width: 100%;
        box-sizing: border-box;
        font-size: 18px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.button("Submit Feedback"):
        if name and email and message:
            # Save feedback to Excel
            save_feedback_to_excel(name, email, message)

            # Send Thank You email
            send_email(email, name)
            st.success("✅ Feedback submitted successfully! Thank you for your feedback.")
        else:
            st.error("⚠️ Please fill all fields before submitting.")

    # Button to go back to home page
    if st.button("🏠 Back to Home"):
        st.session_state.page = "home"
        st.rerun()

# Initialize session state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "home"

# Define a function to change the page when a button is clicked
def navigate(page):
    st.session_state.page = page
    st.rerun()  # Ensure page updates instantly

# Sidebar navigation buttons with unique keys
st.sidebar.header("Navigation")

if st.sidebar.button("🏠 Home", key="home_btn"):
    navigate("home")
if st.sidebar.button("📝 Signup", key="signup_btn"):
    navigate("signup")
if st.sidebar.button("🔑 Login", key="login_btn"):
    navigate("login")
if st.sidebar.button("📊 Main", key="main_btn"):
    navigate("main")
if st.sidebar.button("📖 About", key="about_btn"):
    navigate("about")
if st.sidebar.button("💬 Feedback", key="feedback_btn"):
    navigate("feedback")

# Display the selected page content
if st.session_state.page == "home":
    show_home()
elif st.session_state.page == "signup":
    show_signup()
elif st.session_state.page == "login":
    show_login()
elif st.session_state.page == "main":
    show_dashboard()
elif st.session_state.page == "about":
    show_about()
elif st.session_state.page == "feedback":

    show_feedback()

