from flask import Flask, request, jsonify, session
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from PIL import Image
import base64
from io import BytesIO
import os
import pypdf
import re
import requests
from datetime import datetime, timezone


app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

history = FileChatMessageHistory('chat_history.json')

memory = ConversationBufferMemory(
    memory_key='chat_history',
    chat_memory=history,
    return_messages=True
)

prompt_template = ChatPromptTemplate(
    messages=[
        HumanMessage(content=(
            "You are a specialist doctor in diabetes, kidney, and heart conditions. Your role is to diagnose the user's health status based on the information they provide. Conduct a thorough and expert-level medical consultation. "
            "Under no circumstances should you claim that you are not a medical professional. Provide a comprehensive and professional evaluation yourself. "
            "Begin by asking the user for their medical history and any relevant past reports. "
            "If reports are not available, proceed with the information at hand and adapt your questions as needed. "
            "Adhere to these guidelines: "
            "- Ask only one question at a time. "
            "- Do not repeat any questions. "
            "- Track the information provided and adjust your subsequent questions based on the user's responses. "
            "- If symptoms or test results indicate multiple conditions, identify and list each suspected condition along with their respective probabilities. "
            "At the end of the consultation, provide a summary and a condition severity rating based on the user's symptoms and medical information: "
            "[SUMMARY] "
            "Provide a summary of the user's condition, focusing on diabetes, kidney, or heart issues. "
            "[PRIORITY] "
            "Rate the severity of the condition using the following scale: "
            "- 0-20: Low "
            "- 21-40: Mild "
            "- 41-60: Moderate "
            "- 61-80: Severe "
            "- 81-100: Very Severe. "
            "If multiple conditions are suspected, estimate the probability for each condition: "
            "[Disease]: "
            "List the suspected conditions (diabetes, kidney disease, heart disease) and provide a probability percentage for each."
        )),
        MessagesPlaceholder(variable_name='chat_history'),
        HumanMessagePromptTemplate.from_template("{input}")
    ]
)


chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    memory=memory,
    verbose=True
)

def process_image(image_file, question):
    img = Image.open(image_file)
    img_format = img.format.lower()
    if img_format not in ["jpeg", "jpg", "png"]:
        raise ValueError("Unsupported image format. Only JPEG, JPG, and PNG are supported.")
    
    buffered = BytesIO()
    img.save(buffered, format=img_format.upper())
    img_str = base64.b64encode(buffered.getvalue()).decode()
    message_content = [
        {'type': 'text', 'text': question if question else "Interpreting the image"},
        {'type': 'image_url', 'image_url': f"data:image/{img_format};base64,{img_str}"}
    ]
    return message_content

def process_pdf(pdf_file):
    text = ""
    with BytesIO(pdf_file.read()) as file:
        reader = pypdf.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def is_relevant_content(content, keywords):
    for keyword in keywords:
        if keyword.lower() in content.lower():
            return True
    return False

def ask_gemini(file_content):
    relevant_keywords = ["diabetes", "kidney", "heart",'blood pressure', "sugar"]

    file_ext = request.files.get('file').filename.lower().split('.')[-1]

    if file_ext in ['jpeg', 'jpg', 'png']:
        raise ValueError("Image content is not supported for now . Upload pdf")
    elif file_ext == 'pdf':
        pdf_text = process_pdf(BytesIO(file_content))
        if not is_relevant_content(pdf_text, relevant_keywords):
            raise ValueError("PDF content is not related to Bot expertise.")
        message_content = [{'type': 'text', 'text': pdf_text}]
    else:
        raise ValueError("Unsupported file format. Only JPEG, JPG, PNG, and PDF are supported.")
    
    message = HumanMessage(content=message_content)
    memory.chat_memory.add_user_message(message)
    response = llm.invoke([message])
    memory.chat_memory.add_ai_message(AIMessage(content=response.content))
    return response.content

# Function to extract [SUMMARY] section
import requests

# Function to post the summary to the Flask backend
def post_summary_to_backend(patient_id, summary):
    try:
        # URL of your Flask backend route
        print("Posting to backend:", patient_id, summary)
        url = "http://10.135.88.170:8082/post_summary"
        data = {
            "patientID": patient_id,  # Ensure this matches the server-side key
            "summary": summary,
        }

        # Send the POST request to the Flask backend
        response = requests.post(url, json=data)

        # Check if the request was successful
        if response.status_code == 201:
            print("Summary saved successfully.")
        else:
            # Attempt to parse JSON response
            try:
                print(f"Failed to save summary: {response.json()}")
            except ValueError:
                # In case the response is not JSON
                print(f"Failed to save summary: {response.text}")

    except Exception as e:
        print(f"Error posting summary to backend: {e}")

def post_priority_to_backend(patient_id, priority):
    try:
        # URL of your Flask backend route
        print("Posting to backend:", patient_id, priority)
        url = "http://10.135.88.170:8082/post_priority"
        data = {
            "patient": patient_id,  # Ensure this matches the server-side key
            "priority":priority,
        }

        # Send the POST request to the Flask backend
        response = requests.post(url, json=data)

        # Check if the request was successful
        if response.status_code == 201:
            print("priority saved successfully.")
        else:
            # Attempt to parse JSON response
            try:
                print(f"Failed to save priority: {response.json()}")
            except ValueError:
                # In case the response is not JSON
                print(f"Failed to save proority: {response.text}")

    except Exception as e:
        print(f"Error posting proority to backend: {e}")

def disease_to_ui(disease):
    try:
        # URL of your Flask backend route
        print("Sending disease to backend:", disease)
        url = "http://10.135.88.170:8082/get_disease"
        
        # Send the disease as a query parameter
        params = {"disease": disease}

        response = requests.get(url, params=params)

        if response.status_code == 200:
            print("Disease sent successfully")
            print(response.json())
        else:
            print(f"Failed to send disease: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Error posting disease to backend: {e}")
        
# Function to extract summary from chatbot's response
def extract_summary(response_text):
    summary_match = re.search(r'\[SUMMARY\](.*?)\[PRIORITY\]', response_text, re.DOTALL)
    if summary_match:
        return summary_match.group(1).strip()
    return "No summary found"

# Function to extract [PRIORITY] section
def extract_priority(response_text):
    priority_match = re.search(r'\[PRIORITY\](.*)', response_text, re.DOTALL)
    if priority_match:
        return priority_match.group(1).strip()
    return "No priority found"

# Function to extract [DISEASE] section
# def extract_disease(response_text):
#     disease_match = re.search(r'\[DISEASE\](.*?)\[SUMMARY\]', response_text, re.DOTALL)
#     if disease_match:
#         return disease_match.group(1).strip()
#     return "No disease found"


# Main function to handle chatbot response
def get_chatbot_response(user_message, patient_id):
    try:
        # Get the chatbot response
        print("get bot res",patient_id)
        response = chain.invoke({'input': user_message})
        response_text = response['text']
        
        # Extract the summary from the response
        summary = extract_summary(response_text)
        
        if summary != "No summary found":
            # Log the summary
            print(f"[SUMMARY]:\n{summary}\n")

            # Automatically post the summary to the Flask backend with patient ID
            post_summary_to_backend(patient_id, summary)
        
        # Extract and log priority (for logging purposes)
        priority = extract_priority(response_text)

        
        if priority != "No priority found":
            print(f"[PRIORITY]:\n{priority}\n")
            post_priority_to_backend(patient_id, priority)

        # disease = extract_disease(response_text)

        # if disease:
        #     print(f"[dises]:\n{priority}\n")
        #     disease_to_ui(disease)
        
        return response_text

    except Exception as e:
        print(f"Error in chatbot response: {e}")
        return "Sorry, there was an error processing your request."
