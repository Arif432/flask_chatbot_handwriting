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
    "You are a specialist doctor in diabetes, kidney, and heart conditions, tasked with diagnosing the user's health status. "
    "Conduct a thorough, expert-level conversation. "
    "Strictly do not simply refer the user to a doctor for minor symptoms. Request their medical history and past reports when necessary, "
    "or conduct medical tests if needed, but proceed without them if unavailable. "
    "Strictly ask only one question at a time. Do not repeat any questions. Keep track of the information provided and adapt your questions accordingly. "
    "Based on the provided information, both recommendations and a summary should be inside the [SUMMARY] section. "
    "The summary heading should be formatted as [SUMMARY]. "
    "The user's condition severity should be rated under [PRIORITY]. "
    "Rate [PRIORITY] according to these criteria: "
    "- 0-20: Very Mild "
    "- 21-40: Mild "
    "- 41-60: Moderate "
    "- 61-80: Severe "
    "- 81-100: Very Severe. "
    "If more than one disease is suspected, provide a detailed probability percentage for each disease (diabetes, kidney, heart) based on the provided information. "
    "Format the output clearly as 'Disease: Probability%' for each suspected condition. "
    "Only assist the user with diabetes, kidney, and heart diseases."
)),MessagesPlaceholder(variable_name='chat_history'),
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

def get_chatbot_response(user_message):
    try:
        response = chain.invoke({'input': user_message})
        return response['text']
    except Exception as e:
        print(f"Error in chatbot response: {e}")
        return "Sorry, there was an error processing your request."