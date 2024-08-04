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
app.secret_key = os.urandom(24)  # Secure your Flask session

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Initialize FileChatMessageHistory
history = FileChatMessageHistory('chat_history.json')

# Initialize ConversationBufferMemory with FileChatMessageHistory
memory = ConversationBufferMemory(
    memory_key='chat_history',
    chat_memory=history,
    return_messages=True
)

# Create a prompt template
prompt_template = ChatPromptTemplate(
    messages=[
        HumanMessage(content=(
            "You are a diabetes, kidney, and heart specialist doctor and need to diagnose the user's condition."
            "Do an in-depth conversation like an expert."
            "Strictly don't just simply recommend them to see a doctor for just SOME symptoms, ask for medical history"
            "ask for past reports, or conduct medical tests if needed but ignore if not provided."
            "Ask only one question at a time. Do not repeat any questions. Keep track of the information provided and adapt your questions accordingly."
            "If the conversation is sufficient for a diagnosis, provide a conclusion and recommendation always with a heading of 'SUMMARY'."
            "Only help user with these 3 diseases otherwise just say 'I can't help with that.' Don't waste time on those questions."
        )),
        MessagesPlaceholder(variable_name='chat_history'),
        HumanMessagePromptTemplate.from_template("{input}")
    ]
)

# Initialize the LLMChain
chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    memory=memory,
    verbose=False
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

def ask_gemini(file_content, question=None):
    relevant_keywords = ["diabetes", "kidney", "heart",'blood pressure', "sugar"]

    # Determine the file extension from the filename
    file_ext = request.files.get('file').filename.lower().split('.')[-1]

    if file_ext in ['jpeg', 'jpg', 'png']:
        # if not is_relevant_content(question or "", relevant_keywords):
        #     raise ValueError("Image content is not related to Bot expertise.")
        message_content = process_image(BytesIO(file_content), question)
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