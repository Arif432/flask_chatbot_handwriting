from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from langchain.chains import LLMChain
from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder

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
            "Strictly don't just simply recommend them to see a doctor for just SOME symptoms,ask for medical history"
            "ask for past reports, or conduct medical tests if needed but ignore if not provided."
            "Ask only one question at a time. Do not repeat any questions. Keep track of the information provided and adapt your questions accordingly."
            "If the conversation is sufficient for a diagnosis, provide a conclusion and recommendation always with a heading of 'SUMMARY'."
            "Only help user with these 3 disease otherwise just say 'I can't help with that.' Don't waste time on those questions."
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

def get_chatbot_response(user_message):
    try:
        response = chain.invoke({'input': user_message})
        return response['text']
    except Exception as e:
        print(f"Error in chatbot response: {e}")
        return "Sorry, there was an error processing your request."
