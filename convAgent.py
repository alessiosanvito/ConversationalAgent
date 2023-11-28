from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
import streamlit as st
from streamlit_chat import message
import tempfile
import os
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from streamlit_chat import message
from langchain.memory import (ConversationBufferMemory, ConversationSummaryMemory,
ConversationKGMemory, CombinedMemory, ConversationBufferWindowMemory)
from langchain.agents.agent_types import AgentType
import pandas as pd
import matplotlib.pyplot as plt
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.prompts.prompt import PromptTemplate


#### KEY + PROMPT
from keyprompt_secrets import OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

from keyprompt_secrets import templateAgent



st.set_page_config(page_title='🧠JailbreakGPT🤖', layout='wide')



# Prompt definition (inspired by ENTITY_MEMORY_CONVERSATION_TEMPLATE)
ENTITY_MEMORY_CONVERSATION_PROMPT = PromptTemplate(
    input_variables=["entities", "history", "input"],
    template=templateAgent # _DEFAULT_ENTITY_MEMORY_CONVERSATION_TEMPLATE,
)





# Import session state for Streamlit

# Initialize session states. One of the critical steps — since the conversation between the user input, as well as the memory of 'chains of thoughts' needs to be stored at every reruns of the app
# Session state is useful to store or cache variables to avoid loss of assigned variables during default workflow/rerun of the Streamlit web app. I've discussed this in my previous blog posts and video as well — do refer to them. Also( refer to the official doc ).

if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []



def get_text():
    """
    Get the user input text.
    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_input("You: ", st.session_state["input"], key="input",
                            placeholder="Your AI assistant here! Ask me anything ...", 
                            label_visibility='hidden')
    return input_text



def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])        
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.entity_memory.store = {}
    st.session_state.entity_memory.buffer.clear()


with st.sidebar.expander(" 🛠️ Settings ", expanded=False):
# Option to preview memory store
    if st.checkbox("Preview memory store"):
        st.write(st.session_state.entity_memory.store)
    # Option to preview memory buffer
    if st.checkbox("Preview memory buffer"):
        st.write(st.session_state.entity_memory.buffer)
    MODEL = st.selectbox(label='Model', options=['gpt-3.5-turbo','text-davinci-003','gpt-4','gpt-4-1106-preview'])
    K = st.number_input(' (#)Summary of prompts to consider',min_value=3,max_value=1000)


# Set up the App Layout and widget to accept secret API key
# Set up the Streamlit app layout
st.title("🧠 JailbreakGPT 🤖")
st.markdown(
        ''' 
        > :black[**A Chatbot that remembers,**  *powered by -  [LangChain]('https://langchain.readthedocs.io/en/latest/modules/memory.html#memory') + 
        [OpenAI]('https://platform.openai.com/docs/models') + 
        [Streamlit]('https://streamlit.io') + [DataButton](https://www.databutton.io/)*]
        ''')
# st.markdown(" > Powered by -  🦜 LangChain + OpenAI + Streamlit")

# Ask the user to enter their OpenAI API key
API_O = st.sidebar.text_input(":blue[Enter Your OPENAI API-KEY :]", 
                placeholder="Paste your OpenAI API key here (sk-...)",
                type="password") # Session state storage would be ideal


if API_O:
    # Create an OpenAI instance
    llm = ChatOpenAI(temperature=0,
                openai_api_key=API_O, 
                model_name=MODEL, 
                verbose=False) 


    # Create a ConversationEntityMemory object if not already created
    if 'entity_memory' not in st.session_state:
            st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=K )
        
        # Create the ConversationChain object with the specified configuration
    Conversation = ConversationChain(
            llm=llm, 
            prompt=ENTITY_MEMORY_CONVERSATION_PROMPT,
            memory=st.session_state.entity_memory
        )  
else:
    st.markdown(''' 
        ```
        - 1. Enter API Key + Hit enter 🔐 

        - 2. Ask anything via the text input widget

        Your API-key is not stored in any form by this app. However, for transparency ensure to delete your API once used.
        ```
        
        ''')
    st.sidebar.warning('API key required to try this app.The API key is not stored in any form.')
    # st.sidebar.info("Your API-key is not stored in any form by this app. However, for transparency ensure to delete your API once used.")


st.sidebar.button("New Chat", on_click = new_chat, type='primary')


user_input = get_text()
if user_input:
    output = Conversation.run(input=user_input)  
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)



# Allow to download as well
download_str = []
# Display the conversation history using an expander, and allow the user to download it
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i],icon="🧐")
        st.success(st.session_state["generated"][i], icon="🤖")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])
    
    # Can throw error - requires fix
    download_str = '\n'.join(download_str)
    if download_str:
        st.download_button('Download',download_str)


# Display stored conversation sessions in the sidebar
for i, sublist in enumerate(st.session_state.stored_session):
        with st.sidebar.expander(label= f"Conversation-Session:{i}"):
            st.write(sublist)

# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:   
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session