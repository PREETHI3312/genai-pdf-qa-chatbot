
## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:
The goal is to build a chatbot that can accurately extract and provide answers based on the text from a PDF document, allowing users to interact and retrieve specific information from the document without manually reading it.
### DESIGN STEPS:


#### STEP 1:Install Necessary Libraries
Before starting the implementation, ensure that all necessary libraries and dependencies are installed. This includes LangChain for processing the text, PyPDF2 (or similar) for reading PDF files, and an LLM like OpenAI for question-answering functionality.Install Necessary Libraries
#### STEP 2:Extract Text from PDF
Use libraries like PyPDF2 to extract the text from the provided PDF document. The PDF extraction process should handle multiple pages and ensure that the text is clean and usable for further processing.
#### STEP 3: Process Text Using LangChain
Once the PDF text is extracted, it needs to be processed using LangChain’s tools, such as the TextSplitter and QuestionAnsweringChain, to handle large documents and provide accurate answers based on the content.
#### STEP 4: User Interaction
Allow the user to input questions and receive responses based on the content extracted from the PDF document. The user will interact with the chatbot by entering questions, and the bot will provide answers based on the document’s content.
### PROGRAM:
```
import os
import datetime
import panel as pn
pn.extension()

from dotenv import load_dotenv
load_dotenv()

import openai
openai.api_key = os.environ['OPENAI_API_KEY']

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import param

# === Select LLM ===
current_date = datetime.datetime.now().date()
llm_name = "gpt-3.5-turbo" if current_date >= datetime.date(2023, 9, 2) else "gpt-3.5-turbo-0301"
print("Using LLM:", llm_name)

# === Load PDF and create QA chain ===
def load_db(file_path, k=4):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0),
        chain_type="stuff",
        retriever=retriever,
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    )
    return qa

# === Chatbot Class ===
class ChatBot(param.Parameterized):
    chat_history = param.List([])
    answer = param.String("")
    
    def __init__(self, pdf_path, **params):
        super().__init__(**params)
        self.qa = load_db(pdf_path)
        self.panels = []
    
    def ask(self, query):
        if not query:
            return pn.pane.Markdown("Please enter a question.")
        result = self.qa({"question": query})
        self.answer = result["answer"]
        self.chat_history.append((query, self.answer))
        # Build Panel rows
        self.panels.append(pn.Row("User:", pn.pane.Markdown(query, width=600)))
        self.panels.append(pn.Row("Bot:", pn.pane.Markdown(self.answer, width=600, style={'background-color':'#F6F6F6'})))
        return pn.WidgetBox(*self.panels, scroll=True)

# === Initialize ChatBot ===
pdf_file = "docs/cs229_lectures/exp3IT.pdf"  # check path
bot = ChatBot(pdf_file)

# === Panel UI ===
input_box = pn.widgets.TextInput(placeholder="Enter your question…")
send_button = pn.widgets.Button(name="Send", button_type="primary")
conversation_panel = pn.WidgetBox()

def on_send(event):
    conversation_panel.objects = [bot.ask(input_box.value)]
    input_box.value = ""

send_button.on_click(on_send)

ui = pn.Column(
    pn.pane.Markdown("# Chat with PDF Bot"),
    pn.Row(input_box, send_button),
    conversation_panel
)

ui.servable()


```
### OUTPUT:
<img width="879" height="405" alt="image" src="https://github.com/user-attachments/assets/38c5c38a-07b8-420d-83cc-f331b6efc1de" />


### RESULT:
The chatbot successfully extracts content from the provided PDF document and answers user queries based on the text. The results can vary depending on the complexity and clarity of the document, but the chatbot aims to provide accurate and relevant answers. The system can be further enhanced with more advanced features like document summarization or handling more complex question-answering scenarios.
