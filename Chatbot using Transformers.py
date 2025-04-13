#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install gradio


# In[3]:


import gradio as gr


# In[4]:


from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch



# In[5]:


model_name = "distilbert-base-cased-distilled-squad"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForQuestionAnswering.from_pretrained(model_name)



# In[6]:


# Define the function to get the answer
def answer_question(question, context):
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores) + 1
    answer_tokens = inputs['input_ids'][0][start_index:end_index]
    answer = tokenizer.decode(answer_tokens)
    return answer

# Load the context from a file or set it as a string
with open("C:/Users/AKASH/Desktop/Question answering chatbot using Transformers/Bike repair data.txt", 'r') as file:
    context = file.read()

# Define the chatbot function that takes in a user question and returns the answer
def chatbot_response(user_question):
    if user_question.lower() in ["thank you", "thanks", "bye", "ok"]:
        return "Have a nice day! 😊"
    elif user_question.lower() in ["hello", "hi", "greetings", "hey"]:
        return "Hello! I am here to answer your questions. Ask me anything!"
    else:
        return answer_question(user_question, context)

# Create a Gradio interface for the chatbot
# Create a Gradio interface for the chatbot
interface = gr.Interface(
    fn=chatbot_response,
    inputs=gr.Textbox(lines=2, placeholder="Type your question here..."),
    outputs="text",
    title="Interactive DIY Bike Repair Chatbot",
    description="Ask me any questions about bike repairs and I’ll provide solutions to help you!",
    theme="huggingface"  # Change this to a valid theme like 'default' or 'huggingface'
)

# Launch the interface
interface.launch()


# In[ ]:




