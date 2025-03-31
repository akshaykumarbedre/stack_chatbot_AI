import slack
from dotenv import load_dotenv

import os
load_dotenv()
SLACK_TOKEN=os.environ['SLACK_TOKEN']

client = slack.WebClient(token=SLACK_TOKEN)
client.chat_postMessage(channel='#first_chatbot',text='i am akshay , i am sending the massge to my channel using slack ')