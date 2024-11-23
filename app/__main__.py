from dotenv import load_dotenv
import os
from transformers import AutoTokenizer,TFAutoModel

load_dotenv('.env.local')
tokeniser=AutoTokenizer.from_pretrained(os.getenv('MODEL_PATH'))
model=TFAutoModel.from_pretrained(os.getenv('MODEL_PATH'))