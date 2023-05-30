# Databricks notebook source
# MAGIC %pip install bitsandbytes transformers accelerate langchain openai

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Read in `alpaca_zh` data

# COMMAND ----------

import json
fp = "/dbfs/jeanne.choo@databricks.com/multilingual_llm/alpaca_data_zh_51k.json"
with open(fp, 'r') as f:
  zh = json.load(f)

# COMMAND ----------

from pyspark.sql import Row
zh_df = spark.createDataFrame(Row(**z) for z in zh)
display(zh_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Generate english translation

# COMMAND ----------

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import os 

os.environ["OPENAI_API_KEY"] = "..."

template = """Translate the input below to {tgt}. 
            {input}
            If you are unsure, output None"""

prompt = PromptTemplate(input_variables=["input", "tgt"], template=template)

def translate(inp, tgt="english"):
  llm = OpenAI()
  chain = LLMChain(llm=llm, prompt=prompt)
  return chain.run({"input":inp, "tgt":tgt})

def save_translation(obj, fp):
  with open(fp, "w") as f:
    json.dump(obj, f)
    print(f"translation saved to {fp}")

# COMMAND ----------

en_translation = []
for z in zh[0:50]:
  instruction = translate(z["instruction"])
  inp = translate(z["input"])
  output = translate(z["output"])
  en_translation.append({"instruction": instruction, "input": inp, "output": output})

# COMMAND ----------

en_fp = "/dbfs/jeanne.choo@databricks.com/multilingual_llm/instructions_en.json"
save_translation(en_translation, en_fp)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Generate Indonesian translation

# COMMAND ----------

indo_translation = []
for s in zh[0:50]:
  instruction = translate(s["instruction"], tgt="indonesian")
  inp = translate(s["input"], tgt="indonesian")
  output = translate(s["output"], tgt="indonesian")
  indo_translation.append({"instruction": instruction, "input": inp, "output": output})

# COMMAND ----------

id_fp = "/dbfs/jeanne.choo@databricks.com/multilingual_llm/instructions_id.json"
save_translation(indo_translation, id_fp)

# COMMAND ----------

indo_translation

# COMMAND ----------

