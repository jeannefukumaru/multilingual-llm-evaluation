# Databricks notebook source
# MAGIC %pip install bitsandbytes transformers accelerate langchain openai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import torch
import bitsandbytes as bnb 
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

def build_llm_chain(model_name):
  torch.cuda.empty_cache()
 
  instruct_pipeline = pipeline(model=model_name, model_kwargs= {"device_map": "auto", "load_in_8bit": True})
  
  template = """Below is an instruction and an optional input. Based on the instruction and optional input, generate a suitable response
  Instruction: {instruction}
  Input: {input}
  Response:
  """

  zh_template = """以下是说明和可选输入。根据指令和可选输入，生成合适的响应
  指令：{instruction}
  输入： {input}
  响应:
  """

  prompt = PromptTemplate(input_variables=['instruction', 'input'], template=zh_template)
 
  hf_pipe = HuggingFacePipeline(pipeline=instruct_pipeline)
  # Set verbose=True to see the full prompt:
  return LLMChain(llm=hf_pipe, prompt=prompt, verbose=True)

# COMMAND ----------

llm_chain = build_llm_chain("bigscience/mt0-xxl-mt")

# COMMAND ----------

llm_chain({"instruction":"我们如何在日常生活中减少用水? 请用中文答复", "input": ""})

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
from pyspark.sql.functions import monotonically_increasing_id
zh_df = spark.createDataFrame(Row(**z) for z in zh)
zh_df = zh_df.withColumn("id", monotonically_increasing_id())
display(zh_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Conduct inference using Pandas UDF 

# COMMAND ----------

from typing import Iterator
import pandas as pd
from pyspark.sql import functions as F

def follow_instructions_udf(inputs: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
  os.environ["TRANSFORMERS_CACHE"] = hugging_face_cache
  for i in inputs:
    i["generated_answer"] = llm_chain({"instruction":i["instruction"], "input": i["input"]})
    yield inputs

# COMMAND ----------

zh_df_small = zh_df.limit(10)
zh_df_small.select(F.col("instruction"), F.col("input")).mapInPandas(follow_instructions_udf, schema="instruction string, input string, generated_answer string").show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Naive iterator without Pandas UDF

# COMMAND ----------

responses = []
for inp in zh[0:50]:
  response = llm_chain({"instruction":inp["instruction"], "input": inp["input"]})
  response["original_output"] = inp["output"]
  responses.append(response)

# COMMAND ----------

pd.DataFrame(responses).to_csv("/dbfs/jeanne.choo@databricks.com/multilingual_llm/zh_mt0-xxl-mt.csv", sep="\t")

# COMMAND ----------

# MAGIC %sh head /dbfs/jeanne.choo@databricks.com/multilingual_llm/zh_mt0-xxl-mt.csv

# COMMAND ----------

pd.read_csv("/dbfs/jeanne.choo@databricks.com/multilingual_llm/zh_mt0-xxl-mt.csv", sep="\t")

# COMMAND ----------

