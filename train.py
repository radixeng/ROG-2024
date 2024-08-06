# Databricks notebook source
# MAGIC %md
# MAGIC # install libraries

# COMMAND ----------

!pip install ultralytics





# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # imports

# COMMAND ----------

from ultralytics import YOLO

# COMMAND ----------

# MAGIC %md
# MAGIC ## Models

# COMMAND ----------

# MAGIC %md
# MAGIC Modelos disponíveis para download na Ultralytics
# MAGIC
# MAGIC Também pode ser encontrado aqui: https://docs.ultralytics.com/models/

# COMMAND ----------

from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS
GITHUB_ASSETS_STEMS

# COMMAND ----------

from ultralytics import YOLO, settings
settings

# COMMAND ----------

settings.update({'runs_dir': '/Workspace/Users/jhonat.souza@radixeng.com.br/ROG-2024/runs'})

# COMMAND ----------

model = YOLO('yolov9c')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train

# COMMAND ----------

# MAGIC %md
# MAGIC Todos comandos e instruções: https://docs.ultralytics.com/modes/train/#arguments
# MAGIC
# MAGIC Augmentations disponíveis: https://docs.ultralytics.com/usage/cfg/#augmentation

# COMMAND ----------

model.train(data='/Workspace/Users/jhonat.souza@radixeng.com.br/dataset/data.yaml',
            epochs=30,
            patience=8,
            batch=8,
            imgsz=640)

# COMMAND ----------


