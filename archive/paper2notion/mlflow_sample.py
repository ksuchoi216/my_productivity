from typing import Dict, List

import mlflow
import mlflow.langchain
import pandas as pd
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class LLMChain:
    def __init__(self, config):
        self.config = config
        self.__setup__(config)

    def __setup__(
        self, config
    ):  # check whether config has ["experiment_name", "prompt_name", "prompt_num", "model", "temperature"]
        items = ["experiment_name", "prompt_name", "prompt_num", "model", "temperature"]
        if not all(key in config for key in items):
            raise ValueError(
                "Config must contain 'experiment_name', 'prompt_name', 'prompt_num', 'model', and 'temperature'."
            )
        experiment_name = config["experiment_name"]
        self.experiment_id = utils.turn_on_mlflow(experiment_name=experiment_name)
        self.run_name = utils.create_run_name(config)

    def make_input_dict(self, context_dict):
        lecture_titles = context_dict.get("course_titles", [])

        input_dict = {
            "question": self.config.question,
            "context": context_dict["content"],
            "username": context_dict["username"],
            "num_of_steps": context_dict["num_of_steps"],
        }
        return input_dict

    def run(self, context_dict):
        input_dict = self.make_input_dict(context_dict)

        with mlflow.start_run(experiment_id=self.experiment_id, run_name=self.run_name):
            mlflow.langchain.autolog()  # type: ignore
            mlflow.log_params(self.config)

            prompt_url = f"prompts:/{self.config.prompt_name}/{self.config.prompt_num}"
            prompt_template = mlflow.load_prompt(prompt_url).to_single_brace_format()
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=[
                    "question",
                    "context",
                    "username",
                    "num_of_steps",
                ],
            )
            model = ChatOpenAI(
                temperature=self.config.temperature,
                model=self.config.model,
            )
            chain = prompt | model | StrOutputParser()

            out = chain.invoke(input_dict)

            result_df = pd.DataFrame(
                {
                    "context": [input_dict["context"]],
                    "username": [input_dict["username"]],
                    "answer": [out],
                    "question": [input_dict["question"]],
                }
            )
            mlflow.log_table(data=result_df, artifact_file="result.json")


config = edict(
    {
        "experiment_name": "thats_one_reason",
        "prompt_name": "thats_one_reason",
        "prompt_num": 19,
        "model": "gpt-4o-mini",
        # "model": "gpt-4.1-nano",
        "temperature": 0.2,
        "question": "context을 활용해서, 선택된 강의와 문제집에 대해 왜 선택했는지에 대해 설명해줘. -음, -함, -됨과 같은 말끝을 지켜줘. 필수 정보는 내용을 유지한 채로 받드시 넣어줘.",
    }
)

chain = LLMChain(config)

chain.run(context_dicts[4])
