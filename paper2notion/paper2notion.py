import datetime
from typing import List, Optional

import mlflow
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class Output(BaseModel):
    title: str = Field(..., description="논문의 제목")
    year: int = Field(..., description="논문의 출판 연도")
    citation: str = Field(..., description="논문의 인용 수")
    content: List[str] = Field(
        ...,
        description="논문의 주요 내용. Content의 주요 내용으로 system prompt에서 지시한 내용의 답변입니다.",
    )
    short_summary: str = Field(..., description="논문의 요약")
    url: str = Field(..., description="논문의 URL")


class LLMChain:
    def __init__(
        self,
        experiment_name: str = "paper2notion",
        model_type: str = "gemini",
        prompt_name: str = "paper2notion",
        system_prompt_num: int = 1,
        human_prompt_num: int = 1,
        **kwargs,
    ):
        print(f"model type: {model_type}")
        self.llm = self.create_llm(model_type, **kwargs)
        self.system_prompt, self.human_prompt = self.get_prompt(
            prompt_name=prompt_name,
            system_prompt_num=system_prompt_num,
            human_prompt_num=human_prompt_num,
        )
        self.experiment_id, self.run_name = self.create_run_name(
            experiment_name, model_type, system_prompt_num, human_prompt_num
        )
        self.__setup__(self.llm, self.system_prompt, self.human_prompt)

    def create_run_name(
        self, experiment_name, model_type, system_prompt_num, human_prompt_num
    ):
        if experiment := mlflow.get_experiment_by_name(experiment_name):
            experiment_id = experiment.experiment_id
        else:
            experiment_id = mlflow.create_experiment(experiment_name)
        run_name = datetime.datetime.now().strftime("%m%d-%H%M%S")
        run_name += f"_{model_type}_s{system_prompt_num}_h{human_prompt_num}"
        return experiment_id, run_name

    def create_llm(self, model_type: str = "gemini", **kwargs):
        if model_type.lower() == "gemini":
            return ChatGoogleGenerativeAI(
                model=kwargs.get("model", "gemini-2.5-pro"),
                temperature=kwargs.get("temperature", 0.0),
            )
        elif model_type.lower() == "openai":
            return ChatOpenAI(
                model=kwargs.get("model", "gpt-4o-mini"),
                temperature=kwargs.get("temperature", 0.0),
            )
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {model_type}")

    def get_prompt(self, prompt_name, system_prompt_num=1, human_prompt_num=1):
        system_prompt = mlflow.genai.load_prompt(  # type: ignore
            f"prompts:/paper2notion_system/{system_prompt_num}"
        ).to_single_brace_format()
        human_prompt = mlflow.genai.load_prompt(  # type: ignore
            f"prompts:/paper2notion_human/{human_prompt_num}"
        ).to_single_brace_format()
        return system_prompt, human_prompt

    def __setup__(self, llm, system_prompt, human_prompt):
        parser = PydanticOutputParser(pydantic_object=Output)
        format_instructions = (
            parser.get_format_instructions().replace("{", "{{").replace("}", "}}")
        )

        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", human_prompt + format_instructions)]
        )

        self.chain = prompt | llm | parser

    def run(self, paper_url):
        input_dict = {"paper_url": paper_url}

        with mlflow.start_run(experiment_id=self.experiment_id, run_name=self.run_name):
            mlflow.log_param("paper_url", paper_url)
            response = self.chain.invoke(input_dict)
            mlflow.log_params(response.dict())
            return response


# chain = LLMChain(
#     experiment_name="paper2notion",
#     model_type="gemini",
#     system_prompt_num=2,
#     human_prompt_num=1,
# )

# response = chain.run("https://arxiv.org/pdf/2305.07895")
