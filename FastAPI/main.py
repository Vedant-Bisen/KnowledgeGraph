from fastapi import FastAPI
from RAG import Retriever, GraphDatabaseManager, LLMManager, EntityExtractor
from pydantic import BaseModel
import uvicorn
import re


class Question(BaseModel):
    question: str


app = FastAPI()

# Create an instance of Retriever
graph_db_manager = GraphDatabaseManager()  # Initialize as needed
llm_manager = LLMManager()  # Initialize as needed
entity_extractor = EntityExtractor(llm_manager=llm_manager)  # Initialize as needed
retriever_instance = Retriever(graph_db_manager, llm_manager, entity_extractor)


@app.post("/question/")
async def ask_question(question: Question):
    structured_data, unstructured_data = retriever_instance.retriever_json(
        question.question
    )
    structured_data = re.sub(r"\n", ",", structured_data)
    unstructured_data = [re.sub(r"\s+", " ", el).strip() for el in unstructured_data]
    return {"structured_data": structured_data, "unstructured_data": unstructured_data}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
