import os
from typing import List, Tuple

from langchain.embeddings import OllamaEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama import ChatOllama


class Config:
    def __init__(self):
        os.environ["NEO4J_URI"] = "NEO4J_URI"
        os.environ["NEO4J_USERNAME"] = "NEO4J_USERNAME"
        os.environ["NEO4J_PASSWORD"] = "NEO4J_PASSWORD"


class GraphDatabaseManager:
    def __init__(self):
        self.graph = Neo4jGraph()

    def create_fulltext_index(self):
        self.graph.query(
            "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]"
        )

    def query_graph(self, query: str, params: dict):
        return self.graph.query(query, params)


class LLMManager:
    def __init__(self):
        self.llm = ChatOllama(
            model="llama3.1:8b-instruct-q4_0",
            temperature=0,
            base_url="http://10.10.33.105:11434",
        )
        self.embedding = OllamaEmbeddings(
            model="jina/jina-embeddings-v2-base-en:latest"
        )
        self.vector_index = Neo4jVector.from_existing_graph(
            embedding=self.embedding,
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding",
        )

    def transform_to_graph_documents(self, documents):
        llm_transformer = LLMGraphTransformer(llm=self.llm)
        return llm_transformer.convert_to_graph_documents(documents)

    def invoke_llm(self, prompt):
        return self.llm.invoke(prompt)


class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that appear in the text",
    )


class EntityExtractor:
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are extracting organization and person entities from the text. "
                    "Return only a list of string extracted entities and nothing else"
                    "format to return should be something like entity1,entity2...",
                ),
                ("human", "extract information from the following input: {question}"),
            ]
        )

    def extract_entities(self, question: str) -> Entities:
        formatted_prompt = self.prompt_template.format(question=question)
        response = self.llm_manager.invoke_llm(formatted_prompt)
        entity_names = self.process_response(response.content)
        return Entities(names=entity_names)

    @staticmethod
    def process_response(response: str) -> List[str]:
        return [name.strip() for name in response.split(",") if name.strip()]


class QueryGenerator:
    @staticmethod
    def generate_full_text_query(input: str) -> str:
        """
        Generate a full-text search query for a given input string.

        This function constructs a query string suitable for a full-text
        search. It processes the input string by splitting it into words and
        appending a similarity threshold (~2 changed characters) to each
        word, then combines them using the AND operator. Useful for mapping
        entities from user questions to database values, and allows for some
        misspelings.
        """
        full_text_query = ""
        words = [el for el in remove_lucene_chars(input).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()


class Retriever:
    def __init__(
        self,
        graph_db_manager: GraphDatabaseManager,
        llm_manager: LLMManager,
        entity_extractor: EntityExtractor,
    ):
        self.graph_db_manager = graph_db_manager
        self.llm_manager = llm_manager
        self.entity_extractor = entity_extractor

    def structured_retriever(self, question: str) -> str:
        """
        Collects the neighborhood of entities mentioned in the question.
        """
        result = ""
        entities = self.entity_extractor.extract_entities(question)
        for entity in entities.names:
            response = self.graph_db_manager.query_graph(
                """
                CALL db.index.fulltext.queryNodes('entity', $query, {limit:10})
                YIELD node, score
                CALL {
                MATCH (node)-[r:!MENTIONS]->(neighbor)
                RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                UNION
                MATCH (node)<-[r:!MENTIONS]-(neighbor)
                RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": QueryGenerator.generate_full_text_query(entity)},
            )
            result += "\n".join([el["output"] for el in response])
        return result

    def retriever(self, question: str):
        structured_data = self.structured_retriever(question)
        unstructured_data = [
            el.page_content
            for el in self.llm_manager.vector_index.similarity_search(question)
        ]
        final_data = f"""Structured data:
        {structured_data}
        Unstructured data:
        {"#Document ". join(unstructured_data)}
            """
        return final_data

    def retriever_json(self, question: str):
        structured_data = self.structured_retriever(question)
        unstructured_data = [
            el.page_content
            for el in self.llm_manager.vector_index.similarity_search(question)
        ]
        return structured_data, unstructured_data


class ChatHandler:
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
    in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""  # noqa: E501

    def __init__(self, llm_manager: LLMManager, retriever: Retriever):
        self.llm_manager = llm_manager
        self.retriever = retriever

        self._search_query = RunnableBranch(
            (
                RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                    run_name="HasChatHistoryCheck"
                ),
                RunnablePassthrough.assign(
                    chat_history=lambda x: self._format_chat_history(x["chat_history"])
                )
                | ChatPromptTemplate.from_template(self._template)
                | self.llm_manager.llm
                | StrOutputParser(),
            ),
            RunnableLambda(lambda x: x["question"]),
        )

        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        System:

        Use natural language and be concise.
        'The answer to your given question is' Answer:"""
        self.prompt = ChatPromptTemplate.from_template(template)

        self.chain = (
            RunnableParallel(
                {
                    "context": self._search_query | self.retriever.retriever,
                    "question": RunnablePassthrough(),
                }
            )
            | self.prompt
            | self.llm_manager.llm
            | StrOutputParser()
        )

    @staticmethod
    def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
        buffer = []
        for human, ai in chat_history:
            buffer.append(HumanMessage(content=human))
            buffer.append(AIMessage(content=ai))
        return buffer

    def handle_chat(self, question: str, chat_history: List[Tuple[str, str]] = None):
        input_data = {"question": question}
        if chat_history:
            input_data["chat_history"] = chat_history
        return self.chain(input_data)


# Initialize components
config = Config()
graph_db_manager = GraphDatabaseManager()
llm_manager = LLMManager()
entity_extractor = EntityExtractor(llm_manager)
retriever = Retriever(graph_db_manager, llm_manager, entity_extractor)
chat_handler = ChatHandler(llm_manager, retriever)

# Create fulltext index
graph_db_manager.create_fulltext_index()
