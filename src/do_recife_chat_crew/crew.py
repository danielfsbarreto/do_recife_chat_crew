import os

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, before_kickoff, crew, task
from crewai_tools import MongoDBVectorSearchConfig, MongoDBVectorSearchTool

_DATABASE_NAME = "do-recife"
_DEFAULT_COLLECTION = "do-recife-rag-enriched"
_VECTOR_INDEX_NAME = "vector_index"
_EMBEDDING_MODEL = "text-embedding-3-large"
_EMBEDDING_DIMENSIONS = 3072


@CrewBase
class DoRecifeChatCrew:
    """DoRecifeChatCrew crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    def _mongo_tool(self) -> MongoDBVectorSearchTool:
        tool = MongoDBVectorSearchTool(
            connection_string=os.environ["MONGODB_CONNECTION_STRING"],
            database_name=_DATABASE_NAME,
            collection_name=_DEFAULT_COLLECTION,
            vector_index_name=_VECTOR_INDEX_NAME,
            text_key="text",
            embedding_key="embedding",
            embedding_model=_EMBEDDING_MODEL,
            dimensions=_EMBEDDING_DIMENSIONS,
            query_config=MongoDBVectorSearchConfig(limit=6),
        )
        self._mongo_search_tool = tool
        return tool

    @before_kickoff
    def select_collection(self, inputs):
        """Point the Mongo vector search tool at the requested collection.

        The collection is a crew input (``collection_name``); it defaults to the
        enriched collection. The tool is already built by the time this hook
        runs, so we re-point its live collection handle here.
        """
        collection = (inputs or {}).get("collection_name") or _DEFAULT_COLLECTION
        tool = getattr(self, "_mongo_search_tool", None)
        if tool is not None:
            tool.collection_name = collection
            tool._coll = tool._client[_DATABASE_NAME][collection]
        return inputs

    @agent
    def do_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["do_researcher"],  # type: ignore[index]
            tools=[self._mongo_tool()],
            verbose=True,
        )

    @agent
    def do_reporter(self) -> Agent:
        return Agent(
            config=self.agents_config["do_reporter"],  # type: ignore[index]
            verbose=True,
        )

    @task
    def retrieval_task(self) -> Task:
        return Task(
            config=self.tasks_config["retrieval_task"],  # type: ignore[index]
        )

    @task
    def answer_task(self) -> Task:
        return Task(
            config=self.tasks_config["answer_task"],  # type: ignore[index]
            context=[self.retrieval_task()],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the DoRecifeChatCrew crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
