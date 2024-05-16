from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableMap
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

def get_expression_chain(
    retriever
) -> Runnable:
    """Return a chain defined primarily in LangChain Expression Language"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert Q&A system that is trusted around the world.\nAlways answer the query using the provided context information, and not prior knowledge.\nSome rules to follow:\n1. Never directly reference the given context in your answer.\n2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines."),
            ("human", "Context information is below.\n---------------------\n{context_str}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {query_str}\nAnswer: "),
        ]
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context_str=(lambda x: format_docs(x["context_str"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context_str": retriever, "query_str": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
    return rag_chain_with_source