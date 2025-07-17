import base64
from typing import List
import operator
import tempfile

import streamlit as st
from pydantic import BaseModel
from typing_extensions import Annotated
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

########################### MODELOS ###########################


class P_e_R(BaseModel):
    pergunta: str = None
    resposta: str = None
    feedback: str = None
    nota: float = None
    peso: float = None


class ProvaState(BaseModel):
    documento: str = None
    instrucoes: str = None
    imagens: List[str] = None
    prova: List[P_e_R] = None
    prova_corrigida: Annotated[List[P_e_R], operator.add] = None
    nota_final: float = None


########################### LLM ###########################
load_dotenv()

llm = init_chat_model("gpt-4o")
llm_estruturado = llm.with_structured_output(ProvaState)


########################### RAG(futura implementação) ###########################
def RAG_document_loaders(caminho):
    tipo = caminho.split(".")[-1].lower()
    if tipo == "txt":
        loader = TextLoader(caminho)
    elif tipo == "pdf":
        loader = PyPDFLoader(caminho)
    else:
        raise ValueError(f"Tipo de arquivo '{tipo}' não suportado.")
    lista_documentos = loader.load()
    documento = "\n\n".join([doc.page_content for doc in lista_documentos])
    return documento


def RAG_text_splitting(texto):
    chunk_size = 100
    chunk_overlap = 10

    char_split = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "?", " ", ""],
    )
    return char_split.split_text(texto)


def RAG_embeddings_vector_store_as_retriever(texto):
    embedding_model = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents=texto, embedding=embedding_model)
    return vectorstore


########################### NÓS ###########################


def extrair_perguntas_respostas(state: ProvaState):
    image_blocks = [
        {
            "type": "image",
            "source_type": "base64",
            "data": image_b64,
            "mime_type": "image/jpeg",  # ajuste se for PNG
        }
        for image_b64 in state.imagens
    ]

    def montar_mensagem(input_dict):
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": input_dict["input"]}]
                + image_blocks,
            }
        ]

    chain = RunnableLambda(montar_mensagem) | llm_estruturado
    response = chain.invoke(
        {
            "input": f"""
        Você é um especialista em interpretar provas escolares a partir de uma ou mais imagens.

        Você receberá:
        1. A(s) imagem(ns) original(is) da prova

        NÃO É SUA TAREFA ATUALIZAR OS VALORES DE FEEDBACK E NOTA.
        Sua tarefa é:
        - Usar a **imagem como referência primária**
        - SOMENTE EXTRAIR todas as **PERGUNTAS** e **RESPOSTAS** da prova
        - Adicionar o peso de cada questão de acordo com as instruções: {state.instrucoes}
        - Se a instrução estiver vazia, busque na imagem o peso de cada questão
        - Se a imagem nao contiver os pesos, siga a seguinte regra:
            - A prova possui um total de 10 pontos
            - Cada questão tem o mesmo peso
        - Identificar o tipo de questão
        - Formatar com: **negrito entre #** e **// para quebras de linha**
        - Ignorar partes ilegíveis ou irrelevantes
        - NÃO INVENTAR conteúdo não visível
        """
        }
    )

    return {"prova": response.prova}


def spawn_correcao(state: ProvaState):
    return [
        Send("corrigir_prova", {"dados": p_r, "documento": state.documento})
        for p_r in state.prova
    ]


def join_documents(input):
    input["contexto"] = "\n\n".join([c.page_content for c in input["contexto"]])
    return input


def corrigir_prova(inputs: dict):
    dados = inputs["dados"]
    documento = inputs.get("documento", "")

    if documento != "":
        # Divide o texto em chunks
        split_text = RAG_text_splitting(documento)

        # Cria os documentos
        docs = [Document(page_content=chunk) for chunk in split_text]

        # Cria o retriever
        vector_store = RAG_embeddings_vector_store_as_retriever(docs)
        retriever = vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20}
        )

        # Recupera os documentos mais relevantes com base na pergunta
        context_docs = retriever.invoke(dados.pergunta)
        contexto = "\n\n".join([doc.page_content for doc in context_docs])

    else:
        contexto = "Não há contexto adicional."

    # Prompt final (usando ou não o contexto)
    prompt = f"""Você é um professor especialista em corrigir provas escolares e em dar feedbacks construtivos.
                Aqui estão os detalhes da questão:
                Pergunta: {dados.pergunta}
                Resposta do aluno: {dados.resposta}
                Peso da questão: {dados.peso}

                Contexto adicional para basear sua correção:
                {contexto}

                Sua tarefa:
                - Analisar a resposta do aluno                  
                - SEMPRE fornecer um feedback claro e construtivo, mesmo que a resposta esteja correta
                - Se a resposta estiver errada ou parcialmente correta, explique o porquê
                - Atribua uma nota entre 0 e o peso
                Retorne os campos estruturados como:
                - pergunta, resposta, feedback, nota, peso
            """

    response = llm_estruturado.invoke(prompt)
    return {"prova_corrigida": response.prova}


########################### GRAFO ###########################

builder = StateGraph(ProvaState)
builder.add_node("extrair_perguntas_respostas", extrair_perguntas_respostas)
builder.add_node("corrigir_prova", corrigir_prova)
builder.add_edge(START, "extrair_perguntas_respostas")
builder.add_conditional_edges(
    "extrair_perguntas_respostas", spawn_correcao, ["corrigir_prova"]
)
builder.add_edge("corrigir_prova", END)
graph = builder.compile()

########################### STREAMLIT ###########################


def main():
    st.set_page_config("IA Corretor de Provas", layout="wide")
    st.title("📝 IA Corretor de Provas")

    col1, col2, col3, col4 = st.columns([3, 2, 3, 2])

    with col1:
        st.header("Instruções")
        st.write(
            "✍️ Escreva os valores de cada questão, de acordo com o exemplo abaixo:"
        )
        st.write("Ex: 'questao 1 vale 3 pontos, questao 2 vale 5 pontos...'")
        st.write("Se a imagem da prova já contiver os pesos, deixe em branco.")
        instrucoes = st.text_area("", height=100)

        st.divider()
        arquivos = st.file_uploader(
            "📷 Envie as imagens da prova (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

        st.divider()
        corrigir = st.button("Corrigir Prova")
    with col3:
        st.header("Ajuste fino(RAG)")
        st.markdown(
            "Para uma correção mais acertiva, envia as respostas das perguntas no campo abaixo"
        )
        arquivos_RAG = st.file_uploader(
            "📷 Envie um  (TXT, PDF)",
            type=["txt", "pdf"],
        )
        if arquivos_RAG is not None:
            suffix = "." + arquivos_RAG.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
                temp.write(arquivos_RAG.read())
                caminho_temp = temp.name
            documento = RAG_document_loaders(caminho_temp)

    if corrigir and arquivos:
        imagens_base64 = []
        for img in arquivos:
            b64 = base64.b64encode(img.read()).decode("utf-8")
            imagens_base64.append(b64)

        st.info("Corrigindo a prova... isso pode levar alguns segundos.")
        with st.spinner("A IA está analisando as imagens e corrigindo..."):

            estado_inicial = {
                "instrucoes": instrucoes,
                "imagens": imagens_base64,
            }
            if arquivos_RAG:
                estado_inicial["documento"] = documento
            else:
                estado_inicial["documento"] = ""

            resposta = graph.invoke(estado_inicial)

        st.success("✅ Correção finalizada!")

        if resposta.get("prova_corrigida"):
            nota_total = sum(p.nota for p in resposta["prova_corrigida"])
            peso_total = sum(p.peso for p in resposta["prova_corrigida"])
            st.subheader(f"📊 Nota final: {nota_total:.2f} / {peso_total:.2f}")

            for i, pr in enumerate(resposta["prova_corrigida"], 1):
                with st.expander(f"Questão {i}"):
                    st.markdown(f"**Pergunta:** {pr.pergunta}")
                    st.markdown(f"**Resposta:** {pr.resposta}")
                    st.markdown(f"**Feedback:** {pr.feedback}")
                    st.markdown(f"**Nota:** {pr.nota:.2f} / {pr.peso:.2f}")
        else:
            st.error("A IA não conseguiu extrair perguntas e respostas da imagem.")

    elif corrigir and not arquivos:
        st.warning("Você precisa enviar pelo menos uma imagem.")


if __name__ == "__main__":
    main()
