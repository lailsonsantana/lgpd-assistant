from mcp.client.streamable_http import streamable_http_client
from contextlib import AsyncExitStack
from mcp import ClientSession
from cache import *
from llm import *
import asyncio
import json

NOME_ESPECIALISTA = "Assistente LGPD"
BASE_CONHECIMENTO = "conhecimento/lgpd.json"
MODELO = "gemini-3.5-flash-lite"

CACHE = "cache"
MODELO_CACHE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
NOME_CACHE = "progbot_cache"

URL_FERRAMENTAS = "http://localhost:8000/mcp"
LIMIAR_DISTANCIA_CONHECIMENTO = 0.20

def iniciar_persona_assistente(conhecimento):
    persona = [
        {
            "type": "user_input",
            "content": [{
                "type": "text",
                "text": (
                    "Use exclusivamente as respostas do conhecimento fornecido e os "
                    "resultados das ferramentas MCP como fontes factuais. Não complete "
                    "lacunas com conhecimento geral, memória de treinamento ou suposições. "
                    "Se a informação solicitada não estiver nessas fontes, diga que não "
                    "há informação suficiente na base e peça os dados necessários."
                )
            }]
        },
        {
            "type": "user_input",
            "content": [{
                "type": "text",
                "text": (
                    "Quando o usuário perguntar se um tipo específico de dado é pessoal "
                    "ou sensível, você DEVE chamar classificar_dado uma vez para cada "
                    "tipo antes de responder. Não classifique com base no seu conhecimento. "
                    "Após a chamada, use somente a categoria, referência e orientação "
                    "retornadas pela ferramenta; não acrescente conclusões jurídicas."
                )
            }]
        },
        {
            "type": "user_input",
            "content": [{
                "type": "text",
                "text": (
                    "Não execute classificar_dado para perguntas conceituais, como "
                    "'o que é dado sensível'. Responda apenas se a resposta estiver "
                    "na base de conhecimento fornecida; caso contrário, informe que "
                    "não há informação suficiente na base."
                )
            }]
        },
        {
            "type": "user_input",
            "content": [{ "type": "text", "text": "Se a ferramenta classificar_dado retornar 'depende do contexto' ou 'não catalogado', não afirme uma classificação definitiva: explique o motivo da dúvida e peça ao usuário mais detalhes sobre como o dado é coletado e usado" }]
        },
        {
            "type": "user_input",
            "content": [{ "type": "text", "text": "Ao responder com base na ferramenta, cite o artigo da LGPD informado no resultado e lembre que sua orientação não substitui assessoria jurídica" }]
        }
    ]

    for mensagem, resposta in conhecimento:
        persona.append({
            "type": "user_input",
            "content": [{ "type": "text", "text": (
            f"Se o usuário perguntar o mesmo que '{mensagem}', ainda que com outras palavras, "
            f"responda com o conteúdo de '{resposta}'. "
            "Se a pergunta apenas tocar no mesmo tema, mas pedir algo diferente, não use esta resposta."
        )}]
    })

    return persona

async def carregar_conhecimento(base_conhecimento = BASE_CONHECIMENTO):
    carregado, conhecimento = False, []

    try:
        with open(base_conhecimento, "r", encoding = "utf-8") as arquivo:
            conteudo = json.load(arquivo)
            arquivo.close()

        for pergunta_resposta in conteudo:
            pergunta = pergunta_resposta["pergunta"]
            resposta = pergunta_resposta["resposta"]

            conhecimento.append((pergunta, resposta))

        carregado = True
    except Exception as e:
        print(f"erro carregando conhecimento: {e}")

    return carregado, conhecimento

async def conectar_ferramentas(contexto, url_ferramentas = URL_FERRAMENTAS):
    conexao_mcp, ferramentas = None, [] 

    try:
        stream_leitura, stream_escrita = await contexto.enter_async_context(streamable_http_client(url_ferramentas))

        conexao_mcp = await contexto.enter_async_context(ClientSession(stream_leitura, stream_escrita))

        await conexao_mcp.initialize()

        listagem_ferramentas = await conexao_mcp.list_tools()
        for f in listagem_ferramentas.tools:
            ferramentas.append({
                "type": "function",
                "name": f.name,
                "description": f.description,
                "parameters": f.input_schema
            })
    except Exception as e:
        print(f"erro conectando as ferramentas: {e}")

    return conexao_mcp, ferramentas

def extrair_texto_execucao(resultado):
    if resultado.structured_content:
        return json.dumps(resultado.structured_content, ensure_ascii = False)

    conteudo = []
    for c in resultado.content:
        if hasattr(c, "text"):
            conteudo.append(c.text)
        else:
            conteudo.append(str(c))

    return "\n".join(conteudo)

def normalizar_texto(texto):
    import unicodedata

    texto = unicodedata.normalize("NFD", texto.lower())
    return "".join(caractere for caractere in texto if unicodedata.category(caractere) != "Mn")

def mensagem_e_saudacao(mensagem):
    texto = normalizar_texto(mensagem)
    texto = " ".join(
        parte.strip("!?,.;:") for parte in texto.split()
    ).strip()
    return texto in {
        "oi",
        "ola",
        "oi tudo bem",
        "ola tudo bem",
        "bom dia",
        "boa tarde",
        "boa noite",
        "e ai",
        "quem e voce",
        "quem e vc",
        "quem voce e",
        "quem vc e",
    }

def pergunta_sobre_classificacao(mensagem):
    texto = normalizar_texto(mensagem)
    pede_categoria = "pessoal" in texto or "sensivel" in texto
    pergunta_conceitual = (
        "o que e dado pessoal" in texto
        or "o que e dado sensivel" in texto
        or "defina dado pessoal" in texto
        or "defina dado sensivel" in texto
    )
    return pede_categoria and not pergunta_conceitual

def encontrar_resposta_conhecimento(pergunta, conhecimento, embeddings, modelo_cache):
    embedding_pergunta = modelo_cache.encode(pergunta, normalize_embeddings=True)
    distancias = [
        1 - sum(a * b for a, b in zip(embedding_pergunta, embedding))
        for embedding in embeddings
    ]
    indice = min(range(len(distancias)), key=distancias.__getitem__)

    if distancias[indice] > LIMIAR_DISTANCIA_CONHECIMENTO:
        return None

    return conhecimento[indice][1]

def formatar_resultado_classificacao(tipo_dado, texto_resultado):
    resultado = json.loads(texto_resultado)
    if not isinstance(resultado, dict) or "categoria" not in resultado:
        raise ValueError(f"resultado inválido da ferramenta classificar_dado: {texto_resultado}")

    categoria = resultado["categoria"]
    referencia = resultado.get("referencia")
    orientacao = resultado.get("orientacao")

    if categoria == "não catalogado":
        return (
            f"A base não tem uma classificação cadastrada para '{tipo_dado}'. "
            f"{orientacao or 'Forneça mais detalhes sobre o dado e seu contexto.'}"
        )

    resposta = f"A base classifica '{tipo_dado}' como '{categoria}'."
    if referencia:
        resposta += f" Referência: {referencia}."
    if orientacao:
        resposta += f" {orientacao}"
    elif categoria == "pessoal":
        resposta += " A base não o classifica como dado sensível."

    return resposta

async def executar_ferramentas(conexao_mcp, resposta, mensagens):
    ferramentas_para_executar = [passo for passo in resposta.steps if passo.type == "function_call"]
    resultados = []
    if ferramentas_para_executar:
        for ferramenta in ferramentas_para_executar:
            execucao = await conexao_mcp.call_tool(ferramenta.name, arguments = ferramenta.arguments)
            texto_resultado = extrair_texto_execucao(execucao)

            mensagens.append({
                "type": "function_result",
                "name": ferramenta.name,
                "call_id": ferramenta.id,
                "result": [
                    {
                        "type": "text",
                        "text": texto_resultado
                    }
                ]
            })
            resultados.append({
                "name": ferramenta.name,
                "arguments": ferramenta.arguments,
                "text": texto_resultado,
                "is_error": getattr(execucao, "isError", False)
            })

    return resultados

async def executar():
    contexto = AsyncExitStack()
    conexao_mcp, ferramentas = await conectar_ferramentas(contexto)
    if not ferramentas:
        print("não foram encontradas ferramentas")
    else:
        for ferramenta in ferramentas:
            print(f"ferramenta encontrada: {ferramenta['name']}, descrição: {ferramenta['description']}")

    iniciado, llm = await iniciar_llm()
    if not iniciado:
        await contexto.aclose()

        print("erro iniciando llm")
        return    

    carregado, conhecimento = await carregar_conhecimento()
    if not carregado or not conhecimento:
        await contexto.aclose()

        print("erro carregando conhecimento")
        return    

    iniciado, modelo_cache, colecao_cache = iniciar_cache(MODELO_CACHE, CACHE, NOME_CACHE)
    if not iniciado:
        await contexto.aclose()

        print("erro iniciando cache")
        return    

    embeddings_conhecimento = modelo_cache.encode(
        [pergunta for pergunta, _ in conhecimento],
        normalize_embeddings=True
    )
    persona = iniciar_persona_assistente(conhecimento)


    while True:
        mensagem = input("👤 ")
        if mensagem in ["sair", "/s"]:
            await contexto.aclose()

            break

        persona.append({
            "type": "user_input",
            "content": [{ "type": "text", "text": mensagem }]
        })

        if mensagem_e_saudacao(mensagem):
            resposta = (
                f"Olá! Eu sou o {NOME_ESPECIALISTA}, um especialista na LGPD "
                "para pequenas empresas e profissionais autônomos. Como posso ajudar?"
            )
            persona.append({"type": "model_output", "text": resposta})
            print(f"🤖 {resposta}")
            continue

        pergunta_classificacao = pergunta_sobre_classificacao(mensagem)
        if not pergunta_classificacao:
            resposta_base = encontrar_resposta_conhecimento(
                mensagem, conhecimento, embeddings_conhecimento, modelo_cache
            )
            if resposta_base is None:
                sucesso, resposta_llm = await obter_resposta(
                    llm,
                    persona,
                    [],
                    modelo=MODELO
                )
                resposta = (
                    resposta_llm.output_text.strip()
                    if sucesso and resposta_llm.output_text
                    else "Não foi possível obter uma resposta do assistente."
                )
                persona.append({"type": "model_output", "text": resposta})
                print(f"🤖 {resposta}")
                continue

            encontrada, resposta_cache = buscar_cache(
                mensagem, modelo_cache, colecao_cache
            )
            if encontrada and resposta_cache == resposta_base:
                resposta = resposta_cache
                origem = "cache validado pela base"
            else:
                resposta = resposta_base
                gravar_cache(mensagem, resposta, modelo_cache, colecao_cache)
                origem = "base de conhecimento"

            persona.append({"type": "model_output", "text": resposta})
            print(f"🤖 {resposta} [resposta da {origem}]")
            continue

        ferramenta_classificacao_disponivel = any(
            ferramenta["name"] == "classificar_dado" for ferramenta in ferramentas
        )
        if pergunta_classificacao and not ferramenta_classificacao_disponivel:
            resposta = (
                "Não consigo classificar esse dado porque a ferramenta MCP de "
                "classificação não está disponível. Não vou inferir a resposta."
            )
            persona.append({"type": "model_output", "text": resposta})
            print(f"🤖 {resposta}")
            continue

        ferramentas_executadas = False

        while True:
            sucesso, resposta = await obter_resposta(
                llm,
                persona,
                ferramentas,
                modelo=MODELO
            )
            if not sucesso:
                resposta = (
                    "Não foi possível consultar a classificação na base MCP. "
                    "Não vou inferir a resposta."
                )
                break

            persona.extend(resposta.steps)
            consumo_chamada = get_consumo_tokens(resposta)
            print(f"🤖 tokens consumidos: {consumo_chamada}")

            resultados_ferramentas = await executar_ferramentas(
                conexao_mcp, resposta, persona
            )
            if resultados_ferramentas:
                ferramentas_executadas = True
                if all(
                    resultado["name"] == "classificar_dado"
                    for resultado in resultados_ferramentas
                ):
                    respostas = []
                    for resultado in resultados_ferramentas:
                        tipo_dado = resultado["arguments"].get("tipo_dado", "dado")
                        if resultado["is_error"]:
                            respostas.append(
                                f"Não foi possível consultar a classificação de "
                                f"'{tipo_dado}' na base. Não vou inferir a resposta."
                            )
                        else:
                            respostas.append(formatar_resultado_classificacao(
                                tipo_dado, resultado["text"]
                            ))

                    resposta = "\n".join(respostas)
                    persona.append({"type": "model_output", "text": resposta})
                    break
                continue

            resposta = (
                "Não consegui consultar a classificação na base MCP. "
                "Não vou responder com conhecimento externo."
            )
            persona.append({"type": "model_output", "text": resposta})
            break

        origem = "MCP" if ferramentas_executadas else "consulta MCP"
        print(f"🤖 {resposta} [resposta baseada na {origem}]")

    await contexto.aclose()

if __name__ == "__main__":
    asyncio.run(executar())
