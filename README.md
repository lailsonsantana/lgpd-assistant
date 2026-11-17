# Assistente especialista em LGPD

Assistente de linha de comando que responde a dúvidas sobre a Lei Geral de
Proteção de Dados Pessoais (LGPD), com foco em pequenas empresas e profissionais
autônomos. As respostas conceituais são consultadas em uma base local de
perguntas e respostas; a classificação de tipos específicos de dados é feita
pela ferramenta MCP do projeto. O assistente usa a API do Google Gemini e um
cache semântico local para reutilizar respostas que correspondam à base.

> O conteúdo é informativo e não substitui assessoria jurídica. As bases de
> conhecimento devem ser revisadas e mantidas atualizadas antes de uso.

## Requisitos

- Python 3.10 ou superior
- Node.js/npm, para abrir opcionalmente o inspetor MCP
- Uma chave de API do Google AI Studio

## Configuração

Crie e ative um ambiente virtual, instale as dependências e configure a chave:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edite `.env` e substitua o valor de `GOOGLE_API_KEY` por sua chave do Google AI
Studio. Não compartilhe nem versione esse arquivo.

## Como executar

1. Em um terminal, inicie o servidor MCP, que disponibiliza a ferramenta de
   classificação em `http://localhost:8000/mcp`:

   ```powershell
   python ferramentas.py
   ```

2. Em outro terminal, inicie o assistente:

   ```powershell
   python chat.py
   ```

   Digite uma pergunta no prompt. Para encerrar, digite `sair` ou `/s`.

O assistente carrega `conhecimento/lgpd.json` para respostas da base e consulta
o servidor MCP para classificar dados específicos. O primeiro uso também pode
baixar o modelo multilíngue configurado em `chat.py` para gerar embeddings.

## Avaliação opcional

`julgamento.py` executa verificações de respostas adequadas e inadequadas com
modelos Gemini. Para utilizá-lo, configure a mesma chave no `.env` e execute:

```powershell
python julgamento.py
```

Esse processo faz chamadas à API do Google e pode gerar custos conforme o uso.

## Arquivos do projeto

| Arquivo/pasta | Função |
| --- | --- |
| `chat.py` | Fluxo interativo do assistente, busca semântica na base, integração com MCP e apresentação das respostas. |
| `ferramentas.py` | Servidor MCP e ferramenta `classificar_dado`, baseada no catálogo local. |
| `llm.py` | Inicialização do cliente Gemini, envio de mensagens e leitura do consumo de tokens. |
| `cache.py` | Cache semântico persistido localmente com ChromaDB e Sentence Transformers. |
| `julgamento.py` | Avaliação automatizada de respostas do assistente. |
| `conhecimento/lgpd.json` | Base principal de perguntas e respostas, usada pelo chat. |
| `conhecimento/lgpd_menor.json` | Base reduzida usada pelos testes de avaliação. |
| `conhecimento/classificacao_dados.json` | Catálogo de tipos de dados, categorias e referências para a ferramenta MCP. |
| `requirements.txt` | Dependências Python do projeto. |
| `.env.example` | Modelo local da variável de ambiente necessária. |
| `tempCodeRunnerFile.py` | Arquivo auxiliar temporário do editor; não faz parte do fluxo do assistente. |

O diretório `cache/` é criado em tempo de execução para os dados persistentes do
cache semântico.
