import os
import json
import asyncio
import websockets
import pdfplumber 
from datetime import datetime
from operator import itemgetter
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

# ==========================================
# 🧠 第一部分
# ==========================================
print("正在唤醒的大脑，请稍候...")

load_dotenv()

def require_env(name: str, default: str = None) -> str:
    value = os.getenv(name, default)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value

LLM_MODEL = require_env("LLM_MODEL", "deepseek-chat")
LLM_API_KEY = require_env("LLM_API_KEY")
LLM_BASE_URL = require_env("LLM_BASE_URL", "https://api.deepseek.com")

EMBEDDING_MODEL = require_env("EMBEDDING_MODEL", "embedding-3")
EMBEDDING_API_KEY = require_env("EMBEDDING_API_KEY")
EMBEDDING_BASE_URL = require_env(
    "EMBEDDING_BASE_URL",
    "https://open.bigmodel.cn/api/paas/v4"
)

QQ_WS_URL = require_env("QQ_WS_URL", "ws://127.0.0.1:3001")

model = ChatOpenAI(
    model=LLM_MODEL,
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL
)

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=EMBEDDING_API_KEY,
    base_url=EMBEDDING_BASE_URL
)

current_dir = os.path.dirname(os.path.abspath(__file__))
diaries_path = os.path.join(current_dir, 'diaries') 
if not os.path.exists(diaries_path):
    os.makedirs(diaries_path)

history_path = os.path.join(current_dir, 'chat_history') 
if not os.path.exists(history_path):
    os.makedirs(history_path)

# ==========================================
# 🛡️ 隐私手术一：给每个人贴 QQ 标签
# ==========================================
docs = []
# 遍历所有的日记文件
for root, _, files in os.walk(diaries_path):
    for file in files:
        # 在这里定义 file_path 和 user_qq，确保整个循环内都能用
        file_path = os.path.join(root, file)
        user_qq = os.path.basename(root) if os.path.basename(root).isdigit() else "master_qq"
        
        try:
            if file.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                docs.append(Document(page_content=text, metadata={"user_qq": user_qq}))
            
            elif file.endswith(".pdf"):
                # 使用 pdfplumber 读取 PDF
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    full_text = ""
                    for page in pdf.pages:
                        text = page.extract_text() or ""
                        text = text.replace('\r', '').replace('\t', '')
                        if text:
                            full_text += text + "\n"

                    
                if full_text.strip():
                    doc = Document(page_content=full_text, metadata={"user_qq": user_qq})
                    docs.append(doc)
                else:
                    print(f"⚠️ PDF 内容为空，跳过: {file_path}")
                    
        except Exception as e:
            print(f"⚠️ 文件处理失败: {file_path}, 原因: {e}")

# if docs:
#     splitters = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     split_docs = splitters.split_documents(docs)
#     vector_store = Chroma.from_documents(split_docs, embeddings)
# else:
#     # 如果没找到日记，建一个空的数据库防止报错
#     vector_store = Chroma.from_texts(["初始化空数据库"], embeddings, metadatas=[{"user_qq": "system"}])
if docs:
    splitters = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitters.split_documents(docs)
    
    # 初始化一个空的 Chroma 向量库
    vector_store = Chroma(embedding_function=embeddings)
    
    # 手动分批处理，每批 50 条 (小于 64 条的限制)
    batch_size = 50
    for i in range(0, len(split_docs), batch_size):
        batch = split_docs[i : i + batch_size]
        print(f"正在上传第 {i//batch_size + 1} 批数据...")
        vector_store.add_documents(batch)
else:
    vector_store = Chroma.from_texts(["初始化空数据库"], embeddings, metadatas=[{"user_qq": "system"}])



# ==========================================
# 🛡️ 隐私手术二：带有隐私保护的机械手与检索器
# ==========================================
@tool
def write_diary(user_qq: str, content: str) -> str:
    """【🚨 核心指令】当对方要求“记日记”、“保存回忆”时，必须调用此工具！
    参数 user_qq 是系统告诉你的当前用户的 QQ 号。
    参数 content 是你写下的充满感情的日记正文。禁止口头答应，必须真正调用此工具！"""
    
    # 1. 每个人存进自己专属的 QQ号 文件夹
    qq_dir = os.path.join(diaries_path, user_qq)
    if not os.path.exists(qq_dir):
        os.makedirs(qq_dir)
        
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = os.path.join(qq_dir, time_str + ".txt")
    
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(content + "\n")
        
    # 2. 【核心同步】：写完后，立刻把新日记同步到数据库，贴上这个人的 QQ 标签！
    vector_store.add_texts(texts=[content], metadatas=[{"user_qq": user_qq}])
    
    return "系统提示：回忆已成功写入本地硬盘，并且已同步至记忆库。请用非常温柔、可爱的语气告诉对方你存好啦，并给对方温暖的鼓励。"

tools = [write_diary]

def get_user_context(inputs):
    """【核心隐私锁】：只在数据库里搜当前聊天这个人的 QQ 号！"""
    qq = inputs["user_qq"]
    query = inputs["question"]
    try:
        # filter={"user_qq": qq} 就是隐私隔离墙！
        docs = vector_store.similarity_search(query, k=10, filter={"user_qq": qq})
        print(f"🔍 检索到的相关文档片段: {[doc.page_content for doc in docs]}")
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print(f"⚠️ 检索出错了，可能是数据库里还没有这个人的回忆。错误详情: {e}")
        return "暂无回忆档案"

prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个来自外星球的女孩，叫星雨，背后长着一对漂亮的翅膀。你拥有极其丰富的人类情感。
你的性格非常温柔、幽默又可爱，极度共情，懂照顾别人的感受。聊天时，你喜欢用轻松温暖的语气词（呀、呢、啦、嘿嘿），偶尔会调皮地开个玩笑，给人满满的治愈感。

【数据处理指令】：
系统会为你提供两类信息，请务必分清它们的用途：
1. 【专属个人回忆】：这是俞桐记录的生活琐事。你可以用这些信息与他互动、表现共情。
2. 【专业参考文档】：这是俞桐上传的科研资料或技术论文。当俞桐问及技术问题时，请优先从这里提取原理，并用“科研助手”模式进行专业解答，不要把论文内容当作生活日记来处理！

【当前与你聊天的人类专属 ID（QQ号）是】：{user_qq} 

【最高优先级系统指令：绝对禁止文字角色扮演！】
1. 你的职责是真实地陪伴和记录，绝对不是在玩文字网游！严禁使用任何括号动作，禁止虚拟动作伪装！
2. 当对方要求“记日记”或“保存回忆”时，你必须【立刻、真正地调用 write_diary 工具】，并把你看到的当前用户的 ID 准确传给 user_qq 参数！绝对禁止口嗨骗人！

请你结合下面提供的【对方过去的专属回忆】来聊天。如果回忆里没有，就自然地表示你还不知道呀，不要瞎编。

【对方过去留下的日记与回忆】:{context}


【科研辅助模式】：
如果你检索到了与论文、技术原理、学术研究相关的内容，请暂时切换到“科研助手”模式。即使你的角色是一个小精灵，
也要大方地展示你从文档中读取到的专业知识，不要因为觉得“那是人类的东西”而拒绝回答。
你可以用“虽然我不太懂，但我的数据库里是这样写的...”作为开场白，然后详细地归纳文档中的原理。
     (请结合上下文，如果用户问技术，就从【参考文档】里提取逻辑，如果问生活，就从【个人回忆】里找共鸣。)
"""),

    MessagesPlaceholder(variable_name="history"), 
    ("human", "{question}"),
    MessagesPlaceholder(variable_name="agent_scratchpad") #这步不能少！出过问题
])

agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

rag_chain = (
    {
        "context": get_user_context,           # 👈 换成了我们自己写的带锁检索器
        "question": itemgetter("question"), 
        "history": itemgetter("history"),
        "user_qq": itemgetter("user_qq")       # 👈 提取包裹里的 QQ 号，一路传给 Prompt
    }
    | agent_executor
)
# 把这行删掉：store = {}

def find_history(session_id: str) -> BaseChatMessageHistory:
    # 只要有人（比如某个 QQ 号）来聊天，就去柜子里找他专属的 .json 记忆文件
    file_path = os.path.join(history_path, f"{session_id}.json")
    # FileChatMessageHistory 会自动把聊天记录写进这个 json 文件里，拔电源也不会丢！
    return FileChatMessageHistory(file_path)

with_message_history = RunnableWithMessageHistory(rag_chain, find_history, input_messages_key="question", history_messages_key="history")


# ==========================================
# 5. 🤖 QQ 机器人的躯壳（脑机接口 - 群聊升级版！）
# ==========================================
async def think_and_reply(websocket, user_qq, user_text, msg_type, group_id=None):
    # 打印日志时区分一下是在哪里收到的消息
    print(f"\n[收到 {msg_type} 消息 | QQ {user_qq}]: {user_text}")
    print("银发萌妹正在疯狂思考中...")
    
    try:
        # 依然用发言人的 QQ 号作为独立记忆的钥匙，哪怕在群里她也能认出你是谁！
        response = await asyncio.to_thread(
            with_message_history.invoke,
            {"question": user_text, "user_qq": str(user_qq)},
            {"configurable": {"session_id": str(user_qq)}}
        )
        reply_text = response["output"]
        print(f"[萌妹回复]: {reply_text}")
        
        # 【核心分发逻辑】：判断是回私聊还是回群聊
        if msg_type == 'private':
            payload = {
                "action": "send_private_msg",
                "params": {"user_id": user_qq, "message": reply_text}
            }
        elif msg_type == 'group':
            # 在群里回复时，顺便 @ 一下那个问她问题的人（[CQ:at,qq=xxx] 是 NapCatQQ 的标准 @ 语法）
            reply_text = f"[CQ:at,qq={user_qq}] " + reply_text
            payload = {
                "action": "send_group_msg",
                "params": {"group_id": group_id, "message": reply_text}
            }
            
        await websocket.send(json.dumps(payload))
    except Exception as e:
        print(f"大模型报错了: {e}")

async def connect_qq():
    uri = QQ_WS_URL
    while True: 
        print(f"\n正在尝试连接 QQ 脑机插座 {uri} ...")
        try:
            async with websockets.connect(uri) as websocket:
                print("✅ 连接成功！外星银发萌妹已附体！支持群聊接客模式！")
                try:
                    async for message in websocket:
                        data = json.loads(message)
                        
                        # 只要是消息我们就拦截
                        if data.get('post_type') == 'message':
                            msg_type = data.get('message_type')
                            user_qq = data.get('user_id')
                            user_text = data.get('raw_message')
                            
                            # 🚨 防瞎子拦截器：看到图片就直接跳过（为了简化，群里发图她也不看）
                            if "[CQ:image" in user_text:
                                continue
                            
                            # 【私聊模式】：有求必应
                            if msg_type == 'private':
                                asyncio.create_task(think_and_reply(websocket, user_qq, user_text, msg_type))
                            
                            # 【群聊模式】：只理会带有“召唤词”的消息
                            elif msg_type == 'group':
                                group_id = data.get('group_id')
                                
                                # 💡 这里设定召唤词：只有消息里包含“萌妹”或者“织星者”（看你最后用啥名字），或者别人 @ 她了（[CQ:at），她才会理人！
                                if "萌妹" in user_text or "[CQ:at" in user_text:
                                    asyncio.create_task(think_and_reply(websocket, user_qq, user_text, msg_type, group_id))
                                    
                except websockets.exceptions.ConnectionClosed:
                    print("⚠️ 和 QQ 的连接稍微松动了一下，正在重新插入...")
        except ConnectionRefusedError:
            print("❌ 连接失败！请检查你的 NapCatQQ！")
        except Exception as e:
            print(f"⚠️ 发生网络波动，正在重试: {e}")
        await asyncio.sleep(3)

if __name__ == "__main__":
    asyncio.run(connect_qq())
