import os
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# 全局变量缓存 Pipeline，避免重复加载
PROMPT_PIPELINE = None

def load_file_content(file_path):
    return Path(file_path).read_text(encoding='utf-8')

def load_examples(data_dir):
    """加载 data 目录下的范文"""
    examples_content = []
    data_path = Path(data_dir)
    if not data_path.exists():
        return "（警告：未找到 data 目录，无范文数据）"
        
    for file in sorted(data_path.glob('*.*')): 
        if file.is_file() and file.suffix in ['.txt', '.md']:
            content = file.read_text(encoding='utf-8')
            examples_content.append(f"--- 范文: {file.name} ---\n{content}\n")
    return "\n".join(examples_content)

def init_pipeline(base_dir=None):
    """初始化 Prompt 模板 pipeline"""
    if base_dir is None:
        base_dir = Path(__file__).parent

    try:
        sys_path = base_dir / "prompt-system.md"
        user_path = base_dir / "prompt-user.md"
        data_path = base_dir / "data"

        if not sys_path.exists() or not user_path.exists():
            return False, "缺少 prompt-system.md 或 prompt-user.md 文件"

        sys_str = load_file_content(sys_path)
        user_str = load_file_content(user_path)
        examples_str = load_examples(data_path)

        system_message = SystemMessagePromptTemplate.from_template(sys_str)
        human_message = HumanMessagePromptTemplate.from_template(user_str)

        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        
        # 固化范文数据
        global PROMPT_PIPELINE
        PROMPT_PIPELINE = chat_prompt.partial(examples=examples_str)
        
        return True, "系统初始化成功"
    except Exception as e:
        return False, f"初始化出错: {str(e)}"

def generate_article(model_repo, topic, event, requirements):
    """
    核心生成函数
    :param api_key: Hugging Face Access Token
    :param model_repo: 模型ID (如 Qwen/Qwen2.5-72B-Instruct)
    """
    if PROMPT_PIPELINE is None:
        return "Error: Pipeline未初始化", "Error", "Error"
    
    api_key = os.getenv("HF_TOKEN")

    # 1. 构造参数
    runtime_params = {
        "topic": topic,
        "event": event,
        "requirements": requirements
    }

    try:
        # 2. 生成完整的 Prompt 对象 (用于调试展示)
        final_prompt_value = PROMPT_PIPELINE.invoke(runtime_params)
        messages = final_prompt_value.to_messages()
        
        sys_display = messages[0].content
        user_display = messages[1].content
        
        # 截断展示，防止界面卡顿
        if len(sys_display) > 800:
            sys_display = sys_display[:800] + "\n\n...(省略长范文)..."

        # 3. 检查 API Key
        if not api_key or not api_key.startswith("hf_"):
            return sys_display, user_display, "❌ 请输入有效的 Hugging Face API Token"

        # 4. 调用 Hugging Face
        # 使用 HuggingFaceEndpoint 调用 Serverless API
        endpoint = HuggingFaceEndpoint(
            repo_id=model_repo,
            huggingfacehub_api_token=api_key,
            temperature=0.6,     # 稍微高一点以模仿语气
            max_new_tokens=4096, # 生成长度
            top_k=50
        )

        chat_model = ChatHuggingFace(llm=endpoint)
        chain = PROMPT_PIPELINE | chat_model
        
        # 执行
        response = chain.invoke(runtime_params)
        
        return sys_display, user_display, response.content

    except Exception as e:
        return "Prompt生成中...", "Prompt生成中...", f"❌ 调用 AI 失败: {str(e)}\n请检查网络或Token权限。"