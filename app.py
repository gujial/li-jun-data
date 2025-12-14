import gradio as gr
import utils

# 1. 应用启动时尝试初始化数据
success, msg = utils.init_pipeline()
print(f"Server Log: {msg}")

# 2. 定义 Gradio 界面
def run_app():
    with gr.Blocks(title="李立军模拟器 (Hugging Face版)", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🏫 李立军风格演讲生成器")
        gr.Markdown(f"状态: *{msg}*")
        
        with gr.Row():
            # --- 左侧配置区 ---
            with gr.Column(scale=1):
                gr.Markdown("### 🛠️ 配置与输入")
                
                # 模型选择
                model_repo = gr.Dropdown(
                    label="选择模型",
                    choices=[
                        "Qwen/Qwen2.5-72B-Instruct", 
                        "Qwen/Qwen3-Next-80B-A3B-Instruct",
                        "Qwen/Qwen3-235B-A22B-Instruct-2507"
                    ],
                    value="Qwen/Qwen2.5-72B-Instruct",
                    interactive=True
                )

                gr.Markdown("---")
                
                input_topic = gr.Textbox(label="演讲主题", value="关于严禁在实验室玩原神")
                input_event = gr.Textbox(label="导火索事件", value="刚才有个后生做实验的时候在那抽卡", lines=2)
                input_req = gr.Textbox(label="具体要求", value="痛斥玩物丧志，结合阶层固化，结尾强调实验室纪律", lines=3)
                
                btn_submit = gr.Button("🚀 开始生成", variant="primary")

            # --- 右侧结果区 ---
            with gr.Column(scale=2):
                gr.Markdown("### 📝 生成结果")
                
                with gr.Tabs():
                    with gr.TabItem("AI 回复"):
                        output_ai = gr.Markdown(label="生成的文章", min_height=400)
                    
                    with gr.TabItem("调试信息"):
                        output_sys = gr.Textbox(label="System Prompt (含范文)", lines=5)
                        output_user = gr.Textbox(label="User Prompt (指令)", lines=3)

        # --- 事件绑定 ---
        btn_submit.click(
            fn=utils.generate_article, # 调用 utils 里的函数
            inputs=[ model_repo, input_topic, input_event, input_req],
            outputs=[output_sys, output_user, output_ai]
        )

    return demo

if __name__ == "__main__":
    app = run_app()
    app.launch()