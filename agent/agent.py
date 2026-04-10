import inspect
import os
import re
from string import Template
from typing import List, Callable, Tuple

from typing import List
import requests

import platform
import subprocess

from prompt_template import react_system_prompt_template

llm_model_id_local = "ollama.rnd.huawei.com/library/qwen3.5:latest"
llm_base_url_local="http://localhost:11434/api/chat"

project_directory = "yhd0407"

class ReactAgent:
    def __init__(self, tools: List[Callable], model: str, project_dir: str):
            self.tools = { func.__name__: func for func in tools }
            self.model = model
            self.project_dir = project_dir

    def run(self, task: str) -> str:
        messages = [
            {"role": "system", "content": self.render_system_prompt(react_system_prompt_template)},
            {"role": "user", "content": f"<question>{task}</question>"}
        ]

        while True:

            # 调用模型
            content = self.call_model(messages)

            # 检测 Thought
            thought_match = re.search(r"<thought>(.*?)</thought>", content, re.DOTALL)
            if thought_match:
                thought = thought_match.group(1)
                print(f"\n\n💭 Thought: {thought}")

            # 检测模型是否输出 Final Answer，如果是的话，直接返回
            if "<final_answer>" in content:
                final_answer = re.search(r"<final_answer>(.*?)</final_answer>", content, re.DOTALL)
                return final_answer.group(1)

            # 检测 Action
            action_match = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
            if not action_match:
                raise RuntimeError("模型未输出 <action>")
            action = action_match.group(1)
            tool_name, args = self.parse_action(action)

            print(f"\n\n🔧 Action: {tool_name}({', '.join(args)})")

            # 只有终端命令才需要询问用户，其他的工具直接执行
            should_continue = input(f"\n\n是否继续？（Y/N）") if tool_name == "run_terminal_command" else "y"
            if should_continue.lower() != 'y':
                print("\n\n操作已取消。")
                return "操作被用户取消"

            try:
                observation = self.tools[tool_name](*args)
            except Exception as e:
                observation = f"工具执行错误：{str(e)}"
            print(f"\n\n🔍 Observation：{observation}")

            obs_msg = f"<observation>{observation}</observation>"
            messages.append({"role": "user", "content": obs_msg})

    def get_tool_list(self) -> str:
        """生成工具列表字符串，包含函数名与简要说明"""
        tool_list = []
        for func in self.tools.values():
            name = func.__name__
            signature = str(inspect.signature(func))
            doc =inspect.getdoc(func) if inspect.getdoc(func) else ""
            tool_list.append(f"{name}{signature} - {doc}")
        return "\n".join(tool_list)


    def render_system_prompt(self, system_prompt_template:str) -> str:
        """渲染系统提示模板，替换变量"""
        tool_list = self.get_tool_list()
        file_list = ", ".join(
            os.path.abspath(os.path.join(self.project_dir, f))
            for f in os.listdir(self.project_dir)
        )
        return Template(system_prompt_template).substitute(
            operating_system=self.get_operating_system_name(),
            tool_list=tool_list,
            file_list=file_list
        )

    def call_model(self, messages) -> str:
        """调用模型"""

        data = {
            'model': llm_model_id_local,
            'messages': messages,
            # 本地
            'stream': False
        }

        print("\n\n正在请求模型，请稍等...")
        response = requests.post(llm_base_url_local, headers=None, json=data)
        # 本地
        answer = response.json().get('message').get('content')
        print(answer)

        messages.append({'role': 'assistant', 'content': answer})
        return answer

    def get_operating_system_name(self):
        """获取操作系统"""
        os_map = {
            "Darwin": "macOS",
            "Windows": "Windows",
            "Linux": "Linux"
        }

        return os_map.get(platform.system(), "Unknown")

    def parse_action(self, code_str: str) -> Tuple[str, List[str]]:
        match = re.match(r'(\w+)\((.*)\)', code_str, re.DOTALL)
        if not match:
            raise ValueError("Invalid function call syntax")

        func_name = match.group(1)
        args_str = match.group(2).strip()

        # 手动解析参数，特别处理包含多行内容的字符串
        args = []
        current_arg = ""
        in_string = False
        string_char = None
        i = 0
        paren_depth = 0

        while i < len(args_str):
            char = args_str[i]

            if not in_string:
                if char in ['"', "'"]:
                    in_string = True
                    string_char = char
                    current_arg += char
                elif char == '(':
                    paren_depth += 1
                    current_arg += char
                elif char == ')':
                    paren_depth -= 1
                    current_arg += char
                elif char == ',' and paren_depth == 0:
                    # 遇到顶层逗号，结束当前参数
                    args.append(self._parse_single_arg(current_arg.strip()))
                    current_arg = ""
                else:
                    current_arg += char
            else:
                current_arg += char
                if char == string_char and (i == 0 or args_str[i-1] != '\\'):
                    in_string = False
                    string_char = None

            i += 1

        # 添加最后一个参数
        if current_arg.strip():
            args.append(self._parse_single_arg(current_arg.strip()))

        return func_name, args

    def _parse_single_arg(self, arg_str: str):
        """解析单个参数"""
        arg_str = arg_str.strip()

        # 如果是字符串字面量
        if (arg_str.startswith('"') and arg_str.endswith('"')) or \
                (arg_str.startswith("'") and arg_str.endswith("'")):
            # 移除外层引号并处理转义字符
            inner_str = arg_str[1:-1]
            # 处理常见的转义字符
            inner_str = inner_str.replace('\\"', '"').replace("\\'", "'")
            inner_str = inner_str.replace('\\n', '\n').replace('\\t', '\t')
            inner_str = inner_str.replace('\\r', '\r').replace('\\\\', '\\')
            return inner_str
        # 尝试使用 ast.literal_eval 解析其他类型
        try:
            return ast.literal_eval(arg_str)
        except (SyntaxError, ValueError):
            # 如果解析失败，返回原始字符串
            return arg_str

def read_file(file_path: str) -> str:
    """用于读取文件内容"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def write_file(file_path: str, content: str) -> str:
    """将指定内容写入指定文件"""
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
    return f"写入成功: {file_path}"

def run_terminal_command(command: str) -> str:
    """执行终端命令并返回输出，自带错误捕获"""
    try:
        # 执行命令，捕获输出
        result = subprocess.run(
            command,
            shell=True,
            check=True,          # 命令失败自动抛异常
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8"
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        # 命令执行失败时返回错误信息
        return f"命令执行失败：{e.stderr.strip()}"


# @click.command()
# @click.argument('project_directory', type=click.Path(exists=True, file_okay=False, dir_okay=True))
# def main(project_directory: str):
#     project_dir = os.path.abspath(project_directory)
def main():

    # 自动创建目录，防止不存在
    if not os.path.exists(project_directory):
        os.makedirs(project_directory)

    # 工具
    tools = [read_file, write_file, run_terminal_command]

    # 实例化agent
    agent = ReactAgent(tools=tools, model=llm_model_id_local, project_dir=project_directory)

    # 指令
    task = input("请输入任务:")

    # 运行agent
    answer = agent.run(task)

    print(f"结果 ： {answer}")

if __name__ == "__main__":
    main()