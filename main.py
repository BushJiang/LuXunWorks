from generator import Generator
import yaml
# 导入easydict库用于将字典转换为更易用的对象
from easydict import EasyDict

def read_yaml_config(file_path):
    """读取YAML配置文件并返回EasyDict对象"""
    with open(file_path, "r") as file:
        # 读取yaml文件，得到字典
        config_data = yaml.safe_load(file)
    # 把字典转换为EasyDict对象。可以通过属性的形式访问键值对
    return EasyDict(config_data)

class CommandLine:
    def __init__(self, config_path):
        """初始化CommandLine类"""
        self._generator = None
        self.config = read_yaml_config(config_path)

    # 创建字符画的在线网站（字体为oge）：http://patorjk.com/software/taag/
    def show_start_info(self):
        """显示启动信息"""
        with open('./start_info.txt') as fw:
            print(fw.read())

    def run(self):
        """运行命令行工具"""
        self.show_start_info()
        
        # 初始化generator实例，用于搜索
        self._generator = Generator(self.config)
        self._show_instructions()
        self._handle_commands()

    def _show_instructions(self):
        """显示用户指令"""
        print('1. 输入`create`创建集合。')
        print('2. 创建完成后，输入`ask`进入问答模式，输入疑似鲁迅的名言。')
        print('3. `delete`删除已有集合。')
        print('4. `exit`退出。')

    def _handle_commands(self):
        """处理用户输入的命令"""
        while True:
            user_input = input("(ask) ")
            commands = user_input.split(' ')
            # 通过空格分成命令和参数两部分
            command = commands[0]
            args = commands[1:]
            # 进入create模式
            if command == 'create' and not args:
                self.create_vector_db()
            elif command == 'create':
                print('(ask) create不接受参数。')
            # 进入ask模式
            elif command == 'ask' and not args:
                self.generate()
            elif command == 'ask':
                print('(ask) ask不接受参数。')
            # 删除集合
            elif command == 'delete' and not args:
                self.delete_collection()
            elif command == 'delete':
                print('(ask) delete不接受参数。')
            # 退出应用
            elif command == 'exit':
                self._exit()
            else:
                print('(ask) 只有[create|ask|delete|exit]命令, 请重新尝试。')

    def create_vector_db(self):
        """创建集合"""
        self._generator.create_vector_db()
        print('(ask) 创建集合完成')

    def generate(self):
        """搜索相似语句，生成回答"""
        while True:
            user_input = input("(ask) 请输入疑似鲁迅的名言: ")
            if user_input == 'exit':
                print('(ask) 退出问答')
                break
            # 如果命令为空，跳过本次循环
            elif not user_input:
                print('(ask) 请重新输入')
                continue
            else:
                self._generator.generate_response(user_input)


    def delete_collection(self):
        """删除集合"""
        self._generator.delete_collection()

    def _exit(self):
        """退出程序"""
        exit()

if __name__ == '__main__':
    cli = CommandLine('config.yaml')
    cli.run()