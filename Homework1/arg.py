import argparse

def parse():
  # 创建解析器
  parser = argparse.ArgumentParser(description='示例脚本')

  # 添加参数
  parser.add_argument('--input', '-i', type=str, required=True, help='输入文件路径')
  parser.add_argument('--output', '-o', type=str, default='output.txt', help='输出文件路径')

  # 解析参数
  args = parser.parse_args()

  # 使用参数
  print(f"输入文件: {args.input}")
  print(f"输出文件: {args.output}")
  
  return args