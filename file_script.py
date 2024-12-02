#创建相应文件结构的脚本
from pathlib import Path
# 获取脚本所在的父文件夹路径
parent_folder = Path(__file__).resolve().parent
# 创建章节文件夹及其子文件夹
for i in range(3, 19):
    chapter_folder = parent_folder / f'chapter{i}'
    # 创建章节文件夹
    chapter_folder.mkdir(parents=True, exist_ok=True)
    # 创建 data, code, result 文件夹
    for subfolder in ['data', 'code', 'result']:
        (chapter_folder / subfolder).mkdir(parents=True, exist_ok=True)
        
print("文件夹创建完成！")
