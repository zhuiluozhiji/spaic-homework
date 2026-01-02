import gzip
import shutil
import os

# 数据存放目录
data_dir = './MNIST'
files = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz'
]

print(f"开始解压 {data_dir} 下的文件...")

for f in files:
    gz_path = os.path.join(data_dir, f)
    # 去掉 .gz 后缀
    extract_path = os.path.join(data_dir, f.replace('.gz', ''))
    
    if os.path.exists(gz_path):
        print(f"正在解压: {f} -> {os.path.basename(extract_path)}")
        try:
            with gzip.open(gz_path, 'rb') as f_in:
                with open(extract_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        except Exception as e:
            print(f"解压失败 {f}: {e}")
    else:
        print(f"警告：找不到文件 {gz_path}")

print("解压完成！")