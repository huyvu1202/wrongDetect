import os 
import subprocess

path = r"D:\SYLLABUS\AIP491\demo\video_output"
des_path = r"D:\SYLLABUS\AIP491\data_no_audio\temp"
os.makedirs(des_path, exist_ok=True)
for filename in os.listdir(path):
    if filename.lower().endswith('.mp4'):
        input_path = os.path.join(path, filename)
        output_path = os.path.join(des_path, filename)
        print(f'Đang xử lí {filename}')
        subprocess.run([
            'ffmpeg', '-i',  
            input_path,
            '-c', 'copy',
            '-an',           
            output_path
        ])
