import subprocess
import numpy as np
import sys

def save_rgb_images_to_video(images, output_filename, fps=30):
    height, width, layers = images[0].shape
    command = ['ffmpeg',
               '-y',
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-s', f'{width}x{height}',
               '-pix_fmt', 'rgb24',
               '-r', str(fps),
               '-i', '-',
               '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p',
               output_filename]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for image in images:
        process.stdin.write(image.tobytes())
    process.stdin.close()
    process.wait()


def save_rgb_images_to_video_do(images, output_filename, fps=30):
    if not images:
        print("Error: No images provided to save_rgb_images_to_video", file=sys.stderr)
        return

    height, width, layers = images[0].shape

    # 构建 ffmpeg 命令
    command = [
        'ffmpeg',
        '-y',  # 覆盖输出文件而不提示
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',  # 视频分辨率
        '-pix_fmt', 'rgb24',  # 输入像素格式
        '-r', str(fps),  # 帧率
        '-i', '-',  # 从标准输入读取原始视频数据
        '-c:v', 'libx264',  # 使用 H.264 编码
        '-pix_fmt', 'yuv420p',  # 输出像素格式（兼容性更好）
        output_filename  # 输出文件名
    ]

    # 启动 ffmpeg 进程
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,  # 允许向 stdin 写入
        stdout=subprocess.DEVNULL,  # 忽略 stdout
        stderr=subprocess.PIPE  # 捕获 stderr 用于调试
    )

    try:
        for image in images:
            # 检查 ffmpeg 进程是否仍然在运行
            if process.poll() is not None:
                raise Exception("FFmpeg process terminated unexpectedly")

            # 将图像数据写入 ffmpeg 的 stdin
            process.stdin.write(image.tobytes())

        # 写入完成后关闭 stdin
        process.stdin.close()

        # 等待 ffmpeg 进程完成
        process.wait()

        if process.returncode != 0:
            # 如果 ffmpeg 返回非零值，抛出异常
            error_message = process.stderr.read().decode()
            raise Exception(f"FFmpeg exited with code {process.returncode}: {error_message}")

    except BrokenPipeError:
        # 处理BrokenPipeError
        print("BrokenPipeError: FFmpeg might have crashed or closed the pipe prematurely", file=sys.stderr)
        error_message = process.stderr.read().decode()
        print(f"FFmpeg Error Output:\n{error_message}", file=sys.stderr)
    except Exception as e:
        # 处理其他异常
        print(f"Error: {str(e)}", file=sys.stderr)
    finally:
        # 确保释放资源
        if process.stdin:
            process.stdin.close()
        if process.stderr:
            process.stderr.close()
        if process.returncode is None:
            process.terminate()
            process.wait()

def in_workspace(pose):
    """
        pose: 机械臂末端位姿
    """
    if pose[0] < 0.2 or pose[0]**2 + pose[1]**2 > 0.6**2:
        return False
    
    return True

# 示例用法
if __name__ == "__main__":
    # 示例图像列表（随机生成的 RGB 图像）
    images = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(30)]

    # 保存为视频文件
    save_rgb_images_to_video(images, "output.mp4", fps=30)