from piper_sdk import *
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import sys
import time
from datetime import datetime
import termios
import tty

# 初始化变量
photo_folder = "calibration_data/collected_photos"
joint_file = "calibration_data/robot_joint.txt"
endPose_file = "calibration_data/robot_end_pose.txt"


joint_factor = 57295.7795  # 用于弧度转换的因子
endPose_factor = 1000  # 用于末端位姿的转换因子

# 创建文件夹和txt文件（如果不存在）
if not os.path.exists(photo_folder):
    os.makedirs(photo_folder)

if not os.path.exists(joint_file):
    with open(joint_file, 'w') as f:
        f.write("Timestamp,joint1,joint2,joint3,joint4,joint5,joint6\n")

if not os.path.exists(endPose_file):
    with open(endPose_file, 'w') as f:
        f.write("Timestamp,end_x,end_y,end_z,end_roll,end_pitch,end_yaw\n")

# 初始化RealSense相机
def init_realsense():
    try:
        # 创建RealSense配置和管道
        pipeline = rs.pipeline()
        config = rs.config()
        
        # 配置深度和颜色流
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # 启动管道
        pipeline.start(config)
        
        # 等待相机稳定
        for _ in range(30):
            pipeline.wait_for_frames()
            
        print("RealSense相机初始化成功")
        return pipeline
    except Exception as e:
        print(f"RealSense相机初始化失败: {e}")
        return None

# 使用RealSense相机获取当前帧
def get_current_frame(pipeline):
    if pipeline is None:
        return None
    
    try:
        # 等待获取一帧数据
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            return None
        
        # 将图像转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())
        return color_image
    except Exception as e:
        print(f"获取帧时出错: {e}")
        return None

# 使用RealSense相机拍摄照片
def take_photo(pipeline):
    if pipeline is None:
        print("相机未初始化")
        return False
    
    try:
        # 等待获取一帧数据
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            print("无法获取彩色帧")
            return False
        
        # 将图像转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())
        
        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        photo_path = os.path.join(photo_folder, f"photo_{timestamp}.jpg")
        cv2.imwrite(photo_path, color_image)
        print(f"照片已保存至: {photo_path}")
        
        return True
    except Exception as e:
        print(f"拍照时出错: {e}")
        return False

# 删除上一条数据
def delete_last_data():
    if os.path.getsize(joint_file) <= len("Timestamp,joint1,joint2,joint3,joint4,joint5,joint6\n"):
        print("没有数据可删除")
        return
    
    # 读取所有行
    with open(joint_file, 'r') as f:
        lines = f.readlines()
    
    # 如果只有标题行或没有数据，不删除
    if len(lines) <= 1:
        print("没有数据可删除")
        return
    
    # 删除最后一行数据
    lines = lines[:-1]
    
    # 写入剩余行
    with open(joint_file, 'w') as f:
        f.writelines(lines)
    
    print("已从txt文件中删除最后一条姿态数据")
    
    # 删除对应的图片
    if os.listdir(photo_folder):
        last_photo_path = os.path.join(photo_folder, os.listdir(photo_folder)[-1])
        if os.path.exists(last_photo_path):
            os.remove(last_photo_path)
            print(f"已删除最后一张照片: {last_photo_path}")

# 获取单字符输入（无需按Enter）
def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

# 在图像上显示关节数据
def display_joint_data(image, joints):
    if image is None:
        return None
    
    # 创建一个用于显示关节数据的副本
    display_image = image.copy()
    
    # 在图像上添加关节数据文本
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 255, 0)  # 绿色
    thickness = 1
    y_offset = 20
    
    # 显示标题
    cv2.putText(display_image, "Joint Angles (radians):", (10, 20), font, font_scale, color, thickness)
    
    # 显示每个关节的数据
    for i, joint in enumerate(joints):
        text = f"Joint {i+1}: {joint:.4f}"
        cv2.putText(display_image, text, (10, 40 + i * y_offset), font, font_scale, color, thickness)
    
    return display_image

# 主程序
def main():
    # 初始化RealSense相机
    pipeline = init_realsense()
    if pipeline is None:
        return
    
    # 连接机械臂
    try:
        piper = C_PiperInterface()
        piper.ConnectPort()
    except Exception as e:
        print(f"连接机械臂失败: {e}")
        pipeline.stop()  # 关闭相机
        return
    
    print("按 's' 捕获姿态和照片，按 'm' 删除上一条数据，按 'p' 停止")
    
    # 创建显示窗口
    cv2.namedWindow("RealSense Image with Joint Data", cv2.WINDOW_NORMAL)
    # 设置窗口大小（宽度=800像素，高度=600像素）
    cv2.resizeWindow("RealSense Image with Joint Data", 800, 600)

    try:
        while True:
            # 获取当前帧
            frame = get_current_frame(pipeline)
            if frame is None:
                print("无法获取图像帧")
                continue
            
            # 获取关节数据
            try:
                data = piper.GetArmJointMsgs()
                joint1 = data.joint_state.joint_1/joint_factor
                joint2 = data.joint_state.joint_2/joint_factor
                joint3 = data.joint_state.joint_3/joint_factor
                joint4 = data.joint_state.joint_4/joint_factor    
                joint5 = data.joint_state.joint_5/joint_factor
                joint6 = data.joint_state.joint_6/joint_factor
                
                joints = [joint1, joint2, joint3, joint4, joint5, joint6]
                
                # 在图像上显示关节数据
                display_frame = display_joint_data(frame, joints)
                
                # 显示图像
                cv2.imshow("RealSense Image with Joint Data", display_frame)

                # 获取末端信息
                endPoseData = piper.GetArmEndPoseMsgs()
                end_x = endPoseData.end_pose.X_axis/endPose_factor
                end_y = endPoseData.end_pose.Y_axis/endPose_factor
                end_z = endPoseData.end_pose.Z_axis/endPose_factor
                end_roll = endPoseData.end_pose.RX_axis/endPose_factor
                end_pitch = endPoseData.end_pose.RY_axis/endPose_factor
                end_yaw = endPoseData.end_pose.RZ_axis/endPose_factor
                endPose = [end_x, end_y, end_z, end_roll, end_pitch, end_yaw]
                
            except Exception as e:
                print(f"获取关节数据时出错: {e}")
                # 显示原始图像
                cv2.imshow("RealSense Image with Joint Data", frame)
            
            # 检测按键 (非阻塞)
            key = cv2.waitKey(10) & 0xFF
            
            # 也支持通过键盘输入控制
            if key == ord('s') or key == ord('S'):
                print("\n正在捕获数据...")
                try:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"在 {timestamp} 捕获关节数据")

                    # 将姿态信息写入txt文件
                    with open(joint_file, 'a') as f:
                        f.write(f"{timestamp},{joint1},{joint2},{joint3},{joint4},{joint5},{joint6}\n")
                    
                    # 将末端位姿信息写入txt文件
                    with open(endPose_file, 'a') as f:
                        f.write(f"{timestamp},{end_x},{end_y},{end_z},{end_roll},{end_pitch},{end_yaw}\n")
                    print(f"关节数据已保存至 {joint_file} 和 {endPose_file}")
                    # 拍摄照片
                    take_photo(pipeline)
                except Exception as e:
                    print(f"捕获数据时出错: {e}")
                
            elif key == ord('m') or key == ord('M'):
                print("\n正在删除上一条数据...")
                delete_last_data()
                
            elif key == ord('p') or key == ord('P'):
                print("\n停止捕获并保存数据")
                print(f"数据已保存至 {joint_file}")
                break
                
            # 也支持原始的getch()方式获取按键
            if cv2.getWindowProperty("RealSense Image with Joint Data", cv2.WND_PROP_VISIBLE) < 1:
                break
                
    finally:
        # 释放资源
        cv2.destroyAllWindows()
        pipeline.stop()
        print("RealSense相机已关闭")

if __name__ == "__main__":
    main()
