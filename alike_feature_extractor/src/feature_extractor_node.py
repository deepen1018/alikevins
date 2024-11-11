
# #!/usr/bin/env python

# import rospy
# import cv2
# import numpy as np
# from sensor_msgs.msg import Image, PointCloud
# from geometry_msgs.msg import Point32
# from cv_bridge import CvBridge
# import torch
# import sys

# # 添加 ALIKE 模型的路径
# sys.path.append('/home/jaron/deep_ws/src/ALIKE')  # 确保路径正确
# from alike import ALike, configs  # 导入 ALIKE 模型
# from std_msgs.msg import Header  # 确保正确导入 Header

# class FeatureExtractorNode:
#     def __init__(self):
#         self.bridge = CvBridge()

#         # 初始化ALike模型，使用configs中的model_path
#         self.model = ALike(
#             **configs['alike-t'],  # 使用configs中的model_path
#             device='cuda', 
#             top_k=80, 
#             scores_th=0.5, 
#             n_limit=5000
#         )

#         # 订阅相机图像的topic
#         self.image_sub = rospy.Subscriber("/cam0/image_raw", Image, self.image_callback)

#         # 发布特征点为二维点云
#         self.cloud_pub = rospy.Publisher("/feature_points", PointCloud, queue_size=10)
#         self.image_pub = rospy.Publisher("/feature_image", Image, queue_size=10)

#     def image_callback(self, msg):
#         rospy.loginfo("Image callback triggered")  # 添加日志信息
#         try:
#             # 将ROS Image消息转换为OpenCV图像
#             cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
#         except CvBridgeError as e:
#             rospy.logerr(f"Failed to convert image: {e}")
#             return

#         # 将BGR图像转换为RGB格式以适配ALike模型
#         img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

#         # 提取特征点
#         pred = self.model(img_rgb)
#         keypoints = pred['keypoints']
#         descriptors = pred['descriptors']
#         scores = pred['scores']

#         # 打印调试信息
#         rospy.loginfo(f"Extracted {len(keypoints)} keypoints")
        
#         for pt in keypoints:
#             # 将特征点绘制为小圆圈，颜色为红色，半径为3，线宽为1
#             cv2.circle(cv_image, (int(pt[0]), int(pt[1])), radius=3, color=(0, 0, 255), thickness=1)

#         # 发布带有特征点的图像
#         try:
#             feature_image_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
#             self.image_pub.publish(feature_image_msg)
#             rospy.loginfo("Published feature image with keypoints")
#         except CvBridgeError as e:
#             rospy.logerr(f"Failed to convert and publish image with keypoints: {e}")

#         # 将特征点转换为PointCloud消息
#         cloud_msg = self.convert_to_pointcloud(keypoints)

#         # 发布点云消息
#         if cloud_msg:
#             self.cloud_pub.publish(cloud_msg)
#             rospy.loginfo("Published point cloud message")

#     def convert_to_pointcloud(self, keypoints):
#         # 创建点云消息的header
#         header = Header()  # 使用 Header 创建头部
#         header.stamp = rospy.Time.now()
#         header.frame_id = "camera_link"  # 请根据实际情况修改frame_id

#         # 准备要发布的二维点云数据
#         points = []
#         for pt in keypoints:
#             point = Point32()
#             point.x = pt[0]
#             point.y = pt[1]
#             point.z = 0  # 保持 z 为 0，表示二维点云
#             points.append(point)

#         if len(points) == 0:
#             rospy.logwarn("No points to convert to point cloud")
#             return None

#         # 创建 PointCloud 消息
#         cloud_msg = PointCloud()
#         cloud_msg.header = header
#         cloud_msg.points = points

#         return cloud_msg


# if __name__ == '__main__':
#     rospy.init_node('feature_extractor_node')
#     node = FeatureExtractorNode()
#     rospy.spin()

#################################test ros alike############################################
#!/usr/bin/env python
import rospy
import cv2
import uuid 
import numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud
from geometry_msgs.msg import Point32
from cv_bridge import CvBridge
import torch
import sys
from my_custom_msgs.msg import LeftImageWithFeatures  # 自定义消息

# 添加 ALIKE 模型路径
sys.path.append('/home/jaron/deep_ws/src/ALIKE')  
from alike import ALike, configs 
from cost_function import select_optimal_features

class LeftFeatureExtractorNode:
    def __init__(self):
        self.bridge = CvBridge()
        # 初始化 ALIKE 模型
        self.model = ALike(**configs['alike-l'], device='cuda', top_k=500, scores_th=0.5, n_limit=5000)

        # 订阅左目图像的 topic
        self.left_image_sub = rospy.Subscriber("/cam0/image_raw", Image, self.left_image_callback)

        # 发布包含左目图像和特征点的消息
        self.left_feature_pub = rospy.Publisher("/left_image_with_features", LeftImageWithFeatures, queue_size=10)

        self.feature_image_pub = rospy.Publisher("/feature_image", Image, queue_size=10)


    def left_image_callback(self, msg):
        try:
            # 将 ROS Image 消息转换为 OpenCV 图像
            left_cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"Failed to convert image: {e}")
            return

        # 转换为 RGB 格式
        left_rgb = cv2.cvtColor(left_cv_image, cv2.COLOR_BGR2RGB)

        # 提取左目图像的特征点
        left_pred = self.model(left_rgb)
        left_keypoints = left_pred['keypoints']
        scores = left_pred['scores']

        # 使用 cost_function 筛选出优化的特征点
        selected_keypoints, selected_scores = select_optimal_features(left_keypoints, scores, num_features=150, min_distance=10)

        # 在图像上绘制特征点
        for pt in selected_keypoints:
            cv2.circle(left_cv_image, (int(pt[0]), int(pt[1])), radius=3, color=(0, 255, 0), thickness=-1)

        # 将包含特征点的图像发布到 ROS
        try:
            feature_image_msg = self.bridge.cv2_to_imgmsg(left_cv_image, "bgr8")
            self.feature_image_pub.publish(feature_image_msg)
        except CvBridgeError as e:
            rospy.logerr(f"Failed to convert and publish feature image: {e}")

        # 将特征点转换为 PointCloud 消息
        left_feature_msg = self.convert_to_pointcloud(selected_keypoints, selected_scores ,msg.header.stamp)

        # 创建同步消息并发布
        sync_msg = LeftImageWithFeatures()
        sync_msg.header.stamp = msg.header.stamp
        sync_msg.left_image = msg
        sync_msg.left_feature_points = left_feature_msg

        self.left_feature_pub.publish(sync_msg)

    def convert_to_pointcloud(self, keypoints,scores, timestamp):
        header = Header(stamp=timestamp, frame_id="camera_link")
        points = [Point32(x=pt[0], y=pt[1], z=0) for pt in keypoints]
        # cloud_msg = PointCloud(header=header, points=points)
        # return cloud_msg
        # 创建自定义的 channels 来存储额外信息（ID 和 score）
        #ids = [uuid.uuid4().int & (1<<32)-1 for _ in keypoints]  # 生成32位唯一ID
        point_cloud = PointCloud(header=header, points=points)
        
        # 定义通道

        #point_cloud.channels.append({"name": "id", "values": ids})
        #point_cloud.channels.append({"name": "score", "values": scores.tolist()})
        
        return point_cloud
    
if __name__ == '__main__':
    rospy.init_node('left_feature_extractor_node')
    node = LeftFeatureExtractorNode()
    rospy.spin()
