import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from turtlesim.srv import SetPen
import math

from .cv_pipeline import processar_imagem_para_turtlesim

class TurtleController(Node):
    def __init__(self):
        super().__init__('turtle_controller')
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.pose_sub = self.create_subscription(Pose, '/turtle1/pose', self.pose_callback, 10)
        self.pen_client = self.create_client(SetPen, '/turtle1/set_pen')
        
        self.current_pose = None
        self.paths = []
        self.current_path_idx = 0
        self.current_point_idx = 0
        self.state = "INIT"
        
        try:
            self.get_logger().info("Iniciando processamento em alta resolução...")
            image_path = 'dog.jpeg' 
            self.paths = processar_imagem_para_turtlesim(image_path)
            self.get_logger().info(f"Processamento concluído. {len(self.paths)} caminhos gerados.")
        except Exception as e:
            self.get_logger().error(f"Erro no processamento da imagem: {e}")
            return

        self.timer = self.create_timer(0.02, self.control_loop)

    def pose_callback(self, msg):
        self.current_pose = msg

    def set_pen(self, off):
        if not self.pen_client.wait_for_service(timeout_sec=1.0):
            return
        
        req = SetPen.Request()
        req.off = int(off)
        req.r, req.g, req.b = 255, 255, 255
        req.width = 2
        self.pen_client.call_async(req)

    def control_loop(self):
        if self.current_pose is None or not self.paths:
            return

        if self.current_path_idx >= len(self.paths):
            if self.state != "DONE":
                self.get_logger().info("Desenho concluído com sucesso!")
                self.cmd_vel_pub.publish(Twist()) 
                self.state = "DONE"
            return

        current_path = self.paths[self.current_path_idx]
        target_point = current_path[self.current_point_idx]
        
        target_x, target_y = target_point
        dist = math.sqrt((target_x - self.current_pose.x)**2 + (target_y - self.current_pose.y)**2)
        angle_to_target = math.atan2(target_y - self.current_pose.y, target_x - self.current_pose.x)
        
        angle_error = angle_to_target - self.current_pose.theta
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

        msg = Twist()

        TOLERANCIA_DISTANCIA = 0.05 

        if self.state == "INIT":
            self.set_pen(off=1)
            self.state = "MOVING_TO_START"
            
        elif self.state == "MOVING_TO_START":
            if dist < TOLERANCIA_DISTANCIA:
                self.set_pen(off=0)
                self.state = "DRAWING"
            else:
                # Movimento rápido para buscar o início da linha
                msg.angular.z = 10.0 * angle_error
                if abs(angle_error) < 0.5:
                    msg.linear.x = 4.0 * dist

        elif self.state == "DRAWING":
            if dist < TOLERANCIA_DISTANCIA:
                self.current_point_idx += 1
                if self.current_point_idx >= len(current_path):
                    self.current_path_idx += 1
                    self.current_point_idx = 0
                    self.state = "INIT" 
            else:
                msg.angular.z = 12.0 * angle_error
                
                if abs(angle_error) < 0.6: 
                    msg.linear.x = 6.0 * dist

        self.cmd_vel_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TurtleController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
