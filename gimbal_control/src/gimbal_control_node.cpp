#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <px4_msgs/msg/gimbal_manager_set_attitude.hpp>
#include <tf2/LinearMath/Quaternion.h>

#include <algorithm>
#include <cmath>

class GimbalControlNode : public rclcpp::Node {
public:
    GimbalControlNode() : Node("gimbal_control_node") {
        this->declare_parameter("gain_p", 0.01);
        this->declare_parameter("gain_y", 0.01);
        this->declare_parameter("deadzone", 0.05); // 5% of screen
        this->declare_parameter("img_width", 640);
        this->declare_parameter("img_height", 480);

        gain_p_ = this->get_parameter("gain_p").as_double();
        gain_y_ = this->get_parameter("gain_y").as_double();
        deadzone_ = this->get_parameter("deadzone").as_double();
        width_ = this->get_parameter("img_width").as_int();
        height_ = this->get_parameter("img_height").as_int();

        gimbal_pub_ = this->create_publisher<px4_msgs::msg::GimbalManagerSetAttitude>(
            "/fmu/in/gimbal_manager_set_attitude", 10);

        target_sub_ = this->create_subscription<geometry_msgs::msg::Point>(
            "/vision/targets", 10, std::bind(&GimbalControlNode::target_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Gimbal Control Node initialized.");
    }

private:
    void target_callback(const geometry_msgs::msg::Point::SharedPtr msg) {
        // Normalized error from center (-1 to 1)
        double center_x = width_ / 2.0;
        double center_y = height_ / 2.0;

        double err_x = (msg->x - center_x) / center_x;
        double err_y = (msg->y - center_y) / center_y;

        // Apply dead-zone
        if (std::abs(err_x) < deadzone_) err_x = 0;
        if (std::abs(err_y) < deadzone_) err_y = 0;

        if (err_x == 0 && err_y == 0) return;

        // Update target angles (incremental control)
        // err_x corresponds to Yaw, err_y to Pitch
        // Note: Coordinates might need sign flip depending on gimbal mounting
        cur_yaw_ -= err_x * gain_y_; 
        cur_pitch_ += err_y * gain_p_;

        // Clamp angles
        cur_pitch_ = std::clamp(cur_pitch_, -M_PI_2, M_PI_2);
        
        publish_gimbal_command();
    }

    void publish_gimbal_command() {
        px4_msgs::msg::GimbalManagerSetAttitude cmd;
        cmd.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        
        // Convert Euler to Quaternion
        tf2::Quaternion q;
        q.setRPY(0, cur_pitch_, cur_yaw_);

        cmd.q[0] = q.x();
        cmd.q[1] = q.y();
        cmd.q[2] = q.z();
        cmd.q[3] = q.w();

        // Flags for pitch/yaw lock
        cmd.flags = px4_msgs::msg::GimbalManagerSetAttitude::GIMBAL_MANAGER_FLAGS_PITCH_LOCK |
                    px4_msgs::msg::GimbalManagerSetAttitude::GIMBAL_MANAGER_FLAGS_YAW_LOCK;

        gimbal_pub_->publish(cmd);
    }

    rclcpp::Publisher<px4_msgs::msg::GimbalManagerSetAttitude>::SharedPtr gimbal_pub_;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr target_sub_;

    double gain_p_, gain_y_, deadzone_;
    int width_, height_;
    
    double cur_pitch_ = 0.0;
    double cur_yaw_ = 0.0;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<GimbalControlNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
