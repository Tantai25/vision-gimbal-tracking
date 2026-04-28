#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <unistd.h>
#include <poll.h>

#ifdef HAS_TURBOJPEG
#include <turbojpeg.h>
#endif

#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <algorithm>

struct Detection {
    cv::Rect box;
    std::string label;
    float confidence;
};

class VisionNode : public rclcpp::Node {
public:
    VisionNode() : Node("vision_node") {
        this->declare_parameter("device", "/dev/video16");
        this->declare_parameter("width", 640);
        this->declare_parameter("height", 480);
        this->declare_parameter("fps", 30);
        this->declare_parameter("pixel_format", "YUYV");
        this->declare_parameter("model_path", "yolo.onnx");
        this->declare_parameter("conf_threshold", 0.5);
        this->declare_parameter("rtp_url", "rtp://100.81.232.35:5600");

        device_name_ = this->get_parameter("device").as_string();
        width_ = this->get_parameter("width").as_int();
        height_ = this->get_parameter("height").as_int();
        fps_ = this->get_parameter("fps").as_int();
        conf_threshold_ = this->get_parameter("conf_threshold").as_double();
        std::string fmt = this->get_parameter("pixel_format").as_string();
        rtp_url_ = this->get_parameter("rtp_url").as_string();

        if (fmt == "YUYV") {
            pixel_format_ = V4L2_PIX_FMT_YUYV;
        } else if (fmt == "MJPEG") {
            pixel_format_ = V4L2_PIX_FMT_MJPEG;
#ifdef HAS_TURBOJPEG
            tj_handle_ = tjInitDecompress();
#endif
        } else {
            RCLCPP_WARN(this->get_logger(), "Unsupported format, using YUYV");
            pixel_format_ = V4L2_PIX_FMT_YUYV;
        }

        std::string model_path = this->get_parameter("model_path").as_string();
        try {
            net_ = cv::dnn::readNetFromONNX(model_path);
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            RCLCPP_INFO(this->get_logger(), "YOLO Model loaded: %s", model_path.c_str());
        } catch (const cv::Exception& e) {
            RCLCPP_WARN(this->get_logger(), "Could not load ONNX model. Inference will be skipped. %s", e.what());
        }

        target_pub_ = this->create_publisher<geometry_msgs::msg::Point>("/vision/targets", 10);
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/vision/annotated_image", 10);

        if (!init_v4l2()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize V4L2");
            return;
        }

        running_ = true;
        capture_thread_ = std::thread(&VisionNode::capture_loop, this);
        inference_thread_ = std::thread(&VisionNode::inference_loop, this);
        streaming_thread_ = std::thread(&VisionNode::streaming_loop, this);

        RCLCPP_INFO(this->get_logger(), "Vision Node started (Dual-Stream V4L2 + YOLO + FFmpeg)");
    }

    ~VisionNode() {
        running_ = false;
        stream_cv_.notify_all();
        infer_cv_.notify_all();
        
        if (capture_thread_.joinable()) capture_thread_.join();
        if (inference_thread_.joinable()) inference_thread_.join();
        if (streaming_thread_.joinable()) streaming_thread_.join();

        cleanup_v4l2();
#ifdef HAS_TURBOJPEG
        if (tj_handle_) tjDestroy(tj_handle_);
#endif
    }

private:
    struct Buffer {
        void* start;
        size_t length;
    };

    bool init_v4l2() {
        fd_ = open(device_name_.c_str(), O_RDWR | O_NONBLOCK, 0);
        if (fd_ < 0) return false;

        struct v4l2_format fmt;
        memset(&fmt, 0, sizeof(fmt));
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = width_;
        fmt.fmt.pix.height = height_;
        fmt.fmt.pix.pixelformat = pixel_format_;
        fmt.fmt.pix.field = V4L2_FIELD_ANY;

        ioctl(fd_, VIDIOC_S_FMT, &fmt);
        width_ = fmt.fmt.pix.width;
        height_ = fmt.fmt.pix.height;

        struct v4l2_streamparm streamparm;
        memset(&streamparm, 0, sizeof(streamparm));
        streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        streamparm.parm.capture.timeperframe.numerator = 1;
        streamparm.parm.capture.timeperframe.denominator = fps_;
        ioctl(fd_, VIDIOC_S_PARM, &streamparm);

        struct v4l2_requestbuffers req;
        memset(&req, 0, sizeof(req));
        req.count = 4;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;

        if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0) return false;

        buffers_.resize(req.count);
        for (size_t i = 0; i < req.count; ++i) {
            struct v4l2_buffer buf;
            memset(&buf, 0, sizeof(buf));
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            ioctl(fd_, VIDIOC_QUERYBUF, &buf);
            buffers_[i].length = buf.length;
            buffers_[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
            ioctl(fd_, VIDIOC_QBUF, &buf);
        }

        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ioctl(fd_, VIDIOC_STREAMON, &type);
        return true;
    }

    void cleanup_v4l2() {
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ioctl(fd_, VIDIOC_STREAMOFF, &type);
        for (auto& buffer : buffers_) {
            munmap(buffer.start, buffer.length);
        }
        if (fd_ >= 0) close(fd_);
    }

    void capture_loop() {
        struct pollfd pfd;
        pfd.fd = fd_;
        pfd.events = POLLIN;

        while (running_ && rclcpp::ok()) {
            int ret = poll(&pfd, 1, 1000);
            if (ret <= 0) continue;

            if (pfd.revents & POLLIN) {
                struct v4l2_buffer buf;
                memset(&buf, 0, sizeof(buf));
                buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                buf.memory = V4L2_MEMORY_MMAP;

                if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) continue;

                cv::Mat decoded_frame;
                if (pixel_format_ == V4L2_PIX_FMT_YUYV) {
                    cv::Mat yuyv(height_, width_, CV_8UC2, buffers_[buf.index].start);
                    cv::cvtColor(yuyv, decoded_frame, cv::COLOR_YUV2BGR_YUYV);
                } else if (pixel_format_ == V4L2_PIX_FMT_MJPEG) {
#ifdef HAS_TURBOJPEG
                    if (tj_handle_) {
                        decoded_frame.create(height_, width_, CV_8UC3);
                        tjDecompress2(tj_handle_, (unsigned char*)buffers_[buf.index].start, buf.bytesused,
                                      decoded_frame.data, width_, 0, height_, TJPF_BGR, TJFLAG_FASTDCT);
                    } else {
#endif
                        std::vector<char> vec((char*)buffers_[buf.index].start, ((char*)buffers_[buf.index].start) + buf.bytesused);
                        decoded_frame = cv::imdecode(vec, cv::IMREAD_COLOR);
#ifdef HAS_TURBOJPEG
                    }
#endif
                }

                if (!decoded_frame.empty()) {
                    {
                        std::lock_guard<std::mutex> lock(frame_mutex_);
                        current_raw_frame_ = decoded_frame;
                        new_frame_for_stream_ = true;
                        new_frame_for_infer_ = true;
                    }
                    stream_cv_.notify_one();
                    infer_cv_.notify_one();
                }

                ioctl(fd_, VIDIOC_QBUF, &buf);
            }
        }
    }

    void inference_loop() {
        while (running_ && rclcpp::ok()) {
            cv::Mat frame;
            {
                std::unique_lock<std::mutex> lock(frame_mutex_);
                infer_cv_.wait(lock, [this]{ return new_frame_for_infer_ || !running_; });
                if (!running_) break;
                frame = current_raw_frame_.clone();
                new_frame_for_infer_ = false;
            }

            if (net_.empty() || frame.empty()) continue;

            try {
                // YOLOv8/v11 Preprocessing: Square blob with letterbox-style resize
                cv::Mat blob = cv::dnn::blobFromImage(frame, 1/255.0, cv::Size(640, 640), cv::Scalar(), true, false);
                net_.setInput(blob);
                std::vector<cv::Mat> outputs;
                net_.forward(outputs, net_.getUnconnectedOutLayersNames());

                std::vector<Detection> detections;
                if (!outputs.empty()) {
                    cv::Mat out = outputs[0];
                    RCLCPP_INFO(this->get_logger(), "Inference output dims: %d, size: %d x %d x %d", 
                                out.dims, out.size[0], out.size[1], out.size.dims() > 2 ? out.size[2] : 0);
                    
                    if (out.dims == 3 && out.size[1] == 84) {
                        int dims = out.size[1];
                        int rows = out.size[2];
                        
                        float x_scale = (float)width_ / 640.0;
                        float y_scale = (float)height_ / 640.0;

                        for (int i = 0; i < rows; ++i) {
                            // Direct indexing to avoid transpose crash
                            float xc = out.at<float>(0, 0, i);
                            float yc = out.at<float>(0, 1, i);
                            float w = out.at<float>(0, 2, i);
                            float h = out.at<float>(0, 3, i);

                            float max_conf = 0;
                            int class_id = -1;
                            for (int j = 4; j < dims; ++j) {
                                float conf = out.at<float>(0, j, i);
                                if (conf > max_conf) {
                                    max_conf = conf;
                                    class_id = j - 4;
                                }
                            }

                            if (max_conf >= conf_threshold_) {
                                float xc = out.at<float>(0, 0, i);
                                float yc = out.at<float>(0, 1, i);
                                float w = out.at<float>(0, 2, i);
                                float h = out.at<float>(0, 3, i);

                                int left = (xc - w/2) * x_scale;
                                int top = (yc - h/2) * y_scale;
                                int width = w * x_scale;
                                int height = h * y_scale;

                                Detection d;
                                d.box = cv::Rect(left, top, width, height);
                                d.confidence = max_conf;
                                d.label = (class_id == 0) ? "person" : "object";
                                detections.push_back(d);

                                geometry_msgs::msg::Point p;
                                p.x = xc * x_scale;
                                p.y = yc * y_scale;
                                p.z = max_conf;
                                target_pub_->publish(p);
                            }
                        }
                    }
                }

                {
                    std::lock_guard<std::mutex> lock(boxes_mutex_);
                    latest_detections_ = detections;
                }
            } catch (const cv::Exception& e) {
                RCLCPP_ERROR(this->get_logger(), "OpenCV Inference Error: %s", e.what());
            }
        }
    }

    void streaming_loop() {
        std::string ffmpeg_cmd = "ffmpeg -y -f rawvideo -pixel_format bgr24 -video_size " + 
                                 std::to_string(width_) + "x" + std::to_string(height_) + 
                                 " -framerate " + std::to_string(fps_) + 
                                 " -i - -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p" + 
                                 " -b:v 2000k -x264-params \"repeat-headers=1:rc-lookahead=0:keyint=60:no-scenecut=1:bframes=0\"" + 
                                 " -f rtp " + rtp_url_ + " 2>/dev/null";

        RCLCPP_INFO(this->get_logger(), "Starting FFmpeg stream: %s", ffmpeg_cmd.c_str());
        FILE* ffmpeg_pipe = popen(ffmpeg_cmd.c_str(), "w");
        
        if (!ffmpeg_pipe) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open FFmpeg pipe!");
            return;
        }

        while (running_ && rclcpp::ok()) {
            cv::Mat frame;
            {
                std::unique_lock<std::mutex> lock(frame_mutex_);
                stream_cv_.wait(lock, [this]{ return new_frame_for_stream_ || !running_; });
                if (!running_) break;
                frame = current_raw_frame_.clone();
                new_frame_for_stream_ = false;
            }

            // Draw bounding boxes on the streaming frame
            std::vector<Detection> dets_to_draw;
            {
                std::lock_guard<std::mutex> lock(boxes_mutex_);
                dets_to_draw = latest_detections_;
            }

            for (const auto& det : dets_to_draw) {
                cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
                cv::putText(frame, det.label + " " + std::to_string(det.confidence).substr(0, 4), 
                            cv::Point(det.box.x, std::max(0, det.box.y - 10)), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            }

            // Publish annotated frame to ROS for debug
            auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
            msg->header.stamp = this->now();
            msg->header.frame_id = "camera_link";
            image_pub_->publish(*msg);

            // Write raw BGR bytes directly to FFmpeg
            size_t written = fwrite(frame.data, 1, frame.total() * frame.elemSize(), ffmpeg_pipe);
            if (written != frame.total() * frame.elemSize()) {
                RCLCPP_WARN(this->get_logger(), "FFmpeg pipe write incomplete or broken.");
            }
            fflush(ffmpeg_pipe);
        }

        if (ffmpeg_pipe) pclose(ffmpeg_pipe);
    }

    std::string device_name_;
    int width_, height_, fps_;
    uint32_t pixel_format_;
    double conf_threshold_;
    std::string rtp_url_;
    int fd_;
    std::vector<Buffer> buffers_;
#ifdef HAS_TURBOJPEG
    tjhandle tj_handle_ = nullptr;
#endif

    std::thread capture_thread_;
    std::thread inference_thread_;
    std::thread streaming_thread_;
    std::atomic<bool> running_;

    std::mutex frame_mutex_;
    std::condition_variable stream_cv_;
    std::condition_variable infer_cv_;
    cv::Mat current_raw_frame_;
    bool new_frame_for_stream_ = false;
    bool new_frame_for_infer_ = false;

    cv::dnn::Net net_;
    std::mutex boxes_mutex_;
    std::vector<Detection> latest_detections_;

    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr target_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VisionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
