#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>

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
    VisionNode() : Node("vision_node"), env_(ORT_LOGGING_LEVEL_WARNING, "VisionNode") {
        this->declare_parameter("device", "/dev/video0");
        this->declare_parameter("width", 640);
        this->declare_parameter("height", 480);
        this->declare_parameter("fps", 30);
        this->declare_parameter("pixel_format", "MJPEG");
        this->declare_parameter("model_path", "yolo26n.onnx");
        this->declare_parameter("conf_threshold", 0.5);
        this->declare_parameter("rtp_url", "rtp://100.100.209.42:5600");

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
            pixel_format_ = V4L2_PIX_FMT_YUYV;
        }

        std::string model_path = this->get_parameter("model_path").as_string();
        
        // Initialize ONNX Runtime
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(2);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        try {
            session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);
            RCLCPP_INFO(this->get_logger(), "ONNX Runtime Session created successfully: %s", model_path.c_str());
            model_loaded_ = true;
        } catch (const Ort::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load ONNX model via ORT: %s", e.what());
            model_loaded_ = false;
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
    struct Buffer { void* start; size_t length; };

    bool init_v4l2() {
        fd_ = open(device_name_.c_str(), O_RDWR | O_NONBLOCK, 0);
        if (fd_ < 0) return false;
        struct v4l2_format fmt;
        memset(&fmt, 0, sizeof(fmt));
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = width_;
        fmt.fmt.pix.height = height_;
        fmt.fmt.pix.pixelformat = pixel_format_;
        ioctl(fd_, VIDIOC_S_FMT, &fmt);
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
        for (auto& buffer : buffers_) munmap(buffer.start, buffer.length);
        if (fd_ >= 0) close(fd_);
    }

    void capture_loop() {
        struct pollfd pfd;
        pfd.fd = fd_;
        pfd.events = POLLIN;
        while (running_ && rclcpp::ok()) {
            if (poll(&pfd, 1, 1000) <= 0) continue;
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
                { std::lock_guard<std::mutex> lock(frame_mutex_); current_raw_frame_ = decoded_frame; new_frame_for_stream_ = true; new_frame_for_infer_ = true; }
                stream_cv_.notify_one(); infer_cv_.notify_one();
            }
            ioctl(fd_, VIDIOC_QBUF, &buf);
        }
    }

    void inference_loop() {
        // ONNX Runtime allocation
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const char* input_names[] = {"images"};
        const char* output_names[] = {"output0"};
        std::vector<int64_t> input_shape = {1, 3, 640, 640};

        while (running_ && rclcpp::ok()) {
            cv::Mat frame;
            {
                std::unique_lock<std::mutex> lock(frame_mutex_);
                infer_cv_.wait(lock, [this]{ return new_frame_for_infer_ || !running_; });
                if (!running_) break;
                frame = current_raw_frame_.clone();
                new_frame_for_infer_ = false;
            }
            if (!model_loaded_ || frame.empty()) continue;

            try {
                // Preprocessing
                cv::Mat blob = cv::dnn::blobFromImage(frame, 1/255.0, cv::Size(640, 640), cv::Scalar(), true, false);
                
                // Create Input Tensor
                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(), blob.total(), input_shape.data(), input_shape.size());

                // Run Inference
                auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

                std::vector<Detection> detections;
                if (!output_tensors.empty()) {
                    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
                    auto output_type_info = output_tensors.front().GetTensorTypeAndShapeInfo();
                    auto out_shape = output_type_info.GetShape();

                    float x_scale = (float)width_ / 640.0;
                    float y_scale = (float)height_ / 640.0;

                    if (out_shape.size() == 3 && out_shape[1] == 84 && out_shape[2] == 8400) {
                        // Output is 1x84x8400
                        cv::Mat raw_data(84, 8400, CV_32F, floatarr);
                        cv::Mat data = raw_data.t(); // Transpose to 8400x84

                        std::vector<int> class_ids;
                        std::vector<float> confs;
                        std::vector<cv::Rect> boxes;

                        for (int i = 0; i < data.rows; ++i) {
                            float* row = data.ptr<float>(i);
                            cv::Mat scores(1, 80, CV_32F, row + 4);
                            cv::Point class_id_p;
                            double max_s;
                            cv::minMaxLoc(scores, nullptr, &max_s, nullptr, &class_id_p);

                            if (max_s >= conf_threshold_) {
                                float w = row[2] * x_scale;
                                float h = row[3] * y_scale;
                                float left = row[0] * x_scale - w / 2;
                                float top = row[1] * y_scale - h / 2;

                                boxes.push_back(cv::Rect(left, top, w, h));
                                confs.push_back((float)max_s);
                                class_ids.push_back(class_id_p.x);
                            }
                        }

                        // NMS
                        std::vector<int> indices;
                        cv::dnn::NMSBoxes(boxes, confs, conf_threshold_, 0.45f, indices);
                        for (int idx : indices) {
                            Detection d;
                            d.box = boxes[idx];
                            d.confidence = confs[idx];
                            d.label = (class_ids[idx] == 0) ? "person" : "object";
                            detections.push_back(d);

                            geometry_msgs::msg::Point p;
                            p.x = d.box.x + d.box.width / 2.0;
                            p.y = d.box.y + d.box.height / 2.0;
                            p.z = d.confidence;
                            target_pub_->publish(p);
                        }
                    }
                }

                { std::lock_guard<std::mutex> lock(boxes_mutex_); latest_detections_ = detections; }

            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "ORT Inference Error: %s", e.what());
            }
        }
    }

    void streaming_loop() {
        std::string ffmpeg_cmd = "ffmpeg -y -f rawvideo -pixel_format bgr24 -video_size " + std::to_string(width_) + "x" + std::to_string(height_) + " -framerate " + std::to_string(fps_) + " -i - "
    "-c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p "
    "-b:v 1200k -maxrate 1500k -bufsize 500k "
    "-g 15 -x264-params \"slice-max-size=1400:intra-refresh=1:rc-lookahead=0:no-scenecut=1:bframes=0\" "
    "-f rtp " + rtp_url_ + " 2>/dev/null";
        FILE* ffmpeg_pipe = popen(ffmpeg_cmd.c_str(), "w");
        if (!ffmpeg_pipe) return;
        while (running_ && rclcpp::ok()) {
            cv::Mat frame;
            { std::unique_lock<std::mutex> lock(frame_mutex_); stream_cv_.wait(lock, [this]{ return new_frame_for_stream_ || !running_; }); if (!running_) break; frame = current_raw_frame_.clone(); new_frame_for_stream_ = false; }
            std::vector<Detection> dets; { std::lock_guard<std::mutex> lock(boxes_mutex_); dets = latest_detections_; }
            for (const auto& det : dets) {
                cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
                cv::putText(frame, det.label + " " + std::to_string(det.confidence).substr(0, 4), cv::Point(det.box.x, std::max(0, det.box.y - 10)), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            }
            auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
            msg->header.stamp = this->now(); msg->header.frame_id = "camera_link"; image_pub_->publish(*msg);
            fwrite(frame.data, 1, frame.total() * frame.elemSize(), ffmpeg_pipe); fflush(ffmpeg_pipe);
        }
        if (ffmpeg_pipe) pclose(ffmpeg_pipe);
    }

    std::string device_name_, rtp_url_;
    int width_, height_, fps_, fd_;
    uint32_t pixel_format_;
    double conf_threshold_;
    std::vector<Buffer> buffers_;
#ifdef HAS_TURBOJPEG
    tjhandle tj_handle_ = nullptr;
#endif
    std::thread capture_thread_, inference_thread_, streaming_thread_;
    std::atomic<bool> running_;
    std::mutex frame_mutex_, boxes_mutex_;
    std::condition_variable stream_cv_, infer_cv_;
    cv::Mat current_raw_frame_;
    bool new_frame_for_stream_ = false, new_frame_for_infer_ = false;
    
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    bool model_loaded_ = false;

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
